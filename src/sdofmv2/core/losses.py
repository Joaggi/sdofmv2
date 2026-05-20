from typing import Literal

import torch
import torch.nn.functional as functional
from einops import rearrange


def _get_group_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute mean loss for a group defined by a mask, safely handling empty groups.

    Args:
        loss: Element-wise loss tensor.
        mask: Boolean mask of same shape as loss (for 2D) or broadcastable.

    Returns:
        Scalar mean loss for the group, or 0.0 if the group is empty.
    """
    return loss[mask].mean() if mask.any() else torch.tensor(0.0, device=loss.device)


def mae_loss(pred, target) -> torch.Tensor:
    """Calculates the mean absolute error between predictions and targets.

    Args:
        pred (torch.Tensor): Predicted values from the model.
        target (torch.Tensor): Ground truth values to compare against.

    Returns:
        torch.Tensor: The calculated mean absolute error as a scalar tensor.
    """
    err = pred - target
    return torch.mean(torch.abs(err))


def vector_aware_loss(pred, target, base_loss) -> torch.Tensor:
    """Calculates a loss that combines magnitude and orientation for vector fields.

    This method computes a base loss (MSE or MAE) and adds a weighted cosine
    similarity term. The cosine similarity component enforces directional
    alignment between the predicted and target vectors.

    Args:
        pred (torch.Tensor): Predicted vector field of shape (B, 3, F, H, W).
        target (torch.Tensor): Ground truth vector field of shape (B, 3, F, H, W).
        base_loss (str): The type of base loss to compute, either "mse" or "mae".

    Returns:
        torch.Tensor: A scalar tensor representing the combined loss.

    Raises:
        ValueError: If the provided base_loss type isn't supported.
    """
    if base_loss == "mse":
        baseloss = ((pred - target) ** 2).mean()
    elif base_loss == "mae":
        baseloss = (torch.abs(pred - target)).mean()
    else:
        raise ValueError(f"Not supported loss type: {base_loss}")

    # preds, target: [B, 3, Frame, H, W]
    cos_sim = 1 - functional.cosine_similarity(pred, target, dim=1).mean()
    loss = baseloss + 0.1 * cos_sim
    return loss


def pixel_weight_loss(
    pred,
    target_norm,
    target,
    base_loss,
    threshold,
    ar_weight_ratio: float,
):
    """
    Args:
        pred (4d tensor): output from model
        target_norm (4d tensor): re-normalized target by norm_pix_loss
        target (4d tensor): normalized target
        base_loss (str): baseline loss function
        threshold (float): threshold for pixels which have strong magnetic field
        ar_weight_ratio (float): weight for the pixesl greater than threshold

    Returns:
        _type_: torch.float
    """

    loss = _get_base_loss(pred, target_norm, base_loss)

    # Create masks for high magnetic field (ar) and low/no field (noise) regions
    is_ar_region = torch.abs(target) > threshold
    is_noise_region = ~is_ar_region

    mean_ar = _get_group_mean(loss, is_ar_region)
    mean_noise = _get_group_mean(loss, is_noise_region)

    # Normalize weights
    weight_for_ar = ar_weight_ratio / (ar_weight_ratio + 1)
    weight_for_noise = 1 / (ar_weight_ratio + 1)

    return (weight_for_ar * mean_ar) + (weight_for_noise * mean_noise)


def patch_weight_loss(pred, target, loss_dict, mask_hidden, mask_off_limb):
    """Calculates a three-tier weighted reconstruction loss for solar data.

    This function separates patches into three categories (masked inner disk,
    visible inner disk, and off-limb space) and applies independent weights
    to each group's mean loss. This prevents the large population of space
    pixels or masked patches from disproportionately biasing the gradients.

    Args:
        pred (torch.Tensor): Predicted patch values [B, L, D].
        target (torch.Tensor): Ground truth (potentially normalized) patches [B, L, D].
        loss_dict (dict or object): Config object containing:
            * base_loss (dict): Must have 'type' ('mse', 'mae', or 'huber')
              and 'delta' (for huber).
            * weight_on_patches (list[float]): A three-element list:
              [weight_masked_inner, weight_visible_inner, weight_off_limb].
              Example: [0.7, 0.2, 0.1].
        mask_hidden (torch.Tensor): Binary/bool mask from encoder [B, L].
            1 (True) indicates a masked/hidden patch.
        mask_off_limb (torch.Tensor): Binary/bool spatial mask [B, L].
            1 (True) indicates a patch outside the solar disk.

    Returns:
        torch.Tensor: Scalar weighted mean loss.

    Raises:
        ValueError: If an unsupported loss type is provided.
        IndexError: If weight_on_patches does not contain exactly three elements.
    """
    base_loss_type = loss_dict.base_loss.get("type", "mse")
    huber_delta = loss_dict.base_loss.get("delta", 1.0)

    # Extract 3-tier weights
    weights_raw = loss_dict.get("weight_on_patches", [0.7, 0.2, 0.1])
    if len(weights_raw) < 3:
        raise IndexError("weight_on_patches must have 3 elements for the 3-tier strategy.")

    # Base Loss Calculation
    loss = _get_base_loss(pred, target, base_loss_type, huber_delta)

    # Define the three tiers using boolean logic
    # Tier 1: Hidden patches inside the solar disk
    if mask_off_limb is not None:
        is_masked_inner = mask_hidden.bool() & (~mask_off_limb.bool())
        # Tier 2: Visible patches inside the solar disk
        is_visible_inner = (~mask_hidden.bool()) & (~mask_off_limb.bool())
        # Tier 3: All patches outside the solar disk (space)
        is_space = mask_off_limb.bool()
    else:
        raise ValueError("off limb mask is None!")

    mean_masked = _get_group_mean(loss, is_masked_inner)
    mean_visible = _get_group_mean(loss, is_visible_inner)
    mean_space = _get_group_mean(loss, is_space)

    # Apply Normalized Weights
    w_sum = sum(weights_raw)
    w_m, w_v, w_l = [w / w_sum for w in weights_raw]

    final_loss = (w_m * mean_masked) + (w_v * mean_visible) + (w_l * mean_space)

    return final_loss


def _get_base_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    base_type: Literal["mse", "mae", "huber"],
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Compute element-wise base loss between predictions and targets.

    Args:
        pred: Predicted tensor [B, L, D]
        target: Target tensor [B, L, D]
        base_type: Type of loss - "mse", "mae", or "huber"
        huber_delta: Delta parameter for Huber loss

    Returns:
        Element-wise loss tensor of same shape as pred/target
    """
    if base_type == "mse":
        return (pred - target) ** 2
    elif base_type == "mae":
        return torch.abs(pred - target)
    elif base_type == "huber":
        return functional.huber_loss(pred, target, reduction="none", delta=huber_delta)
    else:
        raise ValueError(f"Not supported base loss type: {base_type}")


def _get_zero_pixel_mask_from_target(
    imgs, patch_size=16, tubelet_size=1, corner_size=4, corner_ratio=0.25
):
    b, c, t, h, w = imgs.shape
    p = patch_size
    tub = tubelet_size

    # Calculate corners per frame, channel and batch
    # Input imgs: [b, c, t, h, w]
    corners = torch.cat(
        [
            imgs[:, :, :, :corner_size, :corner_size].reshape(b, c, t, -1),
            imgs[:, :, :, :corner_size, -corner_size:].reshape(b, c, t, -1),
            imgs[:, :, :, -corner_size:, :corner_size].reshape(b, c, t, -1),
            imgs[:, :, :, -corner_size:, -corner_size:].reshape(b, c, t, -1),
        ],
        dim=-1,
    )

    threshold = corner_ratio * corners.mean(dim=-1).unsqueeze(-1).unsqueeze(-1)
    is_zero_pixel = imgs < threshold

    # Rearrange to match patchify: [b, l, d] where d = tub * p * q * c
    is_zero_pixel = rearrange(
        is_zero_pixel,
        "b c (t tub) (h p) (w q) -> b (t h w) (tub p q c)",
        tub=tub,
        p=p,
        q=p,
    )

    return is_zero_pixel


def split_pixel_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    base_type: Literal["mse", "mae", "huber"] = "mse",
    huber_delta: float = 1.0,
    imgs: torch.Tensor | None = None,
    patch_size: int = 16,
    tubelet_size: int = 1,
    corner_size: int = 4,
    corner_ratio: float = 0.25,
) -> torch.Tensor:
    element_loss = _get_base_loss(pred, target, base_type, huber_delta)
    is_zero_pixel = _get_zero_pixel_mask_from_target(
        imgs,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        corner_size=corner_size,
        corner_ratio=corner_ratio,
    )
    is_nonzero_pixel = ~is_zero_pixel

    loss_nonzero = _get_group_mean(element_loss, is_nonzero_pixel)
    loss_zero = _get_group_mean(element_loss, is_zero_pixel)

    total_loss = alpha * loss_nonzero + beta * loss_zero

    return total_loss


def sparse_dense_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    base_type: Literal["mse", "mae", "huber"] = "mse",
    huber_delta: float = 1.0,
    imgs: torch.Tensor | None = None,
    patch_size: int = 16,
    corner_size: int = 4,
    corner_ratio: float = 0.25,
) -> torch.Tensor:
    element_loss = _get_base_loss(pred, target, base_type, huber_delta)
    is_zero_pixel = _get_zero_pixel_mask_from_target(
        imgs, patch_size=patch_size, corner_size=corner_size, corner_ratio=corner_ratio
    )
    is_nonzero_pixel = ~is_zero_pixel

    recon_loss = _get_group_mean(element_loss, is_nonzero_pixel)

    zero_pred = pred[is_zero_pixel]
    embedding_size = (
        (zero_pred**2).sum(dim=-1).mean() / zero_pred.shape[-1]
        if zero_pred.numel() > 0
        else torch.tensor(0.0, device=pred.device)
    )

    total_loss = alpha * recon_loss + beta * embedding_size

    return total_loss


def bright_patch_weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    imgs: torch.Tensor,
    weight_bright: float = 1.0,
    weight_dark: float = 0.1,
    zero_threshold: float = 0.9,
    base_type: Literal["mse", "mae", "huber"] = "mse",
    huber_delta: float = 1.0,
    patch_size: int = 16,
    tubelet_size: int = 1,
    corner_size: int = 4,
    corner_ratio: float = 0.25,
    mask_hidden: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculates weighted loss prioritizing bright patches per channel/frame."""
    element_loss = _get_base_loss(pred, target, base_type, huber_delta)

    # Get pixel-level zero mask in flattened shape [b, l, d]
    is_zero_pixel = _get_zero_pixel_mask_from_target(
        imgs,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        corner_size=corner_size,
        corner_ratio=corner_ratio,
    )

    # D = tub * p * q * C. The channels C are the innermost dimension.
    b, seq_len, d = target.shape
    c = imgs.shape[1]
    d_spatial_temporal = d // c

    # Reshape to separate channel dimension: [b, seq_len, d_spatial_temporal, c]
    # utils.patchify re-arranges to put c as the innermost dim
    is_zero_pixel_reshaped = is_zero_pixel.reshape(b, seq_len, d_spatial_temporal, c)

    # Calculate zero ratio per (patch, channel)
    # Average across the spatial-temporal pixel dimension
    dark_ratio_per_chan = is_zero_pixel_reshaped.float().mean(dim=2)  # [b, seq_len, c]

    # Per-channel, per-patch decision
    is_dark_chan = dark_ratio_per_chan > zero_threshold  # [b, seq_len, c]

    # Expand decision back to [b, seq_len, d_spatial_temporal, c] then back to [b, seq_len, d]
    is_dark_pixel_mask = is_dark_chan.unsqueeze(2).expand(-1, -1, d_spatial_temporal, -1)
    is_dark_pixel_mask = is_dark_pixel_mask.reshape(b, seq_len, d)
    is_bright_pixel_mask = ~is_dark_pixel_mask

    # Restrict loss calculation to hidden/masked patches if applicable
    if mask_hidden is not None:
        # mask_hidden is [b, l], broadcast it over d
        is_masked = mask_hidden.bool().unsqueeze(-1)
        is_dark_pixel_mask = is_dark_pixel_mask & is_masked
        is_bright_pixel_mask = is_bright_pixel_mask & is_masked

    # Compute group means and weighted sum
    loss_dark = _get_group_mean(element_loss, is_dark_pixel_mask)
    loss_bright = _get_group_mean(element_loss, is_bright_pixel_mask)

    # Normalize weights
    w_sum = weight_bright + weight_dark
    w_b = weight_bright / w_sum
    w_d = weight_dark / w_sum

    return (w_b * loss_bright) + (w_d * loss_dark)
