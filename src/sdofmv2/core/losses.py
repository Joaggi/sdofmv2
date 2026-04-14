import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Literal


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
    cos_sim = 1 - F.cosine_similarity(pred, target, dim=1).mean()
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

    if base_loss == "mse":
        loss = (pred - target_norm) ** 2
    elif base_loss == "mae":
        loss = torch.abs(pred - target_norm)
    else:
        raise ValueError(f"Not supported loss type: {base_loss}")

    # Calculate weights. Adding + 0.1 is critical so quiet regions still have some weight.
    # weights = torch.abs(imgs) + 0.1
    # weights = (torch.abs(imgs) ** 2) + 0.1
    weight_for_ar = ar_weight_ratio / (ar_weight_ratio + 1)
    weight_for_noise = 1 / (ar_weight_ratio + 1)
    weights = torch.where(
        torch.abs(target) > threshold, weight_for_ar, weight_for_noise
    )
    return (loss * weights).mean()


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

    # Extract 3-tier weights
    weights_raw = loss_dict.get("weight_on_patches", [0.7, 0.2, 0.1])
    if len(weights_raw) < 3:
        raise IndexError(
            "weight_on_patches must have 3 elements for the 3-tier strategy."
        )

    # Base Loss Calculation
    if base_loss_type == "mse":
        loss = (pred - target) ** 2
    elif base_loss_type == "mae":
        loss = torch.abs(pred - target)
    elif base_loss_type == "huber":
        delta = loss_dict.base_loss.get("delta", 1.0)
        loss = torch.nn.functional.huber_loss(
            pred, target, reduction="none", delta=delta
        )
    else:
        raise ValueError(f"Not supported loss type: {base_loss_type}")

    # Define the three tiers using boolean logic
    # Tier 1: Hidden patches inside the solar disk
    is_masked_inner = mask_hidden.bool() & (~mask_off_limb.bool())
    # Tier 2: Visible patches inside the solar disk
    is_visible_inner = (~mask_hidden.bool()) & (~mask_off_limb.bool())
    # Tier 3: All patches outside the solar disk (space)
    is_space = mask_off_limb.bool()

    # Compute group means safely (avoiding NaN on empty tensors)
    def get_group_mean(l, m):
        # Indexing [B, L, D] with [B, L] mask results in [N_pixels, D]
        return l[m].mean() if m.any() else torch.tensor(0.0, device=l.device)

    mean_masked = get_group_mean(loss, is_masked_inner)
    mean_visible = get_group_mean(loss, is_visible_inner)
    mean_space = get_group_mean(loss, is_space)

    # Apply Normalized Weights
    w_sum = sum(weights_raw)
    w_m, w_v, w_l = [w / w_sum for w in weights_raw]

    final_loss = (w_m * mean_masked) + (w_v * mean_visible) + (w_l * mean_space)

    return final_loss


# =============================================================================
# Patch-level loss functions for non-zero vs all-zero patches
# =============================================================================


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
        return F.huber_loss(pred, target, reduction="none", delta=huber_delta)
    else:
        raise ValueError(f"Not supported base loss type: {base_type}")


def _get_zero_patch_mask_from_target(imgs, patch_size=16, corner_ratio=0.25):
    B, C, T, H, W = imgs.shape
    p = patch_size

    imgs_avg = imgs.mean(dim=2)

    corner_size = 4
    corners = torch.cat(
        [
            imgs_avg[:, :, :corner_size, :corner_size].reshape(B, C, -1),
            imgs_avg[:, :, :corner_size, -corner_size:].reshape(B, C, -1),
            imgs_avg[:, :, -corner_size:, :corner_size].reshape(B, C, -1),
            imgs_avg[:, :, -corner_size:, -corner_size:].reshape(B, C, -1),
        ],
        dim=-1,
    )

    corner_mean = corners.mean(dim=-1)
    threshold = corner_ratio * corner_mean

    threshold_expanded = threshold.unsqueeze(-1).unsqueeze(-1)
    is_zero_pixel = imgs_avg < threshold_expanded

    is_zero_pixel = rearrange(
        is_zero_pixel, "b c (h p) (w q) -> b (h w) (p q c)", p=p, q=p
    )
    is_zero_patch = is_zero_pixel.any(dim=-1)

    return is_zero_patch


def split_patch_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    base_type: Literal["mse", "mae", "huber"] = "mse",
    huber_delta: float = 1.0,
    off_limb_mask: torch.Tensor | None = None,
    use_4corner_detection: bool = False,
    imgs: torch.Tensor | None = None,
    patch_size: int = 16,
    corner_ratio: float = 0.25,
) -> torch.Tensor:
    element_loss = _get_base_loss(pred, target, base_type, huber_delta)

    if off_limb_mask is not None:
        is_zero_patch = off_limb_mask
    elif use_4corner_detection:
        is_zero_patch = _get_zero_patch_mask_from_target(
            imgs, patch_size=patch_size, corner_ratio=corner_ratio
        )
    else:
        patch_mean_abs = target.abs().mean(dim=-1)
        is_zero_patch = patch_mean_abs < 1e-3

    is_nonzero_patch = ~is_zero_patch

    if is_nonzero_patch.any():
        loss_nonzero = element_loss[is_nonzero_patch].mean()
    else:
        loss_nonzero = torch.tensor(0.0, device=pred.device)

    if is_zero_patch.any():
        loss_zero = element_loss[is_zero_patch].mean()
    else:
        loss_zero = torch.tensor(0.0, device=pred.device)

    total_loss = alpha * loss_nonzero + beta * loss_zero

    return total_loss


def sparse_dense_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    base_type: Literal["mse", "mae", "huber"] = "mse",
    huber_delta: float = 1.0,
    off_limb_mask: torch.Tensor | None = None,
    use_4corner_detection: bool = False,
    imgs: torch.Tensor | None = None,
    patch_size: int = 16,
    corner_ratio: float = 0.25,
) -> torch.Tensor:
    element_loss = _get_base_loss(pred, target, base_type, huber_delta)

    if off_limb_mask is not None:
        is_zero_patch = off_limb_mask
    elif use_4corner_detection:
        is_zero_patch = _get_zero_patch_mask_from_target(
            imgs, patch_size=patch_size, corner_ratio=corner_ratio
        )
    else:
        patch_mean_abs = target.abs().mean(dim=-1)
        is_zero_patch = patch_mean_abs < 1e-3

    is_nonzero_patch = ~is_zero_patch

    if is_nonzero_patch.any():
        recon_loss = element_loss[is_nonzero_patch].mean()
    else:
        recon_loss = torch.tensor(0.0, device=pred.device)

    zero_target = target[is_zero_patch]
    if zero_target.numel() > 0:
        embedding_size = (zero_target**2).sum(dim=-1).mean() / target.shape[-1]
    else:
        embedding_size = torch.tensor(0.0, device=pred.device)

    # Combined weighted loss
    total_loss = alpha * recon_loss + beta * embedding_size

    return total_loss
