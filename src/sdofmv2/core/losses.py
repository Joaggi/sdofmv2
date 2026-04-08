import torch
import torch.nn.functional as F


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
