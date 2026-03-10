import torch
import torch.nn.functional as F


def mae_loss(pred, target) -> torch.Tensor:
    err = pred - target
    return torch.mean(torch.abs(err))


def vector_aware_loss(pred, target, base_loss) -> torch.Tensor:

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
    loss = (loss * weights).mean()

    return loss


def patch_weight_loss(pred, target, loss_dict, filtered_mask):
    base_loss_type = loss_dict.base_loss.get("type", "mse")
    weight_on_patches = loss_dict.get("weight_on_patches", [0.7, 0.3])

    # Base Loss calculation using vectorized operations
    if base_loss_type == "mse":
        loss = (pred - target) ** 2
    elif base_loss_type == "mae":
        loss = torch.abs(pred - target)
    elif base_loss_type == "huber":
        delta = loss_dict.base_loss.get("delta", 1.0)
        loss = F.huber_loss(pred, target, reduction="none", delta=delta)

    # Weight application
    w_sum = sum(weight_on_patches)
    w_m = weight_on_patches[0] / w_sum
    w_v = weight_on_patches[1] / w_sum

    # Ensure weights are on the correct device/dtype
    weights = torch.where(filtered_mask.bool(), w_m, w_v).to(pred.device).unsqueeze(-1)

    return (loss * weights).mean()
