from typing import Union

import torch
import torch.nn.functional as F


def focal_loss_multiclass(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: Union[float, torch.Tensor] = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Multi-class focal loss based on torchvision.ops.sigmoid_focal_loss.

    Args:
        inputs (Tensor[N, C]): Logits for each class.
        targets (Tensor[N]): Class indices (0 ≤ targets < C).
        alpha (float or Tensor[C]): Balance factor(s). Scalar or per-class.
        gamma (float): Modulating factor exponent.
        reduction (str): 'none', 'mean', or 'sum'.

    Returns:
        Tensor: Loss per sample, or reduced loss.
    """

    ce_loss = F.cross_entropy(inputs, targets, reduction="none")
    p_t = torch.exp(-ce_loss)

    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, device=targets.device, dtype=ce_loss.dtype)

    alpha_t = alpha[targets]
    loss = alpha_t * (1 - p_t) ** gamma * ce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(
            f"Invalid reduction mode: {reduction}. Choose 'none', 'mean', or 'sum'."
        )
