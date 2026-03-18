# Modified from: https://github.com/isaaccorley/prithvi-pytorch/blob/main/prithvi_pytorch/model.py
from einops import rearrange
import torch
import torch.nn as nn


class WrapEncoder(nn.Module):
    """A wrapper for Prithvi-style encoders to handle temporal dimensions.

    This class ensures that 4D input tensors (B, C, H, W) are correctly reshaped
    into 5D tensors (B, C, T, H, W) before being passed to the encoder. It also
    manages the extraction of intermediate features.

    Attributes:
        encoder: The underlying encoder module (e.g., a Prithvi ViT).
    """

    def __init__(
        self,
        encoder: nn.Module,
    ):
        """Initializes the WrapEncoder with a specific encoder.

        Args:
            encoder: The encoder instance to wrap.
        """
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the encoder.

        If the input is 4D, a singleton temporal dimension is added. The
        temporal dimension is squeezed from the output before returning.

        Args:
            x: Input tensor of shape (B, C, H, W) or (B, C, T, H, W).

        Returns:
            The encoded features as a torch.Tensor.
        """
        # add a temporal dimension if num_frames = 1
        if x.ndim == 4:
            x = rearrange(x, "b c h w -> b c () h w")

        x, _, _ = self.encoder.forward_encoder(x, mask_ratio=0.0)

        # Squeeze temporal dim if t=1
        x = x.squeeze(dim=2)
        return x

    def forward_features(
        self,
        x: torch.Tensor,
        n: list[int],
        mask_ratio: float = 0.0,
        reshape: bool = True,
        norm: bool = False,
    ) -> list[torch.Tensor]:
        """Extracts intermediate features from specific layers of the encoder.

        Args:
            x: Input tensor of shape (B, C, H, W) or (B, C, T, H, W).
            n: A list of layer indices from which to extract features.
            mask_ratio: The fraction of patches to mask during the forward pass.
            reshape: Whether to reshape the output features into a spatial grid.
            norm: Whether to apply normalization to the extracted features.

        Returns:
            A list of tensors containing the intermediate features from the
            requested layers.
        """
        # add a temporal dimension if num_frames = 1
        if x.ndim == 4:
            x = rearrange(x, "b c h w -> b c () h w")

        x = self.encoder.get_intermediate_layers(
            x, n=n, mask_ratio=mask_ratio, reshape=reshape, norm=norm
        )
        return x
