# Modified from: https://github.com/isaaccorley/prithvi-pytorch/blob/main/prithvi_pytorch/model.py
from einops import rearrange
import torch
import torch.nn as nn


class WrapEncoder(nn.Module):
    def __init__(
        self,
        encoder,
    ):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add a temporal dimension if num_frames = 1
        if x.ndim == 4:
            x = rearrange(x, "b c h w -> b c () h w")

        # print('input shape', x.shape)
        # x, _, _ = self.encoder.forward_encoder(x, mask_ratio=0.0)
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
        norm=False,
    ):
        # add a temporal dimension if num_frames = 1
        if x.ndim == 4:
            x = rearrange(x, "b c h w -> b c () h w")

        x = self.encoder.get_intermediate_layers(
            x, n=n, mask_ratio=mask_ratio, reshape=reshape, norm=norm
        )
        return x
