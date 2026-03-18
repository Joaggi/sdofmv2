import os
import types
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import sunpy.visualization.colormaps as sunpycm
from sunpy.visualization.colormaps import color_tables

import torch
from loguru import logger as lgr_logger
from omegaconf import OmegaConf
from timm.layers import maybe_add_mask

from sdofmv2.core import MAE, SDOMLDataModule


# Attention patching function
def patch_attn_layers(model: MAE) -> List[torch.Tensor]:
    """
    Monkey-patch the attention layers of a MAE model to store attention maps.

    Returns:
        attn_maps: List[Tensor] with shape [B, num_heads, N, N] per block
    """
    attn_maps: List[torch.Tensor] = []

    def patched_forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Force unfused attention
        if getattr(self, "fused_attn", False):
            self.fused_attn = False

        # Compute attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = maybe_add_mask(attn, attn_mask)
        attn = attn.softmax(dim=-1)
        attn_maps.append(attn.detach().cpu())
        attn = self.attn_drop(attn)
        x_out = attn @ v

        x_out = x_out.transpose(1, 2).reshape(B, N, C)
        x_out = self.norm(x_out)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

    # Patch all encoder blocks
    for blk in model.autoencoder.blocks:
        blk.attn.forward = types.MethodType(patched_forward, blk.attn)

    return attn_maps


def patch_id_to_xy(patch_id, patch_size=16, grid=32):
    row = patch_id // grid
    col = patch_id % grid
    x = col * patch_size
    y = row * patch_size
    return x, y


def visualize_head(attn_head, ids_keep, img_size=512, patch=16):
    """Visualizes attention scores by mapping kept patches back to a 2D heatmap.

    This function takes attention weights from a sparse set of 'kept' patches
    (after masking), normalizes them, and projects them back onto the original
    spatial grid. The resulting sparse heatmap is then upsampled to the full
    image resolution.

    Args:
        attn_head (np.ndarray | torch.Tensor): Attention values for kept patches
            only. Shape: (num_kept_patches,).
        ids_keep (np.ndarray | torch.Tensor): Global indices of the kept
            patches. Shape: (num_kept_patches,).
        img_size (int): The spatial resolution (height/width) of the original
            image. Defaults to 512.
        patch (int): The side length of each square patch. Defaults to 16.

    Returns:
        np.ndarray: A 2D heatmap of shape (img_size, img_size). Values are
            normalized between 0 and 1, with 0 assigned to all masked/missing
            patch locations.

    Note:
        The normalization uses a small epsilon ($1e-8$) to prevent division
        by zero if all attention weights are identical.
    """
    grid_size = img_size // patch  # 32
    heatmap = np.zeros((grid_size, grid_size))

    # Normalize attention values
    attn_norm = (attn_head - attn_head.min()) / (
        attn_head.max() - attn_head.min() + 1e-8
    )

    # Place attention values at the correct patch positions
    for score, patch_idx in zip(attn_norm, ids_keep):
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        heatmap[row, col] = score

    # Resize to full image resolution
    heatmap_full = np.repeat(np.repeat(heatmap, patch, axis=0), patch, axis=1)

    return heatmap_full
