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


def attn_to_image(attn_vector, visible_patch_ids, img_size=512, patch_size=16):
    heatmap = np.zeros((img_size, img_size), dtype=np.float32)

    # Normalize for visualization
    attn_norm = attn_vector / attn_vector.max()

    for w, patch_id in zip(attn_norm, visible_patch_ids):
        x, y = patch_id_to_xy(patch_id, patch_size)
        heatmap[y : y + patch_size, x : x + patch_size] = w

    return heatmap


# Load MAE weights
def load_mae_weights(ckpt_path: str, masking_ratio: float = 0.5) -> MAE:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt["hyper_parameters"]

    # if masking_ratio is not None:
    #     hparams["masking_ratio"] = masking_ratio
    #     print(f"Overriding masking_ratio to {masking_ratio}")

    # Clean hyperparameters for model construction
    for key in ["create_embedding_file", "lr", "num_classes"]:
        if key in hparams:
            hparams.pop(key)

    if "wavelengths" in hparams:
        hparams["chan_types"] = hparams.pop("wavelengths")

    model = MAE(**hparams)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model


def visualize_head(attn_head, ids_keep, img_size=512, patch=16):
    """
    attn_head: [num_kept] attention values for ONLY the kept patches
    ids_keep: [num_kept] indices of which patches were kept
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


# def visualize_head(attn_head, ids_keep, img_size=512, patch=16):
#     heatmap = np.zeros((img_size, img_size))

#     # attn_head: [549] attention from one query token → each key token
#     attn_norm = attn_head / (attn_head.max() + 1e-6)

#     for score, patch_id in zip(attn_norm, ids_keep):
#         y, x = divmod(patch_id.item(), 32)
#         y *= patch
#         x *= patch
#         heatmap[y:y+patch, x:x+patch] = score

#     return heatmap


def plot_heads(
    attn_maps, ids_restore, image, channels=["Bx", "By", "Bz"]
):  # , patch_id=16
    attn = attn_maps[0][0][:, 1:, 1:]  # [num_head, num_patch, num_patch]
    num_heads = attn.shape[0]
    full_order = torch.argsort(ids_restore[0])  # Invert argsort
    num_kept = attn.shape[1]
    ids_keep = full_order[:num_kept]

    if len(channels) == 3:
        cmap = [
            sunpycm.cmlist.get("hmimag"),
            sunpycm.cmlist.get("hmimag"),
            sunpycm.cmlist.get("hmimag"),
        ]
        norm = TwoSlopeNorm(vmin=-4000, vcenter=0, vmax=4000)

    elif len(channels) == 9:
        cmap = [
            sunpycm.cmlist.get("sdoaia131"),
            sunpycm.cmlist.get("sdoaia1600"),
            sunpycm.cmlist.get("sdoaia1700"),
            sunpycm.cmlist.get("sdoaia171"),
            sunpycm.cmlist.get("sdoaia193"),
            sunpycm.cmlist.get("sdoaia211"),
            sunpycm.cmlist.get("sdoaia304"),
            sunpycm.cmlist.get("sdoaia335"),
            sunpycm.cmlist.get("sdoaia94"),
        ]
        norm = None

    else:
        raise ValueError(f"Channel info is wrong")

    attn_received = attn.mean(axis=1)
    # attn_received = attn[:, 0, 1:]  # cls token
    num_images = image.shape[0]
    num_channels = image.shape[1]

    fig, axs = plt.subplots(
        num_images, num_heads + num_channels, figsize=(25, 4), squeeze=False
    )

    for i in range(image.shape[0]):
        for i_ch, ch in enumerate(channels):
            axs[i, i_ch].imshow(image[i, i_ch, :, :], cmap=cmap[i_ch], norm=norm)
            axs[i, i_ch].set_title(f"{ch}")
            axs[i, i_ch].axis("off")

        for h in range(num_heads):
            # head_attn = attn_shuffled[h, patch_id, :] # weights based on patches
            head_attn = attn_received[h, :]
            heatmap = visualize_head(head_attn, ids_keep, 512, 16)
            axs[i, h + num_channels].imshow(heatmap, cmap="jet")
            axs[i, h + num_channels].set_title(f"Head {h}")
            axs[i, h + num_channels].axis("off")

    plt.tight_layout()
    # plt.savefig("attention_map_no_limb.png", dpi=200)
    return fig, axs


def plot_heads_no_limb(attn_maps, ids_restore, image, patch_id=16):
    attn = attn_maps[0][0][:, 1:, 1:]  # [num_head, num_patch, num_patch]
    num_heads = attn.shape[0]
    ids_keep = ids_restore[0, : attn.shape[1]]
    attn_shuffled = attn[:, ids_restore[0]][:, :, ids_restore[0]]
    attn_received = attn_shuffled.mean(axis=1)
    num_images = image.shape[0]
    num_channels = image.shape[1]

    fig, axs = plt.subplots(
        num_images, num_heads + num_channels, figsize=(25, 4), squeeze=False
    )

    for i in range(image.shape[0]):
        for i_ch, ch in enumerate(["Bx", "By", "Bz"]):
            axs[i, i_ch].imshow(image[i, i_ch, :, :].to("cpu").numpy(), cmap="gray")
            axs[i, i_ch].set_title(f"Ch: {ch}")
            axs[i, i_ch].axis("off")

        for h in range(num_heads):
            # head_attn = attn_shuffled[h, patch_id, :] # weights based on patches
            head_attn = attn_received[h, :]  # weights based on mean of patches
            heatmap = visualize_head(head_attn, ids_keep, 512, 16)
            axs[i, h + num_channels].imshow(heatmap, cmap="jet")
            axs[i, h + num_channels].set_title(f"Head {h}")
            axs[i, h + num_channels].axis("off")

    plt.tight_layout()
    # plt.savefig("attention_map_no_limb.png", dpi=200)
    return fig


# Main execution
if __name__ == "__main__":
    cfg = OmegaConf.load(
        "/home/jhong36/Project/2025-HL-Solar-Wind/solar_phenomena_prediction/configs/pretrain_mae.yaml"
    )

    # Setup dataset
    data_module = SDOMLDataModule(
        hmi_path=(
            os.path.join(
                cfg.data.sdoml.base_directory, cfg.data.sdoml.sub_directory.hmi
            )
            if cfg.data.sdoml.sub_directory.hmi
            else None
        ),
        aia_path=(
            os.path.join(
                cfg.data.sdoml.base_directory, cfg.data.sdoml.sub_directory.aia
            )
            if cfg.data.sdoml.sub_directory.aia
            else None
        ),
        eve_path=None,
        components=cfg.data.sdoml.components,
        wavelengths=cfg.data.sdoml.wavelengths,
        ions=cfg.data.sdoml.ions,
        frequency=cfg.data.sdoml.frequency,
        batch_size=cfg.model.opt.batch_size,
        num_workers=cfg.data.num_workers,
        val_months=cfg.data.month_splits.val,
        test_months=cfg.data.month_splits.test,
        holdout_months=cfg.data.month_splits.holdout,
        cache_dir=os.path.join(
            cfg.data.sdoml.save_directory, cfg.data.sdoml.sub_directory.cache
        ),
        min_date=cfg.data.min_date,
        max_date=cfg.data.max_date,
        num_frames=cfg.model.mae.num_frames,
        drop_frame_dim=cfg.data.drop_frame_dim,
        apply_mask=cfg.data.sdoml.apply_mask,
        precision=cfg.experiment.precision,
        normalization=cfg.data.sdoml.normalization,
    )
    data_module.setup()

    # Load model
    base_path = "../../../../assets/check_point/backbone/"
    model_hmi = load_mae_weights(
        os.path.join(base_path, "id_xn2c11go_mae_epoch=25-val_loss=0.00.ckpt"),
        # masking_ratio=0  # full image by default
    )
    # model_hmi.autoencoder.ids_limb_mask = None
    model_hmi.eval()

    # Patch attention layers
    attn_maps = patch_attn_layers(model_hmi)

    # Forward pass
    id_input = 0
    x = data_module.test_ds[id_input][0].unsqueeze(0)
    lgr_logger.info(f"Input shape: {x.shape}")

    with torch.no_grad():
        latent, mask, ids_restore = model_hmi.autoencoder.forward_encoder(
            x, mask_ratio=0
        )

    # Example: check first attention map
    if len(attn_maps) > 0:
        lgr_logger.info(f"First attention map shape: {attn_maps[0].shape}")

    fig = plot_heads(attn_maps, ids_restore, x[:, 0:1, 0, :, :])
    # fig = plot_heads_no_limb(attn_maps, ids_restore, x[:, 0:1, 0, :, :])
