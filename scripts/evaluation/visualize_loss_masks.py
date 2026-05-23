import torch
import hydra
import matplotlib.pyplot as plt
import os
import numpy as np
from omegaconf import DictConfig
from sdofmv2.core import SDOMLDataModule, MAE
from sdofmv2.utils import unpatchify, patchify, ALL_WAVELENGTHS, ALL_COMPONENTS
from sdofmv2.core.losses import _get_zero_pixel_mask_from_target
from einops import rearrange


@hydra.main(
    config_path="../configs/pretrain/", config_name="pretrain_mae_ALL.yaml", version_base=None
)
def visualize(cfg: DictConfig):
    # Setup Data
    data_module = SDOMLDataModule(
        hmi_path=(
            os.path.join(
                cfg.data.sdoml.base_directory,
                cfg.data.sdoml.sub_directory.hmi,
            )
            if cfg.data.sdoml.sub_directory.hmi
            else None
        ),
        aia_path=(
            os.path.join(
                cfg.data.sdoml.base_directory,
                cfg.data.sdoml.sub_directory.aia,
            )
            if cfg.data.sdoml.sub_directory.aia
            else None
        ),
        eve_path=None,
        components=cfg.data.sdoml.components,
        wavelengths=cfg.data.sdoml.wavelengths,
        ions=cfg.data.sdoml.ions,
        batch_size=cfg.model.misc.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
        multiprocessing_context=cfg.data.multiprocessing_context,
        train_index=cfg.data.train_index,
        val_index=cfg.data.val_index,
        test_index=cfg.data.test_index,
        hmi_mask=cfg.data.hmi_mask,
        num_frames=cfg.model.mae.num_frames,
        drop_frame_dim=cfg.data.drop_frame_dim,
        apply_mask=cfg.data.sdoml.apply_mask,
        precision=cfg.experiment.precision,
        normalization=cfg.data.sdoml.normalization,
        normalization_stat_path=cfg.data.normalization_stat_path,
    )
    data_module.setup()
    dataloader = data_module.test_dataloader()
    batch = next(iter(dataloader))
    imgs = batch[0]  # Assuming batch is [imgs, labels, ...]

    # Dynamically construct chan_types
    aia_list = (
        ALL_WAVELENGTHS
        if cfg.data.sdoml.sub_directory.aia and cfg.data.sdoml.wavelengths is None
        else cfg.data.sdoml.wavelengths or []
    )
    hmi_list = (
        ALL_COMPONENTS
        if cfg.data.sdoml.sub_directory.hmi and cfg.data.sdoml.components is None
        else cfg.data.sdoml.components or []
    )
    aia_list.sort()
    hmi_list.sort()
    chan_types = aia_list + hmi_list

    # Setup Model (to get limb mask and chan_types)
    model = MAE(
        **cfg.model.mae,
        chan_types=chan_types,
        limb_mask=torch.Tensor(np.load(cfg.data.hmi_mask)),
        loss_dict=cfg.model.loss,
    )

    # Masking Logic
    patch_size = cfg.model.mae.patch_size
    tubelet_size = cfg.model.mae.tubelet_size

    # Limb mask is a buffer
    mask_off_limb = model.autoencoder.patch_off_limb_mask.unsqueeze(0).expand(imgs.shape[0], -1)

    # Get pixel-level zero mask [b, l, d]
    is_zero_pixel = _get_zero_pixel_mask_from_target(
        imgs,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )

    b, seq_len, d = is_zero_pixel.shape
    c = imgs.shape[1]
    d_spatial_temporal = d // c
    zero_threshold = cfg.model.loss.bright_patch_weighted_loss.zero_threshold

    is_zero_pixel_reshaped = is_zero_pixel.reshape(b, seq_len, d_spatial_temporal, c)
    dark_ratio_per_chan = is_zero_pixel_reshaped.float().mean(dim=2)
    is_dark_chan = dark_ratio_per_chan > zero_threshold
    print(f"Mean dark_ratio_per_chan: {dark_ratio_per_chan.mean().item():.4f}")
    print(f"Max dark_ratio_per_chan: {dark_ratio_per_chan.max().item():.4f}")
    print(f"Min dark_ratio_per_chan: {dark_ratio_per_chan.min().item():.4f}")
    print(
        f"Number of dark channels (zero_threshold={zero_threshold}): {is_dark_chan.sum().item()} / {is_dark_chan.numel()}"
    )

    # AIA detection (AIA contains 'a' or 'aia' in channel name)
    is_aia = torch.tensor(
        [("a" in str(ch).lower()) for ch in model.chan_types], device=imgs.device
    ).view(1, 1, -1)

    # Region identification
    mask_off_limb_expanded = mask_off_limb.unsqueeze(-1).expand(-1, -1, c)
    is_inner_chan = ~mask_off_limb_expanded
    is_outer_bright_chan = mask_off_limb_expanded & ~is_dark_chan & is_aia
    is_outer_dark_chan = mask_off_limb_expanded & (is_dark_chan | ~is_aia)

    def expand_to_d(mask_chan):
        return mask_chan.unsqueeze(2).expand(-1, -1, d_spatial_temporal, -1).reshape(b, seq_len, d)

    is_inner_pixel_mask = expand_to_d(is_inner_chan)
    is_outer_bright_pixel_mask = expand_to_d(is_outer_bright_chan)
    is_outer_dark_pixel_mask = expand_to_d(is_outer_dark_chan)

    # Visualization (5xc)
    # Convert to pixel space
    h_dim = imgs.shape[3]
    inner_mask_pixel = unpatchify(is_inner_pixel_mask.float(), h_dim, patch_size, tubelet_size)
    outer_bright_mask_pixel = unpatchify(
        is_outer_bright_pixel_mask.float(), h_dim, patch_size, tubelet_size
    )
    outer_dark_mask_pixel = unpatchify(
        is_outer_dark_mask_pixel.float(), h_dim, patch_size, tubelet_size
    )
    # Add raw zero pixel mask calculation
    raw_zero_mask_pixel = unpatchify(is_zero_pixel.float(), h_dim, patch_size, tubelet_size)

    fig, axes = plt.subplots(5, c, figsize=(2 * c, 10))

    for chan in range(c):
        # Row 1: Original
        axes[0, chan].imshow(imgs[0, chan, 0].numpy(), cmap="gray")
        if chan == 0:
            axes[0, chan].set_ylabel("Original")
        axes[0, chan].set_title(model.chan_types[chan])

        # Row 2: Inner
        axes[1, chan].imshow(inner_mask_pixel[0, chan, 0].numpy(), cmap="jet", vmin=0, vmax=1)
        if chan == 0:
            axes[1, chan].set_ylabel("Inner Mask")

        # Row 3: Bright
        axes[2, chan].imshow(
            outer_bright_mask_pixel[0, chan, 0].numpy(), cmap="jet", vmin=0, vmax=1
        )
        if chan == 0:
            axes[2, chan].set_ylabel("Bright Outer")

        # Row 4: Dark
        axes[3, chan].imshow(outer_dark_mask_pixel[0, chan, 0].numpy(), cmap="jet", vmin=0, vmax=1)
        if chan == 0:
            axes[3, chan].set_ylabel("Dark Outer")

        # Row 5: Raw Zero Pixel Mask
        axes[4, chan].imshow(raw_zero_mask_pixel[0, chan, 0].numpy(), cmap="jet", vmin=0, vmax=1)
        if chan == 0:
            axes[4, chan].set_ylabel("Raw Zero Pixels")

    plt.tight_layout()
    plt.savefig("loss_mask_verification.png")
    print("Saved to loss_mask_verification.png")


if __name__ == "__main__":
    visualize()
