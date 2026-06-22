import lightning.pytorch as pl
import numpy as np
from typing import Any
import pandas as pd
import torch

from sdofmv2.core.basemodule import BaseModule
from sdofmv2.core.mae3d import MaskedAutoencoderViT3D
from sdofmv2.core.reconstruction import compute_metrics_pytorch
from sdofmv2.utils import spatial_to_patch_mask, unpatchify
from sdofmv2.utils.constants import ALL_WAVELENGTHS


class MAE(BaseModule):
    """Masked Autoencoder (MAE) for 3D/Spatiotemporal data reconstruction.

    This module implements a Vision Transformer-based autoencoder that learns
    representations by reconstructing masked patches of volumetric data. It
    supports custom ROI masking (limb masking) and automated metric tracking
    across training, validation, and testing phases.

    Args:
        img_size: Side length of the input image (assumed square).
        chan_types: List of channel names/wavelengths for logging.
        patch_size: Spatial size of the 2D patches.
        num_frames: Total number of frames (temporal depth) in the input sequence.
        tubelet_size: Temporal size of the 3D tubelets.
        in_chans: Number of input data channels.
        embed_dim: Embedding dimension for the encoder.
        depth: Number of transformer layers in the encoder.
        num_heads: Number of attention heads in the encoder.
        decoder_embed_dim: Embedding dimension for the decoder.
        decoder_depth: Number of transformer layers in the decoder.
        decoder_num_heads: Number of attention heads in the decoder.
        mlp_ratio: Expansion ratio for the MLP hidden dimension.
        norm_layer: Type of normalization layer to use (e.g., "LayerNorm").
        masking_ratio: Fraction of patches to mask (0.0 to 1.0).
        limb_mask: An optional binary ROI mask.
        loss_dict: Configuration for reconstruction losses.
        optimizer_dict: Configuration for the optimizer.
        scheduler_dict: Configuration for the learning rate scheduler.
        noise: Percentage of added noise (by default 0)
        *args: Variable length argument list passed to BaseModule.
        **kwargs: Arbitrary keyword arguments passed to BaseModule.

    Attributes:
        img_size (int): Spatial resolution of the input images (Height and Width).
        patch_size (int): The side length of the square patches extracted from
            each frame.
        tubelet_size (int): The temporal depth of each 3D patch (number of frames).
        masking_ratio (float): The fraction of patches to be masked out during
            the forward pass (typically 0.75).
        chan_types (list[str]): A list of identifiers for each input channel
            (e.g., specific wavelengths), used for per-channel metric logging.
        limb_mask (Optional[torch.Tensor]): A binary spatial mask of shape
            (H, W) used to restrict the model's focus to specific ROIs.
        loss_dict (dict): Configuration parameters and weights for the
            reconstruction loss functions.
        validation_metrics (list[dict]): A transient buffer that accumulates
            metric dictionaries from each `validation_step` to be processed
            at the epoch end.
        test_results (list[dict]): A transient buffer that accumulates metric
            dictionaries from each `test_step`.
        autoencoder (MaskedAutoencoderViT3D): The core transformer architecture
            consisting of the encoder and decoder blocks.
    """

    def __init__(
        self,
        # MAE specific
        img_size=224,
        chan_types=ALL_WAVELENGTHS,
        patch_size=16,
        num_frames=3,
        tubelet_size=1,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer="LayerNorm",
        masking_ratio=0.75,
        mask_only_inner=True,
        limb_mask=None,
        loss_dict=None,
        optimizer_dict=None,
        scheduler_dict=None,
        save_test_results_csv=None,
        noise=0.0,
        # pass to BaseModule
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            optimizer_dict=optimizer_dict if optimizer_dict is not None else {},
            scheduler_dict=scheduler_dict if scheduler_dict is not None else {},
            **kwargs,
        )
        self.save_hyperparameters(ignore=["limb_mask"])
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames
        self.validation_metrics = []
        self.masking_ratio = masking_ratio
        self.mask_only_inner = mask_only_inner
        self.chan_types = chan_types
        self.limb_mask = limb_mask
        self.loss_dict = loss_dict if loss_dict is not None else {}
        self.test_results = []
        self.save_test_results_csv = save_test_results_csv
        self.noise = noise

        # compute limb_mask_ids
        limb_mask_ids = None
        if self.mask_only_inner and limb_mask is not None:
            mask_bool = spatial_to_patch_mask(
                torch.as_tensor(limb_mask), self.patch_size, num_frames
            )
            limb_mask_ids = torch.where(mask_bool)[0]

        # Register circular mask for metrics
        if limb_mask is not None:
            self.register_buffer("disk_mask", torch.as_tensor(limb_mask, dtype=torch.float32))
        else:
            self.register_buffer("disk_mask", torch.ones((img_size, img_size), dtype=torch.float32))

        self.autoencoder = MaskedAutoencoderViT3D(
            img_size,
            patch_size,
            num_frames,
            tubelet_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            decoder_embed_dim,
            decoder_depth,
            decoder_num_heads,
            mlp_ratio,
            norm_layer,
            limb_mask,
            limb_mask_ids,
            loss_dict,
            chan_types=self.chan_types,
            noise=self.noise
        )

    def forward(self, x, mask_ratio=None):
        """Perform a forward pass through the MAE.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).
            mask_ratio (float, optional): Fraction of patches to mask. If None,
                uses the default masking_ratio. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x_hat: Reconstructed images.
                - mask: The applied mask tensor.
        """
        if mask_ratio is None:
            mask_ratio = self.masking_ratio
        loss, x_hat, mask = self.autoencoder(x, mask_ratio=mask_ratio)
        x_hat = unpatchify(x_hat, self.img_size, self.patch_size, self.tubelet_size)
        return x_hat, mask

    def training_step(self, batch, batch_idx):
        """Perform a single training step.

        Args:
            batch: A tuple containing (images, timestamps).
            batch_idx: The index of the current batch.

        Returns:
            torch.Tensor: The training loss value.
        """
        x, timestamps = batch[:2]

        loss, x_hat, mask = self.autoencoder(x, mask_ratio=self.masking_ratio)

        self.log("train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step."""
        x, _ = batch[:2]

        loss, x_hat, mask = self.autoencoder(x, mask_ratio=self.masking_ratio)
        x_hat_reconstructed = unpatchify(x_hat, self.img_size, self.patch_size, self.tubelet_size)

        # Vectorized metrics on GPU
        batch_size, num_channels, num_frames_in, height, width = x.shape
        grid_size = self.img_size // self.patch_size
        num_frames_p = num_frames_in // self.tubelet_size

        # [batch_size, num_frames_p, grid_size, grid_size]
        mask_full = mask.view(batch_size, num_frames_p, grid_size, grid_size)

        # [batch_size, num_frames_in, height, width]
        mask_full = (
            mask_full.repeat_interleave(self.tubelet_size, dim=1)
            .repeat_interleave(self.patch_size, dim=2)
            .repeat_interleave(self.patch_size, dim=3)
            .bool()
        )

        # Intersect with limb_mask if present
        # disk_mask is [height, width]
        if self.limb_mask is not None:
            mask_full = mask_full & (self.disk_mask.bool().unsqueeze(0).unsqueeze(1))

        # Compute metrics
        step_metrics = compute_metrics_pytorch(x, x_hat_reconstructed, mask_full, self.chan_types)

        self.validation_metrics.append(step_metrics)
        self.log("val_loss", loss.detach(), sync_dist=True)

    def on_validation_epoch_end(self):
        """Called at the end of the validation epoch."""
        # Aggregate metrics
        averaged_metrics = {}
        for chan in self.chan_types:
            averaged_metrics[chan] = {}
            for met in [
                "rmse_intensity",
                "flux_difference",
                "ppe10s",
                "ppe50s",
                "r2_score",
                "pixel_correlation",
            ]:
                averaged_metrics[chan][met] = np.mean(
                    [m[chan][met] for m in self.validation_metrics]
                )

        # Logging
        if isinstance(self.logger, pl.loggers.wandb.WandbLogger):
            for chan, metrics in averaged_metrics.items():
                for m_name, val in metrics.items():
                    self.log(f"val_{chan}_{m_name}", val, sync_dist=True)
        else:
            for metrics in averaged_metrics.values():
                self.log_dict(metrics, sync_dist=True)

        self.validation_metrics.clear()

    def test_step(self, batch, batch_idx):
        """Perform a single test step."""
        x, _ = batch[:2]

        loss, x_hat, mask = self.autoencoder(x, mask_ratio=self.masking_ratio)
        x_hat_reconstructed = unpatchify(x_hat, self.img_size, self.patch_size, self.tubelet_size)

        # Vectorized metrics on GPU
        batch_size, num_channels, num_frames_in, height, width = x.shape

        # [batch_size, num_frames_in, height, width]
        mask_full = torch.ones(
                (batch_size, num_frames_in, height, width),
                dtype=torch.bool,
                device=x.device
            )

        # Intersect with limb_mask if present
        if self.limb_mask is not None:
            mask_full = mask_full & (self.disk_mask.bool().unsqueeze(0).unsqueeze(1))

        # Compute metrics
        step_metrics_norm = compute_metrics_pytorch(x, x_hat_reconstructed, mask_full, self.chan_types)

        step_metrics = {}
        for c in self.chan_types:
            step_metrics[c] = {}
            for met, val in step_metrics_norm[c].items():
                step_metrics[c][f"{met}_norm"] = val

        # Determine if we have datamodule and normalization stats
        dm = getattr(self.trainer, "datamodule", None)

        has_norm = False
        try:
            if dm is not None and getattr(dm, "normalization", None) is not None and getattr(dm, "normalization_stat", None) is not None and getattr(dm.normalization, "enabled", False):
                has_norm = True
        except Exception:
            pass

        if has_norm:
            try:
                norm_type = dm.normalization.type
                scaler_factor = getattr(dm.normalization, "scaler_factor", None)
                norm_bool = getattr(dm.normalization, "norm", True)

                x_unnorm = torch.zeros_like(x)
                x_hat_unnorm = torch.zeros_like(x_hat_reconstructed)

                for c_idx, chan in enumerate(self.chan_types):
                    mean = dm.normalization_stat[chan]["mean"]
                    std = dm.normalization_stat[chan]["std"]

                    if norm_type == "log":
                        x_log = (x[:, c_idx] * (std + 1e-8)) + mean if norm_bool else x[:, c_idx]
                        x_orig = torch.sign(x_log) * torch.expm1(torch.abs(x_log))
                        if scaler_factor is not None:
                            x_orig = x_orig / scaler_factor
                        x_unnorm[:, c_idx] = x_orig

                        x_hat_log = (x_hat_reconstructed[:, c_idx] * (std + 1e-8)) + mean if norm_bool else x_hat_reconstructed[:, c_idx]
                        x_hat_orig = torch.sign(x_hat_log) * torch.expm1(torch.abs(x_hat_log))
                        if scaler_factor is not None:
                            x_hat_orig = x_hat_orig / scaler_factor
                        x_hat_unnorm[:, c_idx] = x_hat_orig

                    elif norm_type == "zscore":
                        x_unnorm[:, c_idx] = x[:, c_idx] * std + mean
                        x_hat_unnorm[:, c_idx] = x_hat_reconstructed[:, c_idx] * std + mean

                    elif norm_type == "min-max":
                        min_val = dm.normalization_stat[chan]["min"]
                        max_val = dm.normalization_stat[chan]["max"]
                        diff = max_val - min_val
                        x_unnorm[:, c_idx] = x[:, c_idx] * diff + min_val
                        x_hat_unnorm[:, c_idx] = x_hat_reconstructed[:, c_idx] * diff + min_val
                    else:
                        x_unnorm[:, c_idx] = x[:, c_idx]
                        x_hat_unnorm[:, c_idx] = x_hat_reconstructed[:, c_idx]

                step_metrics_unnorm = compute_metrics_pytorch(x_unnorm, x_hat_unnorm, mask_full, self.chan_types)
                for c in self.chan_types:
                    for met, val in step_metrics_unnorm[c].items():
                        step_metrics[c][f"{met}_unnorm"] = val
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to unnormalize data: {e}", stacklevel=2)
                for c in self.chan_types:
                    for met, val in step_metrics_norm[c].items():
                        step_metrics[c][f"{met}_unnorm"] = val
        else:
            for c in self.chan_types:
                for met, val in step_metrics_norm[c].items():
                    step_metrics[c][f"{met}_unnorm"] = val

        self.test_results.append(step_metrics)
        self.log("test_loss", loss.detach(), sync_dist=True)

    def on_test_epoch_end(self):
        """Called at the end of the test epoch."""
        # Average metrics across samples
        averaged_metrics = {}
        metrics_names = [
            "mse_norm",
            "rmse_intensity_norm",
            "mae_norm",
            "r2_score_norm",
            "pixel_correlation_norm",
            "mse_unnorm",
            "rmse_intensity_unnorm",
            "mae_unnorm",
            "r2_score_unnorm",
            "pixel_correlation_unnorm",
        ]
        for chan in self.chan_types:
            averaged_metrics[chan] = {}
            for met in metrics_names:
                averaged_metrics[chan][met] = np.mean([m[chan][met] for m in self.test_results])

        # Save metrics to CSV
        if getattr(self.trainer, "global_rank", 0) == 0:
            df = pd.DataFrame.from_dict(averaged_metrics, orient="index")
            df.index.name = "channel"
            df.to_csv(self.save_test_results_csv)

        # Logging to WandB if enabled, otherwise just log to trainer
        if isinstance(self.logger, pl.loggers.wandb.WandbLogger):
            for chan, metrics in averaged_metrics.items():
                for m_name, val in metrics.items():
                    self.log(f"test_{chan}_{m_name}", val, sync_dist=True)
        else:
            for metrics in averaged_metrics.values():
                self.log_dict(metrics, sync_dist=True)

        self.test_results.clear()

    def predict_step(self, batch: tuple[Any, ...], batch_idx: int, dataloader_idx: int = 0) -> dict[str, torch.Tensor | np.ndarray]:
        """Extract unmasked embeddings and timestamps.

        Args:
            batch: A tuple containing (images, timestamps).
            batch_idx: The index of the current batch.
            dataloader_idx: The index of the dataloader.

        Returns:
            dict[str, torch.Tensor | np.ndarray]: Dictionary containing:
                - "embeddings": The extracted unmasked embeddings tensor.
                - "timestamps": Numpy array of string timestamps.
        """
        x, timestamps = batch[:2]

        embeddings, _, _ = self.autoencoder.forward_encoder(x, mask_ratio=0.0)

        if isinstance(timestamps, torch.Tensor):
            ts_list = [str(t) for t in timestamps.detach().cpu().tolist()]
        else:
            ts_list = [str(t) for t in timestamps]

        timestamps_np = np.array(ts_list, dtype="<U64")

        return {"embeddings": embeddings, "timestamps": timestamps_np}
