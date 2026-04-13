import os
import h5py
import torch
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch.nn.functional as F
from skimage.measure import block_reduce

from . import reconstruction as bench_recon
from .mae3d import MaskedAutoencoderViT3D
from .basemodule import BaseModule
from ..utils import unpatchify, patchify
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
        limb_mask=None,
        loss_dict={},
        optimizer_dict={},
        scheduler_dict={},
        # pass to BaseModule
        *args,
        **kwargs,
    ):
        super().__init__(
            optimizer_dict=optimizer_dict,
            scheduler_dict=scheduler_dict,
            *args,
            **kwargs,
        )
        self.save_hyperparameters()
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.validation_metrics = []
        self.masking_ratio = masking_ratio
        self.chan_types = chan_types
        self.limb_mask = limb_mask
        self.loss_dict = loss_dict
        self.test_results = []

        # block reduce limb_mask
        limb_mask_ids = None
        if limb_mask is not None:
            new_matrix = block_reduce(
                limb_mask.numpy(),
                block_size=(self.patch_size, self.patch_size),
                func=np.max,
            )
            limb_mask_ids = torch.tensor(
                np.argwhere(
                    new_matrix.reshape((img_size // self.patch_size) ** 2) == 0
                ).reshape(-1)
            )

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
        )

    def training_step(self, batch, batch_idx):
        """Perform a single training step.

        Args:
            batch: A tuple containing (images, timestamps, zero_patch_mask).
            batch_idx: The index of the current batch.

        Returns:
            torch.Tensor: The training loss value.
        """
        # training_step defines the train loop.
        batch_len = len(batch)
        if batch_len >= 3:
            x = batch[0]
            zero_patch_mask = batch[-1]  # zero_patch_mask is last element
        else:
            x, timestamps = batch[:2]
            zero_patch_mask = None

        loss, x_hat, mask = self.autoencoder(
            x, mask_ratio=self.masking_ratio, zero_patch_mask=zero_patch_mask
        )

        # logs
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step.

        Args:
            batch: A tuple containing (images, timestamps, zero_patch_mask).
            batch_idx: The index of the current batch.
        """
        batch_len = len(batch)
        if batch_len >= 3:
            x = batch[0]
            zero_patch_mask = batch[-1]
        else:
            x, timestamps = batch[:2]
            zero_patch_mask = None

        x_patchified = patchify(x, self.patch_size, self.tubelet_size)
        loss, x_hat, mask = self.autoencoder(
            x, mask_ratio=self.masking_ratio, zero_patch_mask=zero_patch_mask
        )
        x_hat_reconstructed = unpatchify(
            x_hat, self.img_size, self.patch_size, self.tubelet_size
        )

        # only masked patches
        active_mask = mask == 1

        # 2d mask tensor
        batch_size = mask.shape[0]
        grid_size = 512 // self.patch_size
        # Reshape to grid: [B, Grid, Grid]
        mask_grid = (
            mask.reshape(batch_size, grid_size, grid_size).detach().cpu().numpy()
        )

        # Use np.repeat instead of np.kron for better control over the batch dimension
        # This inflates [B, 32, 32] to [B, 512, 512]
        mask_full = mask_grid.repeat(self.patch_size, axis=1).repeat(
            self.patch_size, axis=2
        )
        mask_full = mask_full.astype(bool)

        for i in range(batch_size):
            # Get the 2D mask for this specific sample in the batch
            current_mask = mask_full[i]  # [512, 512]

            for frame in range(x.shape[2]):
                # Extract pixels for all channels simultaneously
                # Resulting shape: [C, Num_Masked_Pixels]
                target_pixels = x[i, :, frame, current_mask]
                pred_pixels = x_hat_reconstructed[i, :, frame, current_mask]

                # Calculate metrics (bench_recon should handle the [C, N] input)
                metrics = bench_recon.get_metrics_for_masked_patches(
                    target_pixels.detach().cpu().numpy(),
                    pred_pixels.detach().cpu().numpy(),
                    self.chan_types,
                )
                self.validation_metrics.append(metrics)

        self.log("val_loss", loss)
        self.log(
            "val_MSEloss_in_masked_patches",
            F.mse_loss(x_patchified[active_mask], x_hat[active_mask]),
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

    def forward_encoder(self, x, mask_ratio):
        """Perform a forward pass through the encoder only.

        Args:
            x (torch.Tensor): Input images.
            mask_ratio (float): Fraction of patches to mask.

        Returns:
            torch.Tensor: Encoded features from the encoder.
        """
        return self.autoencoder.forward_encoder(x, mask_ratio=mask_ratio)

    def on_validation_epoch_end(self):
        """Called at the end of the validation epoch.

        Aggregates validation metrics, logs them to the logger (WandB or default),
        and clears the metrics buffer.
        """
        merged_metrics = bench_recon.merge_metrics(self.validation_metrics)
        batch_metrics = bench_recon.mean_metrics(merged_metrics)

        if isinstance(self.logger, pl.loggers.wandb.WandbLogger):
            # this only occurs on rank zero only
            df = pd.DataFrame(batch_metrics)
            df["mean"] = df.mean(numeric_only=True, axis=1)
            df["metric"] = df.index
            cols = df.columns.tolist()
            self.logger.log_table(
                key="val_reconstruction",
                dataframe=df[cols[-1:] + cols[:-1]],
                step=self.validation_step,
            )
            for k, v in batch_metrics.items():
                for i, j in v.items():
                    self.log(f"val_{k}_{i}", j)

        else:
            for k in batch_metrics.keys():
                batch_metrics[k]["channel"] = k
            for k, v in batch_metrics.items():
                self.log_dict(v, sync_dist=True)

        # reset
        self.validation_metrics.clear()

    def test_step(self, batch, batch_idx):
        """Perform a single test step.

        Args:
            batch: A tuple containing (images, timestamps).
            batch_idx: The index of the current batch.
        """
        x, timestamps = batch
        x_patchified = patchify(x, self.patch_size, self.tubelet_size)
        loss, x_hat, mask = self.autoencoder(x, mask_ratio=self.masking_ratio)
        x_hat_reconstructed = unpatchify(
            x_hat, self.img_size, self.patch_size, self.tubelet_size
        )

        active_mask = mask == 1
        batch_size = mask.shape[0]
        grid_size = 512 // self.patch_size
        mask_grid = (
            mask.reshape(batch_size, grid_size, grid_size).detach().cpu().numpy()
        )

        mask_full = mask_grid.repeat(self.patch_size, axis=1).repeat(
            self.patch_size, axis=2
        )
        mask_full = mask_full.astype(bool)

        step_metrics = []
        for i in range(batch_size):
            current_mask = mask_full[i]
            for frame in range(x.shape[2]):
                target_pixels = x[i, :, frame, current_mask].detach().cpu().numpy()
                pred_pixels = (
                    x_hat_reconstructed[i, :, frame, current_mask]
                    .detach()
                    .cpu()
                    .numpy()
                )

                metrics = bench_recon.get_metrics_for_masked_patches(
                    target_pixels, pred_pixels, self.chan_types
                )
                step_metrics.append(metrics)

        masked_mse = F.mse_loss(x_patchified[active_mask], x_hat[active_mask])
        self.log("test_loss", loss)
        self.log("test_MSEloss_in_masked_patches", masked_mse)
        self.test_results.extend(step_metrics)

    def on_test_epoch_end(self):
        """Called at the end of the test epoch.

        Aggregates test metrics, saves them to a CSV file, logs to the logger,
        and clears the results buffer.
        """
        if not self.test_results:
            return

        merged_metrics = bench_recon.merge_metrics(self.test_results)
        batch_metrics = bench_recon.mean_metrics(merged_metrics)

        df = pd.DataFrame(batch_metrics)
        df["mean"] = df.mean(numeric_only=True, axis=1)
        df["metric"] = df.index

        cols = df.columns.tolist()
        final_df = df[[cols[-1]] + cols[:-1]]

        output_path = os.path.join(
            self.trainer.default_root_dir, "test_metrics_summary.csv"
        )
        final_df.to_csv(output_path, index=False)
        print(f"\n[INFO] Test results saved to: {output_path}")

        if isinstance(self.logger, pl.loggers.wandb.WandbLogger):
            self.logger.log_table(key="test_reconstruction_summary", dataframe=final_df)
            for chan, metrics in batch_metrics.items():
                for m_name, val in metrics.items():
                    self.log(f"test_{chan}_{m_name}", val)

        self.test_results.clear()
