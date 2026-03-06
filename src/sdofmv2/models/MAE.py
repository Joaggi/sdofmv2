import os
import h5py
import torch
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch.nn.functional as F
from skimage.measure import block_reduce

from .BaseModule import BaseModule
from ..benchmarks import reconstruction as bench_recon
from ..models import MaskedAutoencoderViT3D
from ..utils import unpatchify, patchify
from sdofmv2.constants import ALL_WAVELENGTHS


class MAE(BaseModule):
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
        # training_step defines the train loop.
        x, timestamps = batch
        loss, x_hat, mask = self.autoencoder(x, mask_ratio=self.masking_ratio)

        # logs
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, timestamps = batch
        x_patchified = patchify(x, self.patch_size, self.tubelet_size)
        loss, x_hat, mask = self.autoencoder(x, mask_ratio=self.masking_ratio)
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
        if mask_ratio is None:
            mask_ratio = self.masking_ratio
        loss, x_hat, mask = self.autoencoder(x, mask_ratio=mask_ratio)
        x_hat = unpatchify(x_hat, self.img_size, self.patch_size, self.tubelet_size)
        return x_hat, mask

    def forward_encoder(self, x, mask_ratio):
        return self.autoencoder.forward_encoder(x, mask_ratio=mask_ratio)

    def on_validation_epoch_end(self):

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
