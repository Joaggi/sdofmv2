import os
import numpy as np
import pandas as pd
from skimage.measure import block_reduce

import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from . import reconstruction as bench_recon
from .mae3d_old import MaskedAutoencoderViT3D_old
from .basemodule import BaseModule
from ..utils import unpatchify, patchify
from sdofmv2.utils.constants import ALL_WAVELENGTHS


class MAE_old(BaseModule):
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
        norm_pix_loss=False,
        masking_ratio=0.75,
        limb_mask=None,
        # pass to BaseModule
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # self.validation_step_outputs = {'x': [], 'x_hat': []}
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.validation_metrics = []
        self.masking_ratio = masking_ratio
        self.chan_types = chan_types
        self.limb_mask = limb_mask
        self.masking_ratio = masking_ratio
        self.test_results = []

        # block reduce limb_mask
        limb_mask_ids = None
        if limb_mask is not None:
            new_matrix = block_reduce(
                limb_mask.numpy(), block_size=(16, 16), func=np.max
            )
            limb_mask_ids = torch.tensor(
                np.argwhere(new_matrix.reshape(1024) == 0).reshape(-1)
            )

        self.autoencoder = MaskedAutoencoderViT3D_old(
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
            norm_pix_loss,
            limb_mask_ids,
        )
        # self.autoencoder = PrithviEncoder(self.mae)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x = batch
        loss, x_hat, mask = self.autoencoder(x, mask_ratio=self.masking_ratio)
        x_hat = unpatchify(x_hat, self.img_size, self.patch_size, self.tubelet_size)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        loss, x_hat, mask = self.autoencoder(x, mask_ratio=self.masking_ratio)
        x_hat = unpatchify(x_hat, self.img_size, self.patch_size, self.tubelet_size)
        loss = F.mse_loss(x_hat, x)
        for i in range(x.shape[0]):
            for frame in range(x.shape[2]):
                self.validation_metrics.append(
                    bench_recon.get_metrics(
                        x[i, :, frame, :, :], x_hat[i, :, frame, :, :], ALL_WAVELENGTHS
                    )
                )

        self.log("val_loss", loss)

    def forward(self, x):
        loss, x_hat, mask = self.autoencoder(x, mask_ratio=self.masking_ratio)
        x_hat = unpatchify(x_hat, self.img_size, self.patch_size, self.tubelet_size)
        return loss, x_hat, mask

    def forward_encoder(self, x, mask_ratio):
        return self.autoencoder.forward_encoder(x, mask_ratio=mask_ratio)

    def on_validation_epoch_end(self):

        merged_metrics = bench_recon.merge_metrics(self.validation_metrics)
        batch_metrics = bench_recon.mean_metrics(merged_metrics)

        if isinstance(self.logger, pl.loggers.wandb.WandbLogger):
            import wandb
            from pandas import DataFrame

            # this only occurs on rank zero only
            df = DataFrame(batch_metrics)
            df["mean"] = df.mean(numeric_only=True, axis=1)
            df["metric"] = df.index
            cols = df.columns.tolist()
            self.logger.log_table(
                key="val_reconstruction",
                dataframe=df[cols[-1:] + cols[:-1]],
                step=self.validation_step,
            )
            for k, v in batch_metrics.items():
                # sync_dist as this tries to include all
                for i, j in v.items():
                    self.log(f"val_{k}_{i}", j)

            # model_artifact = wandb.Artifact("model", type="model")
            # model_artifact.add_reference(f"gs://sdofm-checkpoints/{wandb.run.id}-{wandb.run.name}/model-step{wandb.run.step}.ckpt")
        else:
            for k in batch_metrics.keys():
                batch_metrics[k]["channel"] = k
            for k, v in batch_metrics.items():
                # sync_dist as this tries to include all
                self.log_dict(v, sync_dist=True)  # This doesn't work?

        # reset
        # self.validation_step_outputs['x'].clear()
        # self.validation_step_outputs['x_hat'].clear()
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
