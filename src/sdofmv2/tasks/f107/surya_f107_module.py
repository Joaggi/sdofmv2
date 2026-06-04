import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as functional
import lightning.pytorch as pl
from loguru import logger
from terratorch_surya.downstream_examples.ar_segmentation.models import HelioSpectformer1D
from omegaconf import DictConfig


class SuryaF107Model(pl.LightningModule):
    def __init__(self, config: DictConfig, max_norm):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Configure model
        model_config = {
            "model": {
                "global_average_pooling": config.model.get("global_average_pooling", False),
                "global_max_pooling": config.model.get("global_max_pooling", False),
                "attention_pooling": config.model.get("attention_pooling", False),
                "transformer_pooling": config.model.get("transformer_pooling", True),
                "dropout": config.model.get("dropout", 0.1),
                "penultimate_linear_layer": config.model.get("penultimate_linear_layer", True),
            }
        }

        self.model = HelioSpectformer1D(
            img_size=config.backbone.img_size,
            patch_size=config.backbone.patch_size,
            in_chans=len(config.data.channels),
            embed_dim=config.backbone.embed_dim,
            time_embedding=dict(config.backbone.time_embedding),
            depth=config.backbone.depth,
            n_spectral_blocks=config.backbone.n_spectral_blocks,
            num_heads=config.backbone.num_heads,
            mlp_ratio=config.backbone.mlp_ratio,
            drop_rate=config.backbone.drop_rate,
            window_size=config.backbone.window_size,
            dp_rank=config.backbone.dp_rank,
            num_outputs=1,
            finetune=True,
            config=model_config,
        )

        # Load weights
        if config.backbone.path_weights:
            checkpoint = torch.load(
                config.backbone.path_weights, map_location="cpu", weights_only=True
            )
            self.model.load_state_dict(checkpoint, strict=False)

        # Metrics
        self.val_preds: list[dict] = []
        self.test_preds: list[dict] = []
        self.test_results_path = config.etc.output_dir
        self.test_results_filename = config.etc.test_results_filename
        self.max_norm = self.max_norm
        self.criterion = nn.MSELoss()

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        data_dict, target = batch
        pred = self(data_dict).squeeze()
        loss = self.criterion(pred.view(-1), target.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data_dict, target = batch
        pred = self(data_dict).squeeze()
        loss = self.criterion(pred.view(-1), target.view(-1))
        self.log("val_loss", loss, prog_bar=True)

        preds_real = pred.detach().cpu().float().numpy().flatten()
        labels_real = target.detach().cpu().float().numpy().flatten()

        for label, p in zip(labels_real, preds_real, strict=True):
            self.val_preds.append({"label": label.item(), "prediction": p.item()})
        return loss

    def on_validation_epoch_end(self):
        if self.val_preds:
            df = pd.DataFrame(self.val_preds)
            labels = torch.tensor(df["label"].values)
            preds = torch.tensor(df["prediction"].values)

            r2 = functional.r2_score(preds, labels)
            rmse = functional.mean_squared_error(preds, labels, squared=False)
            mae = functional.mean_absolute_error(preds, labels)
            mse = functional.mean_squared_error(preds, labels)

            # Inverse normalize for reporting
            preds_denorm = preds * self.max_norm
            labels_denorm = labels * self.max_norm

            r2_denorm = functional.r2_score(preds_denorm, labels_denorm)
            rmse_denorm = functional.mean_squared_error(preds_denorm, labels_denorm, squared=False)
            mae_denorm = functional.mean_absolute_error(preds_denorm, labels_denorm)
            mse_denorm = functional.mean_squared_error(preds_denorm, labels_denorm)

            self.log_dict(
                {
                    "val_r2": r2,
                    "val_rmse": rmse,
                    "val_mae": mae,
                    "val_mse": mse,
                    "val_r2_denorm": r2_denorm,
                    "val_rmse_denorm": rmse_denorm,
                    "val_mae_denorm": mae_denorm,
                    "val_mse_denorm": mse_denorm,
                },
                prog_bar=True,
                sync_dist=True,
            )
            self.val_preds.clear()

    def test_step(self, batch, batch_idx):
        data_dict, target = batch
        pred = self(data_dict).squeeze()
        loss = self.criterion(pred.view(-1), target.view(-1))
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)

        preds_real = pred.detach().cpu().float().numpy().flatten()
        labels_real = target.detach().cpu().float().numpy().flatten()

        for label, p in zip(labels_real, preds_real, strict=True):
            self.test_preds.append({"label": label.item(), "prediction": p.item()})
        return loss

    def on_test_epoch_end(self):
        if self.test_preds:
            df = pd.DataFrame(self.test_preds)
            labels = torch.tensor(df["label"].values)
            preds = torch.tensor(df["prediction"].values)

            r2 = functional.r2_score(preds, labels)
            rmse = functional.mean_squared_error(preds, labels, squared=False)
            mae = functional.mean_absolute_error(preds, labels)
            mse = functional.mean_squared_error(preds, labels)

            # Inverse normalize for reporting
            preds_denorm = preds * self.max_norm
            labels_denorm = labels * self.max_norm

            r2_denorm = functional.r2_score(preds_denorm, labels_denorm)
            rmse_denorm = functional.mean_squared_error(preds_denorm, labels_denorm, squared=False)
            mae_denorm = functional.mean_absolute_error(preds_denorm, labels_denorm)
            mse_denorm = functional.mean_squared_error(preds_denorm, labels_denorm)

            self.log_dict(
                {
                    "test_r2": r2,
                    "test_rmse": rmse,
                    "test_mae": mae,
                    "test_mse": mse,
                    "test_r2_denorm": r2_denorm,
                    "test_rmse_denorm": rmse_denorm,
                    "test_mae_denorm": mae_denorm,
                    "test_mse_denorm": mse_denorm,
                },
                prog_bar=True,
                sync_dist=True,
            )

            os.makedirs(self.test_results_path, exist_ok=True)
            output_path = os.path.join(self.test_results_path, self.test_results_filename)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved test results to {output_path}")
            self.test_preds.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
        )
