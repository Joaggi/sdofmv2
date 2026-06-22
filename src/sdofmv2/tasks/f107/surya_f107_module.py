import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as functional
import lightning.pytorch as pl
from loguru import logger
from terratorch_surya.models.helio_spectformer import HelioSpectFormer
from omegaconf import DictConfig


class SuryaF107Model(pl.LightningModule):
    def __init__(self, config: DictConfig, max_norm):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.max_norm = max_norm

        # Configure model
        self.backbone = HelioSpectFormer(
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
            finetune=True,
        )

        # Load weights
        if config.backbone.path_weights:
            # Need to be careful loading weights if architecture changed.
            # HelioSpectformer1D was a wrapper, HelioSpectFormer is the backbone.
            # Loading strict=False should help.
            checkpoint = torch.load(
                config.backbone.path_weights, map_location="cpu", weights_only=True
            )
            self.backbone.load_state_dict(checkpoint, strict=False)

        # Simple MLP Head
        hidden_dims = config.model.get("mlp_hidden_layer_dims", [512, 512, 512])
        self.norm = nn.LayerNorm(config.backbone.embed_dim)
        self.dropout = nn.Dropout(p=config.model.get("dropout", 0.1))

        # Build MLP
        layers = []
        in_dim = config.backbone.embed_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(self.dropout)
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)

        # Metrics
        self.val_preds: list[dict] = []
        self.test_preds: list[dict] = []
        self.test_results_path = config.etc.output_dir
        self.test_results_filename = config.etc.test_results_filename
        self.criterion = nn.MSELoss()

    def forward(self, batch):
        tokens = self.backbone(batch)  # (B, L, D)
        # Mean pooling
        pooled = tokens.mean(dim=1)
        # Apply norm before MLP
        x = self.norm(pooled)
        return self.mlp(x)

    def training_step(self, batch, batch_idx):
        data_dict, timestamp, target = batch
        pred = self(data_dict).squeeze()
        loss = self.criterion(pred.view(-1), target.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data_dict, timestamp, target = batch
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
        data_dict, timestamp, target = batch
        pred = self(data_dict).squeeze()
        loss = self.criterion(pred.view(-1), target.view(-1))
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)

        preds_real = pred.detach().cpu().float().numpy().flatten()
        labels_real = target.detach().cpu().float().numpy().flatten()

        for t, label, p in zip(timestamp, labels_real, preds_real, strict=True):
            self.test_preds.append(
                {"timestamp": t, "label": label.item(), "prediction": p.item()}
            )
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
