import os

import pandas as pd
import torch
import torch.nn as nn
import torchmetrics.functional as functional
from loguru import logger

from sdofmv2.core import BaseModule


class MultiLayerPerceptron(BaseModule):
    """Multi-layer perceptron head for processing backbone features.

    This class implements a regression or classification head that sits on top of a
    pre-trained backbone. It extracts latent representations from the backbone,
    aggregates patch tokens using either mean and max pooling (with optional disk masking)
    or cross-attention pooling, and processes the features through a series of
    fully connected layers.

    Args:
        backbone (nn.Module): The feature extraction model containing an autoencoder.
        freeze (bool): Whether to freeze the backbone parameters to prevent training.
        input_dim (int): The dimensionality of the backbone's latent features.
            If mean/max pooling is used, the internal MLP input dimension is twice this value.
        output_dim (int, optional): The number of output units. Defaults to 1.
        hidden_layer_dims (list[int], optional): Dimensions of the hidden MLP layers.
            Defaults to [512, 512, 512].
        dropout (float, optional): Dropout probability for regularization.
            Defaults to 0.0.
        mask_ratio (float, optional): Fraction of input patches to mask during
            the forward pass. Defaults to 0.0.
        pooling_type (str, optional): The type of pooling to perform.
            Options are "mean_max", "disk_masked_mean_max", "cross_attention".
            Defaults to "mean_max".
        optimizer_dict (dict, optional): Configuration for the optimizer.
            Defaults to None.
        scheduler_dict (dict, optional): Configuration for the learning rate scheduler.
            Defaults to None.
        test_results_path (str, optional): Path to save test results. Defaults to "./".
        test_results_filename (str, optional): Filename for test results. Defaults to "test_results.csv".
        max_norm (float, optional): Maximum value for normalization. Defaults to 1.0.

    Returns:
        torch.Tensor: The output logits or predictions from the final linear layer.
    """

    def __init__(
        self,
        backbone,
        freeze,
        input_dim,
        output_dim=1,
        hidden_layer_dims=None,
        dropout=0.0,
        mask_ratio=0.0,
        pooling_type="mean_max",
        optimizer_dict=None,
        scheduler_dict=None,
        test_results_path: str = "./",
        test_results_filename: str = "test_results.csv",
        max_norm: float = 1.0,
    ):
        super().__init__(optimizer_dict=optimizer_dict, scheduler_dict=scheduler_dict)
        if hidden_layer_dims is None:
            hidden_layer_dims = [512, 512, 512]
        self.backbone = backbone
        self.freeze_backbone = freeze
        self.test_results_path = test_results_path
        self.test_results_filename = test_results_filename
        self.max_norm = max_norm
        self.test_preds: list[dict] = []
        self.val_preds: list[dict] = []

        if self.freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.mask_ratio = mask_ratio
        self.pooling_type = pooling_type

        # Detect backbone embed dim dynamically
        try:
            backbone_embed_dim = backbone.autoencoder.cls_token.shape[-1]
        except AttributeError:
            # Fallback for backbones that don't have cls_token
            backbone_embed_dim = input_dim // 2 if "mean_max" in self.pooling_type else input_dim

        if self.pooling_type == "cross_attention":
            feature_dim = backbone_embed_dim
            self.query = nn.Parameter(torch.zeros(1, 1, backbone_embed_dim))
            nn.init.normal_(self.query, std=0.02)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=backbone_embed_dim, num_heads=8, batch_first=True
            )
        else:
            feature_dim = backbone_embed_dim * 2

        self.norm = nn.LayerNorm(feature_dim)

        # Define the dimensions of the MLP layers
        dims = [feature_dim] + hidden_layer_dims

        # Define the dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Define the fully connected layers
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

        # Define the activation function
        self.acts = nn.ModuleList([nn.LeakyReLU(0.01) for _ in range(len(dims) - 1)])

        # Define the output layer
        self.fc_out = nn.Linear(dims[-1], output_dim)

        # Define the loss function
        self.criterion = nn.MSELoss()

    def forward(self, x):
        """Processes input through the backbone and MLP head.

        Args:
            x (torch.Tensor): Input image or data tensor.

        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_dim).
        """
        if self.freeze_backbone:
            with torch.no_grad():
                # latent shape: [Batch, Num_Patches + 1, Hidden_Dim]
                latent, mask, ids_restore = self.backbone.autoencoder.forward_encoder(
                    x, mask_ratio=self.mask_ratio
                )
        else:
            latent, mask, ids_restore = self.backbone.autoencoder.forward_encoder(
                x, mask_ratio=self.mask_ratio
            )

        patch_tokens = latent[:, 1:, :]

        if self.pooling_type == "disk_masked_mean_max":
            mask_buffer = getattr(self.backbone.autoencoder, "patch_off_limb_mask", None)
            if mask_buffer is not None and isinstance(mask_buffer, torch.Tensor):
                # mask_buffer is True for off-limb patches
                on_disk_weight = (~mask_buffer).float().unsqueeze(0).unsqueeze(-1)  # Shape: [1, L, 1]
                masked_tokens = patch_tokens * on_disk_weight
                num_on_disk = (~mask_buffer).sum().clamp(min=1)
                x_avg = masked_tokens.sum(dim=1) / num_on_disk

                # Fill off-limb tokens with large negative values for max pooling
                max_tokens = patch_tokens.masked_fill(
                    mask_buffer.unsqueeze(0).unsqueeze(-1), float("-inf")
                )
                x_max = max_tokens.max(dim=1).values
                x_cls = torch.cat([x_avg, x_max], dim=-1)
            else:
                x_avg = patch_tokens.mean(dim=1)
                x_max = patch_tokens.max(dim=1).values
                x_cls = torch.cat([x_avg, x_max], dim=-1)
        elif self.pooling_type == "cross_attention":
            q = self.query.expand(patch_tokens.shape[0], -1, -1)
            # mask_buffer = getattr(self.backbone.autoencoder, "patch_off_limb_mask", None)
            # if mask_buffer is not None and isinstance(mask_buffer, torch.Tensor):
            #     # key_padding_mask expects a boolean tensor of shape [B, L] where True means ignored (masked out)
            #     key_padding_mask = mask_buffer.unsqueeze(0).expand(patch_tokens.shape[0], -1)
            #     attn_out, _ = self.cross_attn(
            #         q, patch_tokens, patch_tokens, key_padding_mask=key_padding_mask
            #     )
            # else:
            attn_out, _ = self.cross_attn(q, patch_tokens, patch_tokens)
            x_cls = attn_out.squeeze(1)
        else:
            # Default mean_max
            x_avg = patch_tokens.mean(dim=1)
            x_max = patch_tokens.max(dim=1).values
            x_cls = torch.cat([x_avg, x_max], dim=-1)

        x_cls = self.norm(x_cls)
        for fc, act in zip(self.fcs, self.acts, strict=True):
            x_cls = self.dropout(x_cls)
            x_cls = fc(x_cls)
            x_cls = act(x_cls)

        logits = self.fc_out(x_cls)

        return logits

    def on_train_start(self):
        if self.freeze_backbone:
            self.backbone.eval()

    def training_step(self, batch, batch_idx):
        # Training step
        imgs, timestamps, y = batch
        logits = self(imgs).squeeze(-1)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step
        imgs, timestamps, y = batch
        logits = self(imgs).squeeze(-1)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        preds_real = logits.detach().cpu().numpy()
        labels_real = y.cpu().numpy()

        for label, pred in zip(labels_real, preds_real, strict=True):
            self.val_preds.append({"label": label.item(), "prediction": pred.item()})
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
        # Test step
        imgs, timestamps, y = batch
        logits = self(imgs).squeeze(-1)
        loss = self.criterion(logits, y)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)

        preds_real = logits.detach().cpu().numpy()
        labels_real = y.cpu().numpy()

        # Save results per timestamp
        for t, label, pred in zip(timestamps, labels_real, preds_real, strict=True):
            self.test_preds.append(
                {"timestamp": t.item(), "label": label.item(), "prediction": pred.item()}
            )
        return loss

    def on_test_epoch_end(self):
        # Save results
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

    def on_before_optimizer_step(self, optimizer):
        # Compute the norm of the gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # Check if gradients are exploding or NaN
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print("SKIPPING STEP: Gradients are NaN/Inf! Weights saved from corruption.")

            # Only unscale if a scaler actually exists (i.e., if using fp16)
            scaler = getattr(self.trainer, "scaler", None)
            if scaler is not None:
                scaler.unscale_(optimizer)

            optimizer.zero_grad()  # Clear the bad gradients (Don't update weights!)
            return
