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
    aggregates patch tokens using both mean and max pooling, and processes the
    combined features through a series of fully connected layers.

    Args:
        backbone (nn.Module): The feature extraction model containing an autoencoder.
        freeze (bool): Whether to freeze the backbone parameters to prevent training.
        input_dim (int): The dimensionality of the backbone's latent features.
            The internal MLP input dimension is twice this value due to the
            concatenation of mean and max pooled features.
        output_dim (int, optional): The number of output units. Defaults to 1.
        hidden_layer_dims (list[int], optional): Dimensions of the hidden MLP layers.
            Defaults to [512, 512, 512].
        dropout (float, optional): Dropout probability for regularization.
            Defaults to 0.0.
        mask_ratio (float, optional): Fraction of input patches to mask during
            the forward pass. Defaults to 0.0.
        optimizer_dict (dict, optional): Configuration for the optimizer.
            Defaults to None.
        scheduler_dict (dict, optional): Configuration for the learning rate scheduler.
            Defaults to None.

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
        optimizer_dict=None,
        scheduler_dict=None,
        test_results_path: str = "./",
        test_results_filename: str = "test_results.csv",
    ):
        super().__init__(optimizer_dict=optimizer_dict, scheduler_dict=scheduler_dict)
        if hidden_layer_dims is None:
            hidden_layer_dims = [512, 512, 512]
        self.backbone = backbone
        self.freeze_backbone = freeze
        self.test_results_path = test_results_path
        self.test_results_filename = test_results_filename
        self.test_preds: list[dict] = []
        self.val_preds: list[dict] = []

        if self.freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.mask_ratio = mask_ratio
        self.norm = nn.LayerNorm(int(input_dim * (1 - mask_ratio) * 2))

        # Define the dimensions of the MLP layers
        dims = [input_dim * 2] + hidden_layer_dims

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

        x_avg = patch_tokens.mean(dim=2)
        x_max = patch_tokens.max(dim=2).values
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
            self.val_preds.append(
                {"label": label.item(), "prediction": pred.item()}
            )
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

            self.log_dict(
                {
                    "val_r2": r2,
                    "val_rmse": rmse,
                    "val_mae": mae,
                    "val_mse": mse,
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

            self.log_dict(
                {
                    "test_r2": r2,
                    "test_rmse": rmse,
                    "test_mae": mae,
                    "test_mse": mse,
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
