import torch
import torch.nn as nn
from sdofmv2.core import BaseModule


class MultiLayerPerceptron(BaseModule):
    def __init__(
        self,
        backbone,
        freeze,
        input_dim,
        output_dim=1,
        hidden_layer_dims=[512, 512, 512],
        dropout=0.0,
        mask_ratio=0.0,
        optimizer_dict=None,
        scheduler_dict=None,
    ):
        super().__init__(optimizer_dict=optimizer_dict, scheduler_dict=scheduler_dict)
        self.backbone = backbone
        self.freeze_backbone = freeze
        self.nans = []

        if self.freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.mask_ratio = mask_ratio
        self.norm = nn.LayerNorm(input_dim * 2)

        # Define the dimensions of the MLP layers
        dims = [input_dim * 2] + hidden_layer_dims

        # Define the dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Define the fully connected layers
        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        # Define the activation function
        self.acts = nn.ModuleList([nn.LeakyReLU(0.01) for _ in range(len(dims) - 1)])

        # Define the output layer
        self.fc_out = nn.Linear(dims[-1], output_dim)

        # Define the loss function
        self.criterion = nn.MSELoss()

        # Initialize a dictionary to store test predictions
        self.test_preds = {}

    def forward(self, x):

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
        # x_cls = patch_tokens.mean(dim=1)

        x_avg = patch_tokens.mean(dim=1)
        x_max = patch_tokens.max(dim=1).values
        x_cls = torch.cat(
            [x_avg, x_max], dim=-1
        )  # (Requires changing input dim of self.fcs[0])

        x_cls = self.norm(x_cls)
        for fc, act in zip(self.fcs, self.acts):
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
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Test step
        imgs, timestamps, y = batch
        logits = self(imgs).squeeze(-1)
        loss = self.criterion(logits, y)
        self.log("test_loss", loss, prog_bar=True)

        preds_real = logits.detach().cpu().numpy()
        labels_real = y.cpu().numpy()

        # Save results per timestamp
        for t, label, pred in zip(timestamps, labels_real, preds_real):

            self.test_preds[t.item()] = [label.item(), pred.item()]
        return loss

    def on_before_optimizer_step(self, optimizer):
        # Compute the norm of the gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # Check if gradients are exploding or NaN
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(
                "SKIPPING STEP: Gradients are NaN/Inf! Weights saved from corruption."
            )

            # Only unscale if a scaler actually exists (i.e., if using fp16)
            if getattr(self.trainer, "scaler", None) is not None:
                self.trainer.scaler.unscale_(optimizer)

            optimizer.zero_grad()  # Clear the bad gradients (Don't update weights!)
            return
