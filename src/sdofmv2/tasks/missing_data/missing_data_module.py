import torch
import torch.nn.functional as F

from sdofmv2.core import reconstruction as bench_recon, BaseModule
from sdofmv2.utils import unpatchify, ALL_WAVELENGTHS, ALL_COMPONENTS


class MissingDataModel(BaseModule):
    def __init__(
        self,
        # Backbone parameters
        optimizer_dict=None,
        scheduler_dict=None,
        # for finetuning
        backbone: object = None,
        freeze_encoder: bool = True,
        # all else
        *args,
        **kwargs,
    ):
        super().__init__(
            optimizer_dict=optimizer_dict,
            scheduler_dict=scheduler_dict,
            *args,
            **kwargs,
        )

        self.backbone = backbone
        self.masking_ratio = backbone.masking_ratio
        self.validation_metrics = []

        if freeze_encoder:
            self.backbone.autoencoder.blocks.eval()
            for param in self.backbone.autoencoder.blocks.parameters():
                param.requires_grad = False

    def forward(self, imgs, mask_ratio=0.5):
        # continue as normal
        loss, x_hat, mask = self.backbone.autoencoder(imgs, mask_ratio)
        x_hat = unpatchify(
            x_hat,
            self.backbone.autoencoder.img_size,
            self.backbone.autoencoder.patch_size,
            self.backbone.autoencoder.tubelet_size,
        )
        return loss, x_hat, mask

    def forward_random_channel_drop(self, imgs, mask_ratio=0.75):
        B, C, T, H, W = imgs.shape

        # Randomly select a channel index to corrupt for this batch
        target_idx = torch.randint(0, C, (1,)).item()

        corrupted_imgs = imgs.clone()
        corrupted_imgs[:, target_idx, :, :] = 0

        loss, x_hat, mask = self.backbone.autoencoder(corrupted_imgs, mask_ratio)

        return loss, x_hat, mask, target_idx

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, timestamps = batch
        _, x_hat, mask, target_idx = self.forward_random_channel_drop(x)

        x_hat = unpatchify(
            x_hat,
            self.backbone.autoencoder.img_size,
            self.backbone.autoencoder.patch_size,
            self.backbone.autoencoder.tubelet_size,
        )

        loss = F.mse_loss(x_hat[:, target_idx, ...], x[:, target_idx, ...])

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, timestamps = batch
        _, x_hat, mask, target_idx = self.forward_random_channel_drop(x)

        x_hat = unpatchify(
            x_hat,
            self.backbone.autoencoder.img_size,
            self.backbone.autoencoder.patch_size,
            self.backbone.autoencoder.tubelet_size,
        )

        loss = F.mse_loss(x_hat[:, target_idx, ...], x[:, target_idx, ...])

        for i in range(x.shape[0]):
            for frame in range(x.shape[2]):
                self.validation_metrics.append(
                    bench_recon.get_metrics(
                        x[i, :, frame, :, :], x_hat[i, :, frame, :, :], ALL_WAVELENGTHS
                    )
                )

        self.log("val_loss", loss)
