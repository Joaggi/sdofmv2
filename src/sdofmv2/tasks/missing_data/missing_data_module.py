import torch
import torch.nn.functional as F

from sdofmv2.core import reconstruction as bench_recon, BaseModule
from sdofmv2.utils import unpatchify, ALL_WAVELENGTHS, ALL_COMPONENTS


class MissingDataModel(BaseModule):
    """A model for reconstructing missing data channels using a backbone autoencoder.

    This class wraps a backbone autoencoder to perform missing data tasks. It
    implements a random channel drop mechanism where one channel is zeroed out,
    and the model is trained to reconstruct that specific channel using MSE loss.

    Args:
        optimizer_dict (dict, optional): Configuration for the optimizer. Defaults to None.
        scheduler_dict (dict, optional): Configuration for the learning rate scheduler.
            Defaults to None.
        backbone (object, optional): The backbone autoencoder model. Defaults to None.
        freeze_encoder (bool): Whether to freeze the encoder blocks of the backbone.
            Defaults to True.
        *args: Variable length argument list passed to BaseModule.
        **kwargs: Arbitrary keyword arguments passed to BaseModule.

    Attributes:
        backbone (object): The underlying autoencoder model.
        masking_ratio (float): The masking ratio used by the backbone.
        validation_metrics (list): Storage for metrics computed during validation.
    """

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
        """Performs a standard forward pass through the backbone autoencoder.

        Args:
            imgs (torch.Tensor): Input images of shape (B, C, T, H, W).
            mask_ratio (float): Ratio of patches to mask. Defaults to 0.5.

        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): The reconstruction loss.
                - x_hat (torch.Tensor): The unpatchified reconstructed images.
                - mask (torch.Tensor): The mask applied during the forward pass.
        """
        loss, x_hat, mask = self.backbone.autoencoder(imgs, mask_ratio)
        x_hat = unpatchify(
            x_hat,
            self.backbone.autoencoder.img_size,
            self.backbone.autoencoder.patch_size,
            self.backbone.autoencoder.tubelet_size,
        )
        return loss, x_hat, mask

    def forward_random_channel_drop(self, imgs, mask_ratio=0.75):
        """Corrupts a random channel and performs a forward pass.

        Args:
            imgs (torch.Tensor): Input images of shape (B, C, T, H, W).
            mask_ratio (float): Ratio of patches to mask. Defaults to 0.75.

        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): The reconstruction loss.
                - x_hat (torch.Tensor): The reconstructed images.
                - mask (torch.Tensor): The mask applied during the forward pass.
                - target_idx (int): The index of the channel that was zeroed out.
        """
        B, C, T, H, W = imgs.shape

        target_idx = torch.randint(0, C, (1,)).item()

        corrupted_imgs = imgs.clone()
        corrupted_imgs[:, target_idx, :, :] = 0

        loss, x_hat, mask = self.backbone.autoencoder(corrupted_imgs, mask_ratio)

        return loss, x_hat, mask, target_idx

    def training_step(self, batch, batch_idx):
        """Executes a single training step with random channel corruption.

        Args:
            batch (tuple): A tuple containing (images, timestamps).
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The MSE loss calculated on the dropped channel.
        """
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
        """Executes a single validation step and logs metrics.

        Args:
            batch (tuple): A tuple containing (images, timestamps).
            batch_idx (int): The index of the current batch.
        """
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

        self.log("val_loss", loss, sync_dist=True)
