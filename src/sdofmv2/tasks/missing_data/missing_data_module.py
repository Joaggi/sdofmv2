import torch
import torch.nn.functional as F
import pandas as pd
from loguru import logger
from sdofmv2.core import BaseModule
from sdofmv2.utils import unpatchify, ALL_WAVELENGTHS
from sdofmv2.core.reconstruction import compute_metrics_pytorch
from sdofmv2.core.datamodule import inverse_log_norm, inverse_zscore_norm


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
        normalization (dict, optional): Normalization configuration.
        normalization_stat (dict, optional): Normalization statistics.
        wavelengths (list[str], optional): List of wavelengths.
        *args: Variable length argument list passed to BaseModule.
        **kwargs: Arbitrary keyword arguments passed to BaseModule.
    """

    def __init__(
        self,
        optimizer_dict=None,
        scheduler_dict=None,
        backbone: object = None,
        freeze_encoder: bool = True,
        normalization=None,
        normalization_stat=None,
        wavelengths=None,
        masking_ratio=0.0,
        test_result_path=None,
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
        self.normalization = normalization
        self.normalization_stat = normalization_stat
        self.wavelengths = wavelengths
        self.masking_ratio = masking_ratio
        self.val_reconstruction_metrics_batches = []
        self.test_reconstruction_metrics_batches = []
        self.test_result_path = test_result_path

        if freeze_encoder:
            self.backbone.autoencoder.blocks.eval()
            for param in self.backbone.autoencoder.blocks.parameters():
                param.requires_grad = False

    def _inverse_transform(self, data, channel_idx):
        # data: [B, C, T, H, W]
        channel = self.wavelengths[channel_idx]

        # Apply inverse transform based on type
        if self.normalization.type == "log":
            data_ch = inverse_log_norm(
                data[:, channel_idx, :, :, :].detach().cpu(),
                self.normalization_stat,
                channel,
                scaler_factor=self.normalization.scaler_factor,
                norm=self.normalization.norm,
            )
        elif self.normalization.type == "zscore":
            data_ch = inverse_zscore_norm(
                data[:, channel_idx, :, :, :].detach().cpu(),
                None,  # Instrument not used in this simplified call
                channel,
                self.normalization_stat
            )
        elif self.normalization.type == "min-max":
            data_ch = data[:, channel_idx, :, :, :] * (self.normalization_stat["max"] - self.normalization_stat["min"]) + self.normalization_stat["min"]
        else:
            data_ch = data[:, channel_idx, :, :, :]

        return data_ch

    def _apply_inverse_transform(self, data):
        # data: [B, C, T, H, W]
        transformed_channels = []
        for i in range(data.shape[1]):
            transformed_channels.append(self._inverse_transform(data, i).unsqueeze(1))
        return torch.cat(transformed_channels, dim=1)

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
        loss, x_hat, mask = self.backbone.autoencoder(imgs, self.masking_ratio)
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

        loss, x_hat, mask = self.backbone.autoencoder(corrupted_imgs, self.masking_ratio)

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

        loss = F.mse_loss(x_hat, x)

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

        loss = F.mse_loss(x_hat, x)

        # Apply inverse transform
        x_inv = self._apply_inverse_transform(x)
        x_hat_inv = self._apply_inverse_transform(x_hat)

        # Compute metrics
        mask_ones = torch.ones(x.shape[0], x.shape[2], x.shape[3], x.shape[4], device="cpu")
        metrics = compute_metrics_pytorch(x_inv, x_hat_inv, mask_ones, self.wavelengths)

        self.val_reconstruction_metrics_batches.append(metrics)

        self.log("val_loss", loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, timestamps = batch
        _, x_hat, mask, target_idx = self.forward_random_channel_drop(x)

        x_hat = unpatchify(
            x_hat,
            self.backbone.autoencoder.img_size,
            self.backbone.autoencoder.patch_size,
            self.backbone.autoencoder.tubelet_size,
        )

        # Apply inverse transform
        x_inv = self._apply_inverse_transform(x)
        x_hat_inv = self._apply_inverse_transform(x_hat)

        # Compute metrics
        mask_ones = torch.ones(x.shape[0], x.shape[2], x.shape[3], x.shape[4], device="cpu")
        metrics = compute_metrics_pytorch(x_inv, x_hat_inv, mask_ones, self.wavelengths)

        self.test_reconstruction_metrics_batches.append(metrics)

    def on_validation_epoch_end(self):
        if not self.val_reconstruction_metrics_batches:
            return

        data = []
        for batch in self.val_reconstruction_metrics_batches:
            for wave, metrics in batch.items():
                row = {"wavelength": wave}
                row.update(metrics)
                data.append(row)
        df = pd.DataFrame(data)
        agg = df.groupby("wavelength").mean()

        for wave, row in agg.iterrows():
            for metric, value in row.items():
                self.log(f"val/{wave}/{metric}", value, sync_dist=True)

        self.val_reconstruction_metrics_batches.clear()

    def on_test_epoch_end(self):
        if not self.test_reconstruction_metrics_batches:
            return

        data = []
        for batch in self.test_reconstruction_metrics_batches:
            for wave, metrics in batch.items():
                row = {"wavelength": wave}
                row.update(metrics)
                data.append(row)
        df = pd.DataFrame(data)
        agg = df.groupby("wavelength").mean()

        for wave, row in agg.iterrows():
            for metric, value in row.items():
                self.log(f"test/{wave}/{metric}", value, sync_dist=True)
        
        agg.to_csv(self.test_result_path)
        logger.info(f"Saved test results to {self.test_result_path}")
        self.test_reconstruction_metrics_batches.clear()
