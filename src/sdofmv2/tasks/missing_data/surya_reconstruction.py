import os
import random

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from omegaconf import DictConfig, OmegaConf

from terratorch_surya.datasets.helio import HelioNetCDFDataset
from terratorch_surya.models.helio_spectformer import HelioSpectFormer


def safe_collate(batch):
    """
    Intercepts the batch to fix datetime objects before the default
    collator tries to turn them into tensors.
    """
    for sample in batch:
        # Check if the sample is a tuple (data, metadata)
        if isinstance(sample, tuple) and len(sample) == 2:
            metadata = sample[1]
            # Convert datetime64 to strings or unix timestamps
            if "timestamps_input" in metadata:
                metadata["timestamps_input"] = [str(t) for t in metadata["timestamps_input"]]
            if "timestamps_targets" in metadata:
                metadata["timestamps_targets"] = [str(t) for t in metadata["timestamps_targets"]]

    return default_collate(batch)


# def safe_collate(batch):
#     """
#     Strips metadata and returns only the data dictionary,
#     collated into a single batch.
#     """
#     # Extract only the first part of the tuple (the dictionary of tensors)
#     data_only_batch = [sample[0] for sample in batch]

#     # default_collate will now only see standard tensors/numbers
#     return default_collate(data_only_batch)


class SuryaReconstructionDataModule(pl.LightningDataModule):
    """DataModule for Surya channel reconstruction task.

    Args:
        config (DictConfig): Hydra configuration object.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.dataset = None
        self.scalers = OmegaConf.load(self.config.data.scalers_path)

    def setup(self, stage: str | None = None):
        """Sets up the dataset.

        Args:
            stage (str, optional): The stage (train, val, test). Defaults to None.
        """
        common_kwargs = dict(
            time_delta_input_minutes=list(self.config.data.time_delta_input_minutes),
            time_delta_target_minutes=self.config.data.time_delta_target_minutes,
            n_input_timestamps=self.config.data.n_input_timestamps,
            rollout_steps=0,
            num_mask_aia_channels=0,
            channels=list(self.config.data.channels),
            sdo_data_root_path=self.config.data.sdo_data_root_path,
            pooling=self.config.data.pooling,
            random_vert_flip=self.config.data.random_vert_flip,
            scalers=self.scalers,
        )

        # Setup train and val datasets for the 'fit' (training) stage
        if stage == "fit" or stage is None:
            self.train_dataset = HelioNetCDFDataset(
                index_path=self.config.data.train_data_path,
                phase="train",
                **common_kwargs,
            )

            self.val_dataset = HelioNetCDFDataset(
                index_path=self.config.data.valid_data_path,
                phase="val",
                **common_kwargs,
            )

        # Setup test dataset for the 'test' stage
        if stage == "test" or stage is None:
            self.test_dataset = HelioNetCDFDataset(
                index_path=self.config.data.test_data_path,
                phase="test",
                **common_kwargs,
            )

    def train_dataloader(self) -> DataLoader:
        """Returns the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_data_workers,
            pin_memory=self.config.data.pin_memory,
            prefetch_factor=self.config.data.prefetch_factor,
            persistent_workers=self.config.data.persistent_workers,
            collate_fn=safe_collate,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,  # No need to shuffle validation data
            num_workers=self.config.data.num_data_workers,
            pin_memory=self.config.data.pin_memory,
            prefetch_factor=self.config.data.prefetch_factor,
            persistent_workers=self.config.data.persistent_workers,
            collate_fn=safe_collate,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the testing dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,  # No need to shuffle test data
            num_workers=self.config.data.num_data_workers,
            pin_memory=self.config.data.pin_memory,
            prefetch_factor=self.config.data.prefetch_factor,
            persistent_workers=self.config.data.persistent_workers,
            collate_fn=safe_collate,
        )


class SuryaReconstructionModel(pl.LightningModule):
    """LightningModule for fine-tuning Surya for channel reconstruction.

    Args:
        config (DictConfig): Hydra configuration object.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        in_channels = len(config.data.channels)

        self.model = HelioSpectFormer(
            img_size=config.backbone.img_size,
            patch_size=config.backbone.patch_size,
            in_chans=in_channels,
            embed_dim=config.backbone.embed_dim,
            time_embedding=dict(config.backbone.time_embedding),
            depth=config.backbone.depth,
            n_spectral_blocks=config.backbone.n_spectral_blocks,
            num_heads=config.backbone.num_heads,
            mlp_ratio=config.backbone.mlp_ratio,
            drop_rate=config.backbone.drop_rate,
            window_size=config.backbone.window_size,
            dp_rank=config.backbone.dp_rank,
            learned_flow=config.backbone.learned_flow,
            use_latitude_in_learned_flow=config.backbone.use_latitude_in_learned_flow,
            init_weights=config.backbone.init_weights,
            checkpoint_layers=list(config.backbone.checkpoint_layers),
            rpe=config.backbone.rpe,
            finetune=False, # Use original decoder
        )

        pretrained_path = config.backbone.path_weights
        if pretrained_path and os.path.exists(pretrained_path):
            logger.info(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            msg = self.model.load_state_dict(checkpoint, strict=False)
            logger.info(f"Checkpoint load result: {msg}")

        # Freeze encoder, finetune decoder
        logger.info("Freezing encoder (embedding/backbone), finetuning decoder (unembed)")
        for name, param in self.model.named_parameters():
            if "unembed" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def mask_input(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Randomly masks one channel across all time steps.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W).

        Returns:
            tuple[torch.Tensor, int]: The masked tensor and the index of the dropped channel.
        """
        b, c, t, h, w = x.shape
        masked_x = x.clone()
        channel_idx = random.randint(0, c - 1)

        mask = torch.ones((c, 1, 1, 1), device=x.device, dtype=x.dtype)
        mask[channel_idx, ...] = 0.0

        masked_x = masked_x * mask
        return masked_x, channel_idx

    def forward(self, batch: dict) -> torch.Tensor:
        """Forward pass.

        Args:
            batch (dict): Batch dictionary from the dataset.

        Returns:
            torch.Tensor: The predicted reconstructed image.
        """
        return self.model(batch)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch (dict): Batch dictionary.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: The computed loss.
        """
        data, metadata = batch
        original_x = data["ts"]

        masked_x, dropped_channel = self.mask_input(original_x)

        model_input = data.copy()
        model_input["ts"] = masked_x

        predicted_x = self(model_input)

        # Target is the original unmasked channel at the most recent timestep
        target = original_x[:, dropped_channel, -1, :, :]
        pred = predicted_x[:, dropped_channel, :, :]

        loss = F.mse_loss(pred, target)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step.

        Args:
            batch (dict): Batch dictionary.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: The computed loss.
        """
        data, metadata = batch
        original_x = data["ts"]
        masked_x, dropped_channel = self.mask_input(original_x)

        model_input = data.copy()
        model_input["ts"] = masked_x

        predicted_x = self(model_input)

        target = original_x[:, dropped_channel, -1, :, :]
        pred = predicted_x[:, dropped_channel, :, :]

        loss = F.mse_loss(pred, target)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step.

        Args:
            batch (dict): Batch dictionary.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: The computed loss.
        """
        data, metadata = batch
        original_x = data["ts"]
        masked_x, dropped_channel = self.mask_input(original_x)

        model_input = data.copy()
        model_input["ts"] = masked_x

        predicted_x = self(model_input)

        target = original_x[:, dropped_channel, -1, :, :]
        pred = predicted_x[:, dropped_channel, :, :]

        loss = F.mse_loss(pred, target)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """Configures the optimizer.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        # Filter out frozen parameters
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        if self.config.optimizer.type == "adamw":
            return torch.optim.AdamW(
                trainable_params,
                lr=self.config.optimizer.lr,
                weight_decay=self.config.optimizer.weight_decay,
            )
        return torch.optim.Adam(trainable_params, lr=self.config.optimizer.lr)
