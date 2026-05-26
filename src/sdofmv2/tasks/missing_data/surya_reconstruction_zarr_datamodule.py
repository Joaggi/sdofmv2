import os

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from sdofmv2.tasks.helio_zarr_dataset import HelioZarrDataset
from sdofmv2.utils.data_utils import safe_collate


class SuryaReconstructionZarrDataModule(pl.LightningDataModule):
    """DataModule for Surya channel reconstruction task using Zarr data."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.dataset = None
        # Load scalers from the specified path
        self.scalers = OmegaConf.load(self.config.data.scalers_path)

    def setup(self, stage: str | None = None):
        """Sets up the dataset."""
        common_kwargs = dict(
            zarr_root_path=self.config.data.zarr_root_path,
            time_delta_input_minutes=list(self.config.data.time_delta_input_minutes),
            time_delta_target_minutes=self.config.data.time_delta_target_minutes,
            n_input_timestamps=self.config.data.n_input_timestamps,
            channels=list(self.config.data.channels),
            pooling=self.config.data.pooling,
            random_vert_flip=self.config.data.random_vert_flip,
            scalers=self.scalers,
            rollout_steps=0, # Not used in this specific reconstruction task
            num_mask_aia_channels=0, # Not used in this specific reconstruction task
        )

        # Setup train and val datasets for the 'fit' (training) stage
        if stage == "fit" or stage is None:
            self.train_dataset = HelioZarrDataset(
                index_path=self.config.data.train_data_path,
                phase="train",
                **common_kwargs,
            )

            self.val_dataset = HelioZarrDataset(
                index_path=self.config.data.valid_data_path,
                phase="val",
                **common_kwargs,
            )

        # Setup test dataset for the 'test' stage
        if stage == "test" or stage is None:
            self.test_dataset = HelioZarrDataset(
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
