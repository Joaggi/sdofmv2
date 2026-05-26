import os
from omegaconf import DictConfig
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from loguru import logger
import torch.nn.functional as F

from sdofmv2.tasks.helio_zarr_dataset import HelioZarrDataset
from sdofmv2.tasks.missing_data.surya_reconstruction_zarr_datamodule import safe_collate


class HelioF107ZarrDataModule(LightningDataModule):
    """DataModule for F10.7 prediction using Zarr data."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.dataset_kwargs = dict(
            zarr_root_path=self.config.data.zarr_root_path,
            index_path=self.config.data.train_data_path, # This will be overridden in setup
            time_delta_input_minutes=list(self.config.data.time_delta_input_minutes),
            time_delta_target_minutes=self.config.data.time_delta_target_minutes,
            n_input_timestamps=self.config.data.n_input_timestamps,
            channels=list(self.config.data.channels),
            pooling=self.config.data.pooling,
            random_vert_flip=self.config.data.random_vert_flip,
            scalers=OmegaConf.load(self.config.data.scalers_path), # Load scalers here
            f107_csv_path=self.config.data.f107_csv_path,
            rollout_steps=0, # Not used in this specific prediction task
            num_mask_aia_channels=0, # Not used in this specific prediction task
        )

    def setup(self, stage: str | None = None):
        """Sets up the dataset."""
        if stage == "fit" or stage is None:
            self.train_ds = HelioF107ZarrDataset(
                phase="train",
                index_path=self.config.data.train_data_path,
                **self.dataset_kwargs
            )
            self.val_ds = HelioF107ZarrDataset(
                phase="val",
                index_path=self.config.data.valid_data_path,
                **self.dataset_kwargs
            )
        if stage == "test" or stage is None:
            self.test_ds = HelioF107ZarrDataset(
                phase="test",
                index_path=self.config.data.test_data_path,
                **self.dataset_kwargs
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_data_workers,
            pin_memory=self.config.data.pin_memory,
            prefetch_factor=self.config.data.prefetch_factor,
            persistent_workers=self.config.data.persistent_workers,
            collate_fn=safe_collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_data_workers,
            pin_memory=self.config.data.pin_memory,
            prefetch_factor=self.config.data.prefetch_factor,
            persistent_workers=self.config.data.persistent_workers,
            collate_fn=safe_collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_data_workers,
            pin_memory=self.config.data.pin_memory,
            prefetch_factor=self.config.data.prefetch_factor,
            persistent_workers=self.config.data.persistent_workers,
            collate_fn=safe_collate,
        )
