import pandas as pd
import torch
import numpy as np
from datetime import timedelta
from terratorch_surya.datasets.helio import HelioNetCDFDataset
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from terratorch_surya.datasets.helio import RandomChannelMaskerTransform
import random


def safe_collate(batch):
    # Based on the safe_collate function in surya_reconstruction.py
    from torch.utils.data._utils.collate import default_collate

    for sample in batch:
        if isinstance(sample, tuple) and len(sample) == 2:
            metadata = sample[1]
            if "timestamps_input" in metadata:
                metadata["timestamps_input"] = [str(t) for t in metadata["timestamps_input"]]
            if "timestamps_targets" in metadata:
                metadata["timestamps_targets"] = [str(t) for t in metadata["timestamps_targets"]]
    return default_collate(batch)


class HelioF107Dataset(HelioNetCDFDataset):
    def __init__(self, f107_csv_path, **kwargs):
        super().__init__(**kwargs)

        # Load F10.7 data
        self.f107_df = pd.read_csv(f107_csv_path)
        # Handle the space in the column name " f107"
        self.f107_df.columns = self.f107_df.columns.str.strip()
        self.f107_df["timestep"] = pd.to_datetime(self.f107_df["date"], format="%Y%m%d")
        self.f107_df.set_index("timestep", inplace=True)
        self.f107_df.sort_index(inplace=True)

        # Normalize targets
        self.max_f107 = self.f107_df["f107"].max()
        self.f107_df["f107_norm"] = self.f107_df["f107"] / self.max_f107

        # Apply 3-minute buffer filtering to self.valid_indices
        self.filter_indices_with_buffer()
        self.adjusted_length = len(self.valid_indices)

    def filter_indices_with_buffer(self):
        new_valid_indices = []
        for ts in self.valid_indices:
            # SDO image timestamp (ts)
            # Find the nearest F10.7 entry (midnight of a day)
            # The buffer is 3 minutes
            nearest_f107 = self.f107_df.index.get_indexer([ts], method="nearest")[0]
            nearest_ts = self.f107_df.index[nearest_f107]

            if abs(ts - nearest_ts) <= timedelta(minutes=12):
                new_valid_indices.append(ts)
        self.valid_indices = new_valid_indices

    def __getitem__(self, idx):
        # Call super to get the data_dict and metadata
        data_dict, metadata = super().__getitem__(idx)

        # Get reference timestamp (latest in input)
        ref_ts = pd.to_datetime(metadata["timestamps_input"][-1])

        # Match to F10.7 entry
        nearest_idx = self.f107_df.index.get_indexer([ref_ts], method="nearest")[0]
        target_val = self.f107_df.iloc[nearest_idx]["f107_norm"]

        target = torch.tensor(target_val, dtype=torch.float32)

        return data_dict, target


class HelioF107DataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset_kwargs = dict(
            index_path=self.config.data.train_data_path,
            time_delta_input_minutes=list(self.config.data.time_delta_input_minutes),
            time_delta_target_minutes=self.config.data.time_delta_target_minutes,
            n_input_timestamps=self.config.data.n_input_timestamps,
            rollout_steps=0,  # For regression, not forecasting
            channels=list(self.config.data.channels),
            sdo_data_root_path=self.config.data.sdo_data_root_path,
            pooling=self.config.data.pooling,
            f107_csv_path=self.config.data.f107_csv_path,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_ds = HelioF107Dataset(phase="train", **self.dataset_kwargs)
            self.val_ds = HelioF107Dataset(phase="val", **self.dataset_kwargs)
        if stage == "test" or stage is None:
            self.test_ds = HelioF107Dataset(phase="test", **self.dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            collate_fn=safe_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            collate_fn=safe_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            collate_fn=safe_collate,
        )
