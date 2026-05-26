import math
import os
import random

import numpy as np
import pandas as pd
import torch
import xarray as xr
from loguru import logger
from torch.utils.data import Dataset
import torch.nn.functional as F


class HelioZarrDataset(Dataset):
    """Dataset for loading Helio data from Zarr stores."""

    def __init__(
        self,
        index_path: str,
        zarr_root_path: str,
        channels: list[str],
        scalers: dict,
        time_delta_input_minutes: list[int],
        time_delta_target_minutes: int,
        n_input_timestamps: int,
        rollout_steps: int = 0,
        num_mask_aia_channels: int = 0,
        pooling: int | None = None,
        random_vert_flip: bool = False,
        phase: str = "train",
    ):
        super().__init__()
        self.index_df = pd.read_csv(index_path, parse_dates=["timestamp"])
        self.index_df.sort_values(by="timestamp", inplace=True)  # Added explicit sorting
        self.zarr_root_path = zarr_root_path
        self.channels = channels
        self.scalers = scalers
        self.time_delta_input_minutes = time_delta_input_minutes
        self.time_delta_target_minutes = time_delta_target_minutes
        self.n_input_timestamps = n_input_timestamps
        self.rollout_steps = rollout_steps
        self.num_mask_aia_channels = num_mask_aia_channels
        self.pooling = pooling
        self.random_vert_flip = random_vert_flip
        self.phase = phase

        # Filter index_df based on 'present' column
        if "present" in self.index_df.columns:
            initial_count = len(self.index_df)
            self.index_df = self.index_df[self.index_df['present'] == 1].reset_index(drop=True)
            if len(self.index_df) < initial_count:
                logger.info(f"Filtered out {initial_count - len(self.index_df)} non-present timestamps from index.")
        if selfa.index_df.empty:
            logger.error(f"Index DataFrame is empty after filtering for 'present=1'. Check index_path: {index_path}")
            raise ValueError("Empty index DataFrame after filtering.")

        # Removed xr.open_zarr from here, will open month-specific groups dynamically

    def __len__(self) -> int:
        return len(self.index_df)

    def _apply_scaling(self, data: np.ndarray, channel_name: str) -> np.ndarray:
        """Applies min-max scaling to data based on pre-computed scalers."""
        if self.scalers and channel_name in self.scalers:
            scaler_params = self.scalers[channel_name]
            min_val = scaler_params["min"]
            max_val = scaler_params["max"]
            if max_val - min_val > 0:
                data = (data - min_val) / (max_val - min_val)
            else:
                logger.warning(
                    f"Scaler for {channel_name} has min_val == max_val. Returning zeros."
                )
                data = np.zeros_like(data)
        return data

    def _get_data_for_timestamp(self, timestamp: pd.Timestamp) -> np.ndarray:
        """Retrieves and processes data for a single timestamp from its month group."""
        year_month_path = os.path.join(self.zarr_root_path, str(timestamp.year), f"{timestamp.month:02d}")

        try:
            # Open the specific month/year Zarr group (removed consolidated=True)
            month_zarr_data = xr.open_zarr(year_month_path, chunks="auto")
        except Exception as e:
            logger.warning(f"Error opening Zarr group {year_month_path}: {e}. Returning NaN array for {timestamp}.")
            return np.full((len(self.channels), 4096, 4096), np.nan, dtype=np.float32)

        # Ensure required channels are present in this specific month group
        try:
            available_channels = month_zarr_data.coords['channel'].values
        except KeyError:
            logger.warning(f"'channel' coordinate not found in Zarr group {year_month_path}. Returning NaN array for {timestamp}.")
            return np.full((len(self.channels), 4096, 4096), np.nan, dtype=np.float32)
        
        missing_channels = set(self.channels) - set(available_channels)
        if missing_channels:
            logger.warning(
                f"Channels {missing_channels} not found in Zarr group {year_month_path}. Returning NaN array for {timestamp}."
            )
            return np.full((len(self.channels), 4096, 4096), np.nan, dtype=np.float32)

        try:
            # Select data for the given timestamp and requested channels from the month group
            data_xr = month_zarr_data.sel(time=timestamp, channel=self.channels).compute()
        except KeyError:
            logger.warning(f"Timestamp {timestamp} not found in Zarr group {year_month_path}. Returning NaN array.")
            return np.full((len(self.channels), 4096, 4096), np.nan, dtype=np.float32)

        # Convert to numpy array and reorder dimensions to (channel, y, x)
        data_np = data_xr["data"].transpose("channel", "y", "x").values

        # Apply scaling
        for i, channel_name in enumerate(self.channels):
            data_np[i] = self._apply_scaling(data_np[i], channel_name)

        return data_np

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], dict] | None:
        row = self.index_df.iloc[idx]
        current_time = row["timestamp"]

        # Determine input timestamps
        input_timestamps = [
            current_time + pd.Timedelta(minutes=td)
            for td in self.time_delta_input_minutes
        ]

        # Determine target timestamp
        target_timestamp = current_time + pd.Timedelta(
            minutes=self.time_delta_target_minutes
        )

        input_data_list = []
        for ts in input_timestamps:
            data = self._get_data_for_timestamp(ts)
            # Check if the retrieved input data is all NaNs
            if np.isnan(data).all():
                logger.warning(f"Input timestamp {ts} for sample {idx} is all NaNs. Skipping sample.")
                return None # Return None if any input is NaN
            input_data_list.append(data)

        # Stack input data along a new time dimension
        # Shape: (T, C, H, W) where T is n_input_timestamps
        input_tensor = torch.from_numpy(np.stack(input_data_list, axis=0)).float()

        target_data_np = self._get_data_for_timestamp(target_timestamp)
        # Check if the retrieved target data is all NaNs
        if np.isnan(target_data_np).all():
            logger.warning(f"Target timestamp {target_timestamp} for sample {idx} is all NaNs. Skipping sample.")
            return None # Return None if target is NaN

        target_tensor = torch.from_numpy(target_data_np).float()

        # Apply pooling if specified
        if self.pooling and self.pooling > 1:
            pool_factor = self.pooling
            # Input tensor shape: (T, C, H, W)
            input_tensor = F.avg_pool3d(
                input_tensor.unsqueeze(0),  # Add batch dim for pooling: (1, T, C, H, W)
                kernel_size=(1, pool_factor, pool_factor),
                stride=(1, pool_factor, pool_factor)
            ).squeeze(0)  # Remove batch dim: (T, C, H // pool_factor, W // pool_factor)

            # Target tensor shape: (C, H, W)
            target_tensor = F.avg_pool2d(
                target_tensor.unsqueeze(0),  # Add batch dim for pooling: (1, C, H, W)
                kernel_size=pool_factor,
                stride=pool_factor
            ).squeeze(0)  # Remove batch dim: (C, H // pool_factor, W // pool_factor)

        # Apply random vertical flip if in training phase
        if self.random_vert_flip and self.phase == "train" and random.random() > 0.5:
            input_tensor = torch.flip(input_tensor, dims=[2])  # Flip height dimension (H)
            target_tensor = torch.flip(target_tensor, dims=[1])  # Flip height dimension (H)

        # Output data and metadata
        data = {"ts": input_tensor, "target": target_tensor}
        metadata = {
            "timestamps_input": input_timestamps,
            "timestamps_targets": [target_timestamp],
        }

        return data, metadata
