import time
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
        self.index_df = pd.read_csv(index_path)
        self.index_df = self.index_df[self.index_df["present"] == 1]
        self.index_df["timestep"] = pd.to_datetime(self.index_df["timestep"]).values.astype(
            "datetime64[ns]"
        )
        self.index_df.set_index("timestep", inplace=True)
        self.index_df.sort_index(inplace=True)
        self.zarr_root_path = zarr_root_path
        self.channels = channels
        self.scalers = scalers
        # Convert time delta to numpy timedelta64
        self.time_delta_input_minutes = sorted(
            np.timedelta64(t, "m") for t in time_delta_input_minutes
        )
        self.time_delta_target_minutes = [
            np.timedelta64(iroll * time_delta_target_minutes, "m")
            for iroll in range(1, rollout_steps + 2)
        ]
        self.n_input_timestamps = n_input_timestamps
        self.rollout_steps = rollout_steps
        self.num_mask_aia_channels = num_mask_aia_channels
        self.pooling = pooling
        self.random_vert_flip = random_vert_flip
        self.phase = phase
        self.valid_indices = self.filter_valid_indices()

    def filter_valid_indices(self):
        """
        Extracts timestamps from the index of self.index that define valid
        samples.

        Args:
        Returns:
            List of timestamps.
        """

        valid_indices = []
        time_deltas = np.unique(self.time_delta_input_minutes + self.time_delta_target_minutes)

        for reference_timestep in self.index_df.index:
            required_timesteps = reference_timestep + time_deltas

            if all(t in self.index_df.index for t in required_timesteps):
                valid_indices.append(reference_timestep)

        return valid_indices

    def __len__(self) -> int:
        return len(self.valid_indices)

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
        year_month_path = os.path.join(
            self.zarr_root_path, str(timestamp.year), f"{timestamp.month:02d}"
        )

        try:
            # Open the specific month/year Zarr group (removed consolidated=True)
            month_zarr_data = xr.open_zarr(year_month_path, chunks="auto", consolidated=False)
        except Exception as e:
            logger.warning(
                f"Error opening Zarr group {year_month_path}: {e}. Returning NaN array for {timestamp}."
            )
            return np.full((len(self.channels), 4096, 4096), np.nan, dtype=np.float32)

        # Ensure required channels are present in this specific month group
        try:
            available_channels = month_zarr_data.coords["channel"].values
        except KeyError:
            logger.warning(
                f"'channel' coordinate not found in Zarr group {year_month_path}. Returning NaN array for {timestamp}."
            )
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
            logger.warning(
                f"Timestamp {timestamp} not found in Zarr group {year_month_path}. Returning NaN array."
            )
            return np.full((len(self.channels), 4096, 4096), np.nan, dtype=np.float32)

        # Convert to numpy array and reorder dimensions to (channel, y, x)
        data_np = data_xr["data"].transpose("channel", "y", "x").values

        # Apply scaling
        for i, channel_name in enumerate(self.channels):
            data_np[i] = self._apply_scaling(data_np[i], channel_name)

        return data_np

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], dict] | None:
        start_time = time.time()
        time_deltas = np.array(
            sorted(random.sample(self.time_delta_input_minutes[:-1], self.n_input_timestamps - 1))
            + [self.time_delta_input_minutes[-1]]
            + self.time_delta_target_minutes
        )
        reference_timestep = self.valid_indices[idx]
        required_timesteps = reference_timestep + time_deltas

        sequence_data = []
        for ts in required_timesteps:
            ts = pd.Timestamp(ts)
            data = self._get_data_for_timestamp(ts)
            # Check if the retrieved input data is all NaNs
            if np.isnan(data).all():
                logger.warning(
                    f"Input timestamp {ts} for sample {idx} is all NaNs. Skipping sample."
                )
                return None  # Return None if any input is NaN
            sequence_data.append(data)

        # Split sequence_data into inputs and target
        inputs = sequence_data[: -self.rollout_steps - 1]
        targets = sequence_data[-self.rollout_steps - 1 :]
        stacked_inputs = np.stack(inputs, axis=1)
        stacked_targets = np.stack(targets, axis=1)

        timestamps_input = required_timesteps[: -self.rollout_steps - 1]
        timestamps_targets = required_timesteps[-self.rollout_steps - 1 :]

        # Stack input data along a new time dimension
        # Shape: (T, C, H, W) where T is n_input_timestamps
        # Measure time taken for input data loading
        input_data_load_time = time.time() - start_time
        logger.debug(f"Sample {idx}: Input data loading took {input_data_load_time:.4f} seconds.")

        time_delta_input_float = (
            time_deltas[-self.rollout_steps - 2] - time_deltas[: -self.rollout_steps - 1]
        ) / np.timedelta64(1, "h")
        time_delta_input_float = time_delta_input_float.astype(np.float32)

        lead_time_delta_float = (
            time_deltas[-self.rollout_steps - 2] - time_deltas[-self.rollout_steps - 1 :]
        ) / np.timedelta64(1, "h")
        lead_time_delta_float = lead_time_delta_float.astype(np.float32)

        metadata = {
            "timestamps_input": timestamps_input,
            "timestamps_targets": timestamps_targets,
        }

        if np.isnan(stacked_targets).all():
            logger.warning(
                f"Target timestamp {stacked_targets} for sample {idx} is all NaNs. Skipping sample."
            )
            return None  # Return None if target is NaN

        # Apply pooling if specified
        if self.pooling and self.pooling > 1:
            pool_factor = self.pooling
            # Input tensor shape: (T, C, H, W)
            stacked_inputs = F.avg_pool3d(
                stacked_inputs.unsqueeze(0),  # Add batch dim for pooling: (1, T, C, H, W)
                kernel_size=(1, pool_factor, pool_factor),
                stride=(1, pool_factor, pool_factor),
            ).squeeze(
                0
            )  # Remove batch dim: (T, C, H // pool_factor, W // pool_factor)

            # Target tensor shape: (C, H, W)
            stacked_targets = F.avg_pool2d(
                stacked_targets.unsqueeze(0),  # Add batch dim for pooling: (1, C, H, W)
                kernel_size=pool_factor,
                stride=pool_factor,
            ).squeeze(
                0
            )  # Remove batch dim: (C, H // pool_factor, W // pool_factor)

        # Apply random vertical flip if in training phase
        if self.random_vert_flip and self.phase == "train" and random.random() > 0.5:
            stacked_inputs = torch.flip(stacked_inputs, dims=[2])  # Flip height dimension (H)
            stacked_targets = torch.flip(stacked_targets, dims=[1])  # Flip height dimension (H)

        return {
            "ts": stacked_inputs,
            "time_delta_input": time_delta_input_float,
            "forecast": stacked_targets,
            "lead_time_delta": lead_time_delta_float,
        }, metadata

        return data, metadata
