import pandas as pd
import torch
import numpy as np
from datetime import timedelta
from loguru import logger
import torch.nn.functional as F

from sdofmv2.tasks.helio_zarr_dataset import HelioZarrDataset


class HelioF107ZarrDataset(HelioZarrDataset):
    """Dataset for loading Helio data from Zarr stores for F10.7 prediction."""

    def __init__(self, f107_csv_path: str, **kwargs):
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

        # Apply 12-minute buffer filtering to self.index_df
        self.filter_indices_with_buffer()

    def filter_indices_with_buffer(self):
        new_index_df = []
        for _, row in self.index_df.iterrows():
            ts = row["timestamp"]
            # Find the nearest F10.7 entry (midnight of a day)
            # The buffer is 12 minutes as per original HelioF107Dataset
            try:
                nearest_f107_idx = self.f107_df.index.get_indexer([ts], method="nearest")[0]
            except IndexError:
                logger.warning(f"Could not find nearest F10.7 data index for timestamp {ts}. Skipping.")
                continue

            # Check if get_indexer returned a valid index
            if nearest_f107_idx == -1:
                logger.warning(f"No nearest F10.7 data found for timestamp {ts}. Skipping.")
                continue

            nearest_f107_ts = self.f107_df.index[nearest_f107_idx]

            if abs(ts - nearest_f107_ts) <= timedelta(minutes=12):
                new_index_df.append(row)
            else:
                logger.warning(f"Timestamp {ts} out of F10.7 buffer range (diff: {abs(ts - nearest_f107_ts)}). Skipping.")

        self.index_df = pd.DataFrame(new_index_df) if new_index_df else pd.DataFrame(columns=self.index_df.columns)
        if self.index_df.empty:
            logger.error("After F10.7 buffer filtering, the index_df is empty. Please check your data and f107_csv_path.")

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        # Call super to get the data_dict and metadata (image data)
        data_dict, metadata = super().__getitem__(idx)

        # Get reference timestamp (latest in input)
        # metadata["timestamps_input"] contains datetime objects, not strings due to safe_collate already handling it
        ref_ts = metadata["timestamps_input"][-1]

        # Match to F10.7 entry
        try:
            nearest_f107_idx = self.f107_df.index.get_indexer([ref_ts], method="nearest")[0]
        except IndexError:
            logger.warning(f"Could not find nearest F10.7 data index for reference timestamp {ref_ts}. Returning NaN target.")
            target_val = np.nan # Or some other default value
        else:
            if nearest_f107_idx == -1:
                logger.warning(f"No nearest F10.7 data found for reference timestamp {ref_ts}. Returning NaN target.")
                target_val = np.nan
            else:
                target_val = self.f107_df.iloc[nearest_f107_idx]["f107_norm"]

        target = torch.tensor(target_val, dtype=torch.float32)

        return data_dict, target
