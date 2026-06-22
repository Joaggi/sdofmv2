import os
import shutil

import dask.array as da
import numcodecs
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from loguru import logger
from tqdm import tqdm

# Optimized Blosc configuration
compressor = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)

# Define channels based on previous context
REQUESTED_CHANNELS = [
    "aia94",
    "aia131",
    "aia171",
    "aia193",
    "aia211",
    "aia304",
    "aia335",
    "aia1600",
    "hmi_m",
    "hmi_bx",
    "hmi_by",
    "hmi_bz",
    "hmi_v",
]


def get_zarr_group_path(zarr_root_path: str, timestamp: pd.Timestamp) -> str:
    """Constructs the Zarr group path for a given timestamp."""
    return os.path.join(zarr_root_path, str(timestamp.year), f"{timestamp.month:02d}")

def clean_monthly_zarr_group(monthly_zarr_path: str, channels: list[str]):
    """Cleans a single monthly Zarr group by removing NaN-containing time slices."""
    if not os.path.exists(monthly_zarr_path):
        logger.warning(f"Zarr group not found: {monthly_zarr_path}. Skipping.")
        return

    try:
        # Open the Zarr group lazily with Dask
        monthly_ds = xr.open_zarr(monthly_zarr_path, chunks='auto')
        original_timestamps = pd.to_datetime(monthly_ds.time.values)

        clean_timestamps_for_month = []
        total_timestamps_in_month = len(original_timestamps)

        # Iterate through timestamps and check for NaNs
        for i, ts in enumerate(tqdm(original_timestamps, desc=f"Cleaning {os.path.basename(monthly_zarr_path)}", leave=False)):
            try:
                # Access the data for a single timestamp slice (lazy)
                single_slice_data_array = monthly_ds['data'].sel(time=ts)
                # Compute only this slice to check for NaNs
                if not single_slice_data_array.compute().isnull().any():
                    clean_timestamps_for_month.append(ts)
            except KeyError:
                logger.warning(f"Timestamp {ts} not found in Zarr group {monthly_zarr_path}. Skipping.")
            except Exception as e:
                logger.error(f"Error processing timestamp {ts} in {monthly_zarr_path}: {e}. Skipping.")

        # --- Rewrite Zarr group if cleaning resulted in changes ---
        # Case 1: Entire Month is Dirty (or becomes empty after filtering)
        if not clean_timestamps_for_month:
            logger.info(f"No clean data found for {monthly_zarr_path}. Removing the entire group.")
            shutil.rmtree(monthly_zarr_path)
            return

        # Case 2: Month contains some clean data, but needs rewriting
        # Check if the cleaned list is different from original timestamps
        if len(clean_timestamps_for_month) < total_timestamps_in_month:
            logger.info(f"Found {len(clean_timestamps_for_month)} clean timestamps out of {total_timestamps_in_month} for {monthly_zarr_path}. Rewriting.")
            
            # Create a temporary cleaned dataset
            cleaned_monthly_ds = monthly_ds.sel(time=clean_timestamps_for_month)

            temp_zarr_path = monthly_zarr_path + "_temp"
            encoding = {"data": {"compressor": compressor, "chunks": (1, 1, 4096, 4096)}} # Use chunking for efficiency

            # Write to a temporary location
            cleaned_monthly_ds.to_zarr(temp_zarr_path, mode="w", encoding=encoding, consolidated=True)
            
            # Atomically replace the original group
            shutil.rmtree(monthly_zarr_path)
            os.rename(temp_zarr_path, monthly_zarr_path)
            logger.info(f"Successfully rewrote and replaced {monthly_zarr_path}.")
        else:
            logger.info(f"All {total_timestamps_in_month} timestamps in {monthly_zarr_path} are clean. No rewrite needed.")

    except FileNotFoundError:
        logger.warning(f"Zarr group not found at {monthly_zarr_path}. Skipping.")
    except Exception as e:
        logger.error(f"An unexpected error occurred processing {monthly_zarr_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Clean Zarr files in-place by removing NaN-containing time slices.")
    parser.add_argument("--input_zarr_root_path", type=str, required=True, help="Root path to the input Zarr store.")
    args = parser.parse_args()

    zarr_root = Path(args.input_zarr_root_path)
    if not zarr_root.exists():
        logger.error(f"Input Zarr root path does not exist: {args.input_zarr_root_path}")
        sys.exit(1)

    logger.info(f"Starting Zarr cleaning process for: {args.input_zarr_root_path}")

    # Iterate through all year/month directories
    for year_dir in tqdm(os.listdir(args.input_zarr_root_path), desc="Scanning years"):
        year_path = os.path.join(args.input_zarr_root_path, year_dir)
        if not os.path.isdir(year_path) or not year_dir.isdigit():
            logger.warning(f"Skipping non-directory or non-year directory: {year_path}")
            continue

        for month_dir in tqdm(os.listdir(year_path), desc=f"Scanning months in {year_dir}", leave=False):
            month_path = os.path.join(year_path, month_dir)
            if not os.path.isdir(month_path) or not month_dir.isdigit() or not (1 <= int(month_dir) <= 12):
                logger.warning(f"Skipping non-directory or non-month directory: {month_path}")
                continue
            
            clean_monthly_zarr_group(month_path, REQUESTED_CHANNELS)

    # Root consolidation after all groups are processed
    logger.info("Consolidating metadata for the root Zarr store...")
    try:
        zarr.consolidate_metadata(args.input_zarr_root_path)
        logger.info("Zarr cleaning process completed successfully.")
    except Exception as e:
        logger.error(f"Error during root metadata consolidation: {e}")


if __name__ == "__main__":
    main()
