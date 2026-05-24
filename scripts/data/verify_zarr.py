
import argparse
import os
import re
import sys
from pathlib import Path

import dask.distributed
import xarray as xr
import pandas as pd
import numpy as np
import zarr
from loguru import logger as lgr_logger
from tqdm import tqdm

# Define expected channels and shape from the nc_to_zarr.py script
REQUESTED_CHANNELS = [
    'aia94', 'aia131', 'aia171', 'aia193', 'aia211', 'aia304', 'aia335',
    'aia1600', 'hmi_m', 'hmi_bx', 'hmi_by', 'hmi_bz', 'hmi_v'
]
EXPECTED_H_W = 4096
EXPECTED_DTYPE = np.float32

# --- Helper functions copied and adapted from nc_to_zarr.py ---

def parse_filename(filename):
    match = re.search(r"(\d{8})_(\d{4})\.nc", filename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        try:
            return pd.to_datetime(f"{date_str} {time_str[:2]}:{time_str[2:]}")
        except ValueError as e:
            lgr_logger.warning(f"Could not parse timestamp from {filename}: {e}")
            return None
    return None

def count_nc_files_in_month(input_dir: Path, year: int, month: int) -> int:
    """Counts .nc files in a given directory for a specific year and month."""
    count = 0
    for root, _, files in os.walk(str(input_dir), followlinks=True):
        for f in files:
            if f.endswith(".nc"):
                ts = parse_filename(f)
                if ts and ts.year == year and ts.month == month:
                    count += 1
    return count

# --- Verification Logic ---

def verify_zarr_store(zarr_path: str, input_dir: str, start_year: int, end_year: int):
    """
    Verifies the structure and content of a Zarr store created by nc_to_zarr.py.

    Args:
        zarr_path (str): The absolute path to the root of the Zarr store.
        input_dir (str): The absolute path to the directory containing the original .nc files.
        start_year (int): The starting year for which data is expected.
        end_year (int): The ending year for which data is expected.
    """
    lgr_logger.info(f"Starting Zarr store verification for: {zarr_path}")
    lgr_logger.info(f"Expecting source .nc files from: {input_dir}")

    zarr_path_p = Path(zarr_path)
    input_dir_p = Path(input_dir)

    if not zarr_path_p.exists():
        lgr_logger.error(f"Zarr store path does not exist: {zarr_path}")
        sys.exit(1)
    if not input_dir_p.is_dir():
        lgr_logger.error(f"Input directory path is not a valid directory: {input_dir}")
        sys.exit(1)

    all_checks_passed = True

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            group_path_str = str(zarr_path_p / str(year) / f"{month:02d}")
            
            # Check if the Zarr group exists. It's okay if it doesn't, implies no data for that month.
            if not Path(group_path_str).exists():
                lgr_logger.debug(f"Zarr group not found: {group_path_str}. Skipping month.")
                continue
            
            lgr_logger.info(f"Verifying Zarr group: {group_path_str}")

            try:
                ds = xr.open_zarr(group_path_str)
                lgr_logger.info(f"Successfully opened Zarr group {group_path_str}")

                # 1. Check for 'data' DataArray existence
                if 'data' not in ds.data_vars:
                    lgr_logger.error(f"FAIL: 'data' DataArray not found in {group_path_str}")
                    all_checks_passed = False
                    continue
                
                data_array = ds['data']

                # 2. Check dimensions
                expected_dims = ('time', 'channel', 'y', 'x')
                if data_array.dims != expected_dims:
                    lgr_logger.error(f"FAIL: Dimensions mismatch in {group_path_str}. Expected {expected_dims}, got {data_array.dims}")
                    all_checks_passed = False
                    continue # Skip further checks if dimensions are wrong

                # 3. Check coordinates
                if 'time' not in ds.coords:
                    lgr_logger.error(f"FAIL: 'time' coordinate not found in {group_path_str}")
                    all_checks_passed = False
                else:
                    # Verify time frequency (expected 12 minutes)
                    if len(ds['time']) > 1:
                        time_diff = (ds['time'].isel(time=1) - ds['time'].isel(time=0)).values
                        # Expected 12 minutes, allow some tolerance for datetime arithmetic
                        # Note: pd.Timedelta uses nanoseconds, so direct comparison might be tricky. Check total seconds.
                        time_diff_seconds = pd.to_timedelta(time_diff).total_seconds()
                        expected_12_min_seconds = 12 * 60
                        if not np.isclose(time_diff_seconds, expected_12_min_seconds, atol=1):
                            lgr_logger.warning(f"WARNING: Time coordinate frequency might not be 12 minutes in {group_path_str}. First diff: {pd.to_timedelta(time_diff)}")
                    elif len(ds['time']) == 1:
                        lgr_logger.debug(f"Only one time step in {group_path_str}, cannot verify frequency.")
                    else: # len(ds['time']) == 0
                        lgr_logger.warning(f"No time steps found in {group_path_str} despite group existing.")

                if 'channel' not in ds.coords:
                    lgr_logger.error(f"FAIL: 'channel' coordinate not found in {group_path_str}")
                    all_checks_passed = False
                else:
                    channel_values = data_array['channel'].values
                    if not all(c in channel_values for c in REQUESTED_CHANNELS):
                        missing_channels = set(REQUESTED_CHANNELS) - set(channel_values)
                        lgr_logger.error(f"FAIL: Missing channels in {group_path_str}. Expected subset of {REQUESTED_CHANNELS}, got {channel_values}. Missing: {missing_channels}")
                        all_checks_passed = False
                    
                    # Check order - this assumes REQUESTED_CHANNELS is the definitive order
                    if not np.array_equal(channel_values, REQUESTED_CHANNELS):
                         lgr_logger.warning(f"WARNING: Channel order mismatch in {group_path_str}. Expected {REQUESTED_CHANNELS}, got {channel_values}")

                # 4. Check shape and data type
                # Calculate expected time length based on month and 12-min frequency
                start_date = pd.Timestamp(year, month, 1)
                end_date = start_date + pd.offsets.MonthEnd(0)
                time_index_expected = pd.date_range(start=start_date, end=end_date, freq="12min")
                expected_time_len = len(time_index_expected)

                expected_shape = (expected_time_len, len(REQUESTED_CHANNELS), EXPECTED_H_W, EXPECTED_H_W)

                if data_array.shape != expected_shape:
                    lgr_logger.error(f"FAIL: Shape mismatch in {group_path_str}. Expected {expected_shape}, got {data_array.shape}")
                    all_checks_passed = False

                if data_array.dtype != EXPECTED_DTYPE:
                    lgr_logger.error(f"FAIL: Data type mismatch in {group_path_str}. Expected {EXPECTED_DTYPE}, got {data_array.dtype}")
                    all_checks_passed = False

                # 5. Check compressor (from encoding attributes)
                if 'compressor' in data_array.encoding and data_array.encoding['compressor'] is not None:
                    # Access compressor details like cname. Blosc compressor object has attributes.
                    compressor_obj = data_array.encoding['compressor']
                    if hasattr(compressor_obj, 'cname'):
                        compressor_name = compressor_obj.cname
                        if compressor_name != 'lz4':
                            lgr_logger.warning(f"WARNING: Compressor mismatch in {group_path_str}. Expected 'lz4', got '{compressor_name}'")
                    else:
                         lgr_logger.warning(f"WARNING: Compressor found but cname attribute missing in {group_path_str}. Compressor: {compressor_obj}")
                else:
                    lgr_logger.warning(f"WARNING: Compressor encoding not found or is None for 'data' in {group_path_str}")
                
                # 6. Data Integrity: Compare NC file count with Zarr time steps
                actual_num_time_steps = data_array.shape[0]
                expected_num_nc_files = count_nc_files_in_month(input_dir_p, year, month)
                
                if actual_num_time_steps != expected_num_nc_files:
                    lgr_logger.error(f"FAIL: Number of NC files ({expected_num_nc_files}) does not match Zarr time steps ({actual_num_time_steps}) for {year}/{month:02d} in {group_path_str}")
                    all_checks_passed = False
                else:
                    lgr_logger.info(f"NC file count ({expected_num_nc_files}) matches Zarr time steps ({actual_num_time_steps}) for {year}/{month:02d}")

                # 7. Data Integrity Spot Check (read a small slice and check for NaNs)
                if actual_num_time_steps > 0 and data_array.shape[1] > 0: # Ensure there's data to check
                    try:
                        # Load a tiny slice: first time, first channel, small spatial region
                        # Use .compute() to force loading data into memory for the check
                        sample_data = data_array.isel(time=0, channel=0, y=slice(0, 10), x=slice(0, 10)).compute()
                        if np.all(np.isnan(sample_data.values)):
                            lgr_logger.warning(f"WARNING: Sample data slice is all NaNs in {group_path_str}. This might indicate missing input data for this time/channel or an issue during Zarr writing.")
                        else:
                            lgr_logger.info(f"Sample data slice read successfully from {group_path_str}. Not all NaNs.")
                    except Exception as e:
                        lgr_logger.error(f"FAIL: Error reading sample data from {group_path_str}: {e}")
                        all_checks_passed = False
                elif actual_num_time_steps == 0:
                    lgr_logger.warning(f"No time steps found in {group_path_str}, skipping NaN check.")

            except FileNotFoundError:
                # This is expected if no .nc files were found for this month/year, and thus no Zarr group was created.
                lgr_logger.debug(f"Zarr group {group_path_str} not found, as expected if no input data existed for this month.")
            except Exception as e:
                lgr_logger.error(f"Error processing Zarr group {group_path_str}: {e}")
                all_checks_passed = False

    if all_checks_passed:
        lgr_logger.success(f"All verification checks passed for Zarr store at {zarr_path}")
    else:
        lgr_logger.error(f"Some verification checks FAILED for Zarr store at {zarr_path}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Verify Zarr file structure, content, and data integrity.")
    parser.add_argument("--zarr_path", type=str, required=True, help="Path to the Zarr store (e.g., /mnt/cedar/surya-bench.zarr)")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the directory containing the original .nc files.")
    parser.add_argument("--start_year", type=int, required=True, help="Starting year for verification.")
    parser.add_argument("--end_year", type=int, required=True, help="Ending year for verification.")
    
    args = parser.parse_args()

    # Ensure input directory is absolute
    abs_input_dir = os.path.abspath(args.input_dir)
    if not Path(abs_input_dir).is_dir():
        lgr_logger.error(f"Input directory does not exist or is not a directory: {abs_input_dir}")
        sys.exit(1)

    verify_zarr_store(args.zarr_path, abs_input_dir, args.start_year, args.end_year)

if __name__ == "__main__":
    main()
