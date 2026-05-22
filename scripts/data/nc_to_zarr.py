import argparse
import os
import re
import sys
from pathlib import Path

import dask.array as da
import dask.distributed
import numcodecs
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from tqdm import tqdm

# Optimized Blosc configuration
compressor = numcodecs.Blosc(cname='lz4', clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)

REQUESTED_CHANNELS = [
    'aia94', 'aia131', 'aia171', 'aia193', 'aia211', 'aia304', 'aia335',
    'aia1600', 'hmi_m', 'hmi_bx', 'hmi_by', 'hmi_bz', 'hmi_v'
]

# Map desired Zarr channel names to actual NetCDF variable names.
# Based on the warning and available variables, NetCDF variable names are like 'aia94', 'aia131', etc.
# The user's requested Zarr names also match these.
CHANNEL_MAP = {
    'aia94': 'aia94', 'aia131': 'aia131', 'aia171': 'aia171', 'aia193': 'aia193', 
    'aia211': 'aia211', 'aia304': 'aia304', 'aia335': 'aia335', 'aia1600': 'aia1600',
    'hmi_m': 'hmi_m', 'hmi_bx': 'hmi_bx', 'hmi_by': 'hmi_by', 'hmi_bz': 'hmi_bz', 'hmi_v': 'hmi_v'
}

def parse_filename(filename):
    match = re.search(r"(\d{8})_(\d{4})\.nc", filename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        return pd.to_datetime(f"{date_str} {time_str[:2]}:{time_str[2:]}")
    return None

def check_channels(filepath, channel_map):
    """Checks if the NetCDF variable names (from channel_map values) are available."""
    try:
        with xr.open_dataset(filepath, engine="h5netcdf") as ds:
            available_vars = set(ds.data_vars)
            # We need to check if the actual NetCDF variable names are present
            nc_vars_to_find = set(channel_map.values())
            missing = nc_vars_to_find - available_vars
            if missing:
                print(f"WARNING: Some required NetCDF variable names missing in {filepath}: {missing}")
                print(f"Available variables in file: {available_vars}")
                return False
    except Exception as e:
        print(f"Error opening {filepath} for channel check: {e}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert NC files to Zarr (Robust Version)")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_zarr", type=str, required=True)
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--end_year", type=int, required=True)
    parser.add_argument("--n_workers", type=int, default=16)
    args = parser.parse_args()

    # Dask setup
    client = dask.distributed.Client(n_workers=args.n_workers, threads_per_worker=1)
    print(f"Dask dashboard: {client.dashboard_link}")

    input_root = Path(args.input_dir)
    file_map = {} # (year, month) -> list of (timestamp, filepath)
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Absolute input_dir: {args.input_dir}")
    print(f"Absolute input_root: {input_root.resolve()}")
    print(f"Walking directory: {args.input_dir}")
    
    total_files = 0
    discovered_files = []
    # Use os.walk to handle symlinks correctly

    for root, dirs, files in os.walk(args.input_dir, followlinks=True):
        print(f"Scanning directory: {root}")
        for f in files:
            if f.endswith(".nc"):
                total_files += 1
                ts = parse_filename(f)
                if ts:
                    print(f"  Parsed: {f} -> Timestamp: {ts}")
                    if args.start_year <= ts.year <= args.end_year:
                        key = (ts.year, ts.month)
                        filepath = os.path.join(root, f)
                        discovered_files.append((ts, filepath))
                        file_map.setdefault(key, []).append((ts, filepath))
                    else:
                        print(f"  Year {ts.year} out of range ({args.start_year}-{args.end_year}), skipping.")
                else:
                    print(f"  Could not parse filename: {f}")
    print(f"Total .nc files found: {total_files}")
    print(f"Files matching year criteria: {len(discovered_files)}")
    
    if not file_map:
        print("ERROR: No files found matching criteria. Check path, year range, and filename regex.")
        sys.exit(1)

    # Diagnostic check on the first discovered file
    if discovered_files:
        first_file_ts, first_file_path = discovered_files[0]
        if not check_channels(first_file_path, CHANNEL_MAP):
            print("ERROR: Channel check failed. Please verify REQUESTED_CHANNELS and CHANNEL_MAP match NetCDF variable names.")
            sys.exit(1)
        else:
            print("Channel check passed.")
    else:
        print("ERROR: No files were successfully parsed for the given year range.")
        sys.exit(1)

    created_groups = []

    # Processing
    for (year, month), files in sorted(file_map.items()):
        print(f"Processing {year}/{month:02d}...")
        
        start_date = pd.Timestamp(year, month, 1)
        end_date = start_date + pd.offsets.MonthEnd(0)
        time_index = pd.date_range(start=start_date, end=end_date, freq="12min")
        
        # Assuming H, W are 4096 based on user input
        shape = (len(time_index), len(REQUESTED_CHANNELS), 4096, 4096)
        
        # Pre-initialize Zarr group with NaNs (Lazy)
        ds_template = xr.Dataset(
            {
                "data": (
                    ("time", "channel", "y", "x"),
                    da.full(shape, np.nan, chunks=(1, 1, 4096, 4096), dtype=np.float32)
                )
            },
            coords={
                "time": time_index,
                "channel": REQUESTED_CHANNELS
            }
        )
        
        encoding = {"data": {"compressor": compressor, "chunks": (1, 1, 4096, 4096)}}
        group_path = f"{year}/{month:02d}"
        
        # Initialize store using mode "a" to append/create if not exists
        ds_template.to_zarr(args.output_zarr, group=group_path, mode="a", encoding=encoding, consolidated=True)
        created_groups.append(os.path.join(args.output_zarr, group_path))
        
        # Write files for the current month
        for ts, path in tqdm(files, desc=f"Writing {year}/{month:02d}"):
            try:
                with xr.open_dataset(path, engine="h5netcdf", chunks=None, cache=False) as ds:
                    # Select and reorder channels using CHANNEL_MAP
                    # This creates a new Dataset with only the desired channels, in the correct order
                    mapped_data_vars = {chan_zarr: ds[nc_var_name] 
                                        for chan_zarr, nc_var_name in CHANNEL_MAP.items()
                                        if nc_var_name in ds} # Use nc_var_name to access ds
                    
                    # Convert to DataArray with 'channel' dimension
                    # Use the requested Zarr channel names for the dimension
                    month_data = xr.Dataset(mapped_data_vars).to_array(dim="channel")
                    
                    # Ensure the channel dimension is ordered according to REQUESTED_CHANNELS
                    month_data = month_data.sel(channel=REQUESTED_CHANNELS)
                    
                    # Find the index for this timestamp in the month's time_index
                    t_idx = time_index.get_loc(ts)
                    
                    # Write to the specific time slice in the Zarr group
                    month_data.to_zarr(
                        args.output_zarr,
                        group=group_path,
                        mode="a",
                        region={"time": slice(t_idx, t_idx + 1)},
                        overwrite=True # Overwrite potentially existing NaN slice
                    )
            except Exception as e:
                print(f"Error processing {path}: {e}")

    # Root consolidation
    if created_groups:
        print("Consolidating metadata...")
        zarr.consolidate_metadata(args.output_zarr)
        print("Finished.")
    else:
        print("No data was written. Exiting.")

if __name__ == "__main__":
    main()
