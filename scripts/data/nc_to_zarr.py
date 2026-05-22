import argparse
import os
import re
from pathlib import Path

import dask
import dask.distributed
import numcodecs
import numpy as np
import pandas as pd
import xarray as xr
import zarr

# Optimized Blosc configuration for speed
compressor = numcodecs.Blosc(cname='lz4', clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)

# Expected channel names. User might need to map these if NetCDF internal names differ.
# Mapping format: {'nc_variable_name': 'desired_name'}
CHANNEL_MAP = {
    'aia94': '94A', 'aia131': '131A', 'aia171': '171A', 'aia193': '193A', 
    'aia211': '211A', 'aia304': '304A', 'aia335': '335A', 'aia1600': '1600A',
    'hmi_m': 'hmi_m', 'hmi_bx': 'hmi_bx', 'hmi_by': 'hmi_by', 'hmi_bz': 'hmi_bz', 'hmi_v': 'hmi_v'
}

def parse_filename(filepath):
    match = re.search(r"(\d{8})_(\d{4})\.nc", filepath.name)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        return pd.to_datetime(f"{date_str} {time_str[:2]}:{time_str[2:]}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Convert NC files to Zarr (High Performance)")
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
    
    for path in input_root.rglob("*.nc"):
        ts = parse_filename(path)
        if ts and args.start_year <= ts.year <= args.end_year:
            key = (ts.year, ts.month)
            file_map.setdefault(key, []).append((ts, path))

    for (year, month), files in sorted(file_map.items()):
        print(f"Processing {year}/{month:02d}...")
        
        # 1. Define full time range for the month
        start_date = pd.Timestamp(year, month, 1)
        end_date = start_date + pd.offsets.MonthEnd(0)
        time_index = pd.date_range(start=start_date, end=end_date, freq="12T")
        
        # 2. Pre-initialize the Zarr group with NaNs (Lazy)
        # Assuming H, W are 4096 based on user input
        shape = (len(time_index), len(CHANNEL_MAP), 4096, 4096)
        
        # Create an empty template Zarr
        ds_template = xr.Dataset(
            {
                "data": (
                    ("time", "channel", "y", "x"),
                    dask.array.full(shape, np.nan, chunks=(1, 1, 4096, 4096), dtype=np.float32)
                )
            },
            coords={
                "time": time_index,
                "channel": list(CHANNEL_MAP.keys())
            }
        )
        
        # Set encoding
        encoding = {
            "data": {
                "compressor": compressor,
                "chunks": (1, 1, 4096, 4096)
            }
        }
        
        group_path = f"{year}/{month:02d}"
        
        # Initialize store
        ds_template.to_zarr(
            args.output_zarr, 
            group=group_path, 
            mode="a", 
            encoding=encoding, 
            consolidated=True
        )
        
        # 3. Write actual data
        for ts, path in files:
            try:
                # Open with h5netcdf (lazy)
                with xr.open_dataset(path, engine="h5netcdf", chunks=None, cache=False) as ds:
                    # Map NC variables to our CHANNEL_MAP keys
                    # This assumes NC variable names are in CHANNEL_MAP.values()
                    # You might need to adjust mapping based on your NC internal structure
                    data_vars = {key: ds[val] for key, val in CHANNEL_MAP.items() if val in ds}
                    
                    # Stack channels
                    month_data = xr.Dataset(data_vars).to_array(dim="channel")
                    
                    # Find time index
                    t_idx = time_index.get_loc(ts)
                    
                    # Write to specific region
                    month_data.to_zarr(
                        args.output_zarr,
                        group=group_path,
                        mode="a",
                        region={"time": slice(t_idx, t_idx + 1)}
                    )
            except Exception as e:
                print(f"Error processing {path}: {e}")

    # Root consolidation
    zarr.consolidate_metadata(args.output_zarr)
    print("Finished.")

if __name__ == "__main__":
    main()
