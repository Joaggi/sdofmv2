import argparse
import os
import re
import sys
from pathlib import Path

import dask
import dask.distributed
import numcodecs
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from tqdm import tqdm

# Optimized Blosc configuration
compressor = numcodecs.Blosc(cname='lz4', clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)

# Mapping: { 'desired_zarr_name': 'internal_nc_variable_name' }
CHANNEL_MAP = {
    'aia94': '94A', 'aia131': '131A', 'aia171': '171A', 'aia193': '193A', 
    'aia211': '211A', 'aia304': '304A', 'aia335': '335A', 'aia1600': '1600A',
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
    with xr.open_dataset(filepath, engine="h5netcdf") as ds:
        available_vars = set(ds.data_vars)
        required_vars = set(channel_map.values())
        missing = required_vars - available_vars
        if missing:
            print(f"WARNING: Some channels missing in {filepath}: {missing}")
            print(f"Available variables: {available_vars}")
            # Try to infer if names are just slightly different
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

    # Robust file discovery
    file_map = {}
    print(f"Walking directory: {args.input_dir}")
    
    total_files = 0
    for root, dirs, files in os.walk(args.input_dir, followlinks=True):
        for f in files:
            if f.endswith(".nc"):
                total_files += 1
                ts = parse_filename(f)
                if ts and args.start_year <= ts.year <= args.end_year:
                    key = (ts.year, ts.month)
                    file_map.setdefault(key, []).append((ts, os.path.join(root, f)))
    
    print(f"Total .nc files found: {total_files}")
    if not file_map:
        print("ERROR: No files found matching criteria. Check path and regex.")
        sys.exit(1)

    # Diagnostic check
    first_file = list(file_map.values())[0][0][1]
    check_channels(first_file, CHANNEL_MAP)

    created_groups = []

    # Processing
    for (year, month), files in sorted(file_map.items()):
        print(f"Processing {year}/{month:02d}...")
        
        start_date = pd.Timestamp(year, month, 1)
        end_date = start_date + pd.offsets.MonthEnd(0)
        time_index = pd.date_range(start=start_date, end=end_date, freq="12T")
        
        shape = (len(time_index), len(CHANNEL_MAP), 4096, 4096)
        
        # Pre-initialize Zarr group
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
        
        encoding = {"data": {"compressor": compressor, "chunks": (1, 1, 4096, 4096)}}
        group_path = f"{year}/{month:02d}"
        
        ds_template.to_zarr(args.output_zarr, group=group_path, mode="a", encoding=encoding, consolidated=True)
        created_groups.append(os.path.join(args.output_zarr, group_path))
        
        # Write files
        for ts, path in tqdm(files, desc=f"Writing {year}/{month:02d}"):
            try:
                with xr.open_dataset(path, engine="h5netcdf", chunks=None, cache=False) as ds:
                    # Construct dataset using mapping
                    data_vars = {key: ds[val] for key, val in CHANNEL_MAP.items() if val in ds}
                    month_data = xr.Dataset(data_vars).to_array(dim="channel").sel(channel=list(CHANNEL_MAP.keys()))
                    
                    t_idx = time_index.get_loc(ts)
                    month_data.to_zarr(
                        args.output_zarr,
                        group=group_path,
                        mode="a",
                        region={"time": slice(t_idx, t_idx + 1)}
                    )
            except Exception as e:
                print(f"Error processing {path}: {e}")

    # Root consolidation
    if created_groups:
        zarr.consolidate_metadata(args.output_zarr)
        print("Finished.")
    else:
        print("No data was written.")

if __name__ == "__main__":
    main()
