import argparse
import os
import re
from pathlib import Path

import dask.distributed
import numcodecs
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from tqdm import tqdm

# Configure Blosc
numcodecs.blosc.use_threads = False  # Avoid thread conflicts
numcodecs.blosc.set_nthreads(1)

CHANNEL_ORDER = [
    'aia94', 'aia131', 'aia171', 'aia193', 'aia211', 'aia304', 'aia335', 
    'aia1600', 'hmi_m', 'hmi_bx', 'hmi_by', 'hmi_bz', 'hmi_v'
]

def parse_filename(filepath):
    # Matches YYYYMMDD_HHMM
    match = re.search(r"(\d{8})_(\d{4})\.nc", filepath.name)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        return pd.to_datetime(f"{date_str} {time_str[:2]}:{time_str[2:]}")
    return None

def process_file(filepath, channels):
    try:
        with xr.open_dataset(filepath, engine="h5netcdf", chunks=None, cache=False) as ds:
            # Select channels and convert to array
            # Assuming variables in netcdf are named similarly to our CHANNEL_ORDER
            # This might need adjustment based on actual file content
            data = ds[channels].to_array(dim="channel").load()
            return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert NC files to Zarr")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_zarr", type=str, required=True)
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--end_year", type=int, required=True)
    parser.add_argument("--n_workers", type=int, default=16)
    args = parser.parse_args()

    # Setup Dask
    client = dask.distributed.Client(n_workers=args.n_workers, threads_per_worker=1)
    print(f"Dask dashboard: {client.dashboard_link}")

    # Discover files
    input_root = Path(args.input_dir)
    file_map = {} # (year, month) -> list of (timestamp, filepath)
    
    for path in tqdm(input_root.rglob("*.nc"), desc="Discovering files"):
        ts = parse_filename(path)
        if ts and args.start_year <= ts.year <= args.end_year:
            key = (ts.year, ts.month)
            if key not in file_map:
                file_map[key] = []
            file_map[key].append((ts, path))

    # Sort and process
    compressor = numcodecs.Blosc(cname='lz4', clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    
    for (year, month), files in sorted(file_map.items()):
        print(f"Processing {year}/{month:02d}...")
        
        # Create full time range for the month
        start_date = pd.Timestamp(year, month, 1)
        end_date = start_date + pd.offsets.MonthEnd(0)
        time_index = pd.date_range(start=start_date, end=end_date, freq="12T")
        
        # Map existing files to index
        files.sort(key=lambda x: x[0])
        file_lookup = {ts: path for ts, path in files}
        
        # Prepare datasets
        datasets = []
        for ts in time_index:
            if ts in file_lookup:
            ds = process_file(file_lookup[ts], CHANNEL_ORDER)
            if ds is not None:
                # Expand time dimension
                ds = ds.expand_dims(time=[ts])
                # Store source path for tracking if needed
                ds.attrs['source_path'] = str(file_lookup[ts])
                datasets.append(ds)
            else:
                datasets.append(None) # Marker for gap

            else:
                datasets.append(None) # Marker for gap

        # Initialize an empty array for the whole month
        # Assuming H, W are 4096, 4096. 
        # We need to know this from file if possible, or assume it.
        # Let's get the shape from the first valid file.
        first_valid_ds = None
        for d in valid_datasets:
            if d is not None:
                first_valid_ds = d
                break
        
        if first_valid_ds is None:
            continue
            
        shape = (len(time_index), len(CHANNEL_ORDER), first_valid_ds.shape[1], first_valid_ds.shape[2])
        # Create a dask array of NaNs for the whole month
        data_template = dask.array.full(shape, np.nan, chunks=(1, 1, 4096, 4096), dtype=np.float32)
        
        # This part is tricky to do efficiently with xarray directly for sparse data.
        # Alternative: build list of datasets including NaNs explicitly.
        
        full_datasets = []
        for ts in time_index:
            if ts in file_lookup and file_lookup[ts] in [d.attrs['source_path'] for d in valid_datasets if 'source_path' in d.attrs]:
                 # ... this is becoming complex.
                 pass
        
        # Revised simple approach:
        # 1. Collect all valid datasets
        # 2. Concat
        # 3. Reindex
        
        combined = xr.concat(valid_datasets, dim="time")
        # Add source path to attrs to track for reindexing if needed, but not strictly required.
        
        combined = combined.reindex(time=time_index, fill_value=np.nan)
        
        # Write to Zarr
        encoding = {
            "data": {
                "compressor": compressor,
                "chunks": (1, 1, 4096, 4096)
            }
        }
        
        combined.name = "data"
        group_path = f"{year}/{month:02d}"
        combined.to_zarr(
            args.output_zarr, 
            group=group_path, 
            mode="w", 
            encoding=encoding, 
            consolidated=True
        )

        # Root consolidation
        zarr.consolidate_metadata(args.output_zarr)
        
        # Simple Verification
        print("Verifying Zarr store...")
        try:
            test_ds = xr.open_zarr(args.output_zarr, group=f"{year}/{month:02d}", consolidated=True)
            print(f"Group {year}/{month:02d} metadata read successfully.")
            print(f"Shape: {test_ds['data'].shape}")
            # Check for NaN values if expected
            print(f"Data contains NaNs: {np.isnan(test_ds['data'].isel(time=0).values).any()}")
        except Exception as e:
            print(f"Verification failed: {e}")
            
    print("Finished.")

if __name__ == "__main__":
    main()
