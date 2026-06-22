import argparse
import os
import re
import sys
from pathlib import Path

import dask.array as da
import dask.distributed
from dask.distributed import as_completed
import numcodecs
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from loguru import logger as lgr_logger
from tqdm import tqdm

# Optimized Blosc configuration
compressor = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)

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

CHANNEL_MAP = {
    "aia94": "aia94",
    "aia131": "aia131",
    "aia171": "aia171",
    "aia193": "aia193",
    "aia211": "aia211",
    "aia304": "aia304",
    "aia335": "aia335",
    "aia1600": "aia1600",
    "hmi_m": "hmi_m",
    "hmi_bx": "hmi_bx",
    "hmi_by": "hmi_by",
    "hmi_bz": "hmi_bz",
    "hmi_v": "hmi_v",
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
            nc_vars_to_find = set(channel_map.values())
            missing = nc_vars_to_find - available_vars
            if missing:
                print(f"WARNING: Required NetCDF variables missing in {filepath}: {missing}")
                print(f"Available variables in file: {available_vars}")
                return False
    except Exception as e:
        print(f"Error opening {filepath} for channel check: {e}")
        return False
    return True


def get_existing_timestamps(zarr_path, group_path):
    """
    Retrieves existing timestamps from a Zarr group efficiently.
    CRITICAL FIX: Loads only a single pixel to check for existence/corruption.
    """
    valid_timestamps = set()
    try:
        if not os.path.exists(os.path.join(zarr_path, group_path)):
            return valid_timestamps

        with xr.open_zarr(zarr_path, group=group_path) as ds:
            all_timestamps = pd.to_datetime(ds.time.values)

            for i, ts in enumerate(all_timestamps):
                try:
                    # Test load a single pixel (time=i, channel=0, y=0, x=0)
                    pixel_val = ds["data"].isel(time=i, channel=0, y=0, x=0).values

                    # Because we use compute=False on creation, unwritten chunks return NaN
                    if not np.isnan(pixel_val):
                        valid_timestamps.add(ts)
                except Exception:
                    # Corrupted chunks (interrupted writes) will throw an error.
                    # We pass here so the timestamp is NOT added, forcing a rewrite.
                    pass

        return valid_timestamps
    except Exception as e:
        lgr_logger.warning(f"Could not read/validate existing timestamps for {group_path}: {e}")
        return set()


def process_and_write_file(
    path, ts, t_idx, output_zarr, group_path, channel_map, requested_channels
):
    """
    Worker function executed by Dask to process a single NetCDF file and write it to Zarr.
    """
    try:
        with xr.open_dataset(path, engine="h5netcdf", chunks=None, cache=False) as ds:
            mapped_data_vars = {
                chan_zarr: ds[nc_var_name]
                for chan_zarr, nc_var_name in channel_map.items()
                if nc_var_name in ds
            }

            month_data = xr.Dataset(mapped_data_vars).to_array(dim="channel")
            month_data = month_data.sel(channel=requested_channels)
            month_data = month_data.expand_dims("time").assign_coords(time=[ts])

            ds_to_write = month_data.to_dataset(name="data").drop_vars(["channel"])

            ds_to_write.to_zarr(
                output_zarr,
                group=group_path,
                mode="a",
                region={"time": slice(t_idx, t_idx + 1)},
                consolidated=False,
            )
        return path, True, None
    except Exception as e:
        return path, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Convert NC files to Zarr (Robust & Parallelized)")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_zarr", type=str, required=True)
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--end_year", type=int, required=True)
    parser.add_argument("--n_workers", type=int, default=16)
    args = parser.parse_args()

    # Dask setup
    client = dask.distributed.Client(n_workers=args.n_workers, threads_per_worker=1)
    lgr_logger.info(f"Dask dashboard available at: {client.dashboard_link}")

    input_root = Path(args.input_dir)
    file_map = {}

    lgr_logger.info(f"Scanning directory: {args.input_dir}")
    total_files = 0
    discovered_files = []

    for root, _dirs, files in os.walk(args.input_dir, followlinks=True):
        for f in files:
            if f.endswith(".nc"):
                total_files += 1
                ts = parse_filename(f)
                if ts:
                    if args.start_year <= ts.year <= args.end_year:
                        key = (ts.year, ts.month)
                        filepath = os.path.join(root, f)
                        discovered_files.append((ts, filepath))
                        file_map.setdefault(key, []).append((ts, filepath))

    lgr_logger.info(f"Total .nc files found: {total_files}")
    lgr_logger.info(f"Files matching year criteria: {len(discovered_files)}")

    if not file_map:
        lgr_logger.error("ERROR: No files found matching criteria.")
        sys.exit(1)

    if discovered_files:
        first_file_ts, first_file_path = discovered_files[0]
        if not check_channels(first_file_path, CHANNEL_MAP):
            lgr_logger.error("ERROR: Channel check failed.")
            sys.exit(1)
        else:
            lgr_logger.info("Diagnostic channel check passed.")

    created_groups = []

    # Processing loop
    for (year, month), files in sorted(file_map.items()):
        lgr_logger.info(f"--- Processing {year}/{month:02d} ---")

        current_month_timestamps = sorted([ts for ts, _ in files])
        if not current_month_timestamps:
            continue

        time_index = pd.DatetimeIndex(current_month_timestamps)
        shape = (len(time_index), len(REQUESTED_CHANNELS), 4096, 4096)

        ds_template = xr.Dataset(
            {
                "data": (
                    ("time", "channel", "y", "x"),
                    da.full(shape, np.nan, chunks=(1, 1, 4096, 4096), dtype=np.float32),
                )
            },
            coords={"time": time_index, "channel": REQUESTED_CHANNELS},
        )

        encoding = {"data": {"compressor": compressor, "chunks": (1, 1, 4096, 4096)}}
        group_path = f"{year}/{month:02d}"

        # CRITICAL FIX: compute=False ensures we write metadata instantly, lazy-loading the empty arrays
        if not os.path.exists(os.path.join(args.output_zarr, group_path)):
            lgr_logger.info(
                f"Initializing Zarr structure for {year}/{month:02d} (compute=False)..."
            )
            ds_template.to_zarr(
                args.output_zarr,
                group=group_path,
                mode="w",
                encoding=encoding,
                consolidated=False,
                compute=False,
            )
        created_groups.append(os.path.join(args.output_zarr, group_path))

        # Check existing data to resume safely
        lgr_logger.info("Scanning existing chunks to determine resume state...")
        existing_timestamps = get_existing_timestamps(args.output_zarr, group_path)

        # Submit tasks to Dask workers
        futures = []
        for ts, path in files:
            if ts in existing_timestamps:
                continue

            t_idx = time_index.get_loc(ts)
            future = client.submit(
                process_and_write_file,
                path,
                ts,
                t_idx,
                args.output_zarr,
                group_path,
                CHANNEL_MAP,
                REQUESTED_CHANNELS,
            )
            futures.append(future)

        # Monitor Dask progress
        if futures:
            lgr_logger.info(f"Submitting {len(futures)} file operations to Dask cluster...")

            # as_completed allows the progress bar to update as workers finish tasks
            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"Writing {year}/{month:02d}"
            ):
                path, success, error_msg = future.result()
                if not success:
                    lgr_logger.error(f"Failed to process {path}: {error_msg}")
        else:
            lgr_logger.info(f"All files for {year}/{month:02d} are already present and validated.")

    if created_groups:
        lgr_logger.info("Consolidating root metadata...")
        zarr.consolidate_metadata(args.output_zarr)
        lgr_logger.info("Process finished successfully.")
    else:
        lgr_logger.info("No data was written. Exiting.")


if __name__ == "__main__":
    main()
