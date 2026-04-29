"""Preprocessing utilities for SDOML data.

This module contains functions for computing aligndata and normalization statistics.
"""

import os
from loguru import logger

import numpy as np
import torch
import yaml
import zarr
import dask.array as da
from dask.diagnostics import ProgressBar
import pandas as pd
from tqdm import tqdm
# from pathlib import Path # Removing unused import if desired, but keeping for safety

from ..utils import ALL_COMPONENTS, ALL_WAVELENGTHS


def compute_stat(key, data):
    """Compute a statistic based on key and log if it's zero."""
    OPS = {
        "sum": lambda x: da.nansum(x),
        "max": lambda x: da.nanmax(x),
        "min": lambda x: da.nanmin(x),
        "std": lambda x: da.nanstd(x),
        "mean": lambda x: da.nanmean(x),
        "median": lambda x: da.percentile(x[~da.isnan(x)], 50),
        "q1": lambda x: da.percentile(x[~da.isnan(x)], 25),
        "q3": lambda x: da.percentile(
            x[~da.isnan(x)],
            75,
        ),
    }

    if key in OPS:
        with ProgressBar():
            return float(OPS[key](data).compute())
    elif key == "image_count":
        return data.shape[0]
    elif key == "pixel_count":
        return data.size
    else:
        raise ValueError(f"Invalid key type: {key}")


def aligntime(
    aia_data,
    hmi_data,
    eve_data,
    wavelengths,
    components,
    ions,
    cadence,
    training_years,
    target_months=None,
    min_date=None,
    max_date=None,
):
    """Extracts common indexes across AIA, HMI, and EVE datasets."""
    join_series = None

    # AIA
    if aia_data is not None:
        logger.info("Aligning AIA data")

        for i, wavelength in enumerate(wavelengths):
            logger.info(f"Aligning AIA data for wavelength: {wavelength}")
            for j, year in enumerate(tqdm((training_years))):
                aia_channel = aia_data[str(year)][wavelength]

                # get observation time
                t_obs_aia_channel = np.array(aia_channel.attrs["T_OBS"])
                if aia_channel.shape[0] != len(t_obs_aia_channel):
                    logger.warning("The length of zarr does not match with T_OBS!")
                    logger.warning(f"year: {year}, wavelength: {wavelength}")

                # Pre-filter by month and date range to save time on NaN checks
                t_obs_dt = pd.to_datetime(t_obs_aia_channel, format="mixed")
                mask = np.ones(len(t_obs_dt), dtype=bool)
                if target_months is not None:
                    mask &= t_obs_dt.month.isin(target_months)
                if min_date is not None:
                    mask &= t_obs_dt >= pd.to_datetime(min_date)
                if max_date is not None:
                    mask &= t_obs_dt <= pd.to_datetime(max_date)

                valid_indices_in_range = np.where(mask)[0]
                if len(valid_indices_in_range) == 0:
                    continue

                # check indices of images without nan
                images = da.from_array(aia_channel, chunks=(512, 512, 512))[valid_indices_in_range]

                # Compute mask of valid images
                valid_mask = ~da.isnan(images).any(axis=(1, 2))

                # Get indices (requires computation)
                logger.info(f"Checking Nans in images, {year} & {wavelength}")
                with ProgressBar():
                    valid_rel_indices = da.nonzero(valid_mask)[0].compute()
                
                valid_indices = valid_indices_in_range[valid_rel_indices]
                total_nan = len(valid_indices)
                logger.info(
                    f"Total {len(valid_indices_in_range) - total_nan} {(len(valid_indices_in_range) - total_nan) * 100 / len(valid_indices_in_range):.0f}% images in range have Nan."
                )

                if j == 0 and (join_series is None or f"idx_{wavelength}" not in join_series.columns):
                    # transform to DataFrame
                    df_t_aia = pd.DataFrame(
                        {
                            "Time": pd.to_datetime(
                                t_obs_aia_channel[valid_indices], format="mixed"
                            ),
                            f"idx_{wavelength}": np.arange(0, len(t_obs_aia_channel))[
                                valid_indices
                            ],
                        }
                    )
                    if df_t_aia[f"idx_{wavelength}"].max() >= len(t_obs_aia_channel):
                        logger.warning("Max index is greater than number of instances in zarr file")

                else:
                    df_tmp_aia = pd.DataFrame(
                        {
                            "Time": pd.to_datetime(
                                t_obs_aia_channel[valid_indices],
                                format="mixed",
                                utc=True,
                            ),
                            f"idx_{wavelength}": np.arange(0, len(t_obs_aia_channel))[
                                valid_indices
                            ],
                        }
                    )
                    if df_tmp_aia[f"idx_{wavelength}"].max() >= len(t_obs_aia_channel):
                        logger.warning("Max index is greater than number of instances in zarr file")
                    
                    # If this is not the first year for this wavelength, concat to df_t_aia
                    # If it IS the first year but we have other wavelengths, we need to handle it.
                    # The original code used 'if j == 0' which assumes j=0 is always the first year with data.
                    # With filtering, j=0 might not have data.
                    if 'df_t_aia' in locals():
                        df_t_aia = pd.concat([df_t_aia, df_tmp_aia], ignore_index=True)
                    else:
                        df_t_aia = df_tmp_aia

            if 'df_t_aia' not in locals():
                continue

            # Enforcing same datetime format
            transform_datetime = lambda x: pd.to_datetime(x, format="mixed").strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            df_t_aia["Time"] = df_t_aia["Time"].apply(transform_datetime)
            df_t_aia["Time"] = pd.to_datetime(df_t_aia["Time"]).dt.tz_localize(
                None
            )  # this is needed for timezone-naive type

            df_t_aia["Time"] = df_t_aia["Time"].dt.round("1min")
            cadence_min = int(pd.to_timedelta(cadence).total_seconds() // 60)
            df_t_aia = df_t_aia.loc[df_t_aia["Time"].dt.minute % cadence_min == 0]
            df_t_obs_aia = df_t_aia.drop_duplicates(
                subset="Time", keep="first"
            )  # removing potential duplicates derived by rounding
            df_t_obs_aia.set_index("Time", inplace=True)

            if join_series is None:
                join_series = df_t_obs_aia
            else:
                join_series = join_series.join(df_t_obs_aia, how="inner")

            # Clean up for next wavelength
            del df_t_aia

        logger.info(f"AIA alignment completed with {join_series.shape[0] if join_series is not None else 0} samples.")

    # HMI
    if hmi_data is not None:
        logger.info("Aligning HMI data")
        for i, component in enumerate(components):
            logger.info(f"Aligning HMI data for component: {component}")
            for j, year in enumerate(tqdm((training_years))):
                hmi_channel = hmi_data[str(year)][component]

                # get observation time
                t_obs_hmi_channel_pre = hmi_channel.attrs["T_OBS"]
                
                # Pre-processing times for filtering
                t_obs_hmi_channel_processed = []
                replacements = {".": "-", "_": "T", "TTAI": "", "60": "59"}
                for word in t_obs_hmi_channel_pre:
                    word = word[:19]
                    for old_char, new_char in replacements.items():
                        word = word.replace(old_char, new_char)
                    t_obs_hmi_channel_processed.append(word)
                
                t_obs_dt = pd.to_datetime(t_obs_hmi_channel_processed, format="mixed", utc=True)
                
                # Pre-filter
                mask = np.ones(len(t_obs_dt), dtype=bool)
                if target_months is not None:
                    mask &= t_obs_dt.month.isin(target_months)
                if min_date is not None:
                    mask &= t_obs_dt >= pd.to_datetime(min_date).tz_localize("UTC")
                if max_date is not None:
                    mask &= t_obs_dt <= pd.to_datetime(max_date).tz_localize("UTC")
                
                valid_indices_in_range = np.where(mask)[0]
                if len(valid_indices_in_range) == 0:
                    continue

                # check indices of images without nan
                images = da.from_array(hmi_channel, chunks=(512, 512, 512))[valid_indices_in_range]

                # Compute mask of valid images
                valid_mask = ~da.isnan(images).any(axis=(1, 2))

                # Get indices (requires computation)
                logger.info(f"Checking Nans in images, {year} & {component}")
                with ProgressBar():
                    valid_rel_indices = da.nonzero(valid_mask)[0].compute()
                
                valid_indices = valid_indices_in_range[valid_rel_indices]
                total_nan = len(valid_indices)
                logger.info(
                    f"Total {len(valid_indices_in_range) - total_nan} {(len(valid_indices_in_range) - total_nan) * 100 / len(valid_indices_in_range):.0f}% images in range have Nan."
                )

                if j == 0 and (join_series is None or f"idx_{component}" not in join_series.columns):
                    # transform to DataFrame
                    df_t_hmi = pd.DataFrame(
                        {
                            "Time": t_obs_dt[valid_indices],
                            f"idx_{component}": np.arange(0, len(t_obs_hmi_channel_pre))[valid_indices],
                        }
                    )
                else:
                    df_tmp_hmi = pd.DataFrame(
                        {
                            "Time": t_obs_dt[valid_indices],
                            f"idx_{component}": np.arange(0, len(t_obs_hmi_channel_pre))[valid_indices],
                        }
                    )
                    if 'df_t_hmi' in locals():
                        df_t_hmi = pd.concat([df_t_hmi, df_tmp_hmi], ignore_index=True)
                    else:
                        df_t_hmi = df_tmp_hmi

            if 'df_t_hmi' not in locals():
                continue

            # Enforcing same datetime format
            transform_datetime = lambda x: pd.to_datetime(x, format="mixed").strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            df_t_hmi["Time"] = df_t_hmi["Time"].apply(transform_datetime)
            df_t_hmi["Time"] = pd.to_datetime(df_t_hmi["Time"]).dt.tz_localize(
                None
            )  # this is needed for timezone-naive type
            df_t_hmi["Time"] = df_t_hmi["Time"].dt.round(cadence)
            df_t_obs_hmi = df_t_hmi.drop_duplicates(
                subset="Time", keep="first"
            )  # removing potential duplicates derived by rounding
            df_t_obs_hmi.set_index("Time", inplace=True)

            if join_series is None:
                join_series = df_t_obs_hmi
            else:
                join_series = join_series.join(df_t_obs_hmi, how="inner")
            
            del df_t_hmi

    logger.info(f"HMI alignment completed with {join_series.shape[0] if join_series is not None else 0} samples.")

    # EVE
    if eve_data is not None:
        logger.info("Aligning EVE data")
        df_t_eve = pd.DataFrame(
            {
                "Time": pd.to_datetime(eve_data["Time"]),
                "idx_eve": np.arange(0, len(eve_data["Time"])),
            }
        )
        df_t_eve["Time"] = pd.to_datetime(df_t_eve["Time"]).dt.round(cadence)
        
        # Apply filters to EVE
        if target_months is not None:
            df_t_eve = df_t_eve[df_t_eve["Time"].dt.month.isin(target_months)]
        if min_date is not None:
            df_t_eve = df_t_eve[df_t_eve["Time"] >= pd.to_datetime(min_date)]
        if max_date is not None:
            df_t_eve = df_t_eve[df_t_eve["Time"] <= pd.to_datetime(max_date)]
            
        df_t_obs_eve = df_t_eve.drop_duplicates(subset="Time", keep="first").set_index("Time")

        if join_series is None:
            join_series = df_t_obs_eve
        else:
            join_series = join_series.join(df_t_obs_eve, how="inner")

        # remove missing eve data (missing values are labeled with negative values)
        # this will remove all but 16 values if the partial year 2014 is included
        for ion in ions:
            if ion == "Fe XVI_2":
                continue
            ion_data = eve_data[ion]
            join_series = join_series.loc[ion_data[join_series["idx_eve"]] > 0]

    if join_series is None:
        raise ValueError("No data found for alignment.")

    join_series.sort_index(inplace=True)

    logger.info(f"Total Alignment Completed with {join_series.shape[0]} Samples.")

    return join_series


def calc_normalizations(
    aligndata,
    hmi_data,
    aia_data,
    eve_data,
    hmi_mask,
    components,
    wavelengths,
    ions,
    training_years,
    normalization_cfg,
    cache_id,
    cache_dir,
):
    """Compute normalization statistics."""
    import json

    normalizations = {}
    normalizations_align = aligndata.copy()

    if eve_data is not None:
        normalizations["EVE"] = _compute_data_statistic(
            normalizations_align,
            eve_data,
            "EVE",
            ions,
            hmi_mask,
            components,
            wavelengths,
            training_years,
            normalization_cfg,
            cache_id,
            cache_dir,
        )

    if aia_data is not None:
        normalizations["AIA"] = _compute_data_statistic(
            normalizations_align,
            aia_data,
            "AIA",
            wavelengths,
            hmi_mask,
            components,
            wavelengths,
            training_years,
            normalization_cfg,
            cache_id,
            cache_dir,
        )

    if hmi_data is not None:
        normalizations["HMI"] = _compute_data_statistic(
            normalizations_align,
            hmi_data,
            "HMI",
            components,
            hmi_mask,
            components,
            wavelengths,
            training_years,
            normalization_cfg,
            cache_id,
            cache_dir,
        )

    return normalizations


def _compute_data_statistic(
    normalizations_align,
    sdoml_data,
    instrument,
    channels,
    hmi_mask,
    components,
    wavelengths,
    training_years,
    normalization_cfg,
    cache_id,
    cache_dir,
):
    import json

    normalizations_stat = {}
    for ch in channels:
        file_name = (
            cache_dir
            + f"/{instrument}/"
            + ch
            + "_"
            + "_".join(cache_id.split("_")[-1:])
            + f"_norm-{normalization_cfg.type}"
            + ".json"
        )

        # check components
        check_list = [
            "sum",
            "max",
            "min",
            "mean",
            "std",
            "median",
            "q1",
            "q3",
            "image_count",
            "pixel_count",
        ]

        # Check existing stat info
        if os.path.exists(file_name):
            logger.info(f"Cache is found: {file_name}")
            with open(file_name, "r") as json_file:
                stat = json.load(json_file)
                normalizations_stat[ch] = stat
                check_list = [k for k in check_list if k not in normalizations_stat[ch].keys()]
                if len(check_list) == 0:
                    continue
        else:
            normalizations_stat[ch] = {}

        # if all the statistics exist, pass.
        if len(check_list) == 0:
            continue

        ch_arr = []
        years_in_data = sorted(normalizations_align.index.year.unique())
        for year in years_in_data:
            ch_data_year = da.from_array(sdoml_data[str(year)][ch])
            ch_idices = normalizations_align.loc[
                normalizations_align.index.year == year, f"idx_{ch}"
            ]
            ch_data_year = ch_data_year[ch_idices]

            # put nan to limb
            if hmi_mask is not None:
                mask_expanded = hmi_mask.cpu().numpy()[None, :, :].astype(bool)
                ch_data_year = da.where(
                    mask_expanded == 1, ch_data_year, np.nan
                )  # outter area to nan

            ch_arr.append(ch_data_year.flatten())
        
        if not ch_arr:
             logger.warning(f"No data for {ch} in the specified range.")
             continue
             
        ch_data = da.concatenate(ch_arr)

        if normalization_cfg.type == "log":
            scaler_factor = normalization_cfg.get("scaler_factor", None)
            ch_data = ch_data * scaler_factor if scaler_factor is not None else ch_data
            ch_data = da.sign(ch_data) * da.log1p(da.abs(ch_data))

        elif normalization_cfg.type == "zscore" and normalization_cfg.clipping.enabled:
            low, high = normalization_cfg.clipping[ch]
            if normalization_cfg.clipping.enabled:
                ch_data = da.clip(ch_data, low, high)

        logger.info(f"\nCalculating normalizations for wavelength {ch}:")
        logger.info("-" * 50)

        for stat_measure in check_list:
            logger.info(f"Computing {stat_measure} of {ch}")
            stat_value = compute_stat(stat_measure, ch_data)
            if stat_value == 0:
                logger.warning(f"Value of {stat_measure} is Zero!, wavelength: {ch}")
            normalizations_stat[ch][stat_measure] = stat_value

        # save statistics of each wavelength
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as json_file:
            json.dump(normalizations_stat[ch], json_file)
            logger.info(f"Saved stats to {file_name}")

    return normalizations_stat


def make_hmi_mask(hmi_data, cache_dir):
    """Generate and save HMI mask."""
    from pathlib import Path
    hmi_mask_cache_filename = f"{cache_dir}/hmi_mask_512x512.npy"
    if Path(hmi_mask_cache_filename).exists():
        loaded_mask = np.load(hmi_mask_cache_filename)
        hmi_mask = torch.Tensor(loaded_mask).to(dtype=torch.uint8)
        logger.info(f"[* CACHE SYSTEM *] Found cached HMI mask data in {hmi_mask_cache_filename}.")
        return hmi_mask
    elif hmi_data is None:
        raise ValueError(
            "Mask could not be found in cache and 2010 HMI data is not available to generate it, stopping..."
        )

    hmi = torch.Tensor(hmi_data[str(2010)][ALL_COMPONENTS[0]][0])
    hmi_mask = (torch.abs(hmi) > 0.0).to(dtype=torch.uint8)
    hmi_mask_ratio = hmi_mask.sum().item() / hmi_mask.numel()
    if np.abs(hmi_mask_ratio - 0.496) > 0.2:
        logger.warning(
            f"WARNING: HMI mask ratio is {hmi_mask_ratio:.2f}, which is significantly different from expected (0.496)"
        )
    logger.info(
        f"[*] Saving HMI mask with ratio {hmi_mask_ratio:.2f} to {hmi_mask_cache_filename}."
    )
    np.save(hmi_mask_cache_filename, hmi_mask.numpy())
    return hmi_mask
