"""Preprocessing script to compute aligndata and statistics for SDOML datasets.

This script performs the heavy data processing (temporal alignment and statistics calculation)
once and saves the results to files. The DataModule can then load these files directly,
significantly speeding up the start of training and evaluation.

Usage:
    python scripts/preprocess.py --config-name pretrain_mae_HMI
"""

import os
from re import L

import hydra
import numpy as np
import pandas as pd
import yaml
import zarr
from loguru import logger
from omegaconf import DictConfig

from sdofmv2.core.preprocessing import aligntime, calc_normalizations, make_hmi_mask
from sdofmv2.utils import ALL_COMPONENTS, ALL_IONS, ALL_WAVELENGTHS


@hydra.main(
    version_base=None, config_path="../configs/pretrain", config_name="pretrain_mae_AIA.yaml"
)
def main(cfg: DictConfig):
    """Main preprocessing function."""
    logger.info("Starting preprocessing script...")

    # Extract configuration
    sdoml_cfg = cfg.data.sdoml
    min_date = cfg.data.get("min_date", None)
    max_date = cfg.data.get("max_date", None)
    
    year_splits = cfg.data.get("year_splits", {})
    month_splits = cfg.data.get("month_splits", {})
    
    train_years = year_splits.get("train", [])
    val_years = year_splits.get("val", [])
    test_years = year_splits.get("test", [])
    holdout_years = year_splits.get("holdout", [])
    
    train_months = month_splits.get("train", [])
    val_months = month_splits.get("val", [])
    test_months = month_splits.get("test", [])
    holdout_months = month_splits.get("holdout", [])

    # Resolve paths
    base_dir = sdoml_cfg.base_directory
    sub_dir = sdoml_cfg.sub_directory
    output_dir = cfg.data.index_save_path

    hmi_path = os.path.join(base_dir, sub_dir.hmi) if sub_dir.hmi else None
    aia_path = os.path.join(base_dir, sub_dir.aia) if sub_dir.aia else None
    eve_path = os.path.join(base_dir, sub_dir.eve) if sub_dir.eve else None

    # Handle null values for paths
    if hmi_path is not None and (hmi_path == "null" or hmi_path == "None"):
        hmi_path = None
    if aia_path is not None and (aia_path == "null" or aia_path == "None"):
        aia_path = None
    if eve_path is not None and (eve_path == "null" or eve_path == "None"):
        eve_path = None

    # Components, Wavelengths, Ions
    components = sdoml_cfg.components
    wavelengths = sdoml_cfg.wavelengths
    ions = sdoml_cfg.ions

    # Frequency
    frequency = sdoml_cfg.frequency

    # Normalization config
    normalization = sdoml_cfg.normalization

    # Determine training years
    all_years_from_splits = sorted(list(set(train_years + val_years + test_years + holdout_years)))
    if not all_years_from_splits:
        if (min_date is None) and (max_date is None):
            years = set()
            if hmi_path:
                hmi_data = zarr.group(zarr.DirectoryStore(hmi_path))
                years.update(hmi_data.keys())
            if aia_path:
                aia_data = zarr.group(zarr.DirectoryStore(aia_path))
                years.update(aia_data.keys())
            training_years = sorted([int(year) for year in years])
        else:
            min_year = pd.to_datetime(min_date).year
            max_year = pd.to_datetime(max_date).year
            training_years = list(range(min_year, max_year + 1))
    else:
        training_years = all_years_from_splits

    logger.info(f"Processing years: {training_years}")

    # Open Zarr stores
    aia_data = zarr.group(zarr.DirectoryStore(aia_path)) if aia_path else None
    hmi_data = zarr.group(zarr.DirectoryStore(hmi_path)) if hmi_path else None
    eve_data = zarr.group(zarr.DirectoryStore(eve_path)) if eve_path else None

    # Determine default channels if null
    if hmi_data is not None:
        if components is None:
            components = ALL_COMPONENTS
    if aia_data is not None:
        if wavelengths is None:
            wavelengths = ALL_WAVELENGTHS
    if eve_data is not None:
        if ions is None:
            ions = ALL_IONS

    # Generate cache_id
    ids = []
    if hmi_data:
        if len(components) == 3:
            component_id = "HMI_all"
        elif len(components) > 0 and len(components) < 3:
            component_id = "_".join(components)
        ids.append(component_id)

    if aia_data:
        if len(wavelengths) == 9:
            wavelength_id = "AIA_all"
        elif len(wavelengths) > 0 and len(wavelengths) < 9:
            wavelength_id = "_".join(wavelengths)
        ids.append(wavelength_id)

    if eve_data:
        if len(ions) == 38:
            ions_id = "EVE_FULL"
        else:
            ions_id = "_".join(ions).replace(" ", "_")
        ids.append(ions_id)

    if (min_date is None) and (max_date is None):
        cache_id = f"{'_'.join(sorted(ids))}_{frequency}_fullyear"
    else:
        cache_id = f"{'_'.join(sorted(ids))}_{frequency}_{str(min_date).replace(' ', '')}-{str(max_date).replace(' ', '')}"

    if aia_path and "small" in aia_path:
        cache_id += "_small"

    # Output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Compute Aligndata
    full_aligndata_path = os.path.join(output_dir, cache_id + "_aligndata_full.csv")
    exist_aligndata_full = False
    if os.path.exists(full_aligndata_path):
        logger.info(f"Aligndata exists in {full_aligndata_path}")
        aligndata = pd.read_csv(full_aligndata_path)
        aligndata["Time"] = pd.to_datetime(aligndata["Time"])
        aligndata.set_index("Time", inplace=True)
        aligndata.sort_index(inplace=True)
        exist_aligndata_full = True
    else:
        logger.info("Aligndata does not exist. Computing aligndata...")
        target_months = sorted(list(set(train_months + val_months + test_months + holdout_months)))
        aligndata = aligntime(
            aia_data,
            hmi_data,
            eve_data,
            wavelengths,
            components,
            ions,
            frequency,
            training_years,
            target_months=target_months if target_months else None,
            min_date=min_date,
            max_date=max_date,
        )

    # Apply date filters
    if min_date:
        aligndata = aligndata[aligndata.index >= min_date]
    if max_date:
        aligndata = aligndata[aligndata.index <= max_date]

    # 2. Compute HMI Mask
    logger.info("Computing HMI mask...")
    hmi_mask = make_hmi_mask(hmi_data, output_dir)

    # 3. Compute Normalizations (on training data only)
    if normalization.enabled:
        logger.info("Computing normalizations...")
        # Filter aligndata for training months AND years
        if not train_years:
            aligndata_train = aligndata.loc[aligndata.index.month.isin(train_months)]
        else:
            aligndata_train = aligndata.loc[
                aligndata.index.month.isin(train_months) & aligndata.index.year.isin(train_years)
            ]

        _ = calc_normalizations(
            aligndata_train,  # Use train split for stats
            hmi_data,
            aia_data,
            eve_data,
            hmi_mask,
            components,
            wavelengths,
            ions,
            training_years,
            normalization,
            cache_id,
            output_dir,
        )

    # 4. Save Aligndata (Split by months and years)
    logger.info("Saving aligndata splits...")

    def save_aligndata(data, filename):
        data.to_csv(os.path.join(output_dir, cache_id + filename))
        logger.info(f"saved {cache_id + filename} with {len(data)} samples.")

    if not train_years:
        aligndata_train = aligndata.loc[aligndata.index.month.isin(train_months)]
        aligndata_val = aligndata.loc[aligndata.index.month.isin(val_months)]
        aligndata_test = aligndata.loc[aligndata.index.month.isin(test_months)]
    else:
        aligndata_train = aligndata.loc[
            aligndata.index.month.isin(train_months) & aligndata.index.year.isin(train_years)
        ]
        aligndata_val = aligndata.loc[
            aligndata.index.month.isin(val_months) & aligndata.index.year.isin(val_years)
        ]
        aligndata_test = aligndata.loc[
            aligndata.index.month.isin(test_months) & aligndata.index.year.isin(test_years)
        ]

    if not exist_aligndata_full:
        save_aligndata(aligndata, "_aligndata_full.csv")
    save_aligndata(aligndata_train, "_aligndata_train.csv")
    save_aligndata(aligndata_val, "_aligndata_val.csv")
    save_aligndata(aligndata_test, "_aligndata_test.csv")

    if holdout_months:
        if not holdout_years:
            aligndata_holdout = aligndata.loc[aligndata.index.month.isin(holdout_months)]
        else:
            aligndata_holdout = aligndata.loc[
                aligndata.index.month.isin(holdout_months) & aligndata.index.year.isin(holdout_years)
            ]
        save_aligndata(aligndata_holdout, "aligndata_holdout.csv")

    logger.info("Preprocessing complete!")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
