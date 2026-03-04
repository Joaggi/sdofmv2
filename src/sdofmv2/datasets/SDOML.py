# Adapted to be general from https://github.com/FrontierDevelopmentLab/2023-FDL-X-ARD-EVE/blob/main/src/irradiance/utilities/data_loader.py

import json
import os
import time
from pathlib import Path
from typing import Optional
from loguru import logger

import dask
import dask.array as da
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import zarr
from dask.array import stats
from dask.diagnostics import ProgressBar
from torch.utils.data import Dataset
from tqdm import tqdm

from ..constants import ALL_COMPONENTS, ALL_IONS, ALL_WAVELENGTHS


def get_dtype_from_precision(precision):
    if str(precision) in ["16", "16-mixed"]:
        return torch.float16
    elif str(precision) in ["bf16", "bf16-mixed"]:
        return torch.bfloat16
    elif str(precision) in ["64", "64-true"]:
        return torch.float64
    else:
        return torch.float32


def zscore_norm(data, instument, channel, normalization_stat, clip_value):
    if clip_value is not None:
        low, high = clip_value
        data = np.clip(data, low, high)
    data -= normalization_stat[instument][channel]["mean"]
    data /= normalization_stat[instument][channel]["std"]
    return data


def min_max_norm(data, instument, channel, normalization_stat):
    data -= normalization_stat[instument][channel]["min"]
    data /= (
        normalization_stat[instument][channel]["max"]
        - normalization_stat[instument][channel]["min"]
    )
    return data


def log_norm(
    data, normalization_stat, instrument, channel, scaler_factor, scaler_div_factor
):

    x = data * scaler_factor if scaler_factor is not None else data

    # Log transform
    x_log = np.sign(x) * np.log1p(np.abs(x))

    # Divide by the SINGLE global scalar
    if scaler_div_factor is None:
        x_transformed = x_log / normalization_stat[instrument][channel]["max"]
    else:
        x_transformed = x_log / scaler_div_factor

    return x_transformed


def inverse_zscore_norm(data, instrument, channel, normalization_stat):
    # Reverse the division
    data = data * normalization_stat[instrument][channel]["std"]

    # Reverse the subtraction
    data = data + normalization_stat[instrument][channel]["mean"]

    return data


def inverse_log_norm(
    data_transformed,
    normalization_stat,
    instrument,
    channel,
    scaler_factor,
    scaler_div_factor,
):
    # Reverse the Division
    if scaler_div_factor is None:
        denom = normalization_stat[instrument][channel]["max"]
    else:
        denom = scaler_div_factor

    x_log = data_transformed * denom

    # Inverse Log Transform
    # The inverse of y = sign(x) * log(1 + |x|) is x = sign(y) * (exp(|y|) - 1)
    # np.expm1 calculates (e^x - 1) with higher precision for small values
    x = np.sign(x_log) * np.expm1(np.abs(x_log))

    # Reverse the Scaling
    if scaler_factor is not None:
        data_original = x / scaler_factor
    else:
        data_original = x

    return data_original


class SDOMLDataset(Dataset):
    def __init__(
        self,
        aligndata,
        hmi_data,
        aia_data,
        eve_data,
        components,
        wavelengths,
        ions,
        freq,
        months,
        normalization=None,
        normalization_stat=None,
        mask=None,
        num_frames=1,
        drop_frame_dim=False,
        min_date=None,
        max_date=None,
        get_header=False,  # Optional[list] = [],
        precision="32",
    ):
        """
        aligndata --> aligned indexes for input-output matching
        aia_data --> zarr: aia data in zarr format
        eve_path --> zarr: eve data in zarr format
        hmi_path --> zarr: hmi data in zarr format
        components --> list: list of magnetic components for hmi (Bx, By, Bz)
        wavelengths   --> list: list of channels for aia (94, 131, 171, 193, 211, 304, 335, 1600, 1700)
        ions          --> list: list of ions for eve (MEGS A and MEGS B)
        freq          --> str: cadence used for rounding time series
        transformation: to be applied to aia in theory, but can stay None here
        use_normalizations: to use or not use normalizations, e.g. if this is test data, we don't want to use normalizations
        mask: to apply or not apply the HMI mask to AIA and HMI data
        """
        super().__init__()

        self.aligndata = aligndata
        self.aia_data = aia_data
        self.eve_data = eve_data
        self.hmi_data = hmi_data

        self.mask = mask
        self.get_header = get_header
        self.precision = precision

        # Select alls
        self.components = components
        self.wavelengths = wavelengths
        self.ions = ions

        # Loading data
        # HMI
        if self.hmi_data is not None:
            if self.components is None:
                self.components = ALL_COMPONENTS
            self.components.sort()
        # AIA
        if self.aia_data is not None:
            if self.wavelengths is None:
                self.wavelengths = ALL_WAVELENGTHS
            self.wavelengths.sort()
        # EVE
        if self.eve_data is not None:
            if self.ions is None:
                self.ions = ALL_IONS
            self.ions.sort()
        self.cadence = freq
        self.months = months
        self.normalization = normalization
        self.normalization_stat = normalization_stat

        # get data from path
        self.aligndata = self.aligndata.loc[
            self.aligndata.index.month.isin(self.months), :
        ]

        self.aligndata = self.aligndata[
            (self.aligndata.index >= min_date) & (self.aligndata.index <= max_date)
        ]

        # number of frames to return per sample
        self.num_frames = num_frames
        self.drop_frame_dim = drop_frame_dim  # for backwards compat
        if self.drop_frame_dim:
            assert self.num_frames == 1

    def __len__(self):
        # report slightly smaller such that all frame sets requested are available
        return self.aligndata.shape[0] - (self.num_frames - 1)

    def __getitem__(self, idx):
        image_stack = None
        header_stack = {}

        if self.aia_data is not None:
            aia_images, aia_headers = self.get_aia_image(idx)
            image_stack = aia_images
            header_stack.update(aia_headers)

        if self.hmi_data is not None:
            hmi_images, hmi_headers = self.get_hmi_image(idx)
            if image_stack is None:
                image_stack = hmi_images
            else:
                image_stack = np.concatenate((image_stack, hmi_images), axis=0)
            header_stack.update(hmi_headers)

        image_stack = torch.from_numpy(image_stack)
        image_stack = image_stack.to(get_dtype_from_precision(self.precision))
        timestamps = self.aligndata.index[idx : idx + self.num_frames].astype("int")
        # timestamps = timestamps.astype("int")
        timestamps = timestamps[0] if self.num_frames <= 1 else timestamps

        if not self.get_header:
            if self.eve_data is not None:
                eve_data = self.get_eve(idx)
                return image_stack, timestamps, eve_data
            else:
                return image_stack, timestamps
        else:
            if self.eve_data is not None:
                eve_data = self.get_eve(idx)
                return image_stack, timestamps, header_stack, eve_data.reshape(-1)
            else:
                return image_stack, timestamps, header_stack

    def _data_norm(self, data, instrument, channel):
        """
        data: numpy array of shape H W
        """
        if self.normalization.type == "log":
            return log_norm(
                data,
                self.normalization_stat,
                instrument,
                channel,
                self.normalization.scaler_factor,
                self.normalization.scaler_div_factor,
            )

        elif self.normalization.type == "zscore":
            return zscore_norm(
                data,
                instrument,
                channel,
                self.normalization_stat,
                (
                    self.normalization.clipping[channel]
                    if self.normalization.clipping.enabled
                    else None
                ),
            )

        elif self.normalization.type == "min-max":
            return min_max_norm(data, instrument, channel, self.normalization_stat)

    def loading_data_retry(
        self,
        data,
        year,
        wavelength,
        id_of_img,
        num_try: int = 10,
        sleep_time: float = 0.5,
    ):
        """
        Tries to load an image from the dataset multiple times to handle transient
        """
        last_error = None

        for attempt in range(num_try):
            try:
                # Attempt to slice the data (triggering decompression)
                img = data[year][wavelength][id_of_img, :, :]
                return img

            except Exception as e:
                # Store error, log warning, and wait before retrying
                last_error = e
                logger.warning(
                    f"Corrupted load (Attempt {attempt+1}/{num_try}) - "
                    f"channel: {wavelength}, year: {year}, idx: {id_of_img}. Error: {e}"
                )
                time.sleep(sleep_time)

        # If the loop finishes, we failed 'num_try' times.
        # Raise the last error to stop execution (or return zeros if preferred).
        logger.error(
            f"PERMANENT FAILURE: Could not load data after {num_try} attempts."
        )
        raise last_error

    def get_aia_image(self, idx):
        """Get AIA image for a given index.
        Returns a numpy array of shape (num_wavelengths, num_frames, height, width).
        """

        aia_image_dict = {}
        aia_header_dict = {}
        for wavelength in self.wavelengths:
            aia_image_dict[wavelength] = []

            if self.get_header:
                aia_header_dict[wavelength] = []
            for frame in range(self.num_frames):
                idx_row_element = self.aligndata.iloc[idx + frame]
                idx_wavelength = idx_row_element[f"idx_{wavelength}"].astype(int)
                year = str(idx_row_element.name.year)
                # img = self.aia_data[year][wavelength][idx_wavelength, :, :]
                img = self.loading_data_retry(
                    self.aia_data, year, wavelength, idx_wavelength, 10, 0.5
                )

                if self.mask is not None:
                    img = img * self.mask

                aia_image_dict[wavelength].append(img)

                if self.get_header:
                    # aia_header_dict[wavelength].append(self.aia_data[year][wavelength].attrs[self.attrs][idx_wavelength])
                    try:
                        aia_header_dict[wavelength].append(
                            {
                                keys: values[idx_wavelength]
                                for keys, values in self.aia_data[year][
                                    wavelength
                                ].attrs.items()
                            }
                        )
                    except:
                        aia_header_dict[wavelength].append(None)

                if self.normalization.enabled:
                    aia_image_dict[wavelength][-1] = self._data_norm(
                        aia_image_dict[wavelength][-1], "AIA", wavelength
                    )

        aia_image = np.array(list(aia_image_dict.values()))

        return (
            (aia_image[:, 0, :, :], aia_header_dict)
            if self.drop_frame_dim
            else (aia_image, aia_header_dict)
        )

    def get_hmi_image(self, idx):
        """Get HMI image for a given index.
        Returns a numpy array of shape (num_channels, num_frames, height, width).
        """
        hmi_image_dict = {}
        hmi_header_dict = {}
        for component in self.components:
            hmi_image_dict[component] = []

            if self.get_header:
                hmi_header_dict[component] = []

            for frame in range(self.num_frames):
                idx_row_element = self.aligndata.iloc[idx + frame]
                idx_component = idx_row_element[f"idx_{component}"].astype(int)
                year = str(idx_row_element.name.year)
                # img = self.hmi_data[year][component][idx_component, :, :]
                img = self.loading_data_retry(
                    self.hmi_data, year, component, idx_component, 10, 0.5
                )

                if self.mask is not None:
                    img = img * self.mask

                hmi_image_dict[component].append(img)

                if self.get_header:
                    hmi_header_dict[component].append(
                        {
                            keys: values[idx_component]
                            for keys, values in self.hmi_data[year][
                                component
                            ].attrs.items()
                        }
                    )

                if self.normalization.enabled:
                    hmi_image_dict[component][-1] = self._data_norm(
                        hmi_image_dict[component][-1], "HMI", component
                    )

        hmi_image = np.array(list(hmi_image_dict.values()))

        return (
            (hmi_image[:, 0, :, :], hmi_header_dict)
            if self.drop_frame_dim
            else (hmi_image, hmi_header_dict)
        )

    def get_eve(self, idx):
        """Get EVE data for a given index.
        Returns a numpy array of shape (num_ions, num_frames, ...).
        """
        eve_ion_dict = {}
        for ion in self.ions:
            eve_ion_dict[ion] = []
            for frame in range(self.num_frames):
                idx_eve = self.aligndata.iloc[idx + frame]["idx_eve"]
                eve_ion_dict[ion].append(self.eve_data[ion][idx_eve])
                if self.normalization.enabled:
                    eve_ion_dict[ion][-1] = self._data_norm(
                        eve_ion_dict[ion][-1], "EVE", ion
                    )

        eve_data = np.array(list(eve_ion_dict.values()), dtype=np.float32)

        return eve_data

    def __str__(self):
        output = ""
        for k, v in self.__dict__.items():
            output += f"{k}: {v}\n"
        return output


class SDOMLDataModule(pl.LightningDataModule):
    """Loads paired data samples of AIA EUV images and EVE irradiance measures.

    Note: Input data needs to be paired.
    Parameters
    ----------
    hmi_path: path to hmi zarr file
    aia_path: path to aia zarr file
    eve_path: path to the EVE zarr data file
    components: list of magnetic field components
    batch_size: batch size (default is 32)
    num_workers: number of workers (needed for the training)
    val_months/test_months/holdout_monts: list of onths used to split the data
    cache_dir: path to directory for cashing data
    """

    def __init__(
        self,
        hmi_path,
        aia_path,
        eve_path,
        components,
        wavelengths,
        ions,
        frequency,
        batch_size: int = 32,
        num_workers=None,
        pin_memory=False,
        persistent_workers=False,
        val_months=[10, 1],
        test_months=[11, 12],
        holdout_months=[],
        predict_months=[],
        normalization=False,
        cache_dir="",
        norm_stat_tag="",
        apply_mask=True,
        num_frames=1,
        drop_frame_dim=False,
        min_date=None,
        max_date=None,
        precision="32",
    ):

        super().__init__()
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count() // 2
        )
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.hmi_path = hmi_path
        self.aia_path = aia_path
        self.eve_path = eve_path
        self.batch_size = batch_size
        self.cadence = frequency
        self.val_months = val_months
        self.test_months = test_months
        self.holdout_months = holdout_months
        self.predict_months = predict_months
        self.cache_dir = cache_dir
        self.norm_stat_tag = norm_stat_tag
        self.apply_mask = apply_mask
        self.num_frames = num_frames
        self.drop_frame_dim = drop_frame_dim
        self.min_date = pd.to_datetime(min_date) if min_date is not None else None
        self.max_date = pd.to_datetime(max_date) if max_date is not None else None
        self.isAIA = True if self.aia_path is not None else False
        self.isHMI = True if self.hmi_path is not None else False
        self.isEVE = True if self.eve_path is not None else False
        self.precision = precision

        # Select alls
        self.components = components
        self.wavelengths = wavelengths
        self.ions = ions

        # checking if AIA is in the dataset
        if self.isAIA:
            self.aia_data = zarr.group(zarr.DirectoryStore(self.aia_path))
            if self.wavelengths is None:
                self.wavelengths = ALL_WAVELENGTHS
        else:
            self.aia_data = None

        # checking if HMI is in the dataset
        if self.isHMI:
            self.hmi_data = zarr.group(zarr.DirectoryStore(self.hmi_path))
            if self.components is None:
                self.components = ALL_COMPONENTS
        else:
            self.hmi_data = None

        # checking if EVE is in the dataset
        if self.isEVE:
            self.eve_data = zarr.group(zarr.DirectoryStore(self.eve_path))
            if self.ions is None:
                self.ions = ALL_IONS
        else:
            self.eve_data = None

        self.train_months = [
            i
            for i in range(1, 13)
            if i not in self.test_months + self.val_months + self.holdout_months
        ]

        # Cache filenames
        ids = []

        if self.isHMI:
            if len(self.components) == 3:
                component_id = "HMI_FULL"
            elif len(self.components) > 0 and len(self.components) < 3:
                component_id = "_".join(self.components)
            ids.append(component_id)

        if self.isAIA:
            if len(self.wavelengths) == 9:
                wavelength_id = "AIA_FULL"
            elif len(self.wavelengths) > 0 and len(self.wavelengths) < 9:
                wavelength_id = "_".join(self.wavelengths)
            ids.append(wavelength_id)

        if self.isEVE:
            if len(self.ions) == 38:  # excluding Fe XVI_2
                ions_id = "EVE_FULL"
            else:
                ions_id = "_".join(self.ions).replace(" ", "_")
            ids.append(ions_id)

        if (self.min_date is None) and (self.max_date is None):
            if not self.isEVE:
                years = set()
                if self.aia_data:
                    years.update(self.aia_data.keys())
                if self.hmi_data:
                    years.update(self.hmi_data.keys())
                self.training_years = sorted([int(year) for year in years])
            else:  # EVE included, limit to 2010-2014
                self.training_years = sorted(
                    [int(year) for year in self.hmi_data.keys() if int(year) < 2015]
                )
            self.cache_id = f"{'_'.join(sorted(ids))}_{self.cadence}_fulldata"
        else:
            min_year = self.min_date.year
            max_year = self.max_date.year
            self.training_years = list(range(min_year, max_year + 1))
            self.cache_id = f"{'_'.join(sorted(ids))}_{self.cadence}_{str(self.min_date).replace(' ', '')}-{str(self.max_date).replace(' ', '')}"

        if self.aia_path is not None:
            if "small" in self.aia_path:
                self.cache_id += "_small"

        self.index_cache_filename = f"{cache_dir}/aligndata_{self.cache_id}.csv"
        self.hmi_mask_cache_filename = f"{cache_dir}/hmi_mask_512x512.npy"

        self.aligndata = (
            self.__aligntime()
        )  # Temporal alignment of hmi, aia and eve data
        # define min-max date after creating align data
        align_min_date = self.aligndata.index.min()
        align_max_date = self.aligndata.index.max()
        self.min_date = max(self.min_date or align_min_date, align_min_date)
        self.max_date = min(self.max_date or align_max_date, align_max_date)
        self.hmi_mask = self.__make_hmi_mask()
        self.normalization = normalization
        self.normalization_stat = (
            self.__calc_normalizations() if normalization.enabled is True else None
        )

    def __str__(self):
        output = ""
        for k, v in self.__dict__.items():
            output += f"{k}: {v}\n"
        return output

    def __aligntime(self):
        """
        This function extracts the common indexes across aia and eve datasets, considering potential missing values.
        """

        # Check the cache
        if Path(self.index_cache_filename).exists():
            print(
                f"[* CACHE SYSTEM *] Found cached index data in {self.index_cache_filename}."
            )
            aligndata = pd.read_csv(self.index_cache_filename)
            aligndata["Time"] = pd.to_datetime(aligndata["Time"])
            aligndata.set_index("Time", inplace=True)
            return aligndata
        print(f"No alignment cache found at {self.index_cache_filename}")
        print("\nData alignment calculation begin:")
        print("-" * 50)

        join_series = None

        # AIA
        if self.isAIA:
            print("Aligning AIA data")

            for i, wavelength in enumerate(self.wavelengths):
                print(f"Aligning AIA data for wavelength: {wavelength}")
                for j, year in enumerate(tqdm((self.training_years))):
                    aia_channel = self.aia_data[str(year)][wavelength]

                    # get observation time
                    t_obs_aia_channel = np.array(aia_channel.attrs["T_OBS"])
                    if aia_channel.shape[0] != len(t_obs_aia_channel):
                        logger.warning(f"The length of zarr does not match with T_OBS!")
                        logger.warning(f"year: {year}, wavelength: {wavelength}")

                    # check indices of images without nan
                    images = da.from_array(aia_channel, chunks=(512, 512, 512))

                    # Compute mask of valid images
                    valid_mask = ~da.isnan(images).any(axis=(1, 2))

                    # Get indices (requires computation)
                    logger.info(f"Checking Nans in images, {year} & {wavelength}")
                    with ProgressBar():
                        valid_indices = da.nonzero(valid_mask)[0].compute()
                    total_nan = len(valid_indices)
                    logger.info(
                        f"Total {images.shape[0]-total_nan} {(images.shape[0]-total_nan)*100/images.shape[0]:.0f}% images have Nan."
                    )

                    if j == 0:
                        # transform to DataFrame
                        # AIA
                        df_t_aia = pd.DataFrame(
                            {
                                "Time": pd.to_datetime(
                                    t_obs_aia_channel[valid_indices], format="mixed"
                                ),
                                f"idx_{wavelength}": np.arange(
                                    0, len(t_obs_aia_channel)
                                )[valid_indices],
                            }
                        )
                        if df_t_aia[f"idx_{wavelength}"].max() >= len(
                            t_obs_aia_channel
                        ):
                            logger.warning(
                                "Max index is greater than number of instances in zarr file"
                            )

                    else:
                        df_tmp_aia = pd.DataFrame(
                            {
                                "Time": pd.to_datetime(
                                    t_obs_aia_channel[valid_indices],
                                    format="mixed",
                                    utc=True,
                                ),
                                f"idx_{wavelength}": np.arange(
                                    0, len(t_obs_aia_channel)
                                )[valid_indices],
                            }
                        )
                        if df_tmp_aia[f"idx_{wavelength}"].max() >= len(
                            t_obs_aia_channel
                        ):
                            logger.warning(
                                "Max index is greater than number of instances in zarr file"
                            )
                        df_t_aia = pd.concat([df_t_aia, df_tmp_aia], ignore_index=True)

                # Enforcing same datetime format
                transform_datetime = lambda x: pd.to_datetime(
                    x, format="mixed"
                ).strftime("%Y-%m-%d %H:%M:%S")
                df_t_aia["Time"] = df_t_aia["Time"].apply(transform_datetime)
                df_t_aia["Time"] = pd.to_datetime(df_t_aia["Time"]).dt.tz_localize(
                    None
                )  # this is needed for timezone-naive type

                df_t_aia["Time"] = df_t_aia["Time"].dt.round("1min")
                cadence_min = int(pd.to_timedelta(self.cadence).total_seconds() // 60)
                df_t_aia = df_t_aia.loc[df_t_aia["Time"].dt.minute % cadence_min == 0]
                df_t_obs_aia = df_t_aia.drop_duplicates(
                    subset="Time", keep="first"
                )  # removing potential duplicates derived by rounding
                df_t_obs_aia.set_index("Time", inplace=True)

                # if i == 0:
                if join_series is None:
                    join_series = df_t_obs_aia
                else:
                    join_series = join_series.join(df_t_obs_aia, how="inner")

                # after all years for this wavelength are processed
                idx_col = f"idx_{wavelength}"

                for year in self.training_years:
                    if (
                        join_series.loc[join_series.index.year == year, idx_col]
                    ).max() >= self.aia_data[str(year)][wavelength].shape[0]:

                        logger.warning(
                            f"Max id is greater than number instances in zarr file"
                        )
                        logger.warning(f"year: {year}, channel: {wavelength}")

            logger.info(f"AIA alignment completed with {join_series.shape[0]} samples.")

        # ----------------------------------------------------------------------------------------------------------------------------------

        # HMI
        if self.isHMI:
            print("Aligning HMI data")
            for i, component in enumerate(self.components):
                print(f"Aligning HMI data for component: {component}")
                for j, year in enumerate(
                    tqdm((self.training_years))
                ):  # EVE data only goes up to 2014

                    hmi_channel = self.hmi_data[year][component]

                    # get observation time
                    t_obs_hmi_channel_pre = hmi_channel.attrs["T_OBS"]

                    for idx, time_val in enumerate(t_obs_hmi_channel_pre):
                        t_obs_hmi_channel_pre[idx] = time_val[:19]

                    # substitute characters
                    replacements = {".": "-", "_": "T", "TTAI": "", "60": "59"}
                    t_obs_hmi_channel = []
                    for word in t_obs_hmi_channel_pre:
                        for old_char, new_char in replacements.items():
                            word = word.replace(old_char, new_char)
                        t_obs_hmi_channel.append(word)
                    t_obs_hmi_channel = np.array(t_obs_hmi_channel)

                    # check indices of images without nan
                    images = da.from_array(hmi_channel, chunks=(512, 512, 512))

                    # Compute mask of valid images
                    valid_mask = ~da.isnan(images).any(axis=(1, 2))

                    # Get indices (requires computation)
                    logger.info(f"Checking Nans in images, {year} & {component}")
                    with ProgressBar():
                        valid_indices = da.nonzero(valid_mask)[0].compute()
                    total_nan = len(valid_indices)
                    logger.info(
                        f"Total {images.shape[0]-total_nan} {(images.shape[0]-total_nan)*100/images.shape[0]:.0f}% images have Nan."
                    )

                    if j == 0:
                        # transform to DataFrame
                        # HMI
                        df_t_hmi = pd.DataFrame(
                            {
                                "Time": pd.to_datetime(
                                    t_obs_hmi_channel[valid_indices],
                                    format="mixed",
                                    utc=True,
                                ),
                                f"idx_{component}": np.arange(
                                    0, len(t_obs_hmi_channel)
                                )[valid_indices],
                            }
                        )

                    else:
                        df_tmp_hmi = pd.DataFrame(
                            {
                                "Time": pd.to_datetime(
                                    t_obs_hmi_channel[valid_indices],
                                    format="mixed",
                                    utc=True,
                                ),
                                f"idx_{component}": np.arange(
                                    0, len(t_obs_hmi_channel)
                                )[valid_indices],
                            }
                        )
                        df_t_hmi = pd.concat([df_t_hmi, df_tmp_hmi], ignore_index=True)

                # Enforcing same datetime format
                transform_datetime = lambda x: pd.to_datetime(
                    x, format="mixed"
                ).strftime("%Y-%m-%d %H:%M:%S")
                df_t_hmi["Time"] = df_t_hmi["Time"].apply(transform_datetime)
                df_t_hmi["Time"] = pd.to_datetime(df_t_hmi["Time"]).dt.tz_localize(
                    None
                )  # this is needed for timezone-naive type
                df_t_hmi["Time"] = df_t_hmi["Time"].dt.round(self.cadence)
                df_t_obs_hmi = df_t_hmi.drop_duplicates(
                    subset="Time", keep="first"
                )  # removing potential duplicates derived by rounding
                df_t_obs_hmi.set_index("Time", inplace=True)

                if join_series is None:
                    join_series = df_t_obs_hmi
                else:
                    join_series = join_series.join(df_t_obs_hmi, how="inner")

        print(f"HMI alignment completed with {join_series.shape[0]} samples.")

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # EVE
        if self.isEVE:
            print("Aligning EVE data")
            df_t_eve = pd.DataFrame(
                {
                    "Time": pd.to_datetime(self.eve_data["Time"]),
                    "idx_eve": np.arange(0, len(self.eve_data["Time"])),
                }
            )
            df_t_eve["Time"] = pd.to_datetime(df_t_eve["Time"]).dt.round(self.cadence)
            df_t_obs_eve = df_t_eve.drop_duplicates(
                subset="Time", keep="first"
            ).set_index("Time")

            if join_series is None:
                join_series = df_t_obs_eve
            else:
                join_series = join_series.join(df_t_obs_eve, how="inner")

            # remove missing eve data (missing values are labeled with negative values)
            # this will remove all but 16 values if the partial year 2014 is included
            for ion in self.ions:
                if ion == "Fe XVI_2":
                    continue
                ion_data = self.eve_data[ion]
                join_series = join_series.loc[ion_data[join_series["idx_eve"]] > 0]

        if join_series is None:
            raise ValueError("No data found for alignment.")

        join_series.sort_index(inplace=True)

        print("")
        print("#" * 50)
        print(f"[*] Total Alignment Completed with {join_series.shape[0]} Samples.")
        print(f"[*] Saving alignment data to {self.index_cache_filename}.")
        print("#" * 50)
        print("")
        # creating csv dataset
        join_series.to_csv(self.index_cache_filename)

        return join_series

    def __calc_normalizations(self):

        normalizations = {}
        normalizations_align = self.aligndata.copy()

        if self.isEVE:
            normalizations["EVE"] = self._compute_data_statistic(
                normalizations_align, self.eve_data, "EVE", self.ions
            )

        if self.isAIA:
            normalizations["AIA"] = self._compute_data_statistic(
                normalizations_align, self.aia_data, "AIA", self.wavelengths
            )

        if self.isHMI:
            normalizations["HMI"] = self._compute_data_statistic(
                normalizations_align, self.hmi_data, "HMI", self.components
            )

        return normalizations

    def compute_stat(self, key, data):
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

    def check_existing_stat_info(self, target_file_name):

        if os.path.exists(target_file_name):
            logger.info(f"Cache is found: {target_file_name}")
            with open(target_file_name, "r") as json_file:
                stat = json.load(json_file)
                return stat
        else:
            return {}

    def _compute_data_statistic(
        self, normalizations_align, sdoml_data, instrument, channels
    ) -> dict[str, dict[str, float]]:

        normalizations_stat: dict[str, dict[str, float]] = {}
        for ch in channels:

            file_name = (
                self.cache_dir
                + f"/{instrument}/"
                + ch
                + "_"
                + "_".join(self.cache_id.split("_")[-1:])
                + f"_norm-{self.normalization.type}"
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

            normalizations_stat[ch] = self.check_existing_stat_info(file_name)
            check_list = [
                k for k in check_list if k not in normalizations_stat[ch].keys()
            ]

            # if all the statistics exist, pass.
            if len(check_list) == 0:
                continue

            ch_arr = []
            for year in self.training_years:
                ch_data_year = da.from_array(sdoml_data[str(year)][ch])
                ch_idices = normalizations_align.loc[
                    normalizations_align.index.year == year, f"idx_{ch}"
                ]
                ch_data_year = ch_data_year[ch_idices]

                # put nan to limb
                mask_expanded = self.hmi_mask.cpu().numpy()[None, :, :].astype(bool)
                ch_data_year = da.where(
                    mask_expanded == 1, ch_data_year, np.nan
                )  # outter area to nan

                ch_arr.append(ch_data_year.flatten())
            ch_data = da.concatenate(ch_arr)

            if self.normalization.type == "log":
                ch_data = (
                    ch_data * self.normalization.scaler_factor
                    if self.normalization.scaler_factor is not None
                    else ch_data
                )
                ch_data = da.sign(ch_data) * da.log1p(da.abs(ch_data))

            elif (
                self.normalization.type == "zscore"
                and self.normalization.clipping.enabled
            ):
                low, high = self.normalization.clipping[ch]
                if self.normalization.clipping.enabled:
                    ch_data = da.clip(ch_data, low, high)

            print(f"\nCalculating normalizations for wavelength {ch}:")
            print("-" * 50)

            for stat_measure in check_list:
                logger.info(f"Computing {stat_measure} of {ch}")
                stat_value = self.compute_stat(stat_measure, ch_data)
                if stat_value == 0:
                    logger.warning(
                        f"Value of {stat_measure} is Zero!, wavelength: {ch}"
                    )
                normalizations_stat[ch][stat_measure] = stat_value

            # save statistics of each wavelength
            with open(file_name, "w") as json_file:
                json.dump(normalizations_stat[ch], json_file)

        return normalizations_stat

    def __make_hmi_mask(self):
        if Path(self.hmi_mask_cache_filename).exists():
            loaded_mask = np.load(self.hmi_mask_cache_filename)
            hmi_mask = torch.Tensor(loaded_mask).to(dtype=torch.uint8)
            print(
                f"[* CACHE SYSTEM *] Found cached HMI mask data in {self.hmi_mask_cache_filename}."
            )
            return hmi_mask
        elif not self.isHMI:
            raise ValueError(
                "Mask could not be found in cache and 2010 HMI data is not available to generate it, stopping..."
            )

        hmi = torch.Tensor(self.hmi_data[str(2010)][ALL_COMPONENTS[0]][0])
        hmi_mask = (torch.abs(hmi) > 0.0).to(dtype=torch.uint8)
        hmi_mask_ratio = hmi_mask.sum().item() / hmi_mask.numel()
        if np.abs(hmi_mask_ratio - 0.496) > 0.2:
            print(
                f"WARNING: HMI mask ratio is {hmi_mask_ratio:.2f}, which is significantly different from expected (0.496)"
            )
        print(
            f"[*] Saving HMI mask with ratio {hmi_mask_ratio:.2f} to {self.hmi_mask_cache_filename}."
        )
        np.save(self.hmi_mask_cache_filename, hmi_mask.numpy())
        return hmi_mask

    def setup(self, stage=None):

        self.train_ds = SDOMLDataset(
            self.aligndata,
            self.hmi_data,
            self.aia_data,
            self.eve_data,
            self.components,
            self.wavelengths,
            self.ions,
            self.cadence,
            self.train_months,
            normalization=self.normalization,
            normalization_stat=self.normalization_stat,
            mask=self.hmi_mask.numpy() if self.apply_mask else None,
            num_frames=self.num_frames,
            drop_frame_dim=self.drop_frame_dim,
            min_date=self.min_date,
            max_date=self.max_date,
            precision=self.precision,
        )
        if stage == "fit" or stage is None:
            logger.info("Train dataloader is ready!")
            logger.info(f"Dataset size: {len(self.train_ds)}")

        self.valid_ds = SDOMLDataset(
            self.aligndata,
            self.hmi_data,
            self.aia_data,
            self.eve_data,
            self.components,
            self.wavelengths,
            self.ions,
            self.cadence,
            self.val_months,
            normalization=self.normalization,
            normalization_stat=self.normalization_stat,
            mask=self.hmi_mask.numpy() if self.apply_mask else None,
            num_frames=self.num_frames,
            drop_frame_dim=self.drop_frame_dim,
            min_date=self.min_date,
            max_date=self.max_date,
            precision=self.precision,
        )
        if stage == "fit" or stage is None:
            logger.info("Validation dataloader is ready!")
            logger.info(f"Dataset size: {len(self.valid_ds)}")

        self.test_ds = SDOMLDataset(
            self.aligndata,
            self.hmi_data,
            self.aia_data,
            self.eve_data,
            self.components,
            self.wavelengths,
            self.ions,
            self.cadence,
            self.test_months,
            normalization=self.normalization,
            normalization_stat=self.normalization_stat,
            mask=self.hmi_mask.numpy() if self.apply_mask else None,
            num_frames=self.num_frames,
            drop_frame_dim=self.drop_frame_dim,
            min_date=self.min_date,
            max_date=self.max_date,
            precision=self.precision,
        )
        if stage == "fit" or stage is None:
            logger.info("test dataloader is ready!")
            logger.info(f"Dataset size: {len(self.test_ds)}")

        self.predict_ds = SDOMLDataset(
            self.aligndata,
            self.hmi_data,
            self.aia_data,
            self.eve_data,
            self.components,
            self.wavelengths,
            self.ions,
            self.cadence,
            self.predict_months,
            normalization=self.normalization,
            normalization_stat=self.normalization_stat,
            mask=self.hmi_mask.numpy() if self.apply_mask else None,
            num_frames=self.num_frames,
            drop_frame_dim=self.drop_frame_dim,
            min_date=self.min_date,
            max_date=self.max_date,
            precision=self.precision,
        )
        if stage == "predict":
            logger.info("test dataloader is ready!")
            logger.info(f"Dataset size: {len(self.predict_ds)}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
