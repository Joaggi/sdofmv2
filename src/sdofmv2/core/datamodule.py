# Adapted to be general from https://github.com/FrontierDevelopmentLab/2023-FDL-X-ARD-EVE/blob/main/src/irradiance/utilities/data_loader.py
import os
from pathlib import Path
import re
import time
from pathlib import Path
from loguru import logger

import torch
import yaml

import dask.array as da
from dask.diagnostics import ProgressBar

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import zarr

from ..utils import ALL_COMPONENTS, ALL_IONS, ALL_WAVELENGTHS


def get_dtype_from_precision(precision):
    if str(precision) in ["16", "16-mixed"]:
        return torch.float16
    elif str(precision) in ["bf16", "bf16-mixed"]:
        return torch.bfloat16
    elif str(precision) in ["64", "64-true"]:
        return torch.float64
    else:
        return torch.float32


def zscore_norm(data, channel, normalization_stat, clip_value):
    if clip_value is not None:
        low, high = clip_value
        data = np.clip(data, low, high)
    data -= normalization_stat[channel]["mean"]
    data /= normalization_stat[channel]["std"]
    return data


def min_max_norm(data, channel, normalization_stat):
    data -= normalization_stat[channel]["min"]
    data /= normalization_stat[channel]["max"] - normalization_stat[channel]["min"]
    return data


def log_norm(data, normalization_stat, channel, scaler_factor):
    x = data * scaler_factor if scaler_factor is not None else data

    # Log transform
    x_log = np.sign(x) * np.log1p(np.abs(x))

    # zscore norm
    x_transformed = (x_log - normalization_stat[channel]["mean"]) / (
        normalization_stat[channel]["std"] + 1e-8
    )

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
    channel,
    scaler_factor=None,
):
    # Retrieve the exact log-domain statistics used during forward normalization
    mean = normalization_stat[channel]["mean"]
    std = normalization_stat[channel]["std"]

    # Reverse the Z-score standardization
    # x_transformed = (x_log - mean) / std  ->  x_log = (x_transformed * std) + mean
    x_log = (data_transformed * (std + 1e-8)) + mean

    # Reverse the SymLog Transform
    # The inverse of y = sign(x) * log(1 + |x|) is x = sign(y) * (exp(|y|) - 1)
    x = np.sign(x_log) * np.expm1(np.abs(x_log))

    # Reverse the Pre-scaling
    if scaler_factor is not None:
        data_original = x / scaler_factor
    else:
        data_original = x

    return data_original


class SDOMLDataset(Dataset):
    """A PyTorch Dataset for Solar Dynamics Observatory (SDO) Machine Learning data.

    This dataset aligns and loads multimodal solar observations from the AIA, HMI,
    and EVE instruments. It supports temporal sequencing, masking, and on-the-fly
    normalization for training deep learning models on solar data.

    Args:
        aligndata (pd.DataFrame): Aligned temporal indexes used
            for matching inputs and outputs across different instruments.
        hmi_data (zarr.hierarchy.Group): Zarr dataset
            HMI magnetogram observations.
        aia_data (zarr.hierarchy.Group): Zarr dataset
            AIA EUV/UV image observations.
        eve_data (zarr.hierarchy.Group): Zarr dataset
            EVE irradiance observations.
        components (list[str]): List of magnetic components to load for HMI
            (e.g., ['Bx', 'By', 'Bz']).
        wavelengths (list[str] or list[int]): List of channels to load for AIA
            (e.g., [94, 131, 171, 193, 211, 304, 335, 1600, 1700]).
        ions (list[str]): List of spectral lines/ions to load for EVE
            (e.g., from MEGS-A and MEGS-B).
        freq (str): The temporal cadence used for rounding and aligning the
            time series (e.g., '12min').
        months (list[int]): List of valid months (1-12) to include in the dataset.
            Useful for creating train/validation/test splits by time.
        normalization (dict): The normalization strategy to apply
            during data loading (e.g., 'zscore', 'minmax'). Defaults to None.
        normalization_stat (dict): Pre-computed statistics (like mean
            and standard deviation) required for the chosen normalization.
            Defaults to None.
        mask (torch.Tensor): Whether to apply the HMI limb
            mask to the AIA and HMI spatial data. Defaults to None.
        num_frames (int, optional): The number of consecutive temporal frames
            to load per sequence sample. Defaults to 1.
        drop_frame_dim (bool, optional): If True and `num_frames` is 1, drops
            the temporal dimension. Defaults to False.
        min_date (str or datetime, optional): The earliest date boundary to
            include in the dataset. Defaults to None.
        max_date (str or datetime, optional): The latest date boundary to
            include in the dataset. Defaults to None.
        get_header (bool or list, optional): Whether to retrieve and return header metadata alongside the image tensors. Defaults to False.
        precision (str, optional): The floating-point precision for the output
            tensors (e.g., "32" for float32, "16" for float16). Defaults to "32".
    """

    def __init__(
        self,
        aligndata,
        hmi_data,
        aia_data,
        eve_data,
        components,
        wavelengths,
        ions,
        normalization=None,
        normalization_stat=None,
        mask=None,
        num_frames=1,
        drop_frame_dim=False,
        get_header=False,  # Optional[list] = [],
        precision="32",
    ):
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
        self.normalization = normalization
        self.normalization_stat = normalization_stat

        # number of frames to return per sample
        self.num_frames = num_frames
        self.drop_frame_dim = drop_frame_dim  # for backwards compat
        if self.drop_frame_dim:
            assert self.num_frames == 1

    def __len__(self):
        # report slightly smaller such that all frame sets requested are available
        return len(self.aligndata) - (self.num_frames - 1)

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
                return (
                    image_stack,
                    timestamps,
                    header_stack,
                    eve_data.reshape(-1),
                )
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
                channel,
                self.normalization.scaler_factor,
            )

        elif self.normalization.type == "zscore":
            return zscore_norm(
                data,
                channel,
                self.normalization_stat,
                (
                    self.normalization.clipping[channel]
                    if self.normalization.clipping.enabled
                    else None
                ),
            )

        elif self.normalization.type == "min-max":
            return min_max_norm(data, channel, self.normalization_stat)

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
                    f"Corrupted load (Attempt {attempt + 1}/{num_try}) - "
                    f"channel: {wavelength}, year: {year}, idx: {id_of_img}. Error: {e}"
                )
                time.sleep(sleep_time)

        # If the loop finishes, we failed 'num_try' times.
        # Raise the last error to stop execution (or return zeros if preferred).
        logger.error(f"PERMANENT FAILURE: Could not load data after {num_try} attempts.")
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
                img = self.loading_data_retry(
                    self.aia_data, year, wavelength, idx_wavelength, 10, 0.5
                )

                if self.mask is not None:
                    img = img * self.mask

                aia_image_dict[wavelength].append(img)

                if self.get_header:
                    try:
                        aia_header_dict[wavelength].append(
                            {
                                keys: values[idx_wavelength]
                                for keys, values in self.aia_data[year][wavelength].attrs.items()
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
                            for keys, values in self.hmi_data[year][component].attrs.items()
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
                    eve_ion_dict[ion][-1] = self._data_norm(eve_ion_dict[ion][-1], "EVE", ion)

        eve_data = np.array(list(eve_ion_dict.values()), dtype=np.float32)

        return eve_data

    def __str__(self):
        output = ""
        for k, v in self.__dict__.items():
            output += f"{k}: {v}\n"
        return output


class SDOMLDataModule(pl.LightningDataModule):
    """A PyTorch Lightning DataModule for paired SDO machine learning data.

    This module orchestrates the downloading, setup, splitting, and batching of
    paired AIA EUV images, HMI magnetograms, and EVE irradiance measures. It
    handles train/val/test splits based on specified months to prevent temporal
    data leakage.

    Note:
        Input data across the different instruments needs to be temporally aligned
        and paired.

    Args:
        hmi_path (str): Path to the HMI Zarr data file.
        aia_path (str): Path to the AIA Zarr data file.
        eve_path (str): Path to the EVE Zarr data file.
        components (list[str]): List of magnetic field components to load from HMI.
        wavelengths (list[int] or list[str]): List of AIA wavelengths to load.
        ions (list[str]): List of EVE ions or spectral lines to load.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        num_workers (int, optional): Number of subprocesses to use for data
            loading. Defaults to None.
        pin_memory (bool, optional): If True, the data loader will copy Tensors
            into CUDA pinned memory before returning them. Defaults to False.
        persistent_workers (bool, optional): If True, the data loader will not
            shutdown worker processes after a dataset has been consumed once.
            Defaults to False.
        normalization (dict): specific normalization strategy to use. Defaults to False.
        hmi_mask (str, optional): Filename for the HMI mask. Defaults to "hmi_mask_512x512.npy".
        apply_mask (bool, optional): Whether to apply the solar limb mask to the
            spatial data. Defaults to True.
        num_frames (int, optional): The number of consecutive temporal frames
            to load per sequence sample. Defaults to 1.
        drop_frame_dim (bool, optional): If True and `num_frames` is 1. Defaults to False.
        min_date (str, optional): The earliest date boundary to include in the
            splits (e.g., '2010-05-01'). Defaults to None.
        max_date (str, optional): The latest date boundary to include in the
            splits. Defaults to None.
        precision (str, optional): The floating-point precision for the output
            tensors (e.g., "32", "16"). Defaults to "32".
    """

    def __init__(
        self,
        hmi_path,
        aia_path,
        eve_path,
        components,
        wavelengths,
        ions,
        batch_size: int = 32,
        num_workers=None,
        pin_memory=False,
        persistent_workers=False,
        normalization={},
        normalization_stat_path="",
        train_index="",
        val_index="",
        test_index="",
        hmi_mask="hmi_mask_512x512.npy",
        apply_mask=True,
        num_frames=1,
        drop_frame_dim=False,
        precision="32",
    ):
        super().__init__()
        self.num_workers = num_workers if num_workers is not None else os.cpu_count() // 2
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.hmi_path = hmi_path
        self.aia_path = aia_path
        self.eve_path = eve_path
        self.batch_size = batch_size
        self.apply_mask = apply_mask
        self.num_frames = num_frames
        self.drop_frame_dim = drop_frame_dim
        self.isAIA = True if self.aia_path is not None else False
        self.isHMI = True if self.hmi_path is not None else False
        self.isEVE = True if self.eve_path is not None else False
        self.precision = precision

        # index data
        # the indices of all data channels with timestamps
        self.train_index = train_index
        self.val_index = val_index
        self.test_index = test_index

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

        # Preprocessed data paths
        self.hmi_mask = hmi_mask
        self.normalization = normalization
        self.normalization_stat_path = normalization_stat_path
        self.timeinterval = re.compile(
            r"(\d{4}-\d{2}-\d{2}\d{2}:\d{2}:\d{2}-\d{4}-\d{2}-\d{2}\d{2}:\d{2}:\d{2})"
        )
        self.normalization_stat = self._load_norm_stats()

    def _load_norm_stats(self):
        match = self.timeinterval.search(self.train_index)
        if not match:
            raise ValueError("Can't find statistics for normalization")

        time_range = match.group(1)

        base_path = Path(self.normalization_stat_path)
        pattern = f"*{time_range}_norm-{self.normalization.type}*.json"

        files = list(base_path.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No normalization stats found for {time_range}")

        stats = {}
        for f in files:
            ch = f.stem.split("_")[0]  # safer than splitting full path

            with f.open("r") as fp:
                stats[ch] = yaml.safe_load(fp)

        return stats

    def __str__(self):
        output = ""
        for k, v in self.__dict__.items():
            output += f"{k}: {v}\n"
        return output

    def setup(self, stage=None):

        # Load mask
        if self.apply_mask:
            if os.path.exists(self.hmi_mask):
                self.hmi_mask = torch.Tensor(np.load(self.hmi_mask))
            else:
                logger.warning(f"HMI mask not found at {self.hmi_mask}, applying no mask.")
                self.hmi_mask = None
        else:
            self.hmi_mask = None

        # Define mask for dataset (numpy array or None)
        mask_np = self.hmi_mask.numpy() if self.hmi_mask is not None else None

        # Note: Dataset now expects a single aligndata and no months filtering (pre-split)
        # We pass the specific split aligndata and None for months to disable filtering

        self.train_ds = SDOMLDataset(
            self._load_aligndata(self.train_index),
            self.hmi_data,
            self.aia_data,
            self.eve_data,
            self.components,
            self.wavelengths,
            self.ions,
            normalization=self.normalization,
            normalization_stat=self.normalization_stat,
            mask=mask_np,
            num_frames=self.num_frames,
            drop_frame_dim=self.drop_frame_dim,
            precision=self.precision,
        )
        if stage == "fit" or stage is None:
            logger.info("Train dataloader is ready!")
            logger.info(f"Dataset size: {len(self.train_ds)}")

        self.valid_ds = SDOMLDataset(
            self._load_aligndata(self.val_index),
            self.hmi_data,
            self.aia_data,
            self.eve_data,
            self.components,
            self.wavelengths,
            self.ions,
            normalization=self.normalization,
            normalization_stat=self.normalization_stat,
            mask=mask_np,
            num_frames=self.num_frames,
            drop_frame_dim=self.drop_frame_dim,
            precision=self.precision,
        )
        if stage == "fit" or stage is None:
            logger.info("Validation dataloader is ready!")
            logger.info(f"Dataset size: {len(self.valid_ds)}")

        self.test_ds = SDOMLDataset(
            self._load_aligndata(self.test_index),
            self.hmi_data,
            self.aia_data,
            self.eve_data,
            self.components,
            self.wavelengths,
            self.ions,
            normalization=self.normalization,
            normalization_stat=self.normalization_stat,
            mask=mask_np,
            num_frames=self.num_frames,
            drop_frame_dim=self.drop_frame_dim,
            precision=self.precision,
        )
        if stage == "fit" or stage is None:
            logger.info("test dataloader is ready!")
            logger.info(f"Dataset size: {len(self.test_ds)}")

        # Handle predict dataset if needed
        # This is optional and depends on the presence of predict months in aligndata_files
        if stage == "predict":
            self.predict_ds = SDOMLDataset(
                self._load_aligndata(self.test_index),
                self.hmi_data,
                self.aia_data,
                self.eve_data,
                self.components,
                self.wavelengths,
                self.ions,
                normalization=self.normalization,
                normalization_stat=self.normalization_stat,
                mask=mask_np,
                num_frames=self.num_frames,
                drop_frame_dim=self.drop_frame_dim,
                precision=self.precision,
            )
            logger.info("Predict dataloader is ready!")
            logger.info(f"Dataset size: {len(self.predict_ds)}")

    def _load_aligndata(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Aligndata file not found: {filename}")

        df = pd.read_csv(filename)
        df["Time"] = pd.to_datetime(df["Time"])
        df.set_index("Time", inplace=True)
        return df

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
