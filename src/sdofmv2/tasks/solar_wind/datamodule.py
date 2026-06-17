import os

import hydra
import numpy as np
import pandas as pd
import torch
import zarr
from loguru import logger

from sdofmv2.core import SDOMLDataModule, SDOMLDataset


def parse_cadence(cadence):
    """Return time group keys based on cadence."""
    return {
        "30s": ["year", "month", "day", "hour", "minute", "second_bool"],
        "1s": ["year", "month", "day", "hour", "minute", "second"],
        "1min": ["year", "month", "day", "hour", "minute"],
        "1h": ["year", "month", "day", "hour"],
        "1D": ["year", "month", "day"],
        "1MS": ["year", "month"],
        "1YS": ["year"],
    }.get(cadence, [])


class SWDataset(SDOMLDataset):
    """Solar Wind dataset for SDOML (Solar Dynamics Observatory Machine Learning).

    This class extends the base SDOMLDataset to handle solar wind-specific
    features, including radial and latitudinal/longitudinal parameters. It
    supports temporal filtering by year/month, class-based undersampling for
    training sets, and automated column index mapping for coordinate features.

    Args:
        aligndata_path (str): Path to the Parquet file containing aligned temporal indices and targets.
        hmi_path (str): Path to HMI Zarr data.
        aia_path (str): Path to AIA Zarr data.
        eve_path (str): Path to EVE Zarr data.
        components (list[str]): List of magnetic field components to load for HMI.
        wavelengths (list[int]): List of AIA wavelengths to include.
        ions (list[str]): List of EVE spectral lines to include.
        mask (torch.Tensor, optional): A precomputed spatial mask to be applied. Defaults to None.
        num_frames (int): Number of consecutive temporal frames per sample. Defaults to 1.
        drop_frame_dim (bool): If True, removes temporal dimension if num_frames is 1. Defaults to False.
        get_header (bool): Whether to retrieve and return FITS headers. Defaults to False.
        normalization (dict): Normalization settings.
        normalization_stat (dict): Precomputed stats for normalization.
        label_type (str): The column name in the alignment file used as the prediction target.
        radial_parameters (list[str], optional): Column names for radial features.
        latlon_parameters (list[str], optional): Column names for coordinate features.
        precision (str): Numerical precision for tensors (e.g., "32"). Defaults to "32".

    Attributes:
        aligndata (pd.DataFrame): The alignment table indexed by observation time.
        id_label (int): Integer index of the target label in aligndata.
        position_list (list[int]): Column indices for coordinate features.
        r_dist_list (list[int]): Column indices for radial distance features.
    """

    def __init__(
        self,
        aligndata_path,
        hmi_path,
        aia_path,
        eve_path,
        components,
        wavelengths,
        ions,
        mask=None,
        num_frames=1,
        drop_frame_dim=False,
        get_header=False,
        normalization={},
        normalization_stat={},
        # set variables for solar wind here
        label_type="",
        radial_parameters=None,
        latlon_parameters=None,
        precision="32",
    ):
        # Load aligndata from Parquet
        aligndata = pd.read_parquet(aligndata_path)
        super().__init__(
            aligndata=aligndata,
            hmi_path=hmi_path,
            aia_path=aia_path,
            eve_path=eve_path,
            components=components,
            wavelengths=wavelengths,
            ions=ions,
            mask=mask,
            num_frames=num_frames,
            drop_frame_dim=drop_frame_dim,
            get_header=get_header,
            normalization=normalization,
            normalization_stat=normalization_stat,
            precision=precision,
        )
        self.radial_parameters = radial_parameters
        self.latlon_parameters = latlon_parameters
        self.aligndata = aligndata

        label_name = label_type
        self.id_label = self.aligndata.columns.get_loc(label_name)

        cols = self.aligndata.columns.to_list()
        # define the position columns
        self.position_list = []
        self.r_dist_list = []

        for para in self.latlon_parameters:
            self.position_list.append(cols.index(f"{para}"))

        for para in self.radial_parameters:
            self.r_dist_list.append(cols.index(f"{para}_norm"))

        logger.info(f"Position list: {self.latlon_parameters}: {self.position_list}")
        logger.info(f"Radial distance: {self.radial_parameters}: {self.r_dist_list}")
        logger.info(f"Label: {self.aligndata[label_name].value_counts()}")

    def __len__(self):
        # report slightly smaller such that all frame sets requested are available
        return self.aligndata.shape[0]

    def __getitem__(self, idx):
        # start = time.time()
        label = self.aligndata.iloc[idx, self.id_label].astype("int64")  # make it start from 0
        position = np.radians(self.aligndata.iloc[idx, self.position_list].values)
        r_distance = self.aligndata.iloc[idx, self.r_dist_list].to_numpy(dtype=np.float32)
        timestamps = self.aligndata.index[idx].value

        # second retrieve input (image, or (image, header)) from parent class
        if self.get_header:
            image_stack, header_stack, _ = super().__getitem__(idx=idx)

            # logger.info(f"end: {time.time()} total: {time.time()-start}")
            return image_stack, timestamps, header_stack, position, r_distance[0], label
        else:
            image_stack, timestamps_parent = super().__getitem__(idx=idx)
            if timestamps_parent != timestamps:
                logger.warning(
                    f"Parent: {pd.to_datetime(timestamps_parent)} &"
                    f"child: {pd.to_datetime(timestamps)} different!"
                )

            # logger.info(f"end: {time.time()} total: {time.time()-start}")
            return image_stack, timestamps, position, r_distance[0], label


class SWDataModule(SDOMLDataModule):
    """PyTorch Lightning DataModule for Solar Wind (SW) prediction.

    This module handles the end-to-end data pipeline for SDOML datasets, including
    loading alignment indices from Zarr, filtering by temporal boundaries (years/months),
    applying spatial longitude cutoffs for solar footpoints, and managing
    normalization for radial distance parameters.

    Args:
        hmi_path (str): Path to HMI Zarr data.
        aia_path (str): Path to AIA Zarr data.
        eve_path (str): Path to EVE Zarr data.
        components (list[str]): HMI magnetic components to use.
        wavelengths (list[int]): AIA wavelengths to use.
        ions (list[str]): EVE spectral lines to use.
        frequency (str): Sampling frequency of the instruments.
        batch_size (int): Number of samples per batch. Defaults to 32.
        num_workers (int): Number of subprocesses for data loading.
        apply_mask (bool): Whether to apply the limb mask to spatial data.
        num_frames (int): Temporal frames per sample.
        drop_frame_dim (bool): Whether to squeeze the temporal dimension.
        precision (str): Numerical precision ("16", "32", "64").
        normalization (dict): Normalization configuration.
        cfg (DictConfig): Hydra configuration object.
        radial_norm (bool): Whether to Z-score normalize radial features.
        radial_parameters (list[str]): Column names for radial features.
        latlon_parameters (list[str]): Column names for coordinate features.
        cadence (str): Data cadence string.
        label_type (str): Prediction target column name.
        sampling_ratio (list[float]): Undersampling ratios per class.
        random_state (int): Seed for reproducibility.
        hmi_mask_path (str): Filename for HMI mask. Defaults to "hmi_mask_512x512.npy".


    Attributes:
        aligndata (pd.DataFrame): The central alignment table indexed by SDO
            observation time, containing indices of data and target labels.
        radial_mean (float): Mean value of the radial parameters used for normalization.
        radial_std (float): Standard deviation of the radial parameters.
        cfg (DictConfig | Any): Configuration object containing hyperparameters
            and data cutoffs (e.g., ``cfg.data.in_situ.lon_cutoff``).
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
        apply_mask=True,
        num_frames=1,
        drop_frame_dim=False,
        precision="32",
        normalization=None,
        normalization_stat_path="",
        train_index="",
        val_index="",
        test_index="",
        # set variables for solar wind here
        cfg=None,
        radial_norm=False,
        radial_parameters=None,
        latlon_parameters=None,
        cadence="1min",
        label_type="",
        sampling_ratio=None,
        random_state=None,
        merged_splits_dir="",
        hmi_mask_path="hmi_mask_512x512.npy",
    ):
        self.hmi_mask_path = hmi_mask_path
        super().__init__(
            hmi_path=hmi_path,
            aia_path=aia_path,
            eve_path=eve_path,
            components=components,
            wavelengths=wavelengths,
            ions=ions,
            batch_size=batch_size,
            num_workers=num_workers,
            normalization=normalization,
            normalization_stat_path=normalization_stat_path,
            hmi_mask_path=hmi_mask_path,
            apply_mask=apply_mask,
            num_frames=num_frames,
            drop_frame_dim=drop_frame_dim,
            precision=precision,
            train_index=train_index,
            val_index=val_index,
            test_index=test_index,
        )
        self.radial_mean = None
        self.radial_std = None
        self.cfg = cfg
        self.cadence = cadence

        self.label_type = label_type
        self.sampling_ratio = sampling_ratio
        self.random_state = random_state

        self.precision = precision
        self.radial_parameters = radial_parameters
        self.latlon_parameters = latlon_parameters
        self.radial_norm = radial_norm
        self.frequency = frequency
        self.hmi_mask_path = hmi_mask_path
        self.merged_splits_dir = merged_splits_dir
        os.makedirs(self.merged_splits_dir, exist_ok=True)

    def setup(self, stage=None):
        super().setup(stage)

        def _merge_and_filter(sdoml_df, psp_df):
            # Merge
            df_merge = pd.merge_asof(
                psp_df,
                sdoml_df.reset_index().rename(columns={"Time": "Time_sdoml"}),
                left_on="time_sdo_loc_est",
                right_on="Time_sdoml",
                direction="nearest",
                allow_exact_matches=True,
                tolerance=pd.Timedelta(minutes=int(self.cfg.data.in_situ.match_tolerance)),
            )

            # Filtering
            if "lon_footpoint" in self.latlon_parameters:
                df_merge = df_merge.loc[
                    df_merge["lon_footpoint"].abs() < self.cfg.data.in_situ.lon_cutoff
                ]
            elif "sc_pos_SH_lon" in self.latlon_parameters:
                df_merge = df_merge.loc[
                    df_merge["sc_pos_SH_lon"].abs() < self.cfg.data.in_situ.lon_cutoff
                ]
            df_merge = df_merge.loc[df_merge["vp_fit_RTN_0_mean"] >= 100]
            df_merge.dropna(subset=["Time_sdoml"], inplace=True)

            # Normalization
            if self.radial_norm:
                for col in self.radial_parameters:
                    mean, std = df_merge[col].mean(), df_merge[col].std()
                    df_merge[f"{col}_norm"] = (df_merge[col] - mean) / std
                    if getattr(self, "radial_mean", None) is None:
                        self.radial_mean = mean
                        self.radial_std = std

            return df_merge.set_index("time_sdo_loc_est")

        # Process splits
        df_psp = None
        for split, ds in [("train", self.train_ds), ("val", self.valid_ds), ("test", self.test_ds)]:
            save_path = os.path.join(self.merged_splits_dir, f"solarwind_{split}.parquet")

            if os.path.exists(save_path):
                logger.info(f"Loading existing merged {split} data from {save_path}")
            else:
                if df_psp is None:
                    # Load PSP data
                    logger.info("Loading and preprocessing PSP data...")
                    path = os.path.join(
                        self.cfg.data.in_situ.base_data_directory,
                        self.cfg.data.in_situ.psp_interpolated_path,
                    )
                    root = zarr.open(path, mode="r")
                    columns = root.attrs["columns"]
                    df_psp = pd.DataFrame(root[:, :], columns=columns)
                    df_psp["time"] = pd.to_datetime(root.attrs["time"])
                    df_psp.dropna(
                        subset=self.radial_parameters + self.latlon_parameters, inplace=True
                    )
                    df_psp[self.cfg.data.propagation_type] = df_psp[
                        self.cfg.data.propagation_type
                    ].apply(lambda x: pd.Timedelta(x, unit="seconds"))
                    df_psp["time_sdo_loc_est"] = (
                        df_psp["time"] - df_psp[self.cfg.data.propagation_type]
                    )
                    df_psp.sort_values(by="time_sdo_loc_est", inplace=True)

                merged_df = _merge_and_filter(ds.aligndata, df_psp)
                merged_df.to_parquet(save_path)
                logger.info(f"Generated and saved merged {split} data to {save_path}")

            # Re-instantiate SWDataset with Parquet path
            setattr(
                self,
                f"{split}_ds",
                SWDataset(
                    aligndata_path=save_path,
                    hmi_path=self.hmi_path,
                    aia_path=self.aia_path,
                    eve_path=self.eve_path,
                    components=self.components,
                    wavelengths=self.wavelengths,
                    ions=self.ions,
                    mask=self.hmi_mask.numpy() if self.hmi_mask is not None else None,
                    num_frames=self.num_frames,
                    drop_frame_dim=self.drop_frame_dim,
                    normalization=self.normalization,
                    normalization_stat=self.normalization_stat,
                    radial_parameters=self.radial_parameters,
                    latlon_parameters=self.latlon_parameters,
                    label_type=self.label_type,
                    precision=self.precision,
                ),
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,  # shuffle true for visualization
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )


@hydra.main(version_base=None, config_path="../configs", config_name="finetune_solarwind_config")
def main(cfg):
    """Initializes the solar wind data module and validates dataset alignment.

    This function sets up the SWDataModule using parameters from the Hydra
    configuration. It verifies the training dataset length, checks frame range
    accessibility, and retrieves a sample to ensure the data pipeline works
    correctly.

    Args:
        cfg (DictConfig): Hydra configuration object containing data paths,
            split definitions, and experiment parameters.

    Returns:
        None
    """
    datamodule = SWDataModule(
        hmi_path=(
            os.path.join(
                cfg.data.sdoml.base_directory,
                cfg.data.sdoml.sub_directory.hmi,
            )
            if cfg.data.sdoml.sub_directory.hmi
            else None
        ),
        aia_path=(
            os.path.join(
                cfg.data.sdoml.base_directory,
                cfg.data.sdoml.sub_directory.aia,
            )
            if cfg.data.sdoml.sub_directory.aia
            else None
        ),
        eve_path=(
            os.path.join(
                cfg.data.sdoml.base_directory,
                cfg.data.sdoml.sub_directory.eve,
            )
            if cfg.data.sdoml.sub_directory.eve
            else None
        ),
        normalization=cfg.data.normalization,
        components=cfg.data.sdoml.components,
        wavelengths=cfg.data.sdoml.wavelengths,
        ions=cfg.data.sdoml.ions,
        frequency=cfg.data.sdoml.frequency,
        batch_size=cfg.model.misc.batch_size,
        num_workers=cfg.data.num_workers,
        num_frames=cfg.data.num_frames,
        drop_frame_dim=cfg.data.drop_frame_dim,
        radial_parameters=cfg.data.in_situ.radial_parameters,
        latlon_parameters=cfg.data.in_situ.latlon_parameters,
        cadence=cfg.data.in_situ.cadence,
        label_type=cfg.data.label_type,
        sampling_ratio=cfg.data.under_sampling.ratio,
        random_state=cfg.data.under_sampling.random_state,
        cfg=cfg,
    )
    datamodule.setup()
    # Check dataset and data alignment
    ds = datamodule.train_ds
    print(f"Dataset __len__: {len(ds)}")
    print(f"Aligndata rows: {len(ds.aligndata)}")

    # Check what index 0
    image, timestamps, position, r_distance, label = datamodule.train_ds[0]
    print(f"Sample retrieved successfully: image shape {image.shape}, label {label}")


if __name__ == "__main__":
    # cfg = omegaconf.OmegaConf.load(
    #     ("/home/jh/project/2025-HL-Solar-Wind/classification"
    #      "/configs/finetune_solarwind_config.yaml")
    # )
    main()
