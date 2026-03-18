import pandas as pd
import torch
from loguru import logger

from sdofmv2.core import SDOMLDataModule, SDOMLDataset


class EmbSolarProxyDataset(SDOMLDataset):
    """A dataset class for solar proxy prediction using SDO multi-instrument data.

    This class extends SDOMLDataset to include the F10.7 solar proxy as the target
    variable for supervised learning tasks. It retrieves aligned image data from
    AIA and HMI instruments and pairs them with the corresponding normalized
    F10.7 index.

    Args:
        aligndata (pd.DataFrame): Aligned temporal indexes and proxy values.
            Must contain a 'f107_norm' column for the target variable.
        hmi_data (zarr.hierarchy.Group): Zarr dataset containing HMI magnetogram
            observations.
        aia_data (zarr.hierarchy.Group): Zarr dataset containing AIA EUV/UV
            image observations.
        eve_data (zarr.hierarchy.Group): Zarr dataset containing EVE irradiance
            observations.
        components (list[str]): List of magnetic components to load for HMI
            (e.g., ['Bx', 'By', 'Bz']).
        wavelengths (list[str] or list[int]): List of channels to load for AIA
            (e.g., [171, 193, 211]).
        ions (list[str]): List of spectral lines/ions to load for EVE.
        freq (str): The temporal cadence used for rounding and aligning the
            time series (e.g., '12min').
        months (list[int]): List of valid months (1-12) to include in the dataset.
        normalization (dict, optional): The normalization strategy to apply
            during data loading. Defaults to None.
        normalization_stat (dict, optional): Pre-computed statistics required
            for the chosen normalization. Defaults to None.
        mask (torch.Tensor, optional): HMI limb mask to apply to the spatial
            data. Defaults to None.
        num_frames (int, optional): The number of consecutive temporal frames
            to load per sequence sample. Defaults to 1.
        drop_frame_dim (bool, optional): If True and `num_frames` is 1, drops
            the temporal dimension. Defaults to False.
        min_date (str or datetime, optional): The earliest date boundary to
            include in the dataset. Defaults to None.
        max_date (str or datetime, optional): The latest date boundary to
            include in the dataset. Defaults to None.
        get_header (bool or list, optional): Whether to retrieve and return
            header metadata alongside the image tensors. Defaults to False.
        precision (str, optional): The floating-point precision for the output
            tensors (e.g., "32", "16"). Defaults to "32".

    Returns:
        tuple: A tuple containing:
            - image_stack (torch.Tensor): Multimodal image data tensor.
            - timestamps (int or np.ndarray): Unix timestamps for the frames.
            - target (torch.Tensor): Normalized F10.7 solar proxy values.
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
        super().__init__(
            aligndata=aligndata,
            hmi_data=hmi_data,
            aia_data=aia_data,
            eve_data=eve_data,
            components=components,
            wavelengths=wavelengths,
            ions=ions,
            freq=freq,
            months=months,
            normalization=normalization,
            normalization_stat=normalization_stat,
            mask=mask,
            num_frames=num_frames,
            drop_frame_dim=drop_frame_dim,
            min_date=min_date,
            max_date=max_date,
            get_header=get_header,  # Optional[list] = [],
            precision=precision,
        )

    def __getitem__(self, idx):
        image_stack, timestamps = super().__getitem__(idx=idx)

        # define target with normalization
        target = torch.tensor(
            self.aligndata.loc[pd.to_datetime(timestamps), "f107_norm"],
            dtype=torch.float32,
        )

        return image_stack, timestamps, target


class EmbSolarProxyDataModule(SDOMLDataModule):
    """PyTorch Lightning DataModule for solar proxy prediction using SDO data.

    This class manages the loading, preprocessing, and splitting of multi-instrument
    SDO data (HMI, AIA, EVE) paired with F10.7 solar proxy values. It handles
    temporal alignment between the SDO observations and the proxy data provided
    in a CSV file.

    Args:
        hmi_path (str): Path to the HMI Zarr dataset.
        aia_path (str): Path to the AIA Zarr dataset.
        eve_path (str): Path to the EVE Zarr dataset.
        components (list[str]): List of HMI magnetic components to load.
        wavelengths (list[str] or list[int]): List of AIA wavelengths to load.
        ions (list[str]): List of EVE spectral lines/ions to load.
        frequency (str): Temporal cadence for data alignment (e.g., '12min').
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        num_workers (int, optional): Number of subprocesses for data loading.
            Defaults to None.
        pin_memory (bool, optional): If True, copies tensors into CUDA pinned
            memory before returning them. Defaults to False.
        persistent_workers (bool, optional): If True, the data loader will not
            shutdown the worker processes after a dataset has been consumed once.
            Defaults to False.
        val_months (list[int], optional): Months to use for the validation set.
            Defaults to [10, 1].
        test_months (list[int], optional): Months to use for the test set.
            Defaults to [11, 12].
        holdout_months (list[int], optional): Months to exclude from all sets.
            Defaults to [].
        normalization (bool or str, optional): Normalization strategy to apply.
            Defaults to False.
        cache_dir (str, optional): Directory to store cached normalization
            statistics. Defaults to "".
        norm_stat_tag (str, optional): Tag for identifying specific normalization
            statistics. Defaults to "".
        apply_mask (bool, optional): Whether to apply the HMI limb mask.
            Defaults to True.
        num_frames (int, optional): Number of consecutive frames per sample.
            Defaults to 1.
        drop_frame_dim (bool, optional): Whether to drop the temporal dimension
            if num_frames is 1. Defaults to False.
        min_date (str or datetime, optional): Earliest date to include.
            Defaults to None.
        max_date (str or datetime, optional): Latest date to include.
            Defaults to None.
        precision (str, optional): Floating-point precision ("32" or "16").
            Defaults to "32".
        ds_data_path (str, optional): Path to the CSV file containing F10.7
            proxy data. Defaults to None.

    Returns:
        DataLoader: The class provides methods (train_dataloader, val_dataloader,
            test_dataloader) that return PyTorch DataLoaders yielding batches
            of (image_stack, timestamps, target).
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
        normalization=False,
        cache_dir="",
        norm_stat_tag="",
        apply_mask=True,
        num_frames=1,
        drop_frame_dim=False,
        min_date=None,
        max_date=None,
        precision="32",
        ds_data_path=None,
    ):
        super().__init__(
            hmi_path=hmi_path,
            aia_path=aia_path,
            eve_path=eve_path,
            components=components,
            wavelengths=wavelengths,
            ions=ions,
            frequency=frequency,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            val_months=val_months,
            test_months=test_months,
            holdout_months=holdout_months,
            normalization=normalization,
            cache_dir=cache_dir,
            norm_stat_tag=norm_stat_tag,
            apply_mask=apply_mask,
            num_frames=num_frames,
            drop_frame_dim=drop_frame_dim,
            min_date=min_date,
            max_date=max_date,
            precision=precision,
        )

        self.df = (
            pd.read_csv(ds_data_path)
            .assign(
                Timestep=lambda x: pd.to_datetime(
                    x["date"].astype(str) + " 00:00:00", format="%Y%m%d %H:%M:%S"
                )
            )
            .set_index("Timestep")
            .sort_index()
        )
        self.df = self.df[~self.df[" f107"].isna()]
        self.max_norm = self.df[" f107"].max()
        self.df["f107_norm"] = self.df[" f107"] / self.max_norm
        self.aligndata = pd.merge_asof(
            self.aligndata,
            self.df,
            left_index=True,
            right_index=True,
            tolerance=pd.Timedelta(12, "min"),
        )

        self.aligndata = self.aligndata.dropna(subset=[" f107", "f107_norm"])

    def setup(self, stage=None):
        self.train_ds = EmbSolarProxyDataset(
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

        self.valid_ds = EmbSolarProxyDataset(
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

        self.test_ds = EmbSolarProxyDataset(
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
