
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
            Must be pre-filtered for the specific data split.
        hmi_path (str | None): Path to the HMI Zarr dataset.
        aia_path (str | None): Path to the AIA Zarr dataset.
        eve_path (str | None): Path to the EVE Zarr dataset.
        components (list[str] | None): List of magnetic components to load for HMI
            (e.g., ['Bx', 'By', 'Bz']).
        wavelengths (list[str] | list[int] | None): List of channels to load for AIA
            (e.g., [171, 193, 211]).
        ions (list[str] | None): List of spectral lines/ions to load for EVE.
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
        get_header (bool | list, optional): Whether to retrieve and return
            header metadata alongside the image tensors. Defaults to False.
        precision (str, optional): The floating-point precision for the output
            tensors (e.g., "32", "16"). Defaults to "32".
    """

    def __init__(
        self,
        aligndata: pd.DataFrame,
        hmi_path: str | None,
        aia_path: str | None,
        eve_path: str | None,
        components: list[str] | None,
        wavelengths: list[str] | list[int] | None,
        ions: list[str] | None,
        normalization: dict | None = None,
        normalization_stat: dict | None = None,
        mask: torch.Tensor | None = None,
        num_frames: int = 1,
        drop_frame_dim: bool = False,
        get_header: bool = False,
        precision: str = "32",
    ) -> None:
        super().__init__(
            aligndata=aligndata,
            hmi_path=hmi_path,
            aia_path=aia_path,
            eve_path=eve_path,
            components=components,
            wavelengths=wavelengths,
            ions=ions,
            normalization=normalization,
            normalization_stat=normalization_stat,
            mask=mask,
            num_frames=num_frames,
            drop_frame_dim=drop_frame_dim,
            get_header=get_header,
            precision=precision,
        )

    def __getitem__(self, idx: int):
        # SDOMLDataset.__getitem__ returns (image_stack, timestamps)
        # OR (image_stack, timestamps, eve_data) if EVE is present.
        data = super().__getitem__(idx=idx)
        image_stack, timestamps = data[0], data[1]

        # define target with normalization
        target = torch.tensor(
            self.aligndata.loc[pd.to_datetime(timestamps), "f107_norm"],
            dtype=torch.float32,
        )

        return image_stack, timestamps, target


class EmbSolarProxyDataModule(SDOMLDataModule):
    """PyTorch Lightning DataModule for solar proxy prediction using SDO data."""

    def __init__(
        self,
        hmi_path: str | None,
        aia_path: str | None,
        eve_path: str | None,
        components: list[str],
        wavelengths: list[str] | list[int],
        ions: list[str],
        batch_size: int = 32,
        num_workers: int | None = None,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        multiprocessing_context: str | None = None,
        normalization: dict | None = None,
        normalization_stat_path: str = "",
        train_index: str = "",
        val_index: str = "",
        test_index: str = "",
        hmi_mask: str = "hmi_mask_512x512.npy",
        apply_mask: bool = True,
        num_frames: int = 1,
        drop_frame_dim: bool = False,
        precision: str = "32",
        ds_data_path: str = "",
    ) -> None:
        if normalization is None:
            normalization = {}
        super().__init__(
            hmi_path=hmi_path,
            aia_path=aia_path,
            eve_path=eve_path,
            components=components,
            wavelengths=wavelengths,
            ions=ions,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
            normalization=normalization,
            normalization_stat_path=normalization_stat_path,
            train_index=train_index,
            val_index=val_index,
            test_index=test_index,
            hmi_mask=hmi_mask,
            apply_mask=apply_mask,
            num_frames=num_frames,
            drop_frame_dim=drop_frame_dim,
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

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage=stage)

        # Prepare HMI mask (base class loads HMI mask as a Tensor)
        mask_tensor = self.hmi_mask if self.apply_mask and isinstance(self.hmi_mask, torch.Tensor) else None

        if stage == "fit" or stage is None:
            self.train_ds = EmbSolarProxyDataset(
                self._load_aligndata(self.train_index),
                hmi_path=self.hmi_path,
                aia_path=self.aia_path,
                eve_path=self.eve_path,
                components=self.components,
                wavelengths=self.wavelengths,
                ions=self.ions,
                normalization=self.normalization,
                normalization_stat=self.normalization_stat,
                mask=mask_tensor,
                num_frames=self.num_frames,
                drop_frame_dim=self.drop_frame_dim,
                precision=self.precision,
            )
            logger.info("Train dataloader is ready!")
            logger.info(f"Dataset size: {len(self.train_ds)}")

            self.valid_ds = EmbSolarProxyDataset(
                self._load_aligndata(self.val_index),
                hmi_path=self.hmi_path,
                aia_path=self.aia_path,
                eve_path=self.eve_path,
                components=self.components,
                wavelengths=self.wavelengths,
                ions=self.ions,
                normalization=self.normalization,
                normalization_stat=self.normalization_stat,
                mask=mask_tensor,
                num_frames=self.num_frames,
                drop_frame_dim=self.drop_frame_dim,
                precision=self.precision,
            )
            logger.info("Validation dataloader is ready!")
            logger.info(f"Dataset size: {len(self.valid_ds)}")

        if stage == "test" or stage is None:
            self.test_ds = EmbSolarProxyDataset(
                self._load_aligndata(self.test_index),
                hmi_path=self.hmi_path,
                aia_path=self.aia_path,
                eve_path=self.eve_path,
                components=self.components,
                wavelengths=self.wavelengths,
                ions=self.ions,
                normalization=self.normalization,
                normalization_stat=self.normalization_stat,
                mask=mask_tensor,
                num_frames=self.num_frames,
                drop_frame_dim=self.drop_frame_dim,
                precision=self.precision,
            )
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
