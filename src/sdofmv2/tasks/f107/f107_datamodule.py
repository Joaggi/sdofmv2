import pandas as pd
import torch
from loguru import logger

from sdofmv2.core import SDOMLDataModule, SDOMLDataset

class EmbSolarProxyDataset(SDOMLDataset):
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
            dtype=torch.float32
        )
        
        return image_stack, timestamps, target
    
class EmbSolarProxyDataModule(SDOMLDataModule):
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
            .assign(Timestep=lambda x: pd.to_datetime(x['date'].astype(str) + ' 00:00:00',
                                                    format='%Y%m%d %H:%M:%S'))
            .set_index('Timestep')
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
            tolerance=pd.Timedelta(12, "min"))

        self.aligndata = self.aligndata.dropna(subset=[' f107', 'f107_norm'])

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