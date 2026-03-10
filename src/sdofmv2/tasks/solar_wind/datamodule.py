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
        years,
        mask=None,
        num_frames=1,
        drop_frame_dim=False,
        min_date=None,
        max_date=None,
        get_header=False,
        normalization={},
        normalization_stat={},
        # set variables for solar wind here
        label_type="",
        radial_parameters=None,
        latlon_parameters=None,
        sampling_ratio=None,
        random_state=None,
        datasplit="train",
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
            mask=mask,
            num_frames=num_frames,
            drop_frame_dim=drop_frame_dim,
            min_date=min_date,
            max_date=max_date,
            get_header=get_header,
            normalization=normalization,
            normalization_stat=normalization_stat,
        )
        self.radial_parameters = radial_parameters
        self.latlon_parameters = latlon_parameters
        # split data based on month
        logger.info(f"{datasplit.upper()} set")
        logger.info(f"Data split, year: {years} & month: {months}")
        month_condition = aligndata.index.month.isin(months)
        year_condition = aligndata.index.year.isin(years)
        self.aligndata = aligndata.loc[month_condition & year_condition, :]

        label_name = label_type
        self.id_label = self.aligndata.columns.get_loc(label_name)

        # undersampling if sampling_ratio true
        if sampling_ratio is not None and datasplit == "train":
            return_df = []
            for class_id, class_ratio in enumerate(sampling_ratio):
                logger.info(
                    f"{class_ratio*100:.0f}% of class: {class_id} instances are sampled!"
                )
                return_df.append(
                    self.aligndata.loc[self.aligndata[label_name] == class_id].sample(
                        frac=class_ratio, replace=False, random_state=random_state
                    )
                )
            self.aligndata = pd.concat(return_df, axis=0, ignore_index=False)

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
        label = self.aligndata.iloc[idx, self.id_label].astype(
            "int64"
        )  # make it start from 0
        position = np.radians(self.aligndata.iloc[idx, self.position_list].values)
        r_distance = self.aligndata.iloc[idx, self.r_dist_list].to_numpy(
            dtype=np.float32
        )
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
        val_months=[10, 1],
        test_months=[11, 12],
        holdout_months=[],
        radial_norm=False,
        cache_dir="",
        apply_mask=True,
        num_frames=1,
        drop_frame_dim=False,
        min_date=None,
        max_date=None,
        precision="32",
        normalization=None,
        # set variables for solar wind here
        cfg=None,
        train_months=[10],
        train_years=2022,
        val_years=2023,
        test_years=2018,
        alignment_indices_path=None,
        radial_parameters=None,
        latlon_parameters=None,
        cadence="1min",
        label_type="",
        sampling_ratio=None,
        random_state=None,
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
            val_months=val_months,
            test_months=test_months,
            holdout_months=holdout_months,
            normalization=normalization,
            cache_dir=cache_dir,
            apply_mask=apply_mask,
            num_frames=num_frames,
            drop_frame_dim=drop_frame_dim,
            min_date=min_date,
            max_date=max_date,
            precision=precision,
        )
        self.cfg = cfg
        self.alignment_indices_path = alignment_indices_path
        self.cadence = cadence
        self.train_months = train_months
        self.label_type = label_type
        self.sampling_ratio = sampling_ratio
        self.random_state = random_state
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years
        self.precision = precision
        self.radial_parameters = radial_parameters
        self.latlon_parameters = latlon_parameters
        self.radial_norm = radial_norm

        # Loading alignment data from zarr file
        if os.path.exists(self.alignment_indices_path):
            logger.info(f"Alignment file is found: {self.alignment_indices_path}")
            root = zarr.open(self.alignment_indices_path, mode="r")
            columns = root.attrs["columns"]
            self.aligndata = pd.DataFrame(root[:, :], columns=columns)
        else:
            self.create_alignment_data()

        self.aligndata["time_sdo_loc_est"] = pd.to_datetime(
            self.aligndata["time_sdo_loc_est"], unit="s"
        )
        self.aligndata["time_sdo_loc_est"] = self.aligndata[
            "time_sdo_loc_est"
        ].dt.round(freq="s")
        self.aligndata.set_index("time_sdo_loc_est", inplace=True)

        # TODO: add switch + angle cutoff to cfg file
        if "lon_footpoint" in self.latlon_parameters:
            self.aligndata = self.aligndata.loc[
                self.aligndata["lon_footpoint"].abs()
                < self.cfg.data.in_situ.lon_cutoff,
                :,
            ]  # Cut data to just those with magnetic footpoints on the visible solar disk
        elif "sc_pos_SH_lon" in self.latlon_parameters:
            self.aligndata = self.aligndata.loc[
                self.aligndata["sc_pos_SH_lon"].abs()
                < self.cfg.data.in_situ.lon_cutoff,
                :,
            ]  # Cut data to just those with PSP position on the visible solar disk
        self.aligndata.sort_index(inplace=True)

        # normalize float values
        if radial_norm is not None:
            for id_col, col in enumerate(self.radial_parameters):
                # self.radial_parameters.append(col)
                logger.info(f"Normalizing column: {col}")
                self.radial_mean = self.aligndata[col].mean()
                self.radial_std = self.aligndata[col].std()
                self.aligndata.loc[:, f"{col}_norm"] = (
                    self.aligndata[col] - self.radial_mean
                ) / self.radial_std

    def create_alignment_data(self):
        logger.info("Creating alignment dataset")
        # loading source files
        path = os.path.join(
            self.cfg.data.in_situ.base_data_directory,
            self.cfg.data.in_situ.psp_interpolated_path,
        )
        root = zarr.open(path, mode="r")

        # preprocessing psp data files
        columns = root.attrs["columns"]
        df_psp = pd.DataFrame(root[:, :], columns=columns)
        df_psp["time"] = pd.to_datetime(root.attrs["time"])
        df_psp.dropna(
            subset=self.radial_parameters + self.latlon_parameters, inplace=True
        )  # missing values from spc data

        # call the propagation type and covert it to datetime dype
        df_psp[self.cfg.experiment.propagation_type] = df_psp[
            self.cfg.experiment.propagation_type
        ].apply(lambda x: pd.Timedelta(x, unit="seconds"))
        # this timestamp ("time_sdo_loc_est") is used for matching sdoml and psp data
        df_psp["time_sdo_loc_est"] = (
            df_psp["time"] - df_psp[self.cfg.experiment.propagation_type]
        )

        # preprocessing sdoml data
        # sdoml data should start from when the psp data start (we use some buffer of 4 days)
        self.aligndata.reset_index(inplace=True)
        self.aligndata.rename({"index": "Time"}, inplace=True)
        self.aligndata = self.aligndata.loc[
            self.aligndata["Time"]
            >= df_psp["time_sdo_loc_est"].min() - pd.Timedelta(days=4),
            :,
        ]

        # sort dataframes before meging them
        df_psp.sort_values(by="time_sdo_loc_est", inplace=True)
        self.aligndata.sort_values(by="Time", inplace=True)

        # find the nearest timstamps (from right dataframe) based on left key
        df_merge = pd.merge_asof(
            df_psp,
            self.aligndata,
            left_on="time_sdo_loc_est",
            right_on="Time",
            direction="nearest",
            allow_exact_matches=True,
            tolerance=pd.Timedelta(minutes=int(self.cfg.data.match_tolerance)),
            suffixes=("", "_sdoml"),
        )

        # sort merged dataframe, which is reodered by merge
        # we set vp_fit_RTN_0_mean < 100 as outlier
        df_merge.sort_values(by="time_sdo_loc_est", inplace=True)
        df_merge = df_merge.loc[df_merge["vp_fit_RTN_0_mean"] >= 100, :]

        # if we do not find nearest timestamps between two dataframe,
        # we drop those rows
        df_merge.dropna(subset=["Time"], inplace=True)
        df_merge[self.cfg.experiment.propagation_type] = df_merge[
            self.cfg.experiment.propagation_type
        ].dt.total_seconds()

        # Save the data to zarr format
        # only numerical columns can be saved in zarr
        obj_cols = df_merge.select_dtypes(exclude="number").columns.to_list()
        logger.info(f"Object columns: {obj_cols} is converted to int")
        for col in obj_cols:
            df_merge[col] = df_merge[col].values.astype(np.int64) / 10**9  #
            # df_merge[col] = df_merge[col].dt.timestamp()
        num_cols = df_merge.select_dtypes(include="number").columns.to_list()

        z1 = zarr.open(
            self.alignment_indices_path,
            mode="w",
            shape=(len(df_merge), len(num_cols)),
            chunks=(20_000, len(num_cols)),
            dtype="f8",
        )

        z1[:, :] = df_merge[num_cols].to_numpy().astype(float)
        z1.attrs["columns"] = num_cols

        logger.info(f"Alignment data is saved: {self.alignment_indices_path}")

        self.aligndata = df_merge[num_cols]

    def setup(self, stage=None):
        # trainset
        self.train_ds = SWDataset(
            self.aligndata,
            self.hmi_data,
            self.aia_data,
            self.eve_data,
            self.components,
            self.wavelengths,
            self.ions,
            self.cadence,
            self.train_months,
            years=self.train_years,
            mask=self.hmi_mask.numpy(),
            num_frames=self.num_frames,
            drop_frame_dim=self.drop_frame_dim,
            min_date=self.min_date,
            max_date=self.max_date,
            normalization=self.normalization,
            normalization_stat=self.normalization_stat,
            # set variables for solar wind here
            radial_parameters=self.radial_parameters,
            latlon_parameters=self.latlon_parameters,
            label_type=self.label_type,
            sampling_ratio=self.sampling_ratio,
            random_state=self.random_state,
            datasplit="train",
        )
        # validation set
        self.valid_ds = SWDataset(
            self.aligndata,
            self.hmi_data,
            self.aia_data,
            self.eve_data,
            self.components,
            self.wavelengths,
            self.ions,
            self.cadence,
            self.val_months,
            years=self.val_years,
            mask=self.hmi_mask.numpy(),
            num_frames=self.num_frames,
            drop_frame_dim=self.drop_frame_dim,
            min_date=self.min_date,
            max_date=self.max_date,
            normalization=self.normalization,
            normalization_stat=self.normalization_stat,
            # set variables for solar wind here
            radial_parameters=self.radial_parameters,
            latlon_parameters=self.latlon_parameters,
            label_type=self.label_type,
            sampling_ratio=None,
            random_state=None,
            datasplit="val",
        )
        # testset
        self.test_ds = SWDataset(
            self.aligndata,
            self.hmi_data,
            self.aia_data,
            self.eve_data,
            self.components,
            self.wavelengths,
            self.ions,
            self.cadence,
            self.test_months,
            years=self.test_years,
            mask=self.hmi_mask.numpy(),
            num_frames=self.num_frames,
            drop_frame_dim=self.drop_frame_dim,
            min_date=self.min_date,
            max_date=self.max_date,
            normalization=self.normalization,
            normalization_stat=self.normalization_stat,
            # set variables for solar wind here
            radial_parameters=self.radial_parameters,
            latlon_parameters=self.latlon_parameters,
            label_type=self.label_type,
            sampling_ratio=None,
            random_state=None,
            datasplit="test",
        )

        # testset
        self.predict_ds = SWDataset(
            self.aligndata,
            self.hmi_data,
            self.aia_data,
            self.eve_data,
            self.components,
            self.wavelengths,
            self.ions,
            self.cadence,
            self.test_months,
            years=self.test_years,
            mask=self.hmi_mask.numpy(),
            num_frames=self.num_frames,
            drop_frame_dim=self.drop_frame_dim,
            min_date=self.min_date,
            max_date=self.max_date,
            normalization=self.normalization,
            normalization_stat=self.normalization_stat,
            # set variables for solar wind here
            radial_parameters=self.radial_parameters,
            latlon_parameters=self.latlon_parameters,
            label_type=self.label_type,
            sampling_ratio=None,
            random_state=None,
            datasplit="predict",
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


@hydra.main(
    version_base=None, config_path="../configs", config_name="finetune_solarwind_config"
)
def main(cfg):

    # cfg = omegaconf.OmegaConf.load(
    #     ("/home/jh/project/2025-HL-Solar-Wind/classification"
    #      "/configs/finetune_solarwind_config.yaml")
    # )
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
        normalization=cfg.data.normalization,
        eve_path=(
            os.path.join(
                cfg.data.sdoml.base_directory,
                cfg.data.sdoml.sub_directory.eve,
            )
            if cfg.data.sdoml.sub_directory.eve
            else None
        ),
        components=cfg.data.sdoml.components,
        wavelengths=cfg.data.sdoml.wavelengths,
        ions=cfg.data.sdoml.ions,
        frequency=cfg.data.sdoml.frequency,
        batch_size=cfg.model.opt.batch_size,
        num_workers=cfg.data.num_workers,
        val_months=cfg.data.month_splits.val,
        train_months=cfg.data.month_splits.train,
        test_months=cfg.data.month_splits.test,
        train_years=cfg.data.year_splits.train,
        val_years=cfg.data.year_splits.val,
        test_years=cfg.data.year_splits.test,
        holdout_months=cfg.data.month_splits.holdout,
        cache_dir=os.path.join(
            cfg.data.sdoml.save_directory, cfg.data.sdoml.sub_directory.cache
        ),
        min_date=cfg.data.min_date,
        max_date=cfg.data.max_date,
        num_frames=cfg.data.num_frames,
        drop_frame_dim=cfg.data.drop_frame_dim,
        alignment_indices_path=cfg.data.in_situ.base_data_directory
        + cfg.data.in_situ.alignment_indices_path,
        parameters=cfg.data.in_situ.parameters,
        cadence="1min",
        label_type=cfg.experiment.label_type,
        sampling_ratio=None,
        cfg=cfg,
    )
    datamodule.setup()
    # Check dataset and data alignment
    ds = datamodule.train_ds
    print(f"Dataset __len__: {len(ds)}")
    print(f"Aligndata rows: {len(ds.aligndata)}")
    print(f"Frame range: {getattr(ds, 'frame_range', [0])}")

    # Check what index 0 + frame would be
    frame_range = getattr(ds, "frame_range", [0])
    for frame in frame_range:
        target_idx = 0 + frame
        print(f"Trying to access index: {target_idx}")
        if target_idx >= len(ds.aligndata):
            print(f"ERROR: Index {target_idx} >= DataFrame size {len(ds.aligndata)}")

    image, timestamps, header, label, position = datamodule.train_ds[0]


if __name__ == "__main__":
    # cfg = omegaconf.OmegaConf.load(
    #     ("/home/jh/project/2025-HL-Solar-Wind/classification"
    #      "/configs/finetune_solarwind_config.yaml")
    # )
    main()
