# Main pretraining and evaluation script for SDO-FM

import os
from loguru import logger as lgr_logger
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    Timer,
    LearningRateMonitor,
)

# from sdofm import utils
from sdofm.datasets import (
    BrightSpotsSDOMLDataModule,
    HelioProjectedSDOMLDataModule,
    NonLinearSDOMLDataModule,
    SDOMLDataModule,
)
from sdofm.pretraining import MAE, MAE_old, NVAE, SAMAE, BrightSpots
from sdofm.constants import ALL_COMPONENTS, ALL_WAVELENGTHS


class Pretrainer(object):
    def __init__(
        self, cfg, logger=None, profiler=None, is_backbone=False, data_module=None
    ):
        self.cfg = cfg
        self.logger = logger  # would be wandb but broken
        self.profiler = profiler  # if profiler is not None else Profiler()
        self.data_module = data_module
        self.model = None
        self.model_class = None
        self.data_module_class = SDOMLDataModule

        self.callbacks = [
            ModelCheckpoint(
                dirpath=cfg.experiment.backbone.ckpt_dir,
                filename=(
                    f"id_{logger.experiment.id}_{cfg.experiment.model}_"
                    "{epoch}-{val_loss:.2f}"
                ),
                verbose=True,
                monitor="val_loss",  # Specify which metric to monitor
                mode="min",  # Use "min" for loss (lower is better)
                save_top_k=3,  # Keep top 3 checkpoints with lowest val_loss
                save_last=True,
                save_weights_only=False,  # Change to True if you only want weights
                enable_version_counter=True,
            ),
            Timer(),
            RichProgressBar(),
            LearningRateMonitor(logging_interval="step"),
        ]

        if self.cfg.experiment.distributed.enabled:
            self.trainer = pl.Trainer(
                accumulate_grad_batches=(
                    self.cfg.model.misc.target_grad_batches
                    // self.cfg.model.misc.batch_size
                ),
                devices=self.cfg.experiment.distributed.devices,
                accelerator=self.cfg.experiment.accelerator,
                max_epochs=self.cfg.model.misc.epochs,
                precision=self.cfg.experiment.precision,
                profiler=self.profiler,
                logger=self.logger,
                enable_checkpointing=True,
                log_every_n_steps=self.cfg.experiment.log_every_n_steps,
                callbacks=self.callbacks,
                limit_train_batches=self.cfg.experiment.limit_train_batches,
            )
        else:
            self.trainer = pl.Trainer(
                accelerator=self.cfg.experiment.accelerator,
                max_epochs=self.cfg.model.misc.epochs,
                logger=self.logger,
                callbacks=self.callbacks,
                limit_train_batches=self.cfg.experiment.limit_train_batches,
            )

        # check input channels
        aia_list = (
            ALL_WAVELENGTHS
            if cfg.data.sdoml.sub_directory.aia and cfg.data.sdoml.wavelengths is None
            else cfg.data.sdoml.wavelengths or []
        )

        hmi_list = (
            ALL_COMPONENTS
            if cfg.data.sdoml.sub_directory.hmi and cfg.data.sdoml.components is None
            else cfg.data.sdoml.components or []
        )

        aia_list.sort()
        hmi_list.sort()
        self.chan_types = aia_list + hmi_list

        # check model name
        model_name = (
            cfg.experiment.model if not is_backbone else cfg.experiment.backbone.model
        )

        if cfg.data.sdoml.feature_engineering.enabled:
            match cfg.data.sdoml.feature_engineering.dclass:
                case "BrightSpots":
                    self.data_module_class = BrightSpotsSDOMLDataModule
                case "HelioProjected":
                    self.data_module_class = HelioProjectedSDOMLDataModule
                case "Log":
                    self.data_module_class = NonLinearSDOMLDataModule

        print(f"Using {self.data_module_class} Data Class")

        match model_name:
            case "mae":
                self.model_class = MAE
                if self.data_module is None:
                    self.data_module = self.data_module_class(
                        hmi_path=(
                            os.path.join(
                                self.cfg.data.sdoml.base_directory,
                                self.cfg.data.sdoml.sub_directory.hmi,
                            )
                            if self.cfg.data.sdoml.sub_directory.hmi
                            else None
                        ),
                        aia_path=(
                            os.path.join(
                                self.cfg.data.sdoml.base_directory,
                                self.cfg.data.sdoml.sub_directory.aia,
                            )
                            if self.cfg.data.sdoml.sub_directory.aia
                            else None
                        ),
                        eve_path=None,
                        components=self.cfg.data.sdoml.components,
                        wavelengths=self.cfg.data.sdoml.wavelengths,
                        ions=self.cfg.data.sdoml.ions,
                        frequency=self.cfg.data.sdoml.frequency,
                        batch_size=self.cfg.model.misc.batch_size,
                        num_workers=self.cfg.data.num_workers,
                        pin_memory=self.cfg.data.pin_memory,
                        persistent_workers=self.cfg.data.persistent_workers,
                        val_months=self.cfg.data.month_splits.val,
                        test_months=self.cfg.data.month_splits.test,
                        holdout_months=self.cfg.data.month_splits.holdout,
                        cache_dir=os.path.join(
                            self.cfg.data.sdoml.save_directory,
                            self.cfg.data.sdoml.sub_directory.cache,
                        ),
                        min_date=self.cfg.data.min_date,
                        max_date=self.cfg.data.max_date,
                        num_frames=self.cfg.model.mae.num_frames,
                        drop_frame_dim=self.cfg.data.drop_frame_dim,
                        apply_mask=self.cfg.data.sdoml.apply_mask,
                        precision=self.cfg.experiment.precision,
                        normalization=self.cfg.data.sdoml.normalization,
                    )
                    self.data_module.setup()

                model_hyperparams_dict = {
                    **cfg.model.mae,
                    "chan_types": self.chan_types,
                    "limb_mask": (
                        self.data_module.hmi_mask
                        if cfg.model.misc.limb_mask is True
                        else None
                    ),
                    "loss_dict": self.cfg.model.loss,
                    "optimizer_dict": self.cfg.model.optimizer,
                    "scheduler_dict": self.cfg.model.scheduler,
                }
                self.model = self.define_model(model_hyperparams_dict)

            case "mae_old":
                self.model_class = MAE_old
                if self.data_module is None:
                    self.data_module = self.data_module_class(
                        hmi_path=(
                            os.path.join(
                                self.cfg.data.sdoml.base_directory,
                                self.cfg.data.sdoml.sub_directory.hmi,
                            )
                            if self.cfg.data.sdoml.sub_directory.hmi
                            else None
                        ),
                        aia_path=(
                            os.path.join(
                                self.cfg.data.sdoml.base_directory,
                                self.cfg.data.sdoml.sub_directory.aia,
                            )
                            if self.cfg.data.sdoml.sub_directory.aia
                            else None
                        ),
                        eve_path=None,
                        components=self.cfg.data.sdoml.components,
                        wavelengths=self.cfg.data.sdoml.wavelengths,
                        ions=self.cfg.data.sdoml.ions,
                        frequency=self.cfg.data.sdoml.frequency,
                        batch_size=self.cfg.model.misc.batch_size,
                        num_workers=self.cfg.data.num_workers,
                        pin_memory=self.cfg.data.pin_memory,
                        persistent_workers=self.cfg.data.persistent_workers,
                        val_months=self.cfg.data.month_splits.val,
                        test_months=self.cfg.data.month_splits.test,
                        holdout_months=self.cfg.data.month_splits.holdout,
                        cache_dir=os.path.join(
                            self.cfg.data.sdoml.save_directory,
                            self.cfg.data.sdoml.sub_directory.cache,
                        ),
                        min_date=self.cfg.data.min_date,
                        max_date=self.cfg.data.max_date,
                        num_frames=self.cfg.model.mae.num_frames,
                        drop_frame_dim=self.cfg.data.drop_frame_dim,
                        apply_mask=self.cfg.data.sdoml.apply_mask,
                        precision=self.cfg.experiment.precision,
                        normalization=self.cfg.data.sdoml.normalization,
                    )
                    self.data_module.setup()

                model_hyperparams_dict = {
                    **cfg.model.mae,
                    "chan_types": self.chan_types,
                    "limb_mask": (
                        self.data_module.hmi_mask
                        if cfg.model.misc.limb_mask is True
                        else None
                    ),
                    "norm_pix_loss": cfg.model.loss.norm_pix_loss,
                    "optimizer_dict": self.cfg.model.optimizer,
                    "scheduler_dict": self.cfg.model.scheduler,
                }
                self.model = self.define_model(model_hyperparams_dict)

            case "samae":
                self.model_class = SAMAE
                self.data_module = self.data_module_class(
                    hmi_path=None,
                    aia_path=os.path.join(
                        cfg.data.sdoml.base_directory, cfg.data.sdoml.sub_directory.aia
                    ),
                    eve_path=None,
                    components=cfg.data.sdoml.components,
                    wavelengths=cfg.data.sdoml.wavelengths,
                    ions=cfg.data.sdoml.ions,
                    frequency=cfg.data.sdoml.frequency,
                    batch_size=cfg.model.misc.batch_size,
                    num_workers=cfg.data.num_workers,
                    val_months=cfg.data.month_splits.val,
                    test_months=cfg.data.month_splits.test,
                    holdout_months=cfg.data.month_splits.holdout,
                    cache_dir=os.path.join(
                        cfg.data.sdoml.base_directory,
                        cfg.data.sdoml.sub_directory.cache,
                    ),
                    min_date=cfg.data.min_date,
                    max_date=cfg.data.max_date,
                    num_frames=cfg.model.mae.num_frames,
                    drop_frame_dim=cfg.data.drop_frame_dim,
                    apply_mask=cfg.data.sdoml.apply_mask,
                )
                self.data_module.setup()

                model_hyperparams_dict = {
                    **cfg.model.mae,
                    **cfg.model.samae,
                    "optimizer": cfg.model.optimizer,
                    "lr": cfg.model.optimizer.learning_rate,
                    "weight_decay": cfg.model.optimizer.weight_decay,
                }
                self.model = self.define_model(model_hyperparams_dict)

            case "brightspots":
                self.model_class = BrightSpots
                self.data_module = BrightSpotsSDOMLDataModule(
                    hmi_path=os.path.join(
                        cfg.data.sdoml.base_directory, cfg.data.sdoml.sub_directory.hmi
                    ),
                    aia_path=os.path.join(
                        cfg.data.sdoml.base_directory, cfg.data.sdoml.sub_directory.aia
                    ),
                    eve_path=None,
                    components=cfg.data.sdoml.components,
                    wavelengths=cfg.data.sdoml.wavelengths,
                    ions=cfg.data.sdoml.ions,
                    frequency=cfg.data.sdoml.frequency,
                    batch_size=cfg.model.misc.batch_size,
                    num_workers=cfg.data.num_workers,
                    blosc_cache=None,  # "/home/walsh/blosc_cache", # disable for TPUs
                    val_months=cfg.data.month_splits.val,
                    test_months=cfg.data.month_splits.test,
                    holdout_months=cfg.data.month_splits.holdout,
                    cache_dir=os.path.join(
                        cfg.data.sdoml.base_directory,
                        cfg.data.sdoml.sub_directory.cache,
                    ),
                    min_date=cfg.data.min_date,
                    max_date=cfg.data.max_date,
                    num_frames=cfg.data.num_frames,
                )
                self.data_module.setup()

                model_hyperparams_dict = {
                    **cfg.model.brightspots,
                    "optimizer": cfg.model.optimizer,
                    "lr": cfg.model.optimizer.learning_rate,
                    "weight_decay": cfg.model.optimizer.weight_decay,
                }
                self.model = self.define_model(model_hyperparams_dict)

            case "nvae":
                self.model_class = NVAE
                if self.data_module is None:
                    self.data_module = SDOMLDataModule(
                        hmi_path=os.path.join(
                            self.cfg.data.sdoml.base_directory,
                            self.cfg.data.sdoml.sub_directory.hmi,
                        ),
                        aia_path=os.path.join(
                            self.cfg.data.sdoml.base_directory,
                            self.cfg.data.sdoml.sub_directory.aia,
                        ),
                        eve_path=None,
                        components=self.cfg.data.sdoml.components,
                        wavelengths=self.cfg.data.sdoml.wavelengths,
                        ions=self.cfg.data.sdoml.ions,
                        frequency=self.cfg.data.sdoml.frequency,
                        batch_size=self.cfg.model.misc.batch_size,
                        num_workers=self.cfg.data.num_workers,
                        val_months=self.cfg.data.month_splits.val,
                        test_months=self.cfg.data.month_splits.test,
                        holdout_months=self.cfg.data.month_splits.holdout,
                        cache_dir=os.path.join(
                            self.cfg.data.sdoml.base_directory,
                            self.cfg.data.sdoml.sub_directory.cache,
                        ),
                        min_date=cfg.data.min_date,
                        max_date=cfg.data.max_date,
                        num_frames=cfg.data.num_frames,
                        drop_frame_dim=cfg.data.num_frames,
                    )

                model_hyperparams_dict = {
                    **cfg.model.nvae,
                    "optimizer": cfg.model.optimizer,
                    "lr": cfg.model.optimizer.learning_rate,
                    "weight_decay": cfg.model.optimizer.weight_decay,
                    "hmi_mask": self.data_module.hmi_mask,
                }
                self.model = self.define_model(model_hyperparams_dict)

    def define_model(self, model_hyperparams):

        # load backbone weights if specified
        # NOTE: weights_only=False is required because we need hyper_parameters
        if self.cfg.experiment.backbone.is_backbone:
            self.ckpt_path = os.path.join(
                self.cfg.experiment.backbone.ckpt_dir,
                self.cfg.experiment.backbone.weight_name,
            )

            if self.cfg.experiment.backbone.weights_only:

                ckpt = torch.load(
                    self.ckpt_path,
                    weights_only=False,
                    map_location="cpu",
                )

                lgr_logger.info("Loading weights only from checkpoint...")
                lgr_logger.info(f"ckpt: {self.cfg.experiment.backbone.weight_name}")
                lgr_logger.info("Using hyperparameters from checkpoint")

                # load weights and hyperparameters
                model = self.model_class(**model_hyperparams)
                model.load_state_dict(ckpt["state_dict"], strict=False)

            else:
                lgr_logger.info("Resuming training from checkpoint...")
                lgr_logger.info(f"ckpt: {self.cfg.experiment.backbone.weight_name}")

        else:
            lgr_logger.info("No checkpoint, training from scratch")

        model = self.model_class(**model_hyperparams)
        return model

    def run(self):
        print("\nPRE-TRAINING\n")
        self.trainer.fit(
            model=self.model,
            datamodule=self.data_module,
            ckpt_path=(
                self.ckpt_path
                if self.cfg.experiment.backbone.is_backbone
                and not self.cfg.experiment.backbone.weights_only
                else None
            ),
            weights_only=False,
        )
        return self.trainer

    def evaluate(self):
        self.trainer.evaluate()

    def test(self):
        self.trainer.test(
            model=self.model,
            datamodule=self.data_module,
            ckpt_path=self.ckpt_path,
            weights_only=False,
        )
