import os

import torch
import numpy as np
import lightning.pytorch as pl
from loguru import logger as lgr_logger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    Timer,
)

from sdofmv2.core import MAE, SDOMLDataModule
from sdofmv2.utils import ALL_COMPONENTS, ALL_WAVELENGTHS, flatten_dict

class Pretrainer:
    """Coordinates the pre-training workflow for Masked Autoencoder (MAE) models.

    This class sets up the training infrastructure by initializing the data
    module, model, and trainer. It manages checkpoint loading and configures
    callbacks for logging and performance monitoring. It's built to handle
    SDOML data and supports both single-gpu and distributed setups.

    Args:
        cfg (DictConfig): The configuration tree for the experiment.
        logger (WandbLogger, optional): Logger for experiment tracking.
            Defaults to None.
        is_backbone (bool, optional): Whether the model is a backbone.
            Defaults to False.

    Attributes:
        cfg (DictConfig): The experiment configuration.
        logger (WandbLogger): The assigned logger.
        ckpt_path (str): Path to the model checkpoint.
        callbacks (list): List of Lightning callbacks.
        trainer (pl.Trainer): The Lightning trainer instance.
        chan_types (list): List of active data channels.
        data_module (SDOMLDataModule): The data handling component.
        model (MAE): The initialized MAE model.
    """

    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger
        self.ckpt_path = (
            os.path.join(
                self.cfg.experiment.backbone.ckpt_dir,
                self.cfg.experiment.backbone.weight_name,
            )
            if self.cfg.experiment.backbone.weight_name is not None
            else None
        )

        self.callbacks = [
            ModelCheckpoint(
                dirpath=cfg.experiment.backbone.ckpt_dir,
                filename=(
                    f"id_{logger.experiment.id}_{cfg.experiment.model}_{{epoch}}-{{val_loss:.2f}}"
                ),
                verbose=True,
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
                save_weights_only=False,
                enable_version_counter=True,
            ),
            ModelCheckpoint(
                dirpath=cfg.experiment.backbone.ckpt_dir,
                filename=(
                    f"id_{logger.experiment.id}_{cfg.experiment.model}_{{epoch}}-{{step}}-{{val_loss:.2f}}"
                ),
                verbose=True,
                monitor="train_loss",
                mode="min",
                save_top_k=10,              # save all checkpoints
                save_last=True,
                save_weights_only=False,
                enable_version_counter=True,
                every_n_train_steps=100,   # save every 100 optimizer steps -> accum * number
                save_on_train_epoch_end=False,
            ),
            Timer(),
            RichProgressBar(),
            LearningRateMonitor(logging_interval="step"),
        ]

        if self.cfg.experiment.distributed.enabled:
            self.trainer = pl.Trainer(
                accumulate_grad_batches=self.cfg.model.misc.accumulate_grad_batches,
                devices=self.cfg.experiment.distributed.devices,
                accelerator=self.cfg.experiment.accelerator,
                max_epochs=self.cfg.model.misc.epochs,
                precision=self.cfg.experiment.precision,
                logger=self.logger,
                enable_checkpointing=True,
                log_every_n_steps=self.cfg.experiment.log_every_n_steps,
                callbacks=self.callbacks,
                limit_train_batches=self.cfg.model.misc.limit_train_batches,
                gradient_clip_algorithm=self.cfg.model.misc.gradient_clip_algorithm,
                gradient_clip_val=self.cfg.model.misc.gradient_clip_val,
            )
        else:
            self.trainer = pl.Trainer(
                accelerator=self.cfg.experiment.accelerator,
                max_epochs=self.cfg.model.misc.epochs,
                logger=self.logger,
                callbacks=self.callbacks,
                limit_train_batches=self.cfg.model.misc.limit_train_batches,
            )

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

        self.data_module = SDOMLDataModule(
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
            batch_size=self.cfg.model.misc.batch_size,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            persistent_workers=self.cfg.data.persistent_workers,
            multiprocessing_context=self.cfg.data.multiprocessing_context,
            train_index=self.cfg.data.train_index,
            val_index=self.cfg.data.val_index,
            test_index=self.cfg.data.test_index,
            hmi_mask_path=self.cfg.data.hmi_mask,
            num_frames=self.cfg.model.mae.num_frames,
            drop_frame_dim=self.cfg.data.drop_frame_dim,
            apply_mask=self.cfg.data.sdoml.apply_mask,
            precision=self.cfg.experiment.precision,
            normalization=self.cfg.data.sdoml.normalization,
            normalization_stat_path=self.cfg.data.normalization_stat_path,
        )
        # self.data_module.setup()

        self.model_hyperparams = {
            **cfg.model.mae,
            "chan_types": self.chan_types,
            "limb_mask": torch.Tensor(np.load(self.cfg.data.hmi_mask)),
            "loss_dict": self.cfg.model.loss,
            "optimizer_dict": self.cfg.model.optimizer,
            "scheduler_dict": self.cfg.model.scheduler,
            "save_test_results_csv": self.cfg.experiment.save_test_results_csv,
        }

    def run(self):
        """Executes the pre-training loop.

        Returns:
            pl.Trainer: The trainer instance after completing the fit process.
        """
        print("\nPRE-TRAINING\n")
        if self.cfg.experiment.backbone.weights_only:
            model = self.load_ckpt_weight_only()
        else:
            model = MAE(
                **self.model_hyperparams
            )

        self.trainer.fit(
            model=model,
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

    def test(self):
        """Runs the test loop on the test set."""
        try:
            model = MAE.load_from_checkpoint(
                self.ckpt_path,
                save_test_results_csv=self.cfg.experiment.save_test_results_csv,
                map_location="cpu", 
                weights_only=False)
            lgr_logger.info(f"Checkpoint is loaded from {self.ckpt_path}")

        except Exception as e:
            lgr_logger.info(f"Checkpoint load failed. Try manaual loading...")
            model = self.load_ckpt_weight_only()

        self.trainer.test(
            model=model,
            datamodule=self.data_module,
            ckpt_path=None,
            weights_only=False,
        )

    def load_ckpt_weight_only(self):
        lgr_logger.info(f"Loading weights manually from {self.ckpt_path}...")
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        hyper_parameters = ckpt.get("hyper_parameters", {})

        extra = {
            "chan_types": self.chan_types,
            "limb_mask": torch.Tensor(np.load(self.cfg.data.hmi_mask)),
            "loss_dict": self.cfg.model.loss,
            "optimizer_dict": self.cfg.model.optimizer,
            "scheduler_dict": self.cfg.model.scheduler,
            "save_test_results_csv": self.cfg.experiment.save_test_results_csv,
        }

        model_hparams = {**self.model_hyperparams, **hyper_parameters, **extra}
        model = MAE(**model_hparams)

        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        lgr_logger.info("missing =", missing)
        lgr_logger.info("unexpected =", unexpected)

        return model