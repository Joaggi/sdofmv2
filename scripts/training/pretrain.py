import os
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.multiprocessing as mp

mp.set_sharing_strategy("file_system")
# PyTorch Lightning imports
import lightning.pytorch as pl
import wandb
from einops import rearrange
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    Timer,
)
from lightning.pytorch.loggers.wandb import WandbLogger
from loguru import logger as lgr_logger
from omegaconf import DictConfig, OmegaConf

from sdofmv2 import utils
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

    def __init__(self, cfg, logger=None, is_backbone=False):
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
        self.data_module.setup()

        model_hyperparams = {
            **cfg.model.mae,
            "chan_types": self.chan_types,
            "limb_mask": torch.Tensor(np.load(self.cfg.data.hmi_mask)),
            "loss_dict": self.cfg.model.loss,
            "optimizer_dict": self.cfg.model.optimizer,
            "scheduler_dict": self.cfg.model.scheduler,
        }

        self.model = self.load_from_ckpt(model_hyperparams)

    def run(self):
        """Executes the pre-training loop.

        Returns:
            pl.Trainer: The trainer instance after completing the fit process.
        """
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
        """Runs the evaluation loop on the validation set."""
        self.trainer.evaluate()

    def test(self):
        """Runs the test loop on the test set."""
        self.trainer.test(
            model=self.model,
            datamodule=self.data_module,
            ckpt_path=self.ckpt_path,
            weights_only=False,
        )

    def load_from_ckpt(self, model_hyperparams):
        # load backbone weights if specified
        # NOTE: weights_only=False is required because we need hyper_parameters
        if self.cfg.experiment.backbone.is_backbone:
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
                model = MAE(**model_hyperparams)
                model.load_state_dict(ckpt["state_dict"], strict=False)

            else:
                lgr_logger.info("Resuming training from checkpoint...")
                lgr_logger.info(f"ckpt: {self.cfg.experiment.backbone.weight_name}")

        else:
            lgr_logger.info("No checkpoint, training from scratch")

        model = MAE(**model_hyperparams)
        return model


@hydra.main(
    config_path="../../configs/pretrain/",
    config_name="pretrain_mae_ALL.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # set seed
    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)
    random.seed(cfg.experiment.seed)
    seed_everything(cfg.experiment.seed)

    # run experiment
    print("\nRunning with config:")
    print(OmegaConf.to_yaml(cfg, resolve=False, sort_keys=False))
    print("\n")

    print(f"Using device: {cfg.experiment.accelerator}")

    # set up wandb logging
    if cfg.experiment.wandb.enable:
        wandb.login()
        output_dir = Path(cfg.experiment.wandb.output_directory)
        output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Created directory for storing results: {cfg.experiment.wandb.output_directory}")
        cache_dir = Path(f"{cfg.experiment.wandb.output_directory}/.cache")
        cache_dir.mkdir(exist_ok=True, parents=True)

        os.environ["WANDB_CACHE_DIR"] = f"{cfg.experiment.wandb.output_directory}/.cache"

        logger = WandbLogger(
            # WandbLogger params
            name=cfg.experiment.wandb.name,
            project=cfg.experiment.wandb.project,
            dir=cfg.experiment.wandb.output_directory,
            log_model=cfg.experiment.wandb.log_model,
            # kwargs for wandb.init
            tags=cfg.experiment.wandb.tags,
            notes=cfg.experiment.wandb.notes,
            group=cfg.experiment.wandb.group,
            save_code=True,
            job_type=cfg.experiment.wandb.job_type,
            config=flatten_dict(cfg),
            id=cfg.experiment.wandb.run_id,
            resume="allow",
            mode="offline" if cfg.experiment.wandb.offline else "online",
        )

    else:
        logger = None

    pretrainer = Pretrainer(
        cfg,
        logger=logger,
        is_backbone=cfg.experiment.backbone.is_backbone,
    )

    pretrainer.run()


if __name__ == "__main__":
    time_start = time.time()

    # errors
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Produce a complete stack trace

    main()
    print(f"\nTotal duration: {utils.days_hours_mins_secs_str(time.time() - time_start)}")
