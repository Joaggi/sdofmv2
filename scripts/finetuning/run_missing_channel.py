import os
import inspect
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import pandas as pd
import torch
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
import sunpy.visualization.colormaps as sunpycm
from sunpy.visualization.colormaps import color_tables
from loguru import logger as lgr_logger

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sdofmv2.core import SDOMLDataModule, MAE, MAE_v1
from sdofmv2.tasks.missing_data import MissingDataModel


@hydra.main(
    version_base=None,
    config_path="../../configs/downstream",
    config_name="missing_channel_sdofmv2_HMI.yaml",
)
def main(cfg: DictConfig):
    # Set seed for reproducibility
    if cfg.experiment.seed is not None:
        pl.seed_everything(cfg.experiment.seed)

    # Initialize WandB Logger
    if cfg.experiment.wandb.enable:
        lgr_logger.info("Initializing Weights & Biases Logger...")
        wandb_logger = WandbLogger(
            project=cfg.experiment.wandb.project,
            entity=cfg.experiment.wandb.entity,
            name=cfg.experiment.wandb.name,
            id=cfg.experiment.wandb.run_id if cfg.experiment.wandb.run_id else None,
            group=cfg.experiment.wandb.group,
            job_type=cfg.experiment.wandb.job_type,
            tags=list(cfg.experiment.wandb.tags),
            notes=cfg.experiment.wandb.notes,
            save_dir=cfg.experiment.wandb.output_directory,
            log_model=cfg.experiment.wandb.log_model,
            offline=cfg.experiment.wandb.offline,
        )
        lgr_logger.info(f"WandB run name: {wandb_logger.experiment.name}")
        lgr_logger.info(f"WandB run ID: {wandb_logger.experiment.id}")
        OmegaConf.resolve(cfg)
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    else:
        wandb_logger = None

    # Load DataModule
    lgr_logger.info("Loading SDOMLDataModule...")
    data_module = SDOMLDataModule(
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
        eve_path=None,
        components=cfg.data.sdoml.components,
        wavelengths=cfg.data.sdoml.wavelengths,
        ions=cfg.data.sdoml.ions,
        batch_size=cfg.model.misc.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
        multiprocessing_context=cfg.data.multiprocessing_context,
        normalization=cfg.data.sdoml.normalization,
        normalization_stat_path=cfg.data.normalization_stat_path,
        train_index=cfg.data.train_index,
        val_index=cfg.data.val_index,
        test_index=cfg.data.test_index,
        hmi_mask_path=cfg.data.hmi_mask,
        apply_mask=cfg.data.sdoml.apply_mask,
        num_frames=cfg.data.num_frames,
        drop_frame_dim=cfg.data.drop_frame_dim,
        precision=cfg.experiment.precision,
    )
    data_module.setup()

    # Visualization setup
    channels = (data_module.wavelengths or []) + (data_module.components or [])

    # load checkpoint
    ckpt_path = os.path.join(cfg.experiment.backbone.ckpt_dir, cfg.experiment.backbone.weight_name)
    lgr_logger.info("Loading checkpoint...")
    if cfg.experiment.backbone.model == "mae_v1":
        backbone = MAE_v1.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            map_location="cpu",
            weights_only=False,
            optimizer_dict=cfg.model.optimizer,
            scheduler_dict=cfg.model.scheduler,
        )
    else:
        try:
            backbone = MAE.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                map_location="cpu",
                weights_only=False,
            )

        except Exception as e:
            print(f"Standard loading failed: {e}. Falling back to manual load...")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            hyper_parameters = ckpt["hyper_parameters"]

            # Get MAE.__init__ argument names (excluding self)
            valid_args = set(inspect.signature(MAE.__init__).parameters.keys()) - {"self"}

            # Keep only parameters accepted by MAE
            model_hparams = {k: v for k, v in hyper_parameters.items() if k in valid_args}

            backbone = MAE(**model_hparams)
            backbone.load_state_dict(ckpt["state_dict"], strict=False)

    # Create MissingDataModel
    lgr_logger.info("Creating MissingDataModel...")
    model_params = {
        "optimizer_dict": cfg.model.optimizer,
        "scheduler_dict": cfg.model.scheduler,
    }
    model = MissingDataModel(
        **model_params,
        backbone=backbone,
        freeze_encoder=True,
        normalization=cfg.data.sdoml.normalization,
        normalization_stat=data_module.normalization_stat,
        wavelengths=channels,
        masking_ratio=0.0,
        test_result_path=os.path.join(
                cfg.experiment.output_dir,
                cfg.experiment.test_results_filename
                ),
        hyperparam_ignore=["backbone"],
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.experiment.ds_ckpt_dir,
        filename=cfg.experiment.ckpt_tag,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
    )

    # Trainer
    lgr_logger.info("Initializing Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=cfg.model.misc.epochs,
        precision=cfg.experiment.precision,
        accelerator=cfg.experiment.accelerator,
        devices=list(cfg.experiment.distributed.devices),
        logger=wandb_logger,
        log_every_n_steps=cfg.experiment.log_every_n_steps,
        callbacks=[checkpoint_callback],
        gradient_clip_val=cfg.model.misc.gradient_clip_val,
        gradient_clip_algorithm=cfg.model.misc.gradient_clip_algorithm,
        accumulate_grad_batches=cfg.model.misc.accumulate_grad_batches,
        limit_train_batches=cfg.model.misc.limit_train_batches,
    )

    ckpt_path = (
        os.path.join(
            cfg.experiment.ds_ckpt_dir,
            cfg.experiment.checkpoint_filename
        )
        if cfg.experiment.checkpoint_filename
        else None
    )

    phase = cfg.experiment.phase

    if phase == "train":
        trainer.fit(model=model, datamodule=data_module)

    elif phase == "resume":
        trainer.fit(
            model=model,
            datamodule=data_module,
            ckpt_path=ckpt_path,
            weights_only=False,
        )

    elif phase == "test":
        lgr_logger.info("Test run from ckpt...")
        trainer.test(
            model=model,
            datamodule=data_module,
            ckpt_path=ckpt_path,
            weights_only=False,
        )
        return

    # train/resume only
    best_ckpt = trainer.checkpoint_callback.best_model_path
    lgr_logger.info("Running evaluation on test set...")
    trainer.test(
        datamodule=data_module,
        ckpt_path=best_ckpt,
        weights_only=False,
    )
    lgr_logger.info("Evaluation complete. Metrics logged to WandB via epoch-end hooks.")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
