import os

import hydra
import numpy as np
import torch
import wandb

# pytorch lightining
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers.wandb import WandbLogger
from loguru import logger as loguru_logger
from omegaconf.base import ContainerMetadata
from omegaconf.listconfig import ListConfig

# from SDOFMv2
from sdofmv2.core import MAE
from sdofmv2.tasks.solar_wind import SWClassifier, SWDataModule
from sdofmv2.utils import ALL_COMPONENTS, ALL_WAVELENGTHS, flatten_dict


@hydra.main(
    version_base=None,
    config_path="../../configs/downstream/",
    config_name="solarwind_sdofmv2_ALL.yaml",
)
def main(cfg):
    """Executes the fine-tuning pipeline for the solar wind classification task.

    It sets up the experiment environment by initializing WandB logging,
    configuring the solar wind data module, and loading the SDO-FM backbone.
    The function constructs the downstream classifier, defines training
    callbacks, and manages the PyTorch Lightning trainer to start the model
    fitting process.

    Args:
        cfg (DictConfig): The Hydra configuration object containing experiment,
            data, and model parameters.

    Returns:
        None
    """
    torch.serialization.add_safe_globals([ListConfig, ContainerMetadata])

    # set logger
    print("Wandb login status:", wandb.login())
    logger = WandbLogger(
        # WandbLogger params
        entity=cfg.experiment.wandb.entity,
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
        resume="allow",
        mode="offline" if cfg.experiment.wandb.offline else "online",
        id=cfg.experiment.wandb.run_id,
    )

    # Load datamodule
    data_module = SWDataModule(
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
        frequency=cfg.data.sdoml.frequency,
        batch_size=cfg.model.misc.batch_size,
        num_workers=cfg.data.num_workers,
        apply_mask=cfg.data.sdoml.apply_mask,
        num_frames=cfg.data.num_frames,
        drop_frame_dim=cfg.data.drop_frame_dim,
        precision=cfg.experiment.precision,
        normalization=cfg.data.sdoml.normalization,
        normalization_stat_path=cfg.data.normalization_stat_path,
        cfg=cfg,
        radial_norm=cfg.data.in_situ.radial_norm,
        radial_parameters=cfg.data.in_situ.radial_parameters,
        latlon_parameters=cfg.data.in_situ.latlon_parameters,
        cadence=cfg.data.in_situ.cadence,
        label_type=cfg.data.label_type,
        sampling_ratio=cfg.data.under_sampling.ratio,
        random_state=cfg.data.under_sampling.random_state,
        train_index=cfg.data.train_index,
        val_index=cfg.data.val_index,
        test_index=cfg.data.test_index,
        merged_splits_dir=cfg.data.index_save_path,
        hmi_mask_path=cfg.data.hmi_mask,
    )

    # Define channels for input/model
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
    channels = aia_list + hmi_list

    # Load backbone from SDO-FM
    if cfg.experiment.backbone.is_backbone:
        backbone = MAE.load_from_checkpoint(
            checkpoint_path=os.path.join(
                cfg.experiment.backbone.ckpt_dir, cfg.experiment.backbone.weight_name
            ),
            map_location="cpu",
            weights_only=cfg.experiment.backbone.weights_only,
        )

    else:
        backbone = MAE(
            **cfg.model.mae,
            chan_types=channels,
            limb_mask=torch.Tensor(np.load(cfg.data.hmi_mask)) if cfg.model.misc.get("limb_mask", False) else None,
        )

    # Downstream model
    model = SWClassifier(
        # Head parameters
        num_classes=cfg.model.linear.num_classes,
        class_names=cfg.data.class_names,
        channels=channels,
        head_type=cfg.model.head.type,
        hidden_dim=cfg.model.head.linear.hidden_dim,
        p_drop=cfg.model.head.dropout_p,
        nhead=cfg.model.head.transformer.nhead,
        embed_dim=cfg.model.mae.embed_dim,
        max_position_element=cfg.model.linear.max_position_element,
        position_size=len(cfg.data.in_situ.latlon_parameters),
        skips=cfg.model.head.skips,
        include_raw_coordinates=cfg.model.head.include_raw_coordinates,
        num_hidden_layers=cfg.model.head.num_hidden_layers,
        # backbone
        backbone=backbone,
        freeze_encoder=cfg.experiment.backbone.freeze,
        hyperparam_ignore=["backbone"],
        # opt
        plt_style=cfg.etc.mpl_style,
        radial_mean=data_module.radial_mean,
        radial_std=data_module.radial_std,
        loss_dict=cfg.model.loss,
        optimizer_dict=cfg.model.optimizer,
        scheduler_dict=cfg.model.scheduler,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.experiment.downstream_model.ckpt_dir,
            filename=(
                f"id_{logger.experiment.id}_{cfg.experiment.backbone.model}_{cfg.model.head.type}_"
                "{epoch}-{val_loss:.2f}-{val_f1:.2f}"
            ),
            verbose=True,
            monitor=cfg.model.misc.ckpt_monitor,
            mode="min",
            save_top_k=3,
            save_weights_only=False,
            save_last=True,
            enable_version_counter=False,
        ),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = Trainer(
        accelerator=cfg.experiment.accelerator,
        devices=cfg.experiment.distributed.devices,
        max_epochs=cfg.model.misc.max_epochs,
        precision=cfg.experiment.precision,
        callbacks=callbacks,
        # profiler=cfg.model.misc.profiler,
        check_val_every_n_epoch=cfg.model.misc.check_val_every_n_epoch,
        log_every_n_steps=cfg.model.misc.log_every_n_steps,
        logger=logger,
        limit_train_batches=cfg.model.misc.limit_train_batches,
        limit_val_batches=cfg.model.misc.limit_val_batches,
        limit_test_batches=cfg.model.misc.limit_test_batches,
        limit_predict_batches=cfg.model.misc.limit_predict_batches,
        accumulate_grad_batches=cfg.model.misc.accumulate_grad_batches,
    )

    if cfg.experiment.downstream_model.resuming and cfg.experiment.downstream_model.weights_only:
        loguru_logger.info("Load weight only from ckpt")
        loguru_logger.info("Model hyperparameters are overridden by ckpt")
        ckpt = torch.load(
            os.path.join(
                cfg.experiment.downstream_model.ckpt_dir,
                cfg.experiment.downstream_model.ckpt_name,
            ),
            map_location="cpu",
        )
        model.load_state_dict(**ckpt["hyper_parameters"], strict=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)

    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=(
            os.path.join(
                cfg.experiment.downstream_model.ckpt_dir,
                cfg.experiment.downstream_model.ckpt_name,
            )
            if cfg.experiment.downstream_model.resuming
            and not cfg.experiment.downstream_model.weights_only
            else None
        ),
        weights_only=False,
    )


if __name__ == "__main__":
    main()
