import os

import hydra
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
    config_path="../configs/downstream/",
    config_name="finetune_solarwind_config.yaml",
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
        name=(
            f"{cfg.experiment.head}-"
            f"radialpos:{'.'.join(cfg.data.in_situ.radial_parameters)}-"
            f"latlonpos:{'.'.join(cfg.data.in_situ.latlon_parameters)}-"
            f"prop:{cfg.experiment.propagation_type}-"
            f"lbl:{cfg.data.label_type}-"
            f"inst:{cfg.data.instrument}-"
            f"backbone:{cfg.experiment.backbone.model}-"
            f"cadence:{cfg.data.cadence}-"
            f"epoch:{cfg.experiment.trainer.max_epochs}-"
            f"lr:{cfg.model.optimizer.learning_rate}-"
            f"wt_decy:{cfg.model.optimizer.weight_decay}-"
            f"scheduler:{cfg.model.scheduler.use}-"
            f"optimiser:{cfg.model.optimizer.use}-"
            f"batch:{cfg.model.misc.batch_size}-"
            f"limit_train_batches:{cfg.experiment.trainer.limit_train_batches}"
            f"limit_val_batches:{cfg.experiment.trainer.limit_val_batches}"
        ),
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
        resume="never",
        mode="offline" if cfg.experiment.wandb.offline else "online",
        id=None,
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
        cache_dir=os.path.join(cfg.data.sdoml.save_directory, cfg.data.sdoml.sub_directory.cache),
        apply_mask=cfg.data.sdoml.apply_mask,
        num_frames=cfg.data.num_frames,
        drop_frame_dim=cfg.data.drop_frame_dim,
        precision=cfg.experiment.trainer.precision,
        normalization=cfg.data.sdoml.normalization,
        cfg=cfg,
        radial_norm=cfg.data.in_situ.radial_norm,
        radial_parameters=cfg.data.in_situ.radial_parameters,
        latlon_parameters=cfg.data.in_situ.latlon_parameters,
        cadence=cfg.data.in_situ.cadence,
        label_type=cfg.data.label_type,
        sampling_ratio=cfg.data.under_sampling.ratio,
        random_state=cfg.data.under_sampling.random_state,
        hmi_mask_path="hmi_mask_512x512.npy",
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
                cfg.experiment.backbone.ckpt_dir, cfg.experiment.backbone.ckpt_name
            ),
            map_location="cpu",
            weights_only=cfg.experiment.backbone.weights_only,
        )

    else:
        backbone = MAE(
            **cfg.model.mae,
            chan_types=channels,
            limb_mask=data_module.hmi_mask if cfg.model.misc.get("limb_mask", False) else None,
        )

    # Downstream model
    model = SWClassifier(
        # Head parameters
        num_classes=cfg.model.linear.num_classes,
        class_names=cfg.data.class_names,
        channels=channels,
        head_type=cfg.experiment.head,
        hidden_dim=cfg.experiment.linear.hidden_dim,
        p_drop=cfg.experiment.dropout_p,
        nhead=cfg.experiment.transformer.nhead,
        embed_dim=cfg.model.mae.embed_dim,
        max_position_element=cfg.model.linear.max_position_element,
        position_size=len(cfg.data.in_situ.latlon_parameters),
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
                f"id_{logger.experiment.id}_{cfg.experiment.backbone.model}_{cfg.experiment.head}_"
                "{epoch}-{val_loss:.2f}-{val_f1:.2f}"
            ),
            verbose=True,
            monitor=cfg.experiment.trainer.ckpt_monitor,
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
        accelerator=cfg.experiment.trainer.accelerator,
        devices=cfg.experiment.trainer.devices,
        strategy=cfg.experiment.trainer.strategy,
        max_epochs=cfg.experiment.trainer.max_epochs,
        precision=cfg.experiment.trainer.precision,
        callbacks=callbacks,
        profiler=cfg.experiment.trainer.profiler,
        check_val_every_n_epoch=cfg.experiment.trainer.check_val_every_n_epoch,
        log_every_n_steps=cfg.experiment.trainer.log_every_n_steps,
        logger=logger,
        limit_train_batches=cfg.experiment.trainer.limit_train_batches,
        limit_val_batches=cfg.experiment.trainer.limit_val_batches,
        limit_test_batches=cfg.experiment.trainer.limit_test_batches,
        limit_predict_batches=cfg.experiment.trainer.limit_predict_batches,
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
