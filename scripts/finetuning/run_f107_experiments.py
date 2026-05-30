import os
import inspect
from pathlib import Path

import hydra
import lightning as l
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from loguru import logger
from omegaconf import DictConfig

from sdofmv2.core import MAE, MAE_old
from sdofmv2.tasks.f107 import EmbSolarProxyDataModule, MultiLayerPerceptron
from sdofmv2.utils import flatten_dict


@hydra.main(
    config_path="../../configs/downstream", config_name="finetune_f107_config_sdofmv2_ALL.yaml"
)
def main(cfg: DictConfig):
    logger.info("Starting F10.7 experiment...")

    # Setup Wandb logger
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

    # Setup DataModule
    datamodule = EmbSolarProxyDataModule(
        hmi_path=(
            os.path.join(cfg.data.sdoml.base_directory, cfg.data.sdoml.sub_directory.hmi)
            if cfg.data.sdoml.sub_directory.hmi
            else None
        ),
        aia_path=(
            os.path.join(cfg.data.sdoml.base_directory, cfg.data.sdoml.sub_directory.aia)
            if cfg.data.sdoml.sub_directory.aia
            else None
        ),
        eve_path=None,
        components=cfg.data.sdoml.components,
        wavelengths=cfg.data.sdoml.wavelengths,
        ions=cfg.data.sdoml.ions,
        batch_size=cfg.model.misc.batch_size,
        num_workers=cfg.data.num_workers,
        train_index=cfg.data.train_index,
        val_index=cfg.data.val_index,
        test_index=cfg.data.test_index,
        num_frames=cfg.model.mae.num_frames,
        drop_frame_dim=cfg.data.drop_frame_dim,
        apply_mask=cfg.data.sdoml.apply_mask,
        precision=cfg.experiment.precision,
        normalization=cfg.data.sdoml.normalization,
        normalization_stat_path=cfg.data.normalization_stat_path,
        ds_data_path=cfg.data.ds_data_path,
    )
    datamodule.setup()

    # Load Backbone
    ckpt_path = os.path.join(cfg.experiment.backbone.ckpt_dir, cfg.experiment.backbone.weight_name)
    if cfg.experiment.backbone.model == "mae_old":
        backbone = MAE_old.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            map_location="cpu",
            weights_only=False,
            optimizer_dict=cfg.model.optimizer,
            scheduler_dict=cfg.model.scheduler,
        )
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hyper_parameters = ckpt["hyper_parameters"]

        # Get MAE.__init__ argument names (excluding self)
        valid_args = set(inspect.signature(MAE.__init__).parameters.keys()) - {"self"}

        # Keep only parameters accepted by MAE
        model_hparams = {k: v for k, v in hyper_parameters.items() if k in valid_args}

        backbone = MAE(**model_hparams)
        backbone.load_state_dict(ckpt["state_dict"], strict=False)

    # Create MLP
    model = MultiLayerPerceptron(
        backbone=backbone,
        freeze=cfg.experiment.backbone.freeze,
        input_dim=cfg.model.head.input_dim,
        mask_ratio=cfg.model.head.mask_ratio,
        optimizer_dict=cfg.model.optimizer,
        scheduler_dict=cfg.model.scheduler,
        test_results_path=cfg.experiment.output_dir,
        test_results_filename=cfg.experiment.test_results_filename,
    )

    # Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.experiment.ds_ckpt_dir,
        filename=cfg.experiment.checkpoint_filename,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = l.Trainer(
        max_epochs=cfg.model.misc.epochs,
        devices=cfg.experiment.distributed.devices,
        precision=cfg.experiment.precision,
        callbacks=[checkpoint_callback],
    )

    # Train
    if not os.path.exists(
        os.path.join(cfg.experiment.output_dir, cfg.experiment.checkpoint_filename + ".ckpt")
    ):
        trainer.fit(model=model, datamodule=datamodule)
    else:
        logger.info("Checkpoint exists, skipping training.")

    # Predict/Evaluate
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
