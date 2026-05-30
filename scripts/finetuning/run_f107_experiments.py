import os
from pathlib import Path

import hydra
import lightning as l
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from loguru import logger
from omegaconf import DictConfig

from sdofmv2.core import MAE, MAE_old
from sdofmv2.tasks.f107 import EmbSolarProxyDataModule, MultiLayerPerceptron


@hydra.main(config_path="../../configs/downstream", config_name="finetune_f107_config_sdofmv2_ALL.yaml")
def main(cfg: DictConfig):
    logger.info("Starting F10.7 experiment...")

    # Setup DataModule
    cache_dir = cfg.data.cache_dir
    train_index = Path(cfg.data.train_index).name
    val_index = Path(cfg.data.val_index).name
    test_index = Path(cfg.data.test_index).name

    datamodule = EmbSolarProxyDataModule(
        hmi_path=os.path.join(cfg.data.sdoml.base_directory, cfg.data.sdoml.sub_directory.hmi)
        if cfg.data.sdoml.sub_directory.hmi
        else None,
        aia_path=os.path.join(cfg.data.sdoml.base_directory, cfg.data.sdoml.sub_directory.aia)
        if cfg.data.sdoml.sub_directory.aia
        else None,
        eve_path=None,
        components=cfg.data.sdoml.components,
        wavelengths=cfg.data.sdoml.wavelengths,
        ions=cfg.data.sdoml.ions,
        batch_size=cfg.model.misc.batch_size,
        num_workers=cfg.data.num_workers,
        train_index=os.path.join(cache_dir, train_index),
        val_index=os.path.join(cache_dir, val_index),
        test_index=os.path.join(cache_dir, test_index),
        num_frames=cfg.model.mae.num_frames,
        drop_frame_dim=cfg.data.drop_frame_dim,
        apply_mask=cfg.data.sdoml.apply_mask,
        precision=cfg.experiment.precision,
        normalization=cfg.data.sdoml.normalization,
        normalization_stat_path=cache_dir,
        ds_data_path=cfg.data.ds_data_path,
    )
    datamodule.setup()

    # Load Backbone
    ckpt_path = cfg.model.backbone_checkpoint_path
    if cfg.model.backbone_type == "MAE_old":
        backbone = MAE_old.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            map_location="cpu",
            weights_only=False,
            optimizer_dict=cfg.model.optimizer,
            scheduler_dict=cfg.model.scheduler,
        )
    else:
        backbone = MAE.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            map_location="cpu",
            weights_only=False,
        )

    # Create MLP
    model = MultiLayerPerceptron(
        backbone=backbone,
        freeze=cfg.model.freeze_backbone,
        input_dim=cfg.model.input_dim,
        mask_ratio=cfg.model.mask_ratio,
        optimizer_dict=cfg.model.optimizer,
        scheduler_dict=cfg.model.scheduler,
    )

    # Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.experiment.output_dir,
        filename=cfg.experiment.checkpoint_filename,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = l.Trainer(
        max_epochs=cfg.model.misc.epochs,
        devices=[0] if torch.cuda.is_available() else "auto",
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
