import os
import hydra
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from sdofmv2.tasks.missing_data.surya_reconstruction_zarr_datamodule import (
    SuryaReconstructionZarrDataModule,
    SuryaReconstructionModel,
)
from sdofmv2.utils.data_utils import safe_collate

torch.set_float32_matmul_precision("medium")


@hydra.main(
    config_path="../configs/downstream",
    config_name="reconstruct_missing_channel",
    version_base=None,
)
def main(config: DictConfig):
    """Main entry point for the reconstruction script using Hydra."""
    print(OmegaConf.to_yaml(config))

    pl.seed_everything(config.data.get("seed", 42))

    datamodule = SuryaReconstructionZarrDataModule(config)
    model = SuryaReconstructionModel(config)

    # Initialize Loggers
    loggers = []
    if "wandb" in config:
        loggers.append(
            WandbLogger(
                name=config.wandb.name,
                project=config.wandb.project,
                dir=config.wandb.output_directory,
                log_model=config.wandb.log_model,
                # kwargs for wandb.init
                tags=config.wandb.tags,
                notes=config.wandb.notes,
                group=config.wandb.group,
                save_code=True,
                job_type=config.wandb.job_type,
                config=flatten_dict(config),
                id=config.wandb.run_id,
                resume="allow",
                mode="offline" if config.wandb.offline else "online",
            )
        )

    loggers.append(
        CSVLogger(save_dir=config.wandb.get("output_directory", "./results"), name="csv_logs")
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        dirpath=config.etc.ckpt_dir,
        filename="reconstruction-best-{epoch:02d}-{val_loss:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=config.etc.max_epochs,
        devices=config.etc.devices,
        accelerator=config.etc.accelerator,
        precision=config.etc.precision,
        accumulate_grad_batches=config.etc.get("accumulate_grad_batches", 1),
        gradient_clip_val=config.etc.get("gradient_clip_val", None),
        gradient_clip_algorithm=config.etc.get("gradient_clip_algorithm", "norm"),
        logger=loggers,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=(
            os.path.join(config.etc.ckpt_dir, config.etc.ckpt_name)
            if config.etc.ckpt_name is not None
            else None
        ),
        weights_only=False,
    )


if __name__ == "__main__":
    main()
