import hydra
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from sdofmv2.tasks.missing_data import (
    SuryaReconstructionDataModule,
    SuryaReconstructionModel,
)


@hydra.main(config_path="../configs/downstream", config_name="reconstruct_missing_channel", version_base="1.3")
def main(config: DictConfig):
    """Main entry point for the reconstruction script using Hydra."""
    print(OmegaConf.to_yaml(config))

    pl.seed_everything(config.data.get("seed", 42))

    datamodule = SuryaReconstructionDataModule(config)
    model = SuryaReconstructionModel(config)

    # Initialize WandB logger if config exists
    logger = None
    if "wandb" in config:
        logger = WandbLogger(
            project=config.wandb.project,
            name=config.wandb.name,
            entity=config.wandb.get("entity", None),
            save_dir=config.wandb.get("save_dir", "./results"),
        )

    trainer = pl.Trainer(
        max_epochs=config.etc.max_epochs,
        devices=config.etc.devices,
        accelerator=config.etc.accelerator,
        precision=config.etc.precision,
        accumulate_grad_batches=config.etc.get("accumulate_grad_batches", 1),
        gradient_clip_val=config.etc.get("gradient_clip_val", None),
        logger=logger,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
