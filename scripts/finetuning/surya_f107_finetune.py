import hydra
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import DictConfig, OmegaConf
from sdofmv2.tasks.f107.helio_f107_zarr_datamodule import HelioF107ZarrDataModule
from sdofmv2.tasks.f107.surya_f107_module import SuryaF107Model

@hydra.main(config_path="../configs/downstream", config_name="f107_surya", version_base=None)
def main(config: DictConfig):
    pl.seed_everything(42)
    
    datamodule = HelioF107ZarrDataModule(config)
    model = SuryaF107Model(config)
    
    loggers = []
    if config.wandb.enable:
        loggers.append(WandbLogger(
            project=config.wandb.project,
            name=config.wandb.name,
            group=config.wandb.group,
            save_dir=config.wandb.output_directory
        ))
    loggers.append(CSVLogger(config.wandb.output_directory, name="csv_logs"))
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=config.wandb.output_directory,
        filename="best-model-{epoch:02d}-{val_loss:.4f}"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    trainer = pl.Trainer(
        max_epochs=config.etc.max_epochs,
        accelerator=config.etc.accelerator,
        devices=config.etc.devices,
        precision=config.etc.precision,
        logger=loggers,
        callbacks=[checkpoint_callback, lr_monitor]
    )
    
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
