import os
from pathlib import Path
import hydra
import lightning as l
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers.wandb import WandbLogger
from loguru import logger as lgr_logger
from omegaconf import DictConfig, OmegaConf
from sdofmv2.tasks.f107.surya_f107_datamodule import HelioF107DataModule
from sdofmv2.tasks.f107.surya_f107_module import SuryaF107Model
from sdofmv2.utils import flatten_dict


@hydra.main(config_path="../configs/downstream", config_name="f107_surya", version_base=None)
def main(cfg: DictConfig):
    l.seed_everything(42)
    lgr_logger.info("Starting Surya F10.7 experiment...")

    # Setup Wandb logger
    if cfg.experiment.wandb.enable:
        wandb.login()
        output_dir = Path(cfg.experiment.wandb.output_directory)
        output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Created directory for storing results: {cfg.experiment.wandb.output_directory}")
        cache_dir = Path(f"{cfg.experiment.wandb.output_directory}/.cache")
        cache_dir.mkdir(exist_ok=True, parents=True)

        os.environ["WANDB_CACHE_DIR"] = f"{cfg.experiment.wandb.output_directory}/.cache"

        logger = WandbLogger(
            name=cfg.experiment.wandb.name,
            project=cfg.experiment.wandb.project,
            dir=cfg.experiment.wandb.output_directory,
            log_model=cfg.experiment.wandb.log_model,
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

    datamodule = HelioF107DataModule(cfg)
    model = SuryaF107Model(cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=cfg.experiment.wandb.output_directory,
        filename="best-model-{epoch:02d}-{val_loss:.4f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = l.Trainer(
        max_epochs=cfg.etc.max_epochs,
        accelerator=cfg.etc.accelerator,
        devices=cfg.etc.devices,
        precision=cfg.etc.precision,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best", weights_only=False)


if __name__ == "__main__":
    main()
