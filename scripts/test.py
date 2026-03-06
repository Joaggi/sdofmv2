import datetime
import os
import random
import time
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from loguru import logger as lgr_logger

# PyTorch Lightning imports
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    Timer,
)

from sdofmv2 import utils
from sdofmv2.utils import flatten_dict
from sdofmv2.models import MAE
from sdofmv2.datasets import SDOMLDataModule
from sdofmv2.constants import ALL_COMPONENTS, ALL_WAVELENGTHS
from .pretrain import Pretrainer


@hydra.main(
    config_path="../configs/pretrain/",
    config_name="pretrain_mae_HMI.yaml",
)
def main(cfg: DictConfig) -> None:

    # set seed
    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)
    random.seed(cfg.experiment.seed)
    seed_everything(cfg.experiment.seed)

    # set precision of torch tensors
    match cfg.experiment.precision:
        case 64:
            torch.set_default_tensor_type(torch.DoubleTensor)
        case 32:
            torch.set_default_tensor_type(torch.FloatTensor)
        case _:
            warnings.warn(
                f"Setting precision {cfg.experiment.precision} will pass through to the trainer but not other operations."
            )

    # run experiment
    print(f"\nRunning with config:")
    print(OmegaConf.to_yaml(cfg, resolve=False, sort_keys=False))
    print("\n")

    print(f"Using device: {cfg.experiment.accelerator}")

    # set up wandb logging
    if cfg.experiment.wandb.enable:
        wandb.login()
        output_dir = Path(cfg.experiment.wandb.output_directory)
        output_dir.mkdir(exist_ok=True, parents=True)
        print(
            f"Created directory for storing results: {cfg.experiment.wandb.output_directory}"
        )
        cache_dir = Path(f"{cfg.experiment.wandb.output_directory}/.cache")
        cache_dir.mkdir(exist_ok=True, parents=True)

        os.environ["WANDB_CACHE_DIR"] = (
            f"{cfg.experiment.wandb.output_directory}/.cache"
        )

        logger = WandbLogger(
            # WandbLogger params
            name=cfg.experiment.name
            + f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            project=cfg.experiment.project,
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
            offline=cfg.experiment.wandb.offline,
        )

    else:
        logger = None

    pretrainer = Pretrainer(
        cfg,
        logger=logger,
        profiler=profiler,
        is_backbone=cfg.experiment.backbone.is_backbone,
    )
    pretrainer.test()


if __name__ == "__main__":

    time_start = time.time()

    # errors
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Produce a complete stack trace

    main()
    print(
        "\nTotal duration: {}".format(
            utils.days_hours_mins_secs_str(time.time() - time_start)
        )
    )
