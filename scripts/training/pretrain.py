import os
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.multiprocessing as mp

mp.set_sharing_strategy("file_system")
# PyTorch Lightning imports
import wandb
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

from sdofmv2 import utils
from sdofmv2.utils import flatten_dict
from sdofmv2.core import Pretrainer


@hydra.main(
    config_path="../../configs/pretrain/",
    config_name="pretrain_mae_ALL.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # set seed
    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)
    random.seed(cfg.experiment.seed)
    seed_everything(cfg.experiment.seed)

    # run experiment
    print("\nRunning with config:")
    print(OmegaConf.to_yaml(cfg, resolve=False, sort_keys=False))
    print("\n")

    print(f"Using device: {cfg.experiment.accelerator}")

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

    pretrainer = Pretrainer(
        cfg,
        logger=logger,
    )

    pretrainer.run()


if __name__ == "__main__":
    time_start = time.time()

    # errors
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Produce a complete stack trace

    main()
    print(f"\nTotal duration: {utils.days_hours_mins_secs_str(time.time() - time_start)}")
