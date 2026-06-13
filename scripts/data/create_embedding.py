import os
import random
import time

import hydra
import numpy as np
import torch
import torch.multiprocessing as mp

mp.set_sharing_strategy("file_system")

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import RichProgressBar
from loguru import logger as lgr_logger
from omegaconf import DictConfig, OmegaConf

from sdofmv2 import utils
from sdofmv2.core import MAE, SDOMLDataModule
from sdofmv2.utils import ALL_COMPONENTS, ALL_WAVELENGTHS


class Predictor:
    """Coordinates the prediction workflow for Masked Autoencoder (MAE) models.

    This class sets up the data module, model, and trainer. It manages
    checkpoint loading and calls the prediction loop to save embeddings.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.ckpt_path = (
            os.path.join(
                self.cfg.experiment.backbone.ckpt_dir,
                self.cfg.experiment.backbone.weight_name,
            )
            if self.cfg.experiment.backbone.weight_name is not None
            else None
        )

        self.callbacks = [RichProgressBar()]

        # Setup Trainer
        # During prediction we can typically use fewer devices or rely on a simpler setup.
        if self.cfg.experiment.distributed.enabled:
            self.trainer = pl.Trainer(
                devices=self.cfg.experiment.distributed.devices,
                accelerator=self.cfg.experiment.accelerator,
                precision=self.cfg.experiment.precision,
                logger=False,  # Skip WandB logging for prediction
                callbacks=self.callbacks,  # type: ignore
            )
        else:
            self.trainer = pl.Trainer(
                accelerator=self.cfg.experiment.accelerator,
                logger=False,
                callbacks=self.callbacks,  # type: ignore
            )

        # Extract channel types
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
        self.chan_types = aia_list + hmi_list

        # Setup DataModule
        self.data_module = SDOMLDataModule(
            hmi_path=(
                os.path.join(
                    self.cfg.data.sdoml.base_directory,
                    self.cfg.data.sdoml.sub_directory.hmi,
                )
                if self.cfg.data.sdoml.sub_directory.hmi
                else None
            ),
            aia_path=(
                os.path.join(
                    self.cfg.data.sdoml.base_directory,
                    self.cfg.data.sdoml.sub_directory.aia,
                )
                if self.cfg.data.sdoml.sub_directory.aia
                else None
            ),
            eve_path=None,
            components=self.cfg.data.sdoml.components,
            wavelengths=self.cfg.data.sdoml.wavelengths,
            ions=self.cfg.data.sdoml.ions,
            batch_size=self.cfg.model.misc.batch_size,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            persistent_workers=self.cfg.data.persistent_workers,
            multiprocessing_context=self.cfg.data.multiprocessing_context,
            train_index=self.cfg.data.train_index,
            val_index=self.cfg.data.val_index,
            test_index=self.cfg.data.test_index,
            hmi_mask_path=self.cfg.data.hmi_mask,
            num_frames=self.cfg.model.mae.num_frames,
            drop_frame_dim=self.cfg.data.drop_frame_dim,
            apply_mask=self.cfg.data.sdoml.apply_mask,
            precision=self.cfg.experiment.precision,
            normalization=self.cfg.data.sdoml.normalization,
            normalization_stat_path=self.cfg.data.normalization_stat_path,
        )
        self.data_module.setup()

        # Check for zarr output path in config or default to "embeddings.zarr"
        zarr_path = getattr(self.cfg.experiment, "zarr_path", "embeddings.zarr")

        model_hyperparams = {
            **cfg.model.mae,
            "chan_types": self.chan_types,
            "limb_mask": torch.Tensor(np.load(self.cfg.data.hmi_mask)),
            "loss_dict": self.cfg.model.loss,
            "optimizer_dict": self.cfg.model.optimizer,
            "scheduler_dict": self.cfg.model.scheduler,
            "zarr_path": zarr_path,
        }

        self.model = self.load_from_ckpt(model_hyperparams)

    def load_from_ckpt(self, model_hyperparams):
        if self.ckpt_path and os.path.exists(self.ckpt_path):
            lgr_logger.info(f"Loading weights from checkpoint: {self.ckpt_path}")
            # ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            # model = MAE(**model_hyperparams)
            # model.load_state_dict(ckpt["state_dict"], strict=False)
            model = MAE.load_from_checkpoint(self.ckpt_path, weights_only=False, map_location="cpu")
        else:
            lgr_logger.warning("No checkpoint found! Initializing model from scratch.")
            model = MAE(**model_hyperparams)
            
        return model

    def run(self):
        """Executes the prediction loop."""
        print("\nPREDICTING EMBEDDINGS...\n")
        # Predict uses test dataloader by default in Lightning unless dataloaders are specified
        self.trainer.predict(
            model=self.model,
            datamodule=self.data_module,
            return_predictions=False
        )


@hydra.main(
    config_path="../../configs/pretrain/",
    config_name="pretrain_mae_ALL_create_embeddings.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # set seed
    # torch.manual_seed(cfg.experiment.seed)
    # np.random.seed(cfg.experiment.seed)
    # random.seed(cfg.experiment.seed)
    # seed_everything(cfg.experiment.seed)

    print("\nRunning Prediction with config:")
    print(OmegaConf.to_yaml(cfg, resolve=False, sort_keys=False))
    print("\n")
    print(f"Using device: {cfg.experiment.accelerator}")

    predictor = Predictor(cfg)
    predictor.run()


if __name__ == "__main__":
    time_start = time.time()
    
    # Produce a complete stack trace on error
    os.environ["HYDRA_FULL_ERROR"] = "1"

    main()
    
    print(f"\nTotal duration: {utils.days_hours_mins_secs_str(time.time() - time_start)}")
