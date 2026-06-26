import os
import time
from typing import Any, cast

import hydra
import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import zarr
from lightning.pytorch.callbacks import BasePredictionWriter, RichProgressBar
from loguru import logger as lgr_logger
from omegaconf import DictConfig, OmegaConf

from sdofmv2 import utils
from sdofmv2.core import MAE, SDOMLDataModule
from sdofmv2.utils import ALL_COMPONENTS, ALL_WAVELENGTHS


class ZarrPredictionWriter(BasePredictionWriter):
    """Callback to gather and write embeddings to a Zarr file iteratively."""

    def __init__(self, zarr_path: str) -> None:
        """Initializes the Zarr prediction writer.

        Args:
            zarr_path: Path to the output Zarr file.
        """
        super().__init__("batch")
        self.zarr_path = zarr_path

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: dict[str, Any],
        batch_indices: list[int] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Gathers predictions from all ranks and writes to Zarr on rank 0."""
        embeddings_tensor = prediction["embeddings"]
        timestamps_np = prediction["timestamps"]

        if trainer.world_size > 1:
            embeddings_gathered = cast(torch.Tensor, pl_module.all_gather(embeddings_tensor))
            if embeddings_gathered.dim() > embeddings_tensor.dim():
                embeddings_gathered = embeddings_gathered.view(-1, *embeddings_gathered.shape[2:])
        else:
            embeddings_gathered = cast(torch.Tensor, embeddings_tensor)

        embeddings_np = embeddings_gathered.detach().cpu().numpy()

        if trainer.world_size > 1:
            gathered_ts: list[Any] = [None for _ in range(trainer.world_size)]
            dist.all_gather_object(gathered_ts, timestamps_np)
            timestamps_gathered = np.concatenate(gathered_ts, axis=0)
        else:
            timestamps_gathered = timestamps_np

        if trainer.global_rank == 0:
            self._write_to_zarr(embeddings_np, timestamps_gathered)

    def _write_to_zarr(self, embeddings_np: np.ndarray, timestamps_np: np.ndarray) -> None:
        """Appends the gathered arrays to the Zarr archive."""
        root = zarr.open_group(self.zarr_path, mode="a")

        if "embeddings" not in root:
            root.create_dataset(
                "embeddings",
                data=embeddings_np,
                shape=embeddings_np.shape,
                chunks=(1, *embeddings_np.shape[1:]),
                dtype=embeddings_np.dtype,
                maxshape=(None, *embeddings_np.shape[1:]),
            )
            root.create_dataset(
                "timestamps",
                data=timestamps_np,
                shape=timestamps_np.shape,
                chunks=(1,),
                dtype=timestamps_np.dtype,
                maxshape=(None,),
            )
        else:
            root["embeddings"].append(embeddings_np)  # type: ignore
            root["timestamps"].append(timestamps_np)  # type: ignore

class Predictor:
    """Coordinates the prediction workflow for Masked Autoencoder (MAE) models.

    This class sets up the data module, model, and trainer. It manages
    checkpoint loading and calls the prediction loop to save embeddings.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initializes the Predictor.

        Args:
            cfg (DictConfig): The configuration dict from Hydra.
        """
        self.cfg = cfg
        self.ckpt_path = self._get_ckpt_path()
        zarr_path = getattr(self.cfg.experiment, "zarr_path", "embeddings.zarr")
        self.callbacks = [RichProgressBar(), ZarrPredictionWriter(zarr_path)]
        self.trainer = self._setup_trainer()
        self.chan_types = self._extract_channel_types()
        self.data_module = self._setup_datamodule()
        self.model = self._setup_model()

    def _get_ckpt_path(self) -> str | None:
        """Retrieves the checkpoint path from the configuration.

        Returns:
            str | None: The path to the checkpoint if specified, otherwise None.
        """
        if self.cfg.experiment.backbone.weight_name is not None:
            return os.path.join(
                self.cfg.experiment.backbone.ckpt_dir,
                self.cfg.experiment.backbone.weight_name,
            )
        return None

    def _setup_trainer(self) -> pl.Trainer:
        """Instantiates the PyTorch Lightning Trainer.

        During prediction we can typically use fewer devices or rely on a simpler setup.

        Returns:
            pl.Trainer: The instantiated trainer.
        """
        if self.cfg.experiment.distributed.enabled:
            return pl.Trainer(
                devices=self.cfg.experiment.distributed.devices,
                accelerator=self.cfg.experiment.accelerator,
                precision=self.cfg.experiment.precision,
                logger=False,  # Skip WandB logging for prediction
                callbacks=self.callbacks,  # type: ignore
            )
        return pl.Trainer(
            accelerator=self.cfg.experiment.accelerator,
            logger=False,
            callbacks=self.callbacks,  # type: ignore
        )

    def _extract_channel_types(self) -> list[str]:
        """Combines and sorts the AIA and HMI channel types based on configuration.

        Returns:
            list[str]: The sorted list of combined channel types.
        """
        aia_list = (
            ALL_WAVELENGTHS
            if self.cfg.data.sdoml.sub_directory.aia and self.cfg.data.sdoml.wavelengths is None
            else self.cfg.data.sdoml.wavelengths or []
        )
        hmi_list = (
            ALL_COMPONENTS
            if self.cfg.data.sdoml.sub_directory.hmi and self.cfg.data.sdoml.components is None
            else self.cfg.data.sdoml.components or []
        )
        aia_list.sort()
        hmi_list.sort()
        return aia_list + hmi_list

    def _setup_datamodule(self) -> SDOMLDataModule:
        """Configures and initializes the SDOMLDataModule.

        Returns:
            SDOMLDataModule: The initialized and setup data module.
        """
        data_module = SDOMLDataModule(
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
        data_module.setup()
        return data_module

    def _setup_model(self) -> MAE:
        """Prepares the hyperparameter dictionary and loads the MAE model.

        Returns:
            MAE: The initialized MAE model.
        """
        model_hyperparams = {
            **self.cfg.model.mae,
            "chan_types": self.chan_types,
            "limb_mask": torch.Tensor(np.load(self.cfg.data.hmi_mask)),
            "loss_dict": self.cfg.model.loss,
            "optimizer_dict": self.cfg.model.optimizer,
            "scheduler_dict": self.cfg.model.scheduler,
        }

        return self.load_from_ckpt(model_hyperparams)

    def load_from_ckpt(self, model_hyperparams: dict[str, Any]) -> MAE:
        """Loads the MAE model from a checkpoint with a dual-loading fallback.

        Args:
            model_hyperparams (dict[str, Any]): The hyperparameters for the model.

        Raises:
            FileNotFoundError: If the checkpoint path is missing or invalid.

        Returns:
            MAE: The loaded MAE model.
        """
        if not self.ckpt_path or not os.path.exists(self.ckpt_path):
            lgr_logger.error("Failed to find checkpoint at: {}", self.ckpt_path)
            raise FileNotFoundError(f"Checkpoint not found at: {self.ckpt_path}")

        lgr_logger.info("Loading weights from checkpoint: {}", self.ckpt_path)
        try:
            return MAE.load_from_checkpoint(self.ckpt_path, weights_only=False, map_location="cpu")
        except Exception as e:
            lgr_logger.warning("load_from_checkpoint failed ({}), falling back to manual load", e)
            ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            model = MAE(**model_hyperparams)
            model.load_state_dict(ckpt["state_dict"], strict=False)
            return model

    def run(self) -> None:
        """Executes the prediction loop."""
        lgr_logger.info("PREDICTING EMBEDDINGS...")
        self.trainer.predict(
            model=self.model, datamodule=self.data_module, return_predictions=False
        )


@hydra.main(
    config_path="../../configs/pretrain/",
    config_name="pretrain_mae_ALL.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # torch.manual_seed(cfg.experiment.seed)
    # np.random.seed(cfg.experiment.seed)
    # random.seed(cfg.experiment.seed)
    # seed_everything(cfg.experiment.seed)

    lgr_logger.info(
        "Running Prediction with config:\n{}",
        OmegaConf.to_yaml(cfg, resolve=False, sort_keys=False),
    )
    lgr_logger.info("Using device: {}", cfg.experiment.accelerator)

    predictor = Predictor(cfg)
    predictor.run()


if __name__ == "__main__":
    mp.set_sharing_strategy("file_system")
    time_start = time.time()

    os.environ["HYDRA_FULL_ERROR"] = "1"

    main()

    lgr_logger.info("Total duration: {}", utils.days_hours_mins_secs_str(time.time() - time_start))
