import os
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import pandas as pd
import torch
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
import sunpy.visualization.colormaps as sunpycm
from loguru import logger as lgr_logger

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from sdofmv2.core import SDOMLDataModule, MAE
from sdofmv2.utils import ALL_WAVELENGTHS
from sdofmv2.tasks.missing_data import MissingDataModel


def main(cfg: DictConfig):
    # Set seed for reproducibility
    if cfg.experiment.seed is not None:
        pl.seed_everything(cfg.experiment.seed)

    # Initialize WandB Logger
    if cfg.experiment.wandb.enable:
        lgr_logger.info("Initializing Weights & Biases Logger...")
        wandb_logger = WandbLogger(
            project=cfg.experiment.wandb.project,
            entity=cfg.experiment.wandb.entity,
            name=cfg.experiment.wandb.name,
            id=cfg.experiment.wandb.run_id if cfg.experiment.wandb.run_id else None,
            group=cfg.experiment.wandb.group,
            job_type=cfg.experiment.wandb.job_type,
            tags=list(cfg.experiment.wandb.tags),
            notes=cfg.experiment.wandb.notes,
            save_dir=cfg.experiment.wandb.output_directory,
            log_model=cfg.experiment.wandb.log_model,
            offline=cfg.experiment.wandb.offline,
        )
        lgr_logger.info(f"WandB run name: {wandb_logger.experiment.name}")
        lgr_logger.info(f"WandB run ID: {wandb_logger.experiment.id}")
        OmegaConf.resolve(cfg)
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    else:
        wandb_logger = None

    # Load DataModule
    lgr_logger.info("Loading SDOMLDataModule...")
    data_module = SDOMLDataModule(
        hmi_path=(
            os.path.join(
                cfg.data.sdoml.base_directory,
                cfg.data.sdoml.sub_directory.hmi,
            )
            if cfg.data.sdoml.sub_directory.hmi
            else None
        ),
        aia_path=(
            os.path.join(
                cfg.data.sdoml.base_directory,
                cfg.data.sdoml.sub_directory.aia,
            )
            if cfg.data.sdoml.sub_directory.aia
            else None
        ),
        eve_path=None,
        components=cfg.data.sdoml.components,
        wavelengths=cfg.data.sdoml.wavelengths,
        ions=cfg.data.sdoml.ions,
        batch_size=cfg.model.misc.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
        multiprocessing_context=cfg.data.multiprocessing_context,
        normalization=cfg.data.sdoml.normalization,
        normalization_stat_path=os.path.join(
            cfg.data.sdoml.base_directory, cfg.data.sdoml.sub_directory.cache
        ),
        train_index=cfg.data.train_index,
        val_index=cfg.data.val_index,
        test_index=cfg.data.test_index,
        hmi_mask=cfg.data.hmi_mask,
        apply_mask=cfg.data.sdoml.apply_mask,
        num_frames=cfg.data.num_frames,
        drop_frame_dim=cfg.data.drop_frame_dim,
        precision=cfg.experiment.precision,
    )
    data_module.setup()

    # Visualization setup
    wavelengths = ALL_WAVELENGTHS
    wavelengths.sort()
    wave_val_list = [int(wave[:-1]) for wave in wavelengths]
    wave_arr = np.array(wave_val_list)
    sort_ids = np.argsort(wave_arr)
    cms = [sunpycm.cmlist.get(f"sdoaia{w[:-1]}") for w in wavelengths]

    # Load pretrained backbone and zero-shot model
    lgr_logger.info("Loading pretrained MAE backbone...")
    backbone = MAE.load_from_checkpoint(
        checkpoint_path=os.path.join(
            cfg.experiment.backbone.ckpt_dir, cfg.experiment.backbone.weight_name
        ),
        map_location="cpu",
        weights_only=cfg.experiment.backbone.weights_only,
    )
    zero_shot = MAE.load_from_checkpoint(
        checkpoint_path=os.path.join(
            cfg.experiment.backbone.ckpt_dir, cfg.experiment.backbone.weight_name
        ),
        map_location="cpu",
        weights_only=cfg.experiment.backbone.weights_only,
    )

    # Create MissingDataModel
    lgr_logger.info("Creating MissingDataModel...")
    model_params = {
        "optimizer_dict": cfg.model.optimizer,
        "scheduler_dict": cfg.model.scheduler,
    }
    model = MissingDataModel(
        **model_params,
        backbone=backbone,
        freeze_encoder=True,
        normalization=cfg.data.sdoml.normalization,
        normalization_stat=data_module.normalization_stat,
        wavelengths=cfg.data.sdoml.wavelengths if cfg.data.sdoml.wavelengths else ALL_WAVELENGTHS,
        hyperparam_ignore=["backbone"],
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.experiment.ds_ckpt_dir,
        filename=cfg.experiment.checkpoint_filename,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
    )

    # Trainer
    lgr_logger.info("Initializing Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=cfg.model.misc.epochs,
        precision=cfg.experiment.precision,
        accelerator=cfg.experiment.accelerator,
        devices=list(cfg.experiment.distributed.devices),
        logger=wandb_logger,
        log_every_n_steps=cfg.experiment.log_every_n_steps,
        callbacks=[checkpoint_callback],
        gradient_clip_val=cfg.model.misc.gradient_clip_val,
        gradient_clip_algorithm=cfg.model.misc.gradient_clip_algorithm,
        accumulate_grad_batches=cfg.model.misc.accumulate_grad_batches,
        limit_train_batches=cfg.model.misc.limit_train_batches,
    )

    # Training
    ckpt_path = os.path.join(cfg.experiment.ds_ckpt_dir, cfg.experiment.checkpoint_filename)
    if not os.path.exists(ckpt_path):
        lgr_logger.info("Starting model training...")
        trainer.fit(model=model, datamodule=data_module)
    else:
        lgr_logger.info("Checkpoint found! Loading model from checkpoint...")
        model = MissingDataModel.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            map_location="cpu",
            weights_only=False,
            backbone=backbone,
        )

    # Evaluation
    lgr_logger.info("Running evaluation on test set...")
    trainer.test(model=model, datamodule=data_module)

    lgr_logger.info("Evaluation complete. Metrics logged to WandB via epoch-end hooks.")

    # Visualization
    # Example image for visualization (from test set)
    timestamps = ["2019-12-25 00:24:00"]
    img_indices = [
        data_module.test_ds.aligndata.index.get_loc(pd.to_datetime(i_time)) for i_time in timestamps
    ]
    x = data_module.test_ds[img_indices[0]][0].unsqueeze(0)
    corrupted_img = x.clone()

    # Use config for corrupted_channel_index, fallback to 5
    corrupted_channel = cfg.experiment.get("corrupted_channel_index", 5)
    corrupted_img[:, corrupted_channel, :, :, :] = 0

    # Forward pass
    loss, x_hat, mask = model(corrupted_img)
    x_hat_zero_shot, mask_zero_shot = zero_shot(corrupted_img)

    # Visualization
    x_hat_np = x_hat.detach().cpu().numpy()
    x_hat_zero_shot_np = x_hat_zero_shot.detach().cpu().numpy()
    limb_mask_np = zero_shot.limb_mask.detach().cpu().numpy()
    ch_info = [str(int(w[:-1])) for w in wavelengths]

    fig, axes = plt.subplots(nrows=4, ncols=9, figsize=(15, 7))

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for i_ch, ch_index in enumerate(sort_ids):
        channel_label = ch_info[ch_index]

        if i_ch == corrupted_channel:
            axes[0, i_ch].set_facecolor("black")
            axes[0, i_ch].set_aspect("equal")
            axes[0, i_ch].set_xlim(0, 1)
            axes[0, i_ch].set_ylim(0, 1)
            axes[0, i_ch].plot([0, 1], [0, 1], color="red", linewidth=2)
            axes[0, i_ch].plot([0, 1], [1, 0], color="red", linewidth=2)
            axes[0, i_ch].text(
                0.5,
                -0.1,
                "missing",
                color="red",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        axes[0, i_ch].imshow(corrupted_img[0, ch_index, 0, :, :], cmap=cms[ch_index])
        axes[1, i_ch].imshow(x[0, ch_index, 0, :, :], cmap=cms[ch_index])
        axes[2, i_ch].imshow(x_hat_np[0, ch_index, 0, :, :], cmap=cms[ch_index])
        axes[3, i_ch].imshow(
            x_hat_zero_shot_np[0, ch_index, 0, :, :] * limb_mask_np, cmap=cms[ch_index]
        )

        axes[0, i_ch].set_title(f"{channel_label} Å", fontsize=12, pad=10)

    plt.tight_layout()
    plt.savefig("DS_missing_data_img_result.pdf", dpi=300, bbox_inches="tight")
    lgr_logger.info("Saved visualization to DS_missing_data_img_result.pdf")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
