import os
import random

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from terratorch_surya.datasets.helio import HelioNetCDFDataset
from terratorch_surya.downstream_examples.ar_segmentation.models import HelioSpectformer2D


class SuryaReconstructionDataModule(pl.LightningDataModule):
    """DataModule for Surya channel reconstruction task.

    Args:
        config (DictConfig): Hydra configuration object.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.dataset = None

    def setup(self, stage: str | None = None):
        """Sets up the dataset.

        Args:
            stage (str, optional): The stage (train, val, test). Defaults to None.
        """
        # Mapping config to HelioNetCDFDataset arguments
        self.dataset = HelioNetCDFDataset(
            index_path=self.config.data.train_data_path,
            time_delta_input_minutes=list(self.config.data.time_delta_input_minutes),
            time_delta_target_minutes=self.config.data.time_delta_target_minutes,
            n_input_timestamps=self.config.data.n_input_timestamps,
            rollout_steps=0, # As per config rollout_steps: 0
            num_mask_aia_channels=0,
            phase="train",
            channels=list(self.config.data.channels),
            sdo_data_root_path=self.config.data.sdo_data_root_path,
            pooling=self.config.data.pooling,
            random_vert_flip=self.config.data.random_vert_flip,
        )

    def train_dataloader(self) -> DataLoader:
        """Returns the training dataloader.

        Returns:
            DataLoader: The training dataloader.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_data_workers,
            pin_memory=self.config.data.pin_memory,
            prefetch_factor=self.config.data.prefetch_factor,
            persistent_workers=self.config.data.persistent_workers,
        )


class SuryaReconstructionModel(pl.LightningModule):
    """LightningModule for fine-tuning Surya for channel reconstruction.

    Args:
        config (DictConfig): Hydra configuration object.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        in_channels = len(config.data.channels)

        model_config = {
            "model": {
                "ft_unembedding_type": "linear",
                "ft_out_chans": in_channels,
            }
        }

        self.model = HelioSpectformer2D(
            img_size=config.backbone.img_size,
            patch_size=config.backbone.patch_size,
            in_chans=in_channels,
            embed_dim=config.backbone.embed_dim,
            time_embedding=dict(config.backbone.time_embedding),
            depth=config.backbone.depth,
            n_spectral_blocks=config.backbone.n_spectral_blocks,
            num_heads=config.backbone.num_heads,
            mlp_ratio=config.backbone.mlp_ratio,
            drop_rate=config.backbone.drop_rate,
            window_size=config.backbone.window_size,
            dp_rank=config.backbone.dp_rank,
            learned_flow=config.backbone.learned_flow,
            use_latitude_in_learned_flow=config.backbone.use_latitude_in_learned_flow,
            init_weights=config.backbone.init_weights,
            dtype=torch.float32,
            checkpoint_layers=list(config.backbone.checkpoint_layers),
            rpe=config.backbone.rpe,
            finetune=config.backbone.finetune,
            config=model_config,
        )

        pretrained_path = config.backbone.path_weights
        if pretrained_path and os.path.exists(pretrained_path):
            logger.info(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            msg = self.model.load_state_dict(checkpoint, strict=False)
            logger.info(f"Checkpoint load result: {msg}")

        # Apply LoRA or handle backbone freezing
        if config.get("lora") and config.lora.get("use"):
            from peft import LoraConfig, get_peft_model

            logger.info("Applying PEFT LoRA")
            lora_cfg = config.lora.config
            peft_config = LoraConfig(
                r=lora_cfg.get("r", 8),
                lora_alpha=lora_cfg.get("lora_alpha", 8),
                target_modules=list(
                    lora_cfg.get(
                        "target_modules",
                        ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
                    )
                ),
                lora_dropout=lora_cfg.get("lora_dropout", 0.1),
                bias=lora_cfg.get("bias", "none"),
            )
            self.model = get_peft_model(self.model, peft_config)

            # Ensure the fresh unembed (decoder) layer is trainable
            for name, param in self.model.named_parameters():
                if "unembed" in name:
                    param.requires_grad = True

        elif config.backbone.get("freeze_backbone"):
            logger.info("Freezing backbone (only training decoder)")
            for param in self.model.parameters():
                param.requires_grad = False
            for name, param in self.model.named_parameters():
                if "unembed" in name:
                    param.requires_grad = True

    def mask_input(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Randomly masks one channel across all time steps.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W).

        Returns:
            tuple[torch.Tensor, int]: The masked tensor and the index of the dropped channel.
        """
        b, c, t, h, w = x.shape
        masked_x = x.clone()
        channel_idx = random.randint(0, c - 1)

        mask = torch.ones((c, 1, 1, 1), device=x.device, dtype=x.dtype)
        mask[channel_idx, ...] = 0.0

        masked_x = masked_x * mask
        return masked_x, channel_idx

    def forward(self, batch: dict) -> torch.Tensor:
        """Forward pass.

        Args:
            batch (dict): Batch dictionary from the dataset.

        Returns:
            torch.Tensor: The predicted reconstructed image.
        """
        return self.model(batch)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch (dict): Batch dictionary.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: The computed loss.
        """
        original_x = batch["ts"]

        masked_x, dropped_channel = self.mask_input(original_x)

        model_input = batch.copy()
        model_input["ts"] = masked_x

        predicted_x = self(model_input)

        # Target is the original unmasked channel at the most recent timestep
        target = original_x[:, dropped_channel, -1, :, :]
        pred = predicted_x[:, dropped_channel, :, :]

        loss = F.mse_loss(pred, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configures the optimizer.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        # Filter out frozen parameters
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        if self.config.optimizer.type == "adamw":
            return torch.optim.AdamW(
                trainable_params,
                lr=self.config.optimizer.lr,
                weight_decay=self.config.optimizer.weight_decay,
            )
        return torch.optim.Adam(trainable_params, lr=self.config.optimizer.lr)
