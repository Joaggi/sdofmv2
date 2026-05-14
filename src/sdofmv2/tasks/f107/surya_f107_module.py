import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from terratorch_surya.downstream_examples.ar_segmentation.models import HelioSpectformer1D
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score
from omegaconf import DictConfig

class SuryaF107Model(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Configure model
        model_config = {
            "model": {
                "global_average_pooling": config.model.get("global_average_pooling", False),
                "global_max_pooling": config.model.get("global_max_pooling", False),
                "attention_pooling": config.model.get("attention_pooling", False),
                "transformer_pooling": config.model.get("transformer_pooling", True),
                "dropout": config.model.get("dropout", 0.1),
                "penultimate_linear_layer": config.model.get("penultimate_linear_layer", True),
            }
        }
        
        self.model = HelioSpectformer1D(
            img_size=config.backbone.img_size,
            patch_size=config.backbone.patch_size,
            in_chans=len(config.data.channels),
            embed_dim=config.backbone.embed_dim,
            time_embedding=dict(config.backbone.time_embedding),
            depth=config.backbone.depth,
            n_spectral_blocks=config.backbone.n_spectral_blocks,
            num_heads=config.backbone.num_heads,
            mlp_ratio=config.backbone.mlp_ratio,
            drop_rate=config.backbone.drop_rate,
            window_size=config.backbone.window_size,
            dp_rank=config.backbone.dp_rank,
            num_outputs=1,
            config=model_config
        )
        
        # Load weights
        if config.backbone.path_weights:
            checkpoint = torch.load(config.backbone.path_weights, map_location="cpu", weights_only=True)
            self.model.load_state_dict(checkpoint, strict=False)
            
        # Metrics
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.r2 = R2Score()
        
    def forward(self, batch):
        return self.model(batch)
        
    def training_step(self, batch, batch_idx):
        data_dict, target = batch
        pred = self(data_dict).squeeze()
        loss = F.mse_loss(pred, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        data_dict, target = batch
        pred = self(data_dict).squeeze()
        loss = F.mse_loss(pred, target)
        self.mse(pred, target)
        self.mae(pred, target)
        self.r2(pred, target)
        self.log("val_loss", loss, prog_bar=True)
        return loss
        
    def on_validation_epoch_end(self):
        self.log("val_mse", self.mse.compute())
        self.log("val_mae", self.mae.compute())
        self.log("val_r2", self.r2.compute())
        self.mse.reset()
        self.mae.reset()
        self.r2.reset()
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.optimizer.lr, weight_decay=self.config.optimizer.weight_decay)
