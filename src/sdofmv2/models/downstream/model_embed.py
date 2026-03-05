import gc
import os
import sys
import torch
import torch.nn.functional as F
from torchmetrics import (
    F1Score, Precision, Recall, AUROC, AveragePrecision,
    MatthewsCorrCoef, CohenKappa, Accuracy
)
from lightning.pytorch.utilities import grad_norm
from torchvision.utils import make_grid

import wandb
from loguru import logger
# import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

from sdofm import BaseModule
# from loguru import logger

from spp.visualization import (
    find_images_labels_embed,
    plot_images_grid,
    plot_disk_distribution,
    plot_ecliptic, 
    plot_tsne, 
    plot_sdoml)
from .head import SimpleLinear, TransformerHead, SkipLinearHead
from .loss import focal_loss_multiclass


class SWClassifierEmbed(BaseModule):
    def __init__(
        self,
        # Backbone parameters
        mask_ratio=0.75,
        # Head parameters
        input_feature_dim=None,
        lr=None,
        weight_decay=None,
        # for finetuning
        wavelengths=None,
        num_classes=None,
        class_names=None,
        max_position_element=None,
        optimiser=None,
        lr_scheduler=None,
        patience=3,
        factor=0.5,
        gamma=None,
        class_weights=None,
        plt_style=None,
        head_type="linear",
        hidden_dim=64,
        position_size=4,
        p_drop=0.1,
        nhead=8,
        embed_dim=512,
        plot_module=None,
        loss_type="cross_entropy",
        alpha=[0.25,0.25,0.25,0.25],
        loss_gamma=2,
        reduction="mean",
        skips=[4],
        include_raw_coordinates=False,
        num_hidden_layers=8,
        radial_mean=None,
        radial_std=None,
        number_of_frames=1,
        # all else
        *args,
        **kwargs,
    ):
        super().__init__(
            optimiser=optimiser,
            lr=lr,
            weight_decay=weight_decay,
            num_classes=num_classes,
            *args,
            **kwargs
        )
        self.lr_scheduler = lr_scheduler
        self.patience = patience
        self.factor = factor
        self.gamma = gamma
        self.class_weights = class_weights
        self.plt_style = plt_style
        self.num_classes = num_classes
        self.class_names = class_names
        self.wavelengths = sorted(wavelengths)
        self.id_193 = self.wavelengths.index("193A")
        self.max_position_element = max_position_element
        self.head_type = head_type
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.include_raw_coordinates = include_raw_coordinates
        self.num_hidden_layers = num_hidden_layers
        self.p_drop = p_drop
        self.nhead = nhead
        self.mask_ratio = mask_ratio
        self.position_size = position_size
        self.loss_type = loss_type
        self.alpha = torch.tensor(alpha)
        self.loss_gamma = loss_gamma
        self.reduction = reduction
        self.plot_module = plot_module
        self.radial_mean = radial_mean
        self.radial_std = radial_std
        self.number_of_frames = number_of_frames

        # evaluation metrics for training
        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes, average="macro")
        self.train_f1 = F1Score(task='multiclass', num_classes=self.num_classes, average="macro")
        self.val_f1 = F1Score(task='multiclass', num_classes=self.num_classes, average="macro")
        self.test_f1 = F1Score(task='multiclass', num_classes=self.num_classes, average="macro")

        # evaluation metrics for validation
        self.val_precision = Precision(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes, average="macro")                     # Single value
        self.val_auroc = AUROC(task="multiclass", num_classes=self.num_classes, average="macro")       # Single value
        self.val_mcc = MatthewsCorrCoef(task="multiclass", num_classes=self.num_classes)                     # Single value
        self.val_kappa = CohenKappa(task="multiclass", num_classes=self.num_classes)                         # Single value

        # evaluation metrics for validation
        self.test_precision = Precision(task="multiclass", num_classes=self.num_classes, average="macro")
        self.test_recall = Recall(task="multiclass", num_classes=self.num_classes, average="macro")
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes, average="macro")                     # Single value
        self.test_auroc = AUROC(task="multiclass", num_classes=self.num_classes, average="macro")       # Single value
        self.test_mcc = MatthewsCorrCoef(task="multiclass", num_classes=self.num_classes)                     # Single value
        self.test_kappa = CohenKappa(task="multiclass", num_classes=self.num_classes) 
        
        # Storage for validation data
        # self.val_all_imgs = []
        self.val_all_timestamps = []
        self.val_all_targets = []
        self.val_all_preds = []
        self.val_all_positions = []

        self.val_barplot_targets = []
        self.val_barplot_preds = []

        match self.head_type:
            case "linear":
                self.head = SimpleLinearEmbed(
                    # virtual eve
                    d_output=self.num_classes,  # number of classes
                    input_feature_dim=input_feature_dim,
                    max_position_element=self.max_position_element,
                    p_drop=self.p_drop,
                    position_size=self.position_size,
                    hidden_dim=self.hidden_dim,
                )
            case "transformer":
                self.head = TransformerHeadEmbed(
                    d_output=self.num_classes,
                    input_token_dim=embed_dim,
                    p_drop=self.p_drop,
                    max_position_element=max_position_element,
                    nhead=nhead,
                )
            case "skip_linear":
                self.head = SkipLinearHead(
                    # virtual eve
                    d_output=self.num_classes,  # number of classes
                    input_feature_dim=input_feature_dim,
                    max_position_element=self.max_position_element,
                    position_size=self.position_size,
                    hidden_dim=self.hidden_dim,
                    skips=self.skips,
                    include_raw_coordinates=self.include_raw_coordinates,
                    num_hidden_layers=self.num_hidden_layers,
                    number_of_frames=self.number_of_frames,
                )
            case _:
                raise ValueError("Invalid head type!")

        # define loss
        match self.loss_type:
            case "cross_entropy":
                self.loss_fn = lambda inputs, targets: F.cross_entropy(
                    inputs, 
                    targets,
                    weight=self.class_weights,
                    reduction=self.reduction
                    )
            
            case "focal":
                self.loss_fn = lambda inputs, targets: focal_loss_multiclass(
                    inputs, 
                    targets,
                    alpha=self.alpha,
                    gamma=self.loss_gamma,
                    reduction=self.reduction
                    )

    def forward(self, x, position, r_distance):
        y_hat = self.head(x, position, r_distance)
        return y_hat
    
    def predict_step(self, batch, batch_idx):
        x, timestamps, position, r_distance, label, unnorm_pos = batch
        y_hat = self(x, position, r_distance)
        preds = torch.argmax(y_hat, dim=1)
        
        return {
            'predictions': preds,
            'logits': y_hat,
            'probabilities': torch.softmax(y_hat, dim=1)
        }

    def training_step(self, batch, batch_idx):
        x, timestamps, position, r_distance, target, unnorm_pos = batch  # [batch, 257, 512]
        y_hat = self(x, position,  r_distance)

        # Calculate loss
        loss = self.loss_fn(y_hat, target)

        # if torch.isnan(x).any():
        #     logger.warning("input has Nans!!")
        # if torch.isnan(y_hat).any():
        #     logger.warning("outputs have Nans!!")
        # if torch.isnan(loss).any():
        #     logger.warning("loss has Nans!!")

        # Update metrics
        preds = torch.argmax(y_hat, dim=1)
        self.train_f1.update(preds, target)
        self.train_acc.update(preds, target)

        self.log(
            'train_loss',
            loss,
            on_step=True,     # Log every step
            on_epoch=True,    # Log at end of epoch
            prog_bar=True,    # Show in progress bar
            logger=True,
            sync_dist=True
        )
        # Log current learning rate from optimizer
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, logger=False, prog_bar=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, timestamps, position, r_distance, target,  unnorm_pos = batch
        y_hat = self(x, position, r_distance)

        # calculate loss
        val_loss = self.loss_fn(y_hat, target)

        # Update metrics
        preds = torch.argmax(y_hat, dim=1)
        self.val_f1.update(preds, target)
        self.val_precision.update(preds, target)
        self.val_recall.update(preds, target)
        self.val_acc.update(preds, target)
        self.val_auroc.update(y_hat, target)
        self.val_mcc.update(preds, target)
        self.val_kappa.update(preds, target)

        self.log(
            'val_loss',
            val_loss,
            on_step=False,     # Log every step
            on_epoch=True,    # Log at end of epoch
            prog_bar=True,    # Show in progress bar
            logger=True,
            sync_dist=True
        )

        # Store data for epoch-end analysis
        if batch_idx < 5:  # Only store first few batches
            # Store data for epoch-end analysis
            self.val_all_targets.append(target.cpu())
            self.val_all_preds.append(preds.cpu())
            self.val_all_timestamps.append(timestamps.cpu())
            self.val_all_positions.append(unnorm_pos.cpu())
        
        self.val_barplot_targets.append(target.cpu())
        self.val_barplot_preds.append(preds.cpu())
    
        return val_loss

    def test_step(self, batch, batch_idx):
        x, timestamps, position, r_distance, target = batch
        y_hat = self(x, position, r_distance)
        test_loss = self.loss_fn(y_hat, target)
        preds = torch.argmax(y_hat, dim=1)
        
        # Update the F1 metric with predictions and targets
        self.test_f1.update(preds, target)
        self.test_precision.update(preds, target)
        self.test_recall.update(preds, target)
        self.test_acc.update(preds, target)
        self.test_auroc.update(y_hat, target)
        self.test_mcc.update(preds, target)
        self.test_kappa.update(preds, target)

        self.log('test_loss', test_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_f1', self.test_f1, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {'test_loss': test_loss, 'preds': preds, 'targets': target}

    def on_validation_epoch_end(self):
        # Compute and log all accumulated metrics
        self.log('val_f1', self.val_f1.compute(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_precision', self.val_precision.compute(), on_epoch=True, logger=True, sync_dist=True)
        self.log('val_recall', self.val_recall.compute(), on_epoch=True, logger=True, sync_dist=True)
        self.log('val_acc', self.val_acc.compute(), on_epoch=True, logger=True, sync_dist=True)
        self.log('val_auroc', self.val_auroc.compute(), on_epoch=True, logger=True, sync_dist=True)
        self.log('val_mcc', self.val_mcc.compute(), on_epoch=True, logger=True, sync_dist=True)
        self.log('val_kappa', self.val_kappa.compute(), on_epoch=True, logger=True, sync_dist=True)

        # WandB logging (only at epoch end to avoid slowdown)
        if len(self.val_barplot_targets) > 0:
            # Concatenate all stored data
            # all_imgs = torch.cat(self.val_all_imgs, dim=0)
            all_timestamps = torch.cat(self.val_all_timestamps, dim=0)
            all_targets = torch.cat(self.val_all_targets, dim=0)
            all_preds = torch.cat(self.val_all_preds, dim=0)
            all_positions = torch.cat(self.val_all_positions, dim=0)
            all_barplot_targets = torch.cat(self.val_barplot_targets, dim=0)
            all_barplot_preds = torch.cat(self.val_barplot_preds, dim=0)

            targets_list = all_barplot_targets.detach().cpu().tolist()
            preds_list = all_barplot_preds.detach().cpu().tolist()

            # confusion matrix
            self.logger.experiment.log(
                {
                    f"conf_mat_epoch_{self.current_epoch}":
                    wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=targets_list,
                        preds=preds_list,
                        class_names=self.class_names,
                        title=f"confusion_matrix_epoch_{self.current_epoch}"
                    )
                })

            # Count occurrences
            true_counts = Counter(targets_list)
            pred_counts = Counter(preds_list)

            # All unique classes
            all_classes = sorted(set(true_counts.keys()) | set(pred_counts.keys()))

            # Create table with separate columns for true/predicted
            data = []
            for cls in all_classes:
                data.append([
                    str(cls),
                    true_counts.get(cls, 0),
                    pred_counts.get(cls, 0)
                ])

            table = wandb.Table(data=data, columns=["class", "true_count", "pred_count"])
            # You can only plot one value column at a time, so make two plots
            self.logger.experiment.log({
                f"true_class_distribution_epoch_{self.current_epoch}": wandb.plot.bar(
                    table, "class", "true_count", title=f"True Class Distribution_epoch_{self.current_epoch}"
                ),
                f"predicted_class_distribution_epoch_{self.current_epoch}": wandb.plot.bar(
                    table, "class", "pred_count", title=f"Predicted Class Distribution_epoch_{self.current_epoch}"
                )
            })

            # If we have a dataloader for image data, use it to plot images
            # if (len(self.val_all_imgs) > 0)&(self.plot_module is not None):
            #     try:
            #         correct_data = {key: [] for key in ["imgs", "timestamps", "targets", "preds", "position"]}
            #         incorrect_data = {key: [] for key in ["imgs", "timestamps", "targets", "preds", "position"]}

            #         for class_id in range(4):
            #             result_correct_dict, result_incorrect_dict = find_images_labels_embed(
            #                 all_imgs, 
            #                 all_timestamps, 
            #                 all_targets, 
            #                 all_preds, 
            #                 all_positions, 
            #                 class_id, 
            #                 self.id_193
            #             )

            #             cor_fig, cor_ax = plot_sdoml(self.plot_module, times = pd.to_datetime(result_correct_dict['timestamps']),
            #                                  n_samples = 4, wsa_footpoint = True, title = f"Class {class_id} Correct")
            #             incor_fig, incor_ax = plot_sdoml(self.plot_module, times = pd.to_datetime(result_incorrect_dict['timestamps']),
            #                                  n_samples = 4, wsa_footpoint = True, title = f"Class {class_id} Inorrect")

            #         self.logger.experiment.log({f"correct_predictions_epoch_{self.current_epoch}": wandb.Image(cor_fig)})
            #         self.logger.experiment.log({f"incorrect_predictions_epoch_{self.current_epoch}": wandb.Image(incor_fig)})
            #         plt.close(cor_fig)
            #         plt.close(incor_fig)

            #         #Plot an image of the sun with source regions highlighted for first image in validation dataset
            #         # disk_fig = plot_disk_distribution(all_imgs[0], self, all_timestamps[0])
            #         # wandb.log({f"validation_disk_epoch_{self.current_epoch}": wandb.Image(disk_fig)})

            #         #Plot an image of the ecliptic with solar wind type colored for first image in validation dataset
            #         # ecliptic_fig = plot_ecliptic(all_imgs[0], self, all_timestamps[0])
            #         # wandb.log({f"validation_ecliptic_epoch_{self.current_epoch}": wandb.Image(ecliptic_fig)})
                    
            #     except Exception as e:
            #         print(f"Plotting error: {e}")
            #     except Exception as e:
            #         print(f"Plotting error: {e}")

        # Clear stored data to free memory
        # self.val_all_imgs.clear()
        self.val_all_timestamps.clear()
        self.val_all_targets.clear()
        self.val_all_preds.clear()
        self.val_all_positions.clear()

        for metric in [self.val_f1,
                       self.val_precision,
                       self.val_recall,
                       self.val_acc,
                       self.val_auroc,
                       self.val_mcc,
                       self.val_kappa]:
            metric.reset()

    def on_train_epoch_end(self):
        gc.collect()
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        # Log all test metrics
        self.log('test_f1', self.test_f1.compute(), on_epoch=True, logger=True, sync_dist=True)
        self.log('test_precision', self.test_precision.compute(), on_epoch=True, logger=True, sync_dist=True)
        self.log('test_recall', self.test_recall.compute(), on_epoch=True, logger=True, sync_dist=True)
        self.log('test_acc', self.test_acc.compute(), on_epoch=True, logger=True, sync_dist=True)
        self.log('test_auroc', self.test_auroc.compute(), on_epoch=True, logger=True, sync_dist=True)
        self.log('test_mcc', self.test_mcc.compute(), on_epoch=True, logger=True, sync_dist=True)
        self.log('test_kappa', self.test_kappa.compute(), on_epoch=True, logger=True, sync_dist=True)

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.head, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        # Select optimizer
        match self.optimiser:
            case "adam":
                optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay
                )

            case "sgd":
                optimizer = torch.optim.SGD(
                    self.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )

            case "adamw":
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )

            case _:
                raise ValueError(f"Unsupported optimizer: {self.optimiser}")

        # Select LR scheduler
        scheduler_config = None
        
        match self.lr_scheduler:
            case "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.trainer.max_epochs,
                )
                scheduler_config = {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
                
            case "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=self.factor, patience=self.patience,
                )
                scheduler_config = {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',  # Add this required parameter!
                    'interval': 'epoch',
                    'frequency': 1
                }
                
            case "exp":
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=self.gamma,
                )
                scheduler_config = {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
                
            case _:
                raise ValueError(f"Unsupported scheduler: {self.lr_scheduler}")

        # Return config based on whether a scheduler is used
        if scheduler_config is not None:
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler_config
            }
        else:
            return optimizer