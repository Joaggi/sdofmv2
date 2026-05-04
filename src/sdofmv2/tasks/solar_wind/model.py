import gc
from collections import Counter

import torch
import torch.nn.functional as F
import wandb

from lightning.pytorch.utilities import grad_norm

from sdofmv2.core import BaseModule
from .focal_loss import focal_loss_multiclass
from .head_networks import ClsLinear, SimpleLinear, SkipLinearHead, TransformerHead
from .metrics import SolarWindMetrics


class SWClassifier(BaseModule):
    """Solar Wind Classifier using a backbone encoder and configurable head.

    This module wraps a pretrained backbone encoder and adds a classification head
    for solar wind prediction tasks. It supports multiple head types (linear,
    transformer, skip_linear), handles coordinate embeddings, and tracks various
    classification metrics during training, validation, and testing.

    Args:
        channels (list[str]): List of data channels to use.
        num_classes (int): Number of output classes.
        class_names (list[str]): Names of the classes for logging.
        max_position_element (int): Maximum power for positional encoding.
        backbone: The pretrained backbone model.
        freeze_encoder (bool): Whether to freeze the backbone encoder.
        plt_style: Plotting style configuration.
        head_type (str): Type of classification head ("linear", "transformer", "skip_linear").
        hidden_dim (int): Hidden dimension for the head network.
        position_size (int): Number of position coordinates.
        p_drop (float): Dropout probability.
        nhead (int): Number of attention heads for transformer head.
        embed_dim (int): Embedding dimension.
        skips (list[int]): Layer indices for skip connections.
        include_raw_coordinates (bool): Whether to include raw coordinates.
        num_hidden_layers (int): Number of hidden layers for skip_linear head.
        radial_mean (float): Mean for radial normalization.
        radial_std (float): Standard deviation for radial normalization.
        loss_dict (dict): Loss function configuration.
        optimizer_dict (dict): Optimizer configuration.
        scheduler_dict (dict): Scheduler configuration.
    """

    def __init__(
        self,
        # for finetuning
        channels=None,
        num_classes=None,
        class_names=None,
        max_position_element=None,
        backbone=None,
        freeze_encoder=True,
        plt_style=None,
        head_type="linear",
        hidden_dim=64,
        position_size=4,
        p_drop=0.1,
        nhead=8,
        embed_dim=512,
        skips=[4],
        include_raw_coordinates=False,
        num_hidden_layers=8,
        radial_mean=None,
        radial_std=None,
        # loss, optimizer, and scheduler
        loss_dict=None,
        optimizer_dict=None,
        scheduler_dict=None,
        # all else
        *args,
        **kwargs,
    ):
        super().__init__(
            optimizer_dict=optimizer_dict,
            scheduler_dict=scheduler_dict,
            *args,
            **kwargs,
        )
        self.freeze_encoder = freeze_encoder
        self.backbone = backbone
        self.mask_ratio = self.backbone.masking_ratio
        self.input_feature_dim = int(
            self.backbone.hparams.embed_dim * (550) * self.mask_ratio
        )
        self.plt_style = plt_style
        self.num_classes = num_classes
        self.class_names = class_names
        self.channels = sorted(channels)
        self.id_193 = self.channels.index("193A") if "193A" in self.channels else None
        self.max_position_element = max_position_element
        self.head_type = head_type
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.include_raw_coordinates = include_raw_coordinates
        self.num_hidden_layers = num_hidden_layers
        self.p_drop = p_drop
        self.nhead = nhead
        self.embed_dim = embed_dim
        self.position_size = position_size
        self.radial_mean = radial_mean
        self.radial_std = radial_std
        self.position_size = position_size
        self.attn_maps = []
        self.loss_dict = loss_dict

        # evaluation metrics
        self.train_metrics = SolarWindMetrics(num_classes=self.num_classes, stage="train")
        self.val_metrics = SolarWindMetrics(num_classes=self.num_classes, stage="val")
        self.test_metrics = SolarWindMetrics(num_classes=self.num_classes, stage="test")

        # Storage for validation data
        self.val_all_imgs = []
        self.val_all_timestamps = []
        self.val_all_targets = []
        self.val_all_preds = []
        self.val_all_positions = []

        self.val_barplot_targets = []
        self.val_barplot_preds = []

        # freeze or unfreeze backbone
        if freeze_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Set the whole backbone to eval mode (dropout, batchnorm, etc.)
            self.backbone.eval()
            torch.set_grad_enabled(False)
        else:
            # backbone trainable
            for param in self.backbone.parameters():
                param.requires_grad = True
            self.backbone.train()

        # Define head network
        match self.head_type:
            case "linear":
                self.head = SimpleLinear(
                    # virtual eve
                    d_output=self.num_classes,  # number of classes
                    input_feature_dim=self.input_feature_dim,
                    max_position_element=self.max_position_element,
                    hidden_dim=self.hidden_dim,
                    position_size=self.position_size,
                    p_drop=self.p_drop,
                )
            case "transformer":
                self.head = TransformerHead(
                    d_output=self.num_classes,
                    input_token_dim=self.embed_dim,
                    p_drop=self.p_drop,
                    max_position_element=self.max_position_element,
                    nhead=self.nhead,
                )

            case "skip_linear":
                self.head = SkipLinearHead(
                    # virtual eve
                    d_output=self.num_classes,  # number of classes
                    input_feature_dim=self.input_feature_dim,
                    max_position_element=self.max_position_element,
                    position_size=self.position_size,
                    hidden_dim=self.hidden_dim,
                    skips=self.skips,
                    include_raw_coordinates=self.include_raw_coordinates,
                    num_hidden_layers=self.num_hidden_layers,
                )
            # TODO: add cls_linear model
            case _:
                raise ValueError("Invalid head type!")

        # define loss
        loss_args = self.loss_dict.get(self.loss_dict.use, "focal")
        match self.loss_dict.use:
            case "cross_entropy":
                self.loss_fn = lambda inputs, targets: F.cross_entropy(
                    inputs,
                    targets,
                    weight=loss_args.class_weights,
                    reduction=loss_args.reduction,
                )

            case "focal":
                self.loss_fn = lambda inputs, targets: focal_loss_multiclass(
                    inputs,
                    targets,
                    alpha=loss_args.alpha,
                    gamma=loss_args.gamma,
                    reduction=loss_args.reduction,
                )

    def forward(self, x, position, r_distance):
        """Perform a forward pass through the classifier.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).
            position (torch.Tensor): Position coordinates of shape (B, position_size).
            r_distance (torch.Tensor): Radial distance values of shape (B,).

        Returns:
            torch.Tensor: Class logits of shape (B, num_classes).
        """
        latent, mask, ids_restore = self.backbone.forward_encoder(x, self.mask_ratio)
        # head layer
        y_hat = self.head(latent, position, r_distance)

        return y_hat

    def forward_analysis(self, x):
        """Perform analysis forward pass for reconstruction visualization.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Reconstructed images.
        """
        loss, x_hat, mask = self.backbone.autoencoder(x, self.mask_ratio)
        x_hat = self.backbone.autoencoder.unpatchify(x_hat)

        return x_hat

    def predict_step(self, batch, batch_idx):
        """Perform a prediction step for inference.

        Args:
            batch: A tuple containing (images, timestamps, position, r_distance, target).
            batch_idx: The index of the current batch.

        Returns:
            dict: A dictionary containing predictions, targets, embeddings, etc.
        """
        imgs, timestamps, position, r_distance, target = batch
        with torch.no_grad():
            y_hat = self(imgs, position, r_distance)
            latent, mask, ids_restore = self.backbone.forward_encoder(
                imgs, self.mask_ratio
            )
            loss, x_hat, mask = self.backbone.autoencoder(imgs, self.mask_ratio)
            x_hat = self.backbone.autoencoder.unpatchify(x_hat)
        preds = torch.argmax(y_hat, dim=1)

        return {
            "timestamps": timestamps,
            "imgs": imgs,
            "x_hat": x_hat,
            "predictions": preds,
            "targets": target,
            "logits": y_hat,
            "probabilities": torch.softmax(y_hat, dim=1),
            "embeddings": latent,
        }

    def training_step(self, batch, batch_idx):
        """Perform a single training step.

        Args:
            batch: A tuple containing (images, timestamps, position, r_distance, target).
            batch_idx: The index of the current batch.

        Returns:
            torch.Tensor: The training loss value.
        """
        imgs, timestamps, position, r_distance, target = batch  # [batch, c, 512, 512]
        y_hat = self(imgs, position, r_distance)

        # Calculate loss
        loss = self.loss_fn(y_hat, target)

        # Update metrics
        self.train_metrics.update(y_hat, target)

        self.log(
            "train_loss",
            loss,
            on_step=True,  # Log every step
            on_epoch=True,  # Log at end of epoch
            prog_bar=True,  # Show in progress bar
            logger=True,
            sync_dist=True,
        )
        # Log current learning rate from optimizer
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step.

        Args:
            batch: A tuple containing (images, timestamps, position, r_distance, target).
            batch_idx: The index of the current batch.

        Returns:
            torch.Tensor: The validation loss value.
        """
        imgs, timestamps, position, r_distance, target = batch
        y_hat = self(imgs, position, r_distance)

        # calculate loss
        val_loss = self.loss_fn(y_hat, target)

        # Update metrics
        self.val_metrics.update(y_hat, target)

        self.log(
            "val_loss",
            val_loss,
            on_step=False,  # Log every step
            on_epoch=True,  # Log at end of epoch
            prog_bar=True,  # Show in progress bar
            logger=True,
            sync_dist=True,
        )

        # Store data for epoch-end analysis
        preds = torch.argmax(y_hat, dim=1)
        if batch_idx < 5:  # Only store first few batches
            # Store data for epoch-end analysis
            self.val_all_imgs.append(imgs.cpu())
            self.val_all_targets.append(target.cpu())
            self.val_all_preds.append(preds.cpu())
            self.val_all_timestamps.append(timestamps.cpu())
            self.val_all_positions.append(position.cpu())

        self.val_barplot_targets.append(target.cpu())
        self.val_barplot_preds.append(preds.cpu())

        return val_loss

    def test_step(self, batch, batch_idx):
        """Perform a single test step.

        Args:
            batch: A tuple containing (images, timestamps, position, r_distance, target).
            batch_idx: The index of the current batch.

        Returns:
            dict: A dictionary containing predictions, targets, logits, and test loss.
        """
        imgs, timestamps, position, r_distance, target = batch  # [batch, c, 512, 512]
        y_hat = self(imgs, position, r_distance)
        test_loss = self.loss_fn(y_hat, target)
        preds = torch.argmax(y_hat, dim=1)

        # Update metrics
        self.test_metrics.update(y_hat, target)

        self.log(
            "test_loss",
            test_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return {
            "predictions": preds,
            "targets": target,
            "logits": y_hat,
            "probabilities": torch.softmax(y_hat, dim=1),
            "attn_maps": self.attn_maps,
            "test_loss": test_loss,
        }

    def on_validation_epoch_end(self):
        """Called at the end of the validation epoch.

        Computes and logs all validation metrics, generates WandB plots
        (confusion matrix, class distributions), and clears stored buffers.
        """
        # Compute and log all accumulated metrics
        self.log_dict(
            self.val_metrics.compute(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # WandB logging (only at epoch end to avoid slowdown)
        if len(self.val_all_preds) > 0 and len(self.val_all_targets) > 0:
            all_barplot_targets = torch.cat(self.val_barplot_targets, dim=0)
            all_barplot_preds = torch.cat(self.val_barplot_preds, dim=0)

            targets_list = all_barplot_targets.detach().cpu().tolist()
            preds_list = all_barplot_preds.detach().cpu().tolist()

            # confusion matrix
            self.logger.experiment.log(
                {
                    f"conf_mat_epoch_{self.current_epoch}": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=targets_list,
                        preds=preds_list,
                        class_names=self.class_names,
                        title=f"confusion_matrix_epoch_{self.current_epoch}",
                    )
                }
            )

            # Count occurrences
            true_counts = Counter(targets_list)
            pred_counts = Counter(preds_list)

            # All unique classes
            all_classes = sorted(set(true_counts.keys()) | set(pred_counts.keys()))

            # Create table with separate columns for true/predicted
            data = []
            for cls in all_classes:
                data.append(
                    [str(cls), true_counts.get(cls, 0), pred_counts.get(cls, 0)]
                )

            table = wandb.Table(
                data=data, columns=["class", "true_count", "pred_count"]
            )
            # You can only plot one value column at a time, so make two plots
            self.logger.experiment.log(
                {
                    f"true_class_distribution_epoch_{self.current_epoch}": wandb.plot.bar(
                        table,
                        "class",
                        "true_count",
                        title=f"True Class Distribution_epoch_{self.current_epoch}",
                    ),
                    f"predicted_class_distribution_epoch_{self.current_epoch}": wandb.plot.bar(
                        table,
                        "class",
                        "pred_count",
                        title=f"Predicted Class Distribution_epoch_{self.current_epoch}",
                    ),
                }
            )

        # Clear stored data to free memory
        self.val_all_imgs.clear()
        self.val_all_timestamps.clear()
        self.val_all_targets.clear()
        self.val_all_preds.clear()
        self.val_all_positions.clear()
        self.val_barplot_targets.clear()
        self.val_barplot_preds.clear()

        self.val_metrics.reset()

        gc.collect()
        torch.cuda.empty_cache()

    def on_train_epoch_end(self):
        """Called at the end of the training epoch.

        Performs garbage collection and clears CUDA cache to free memory.
        """
        self.train_metrics.reset()
        gc.collect()
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        """Called at the end of the test epoch.

        Computes and logs all test metrics including F1, precision, recall,
        accuracy, AUROC, MCC, and Cohen's Kappa.
        """
        # Log all test metrics
        self.log_dict(
            self.test_metrics.compute(),
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.test_metrics.reset()

    def on_before_optimizer_step(self, optimizer):
        """Called before each optimizer step to log gradient norms.

        Args:
            optimizer: The optimizer about to perform an update step.
        """
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.head, norm_type=2)
        self.log_dict(norms)
