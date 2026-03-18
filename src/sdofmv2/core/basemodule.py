from loguru import logger as lgr_logger
import lightning.pytorch as pl
import torch
from transformers import get_cosine_schedule_with_warmup


class BaseModule(pl.LightningModule):
    """A foundational PyTorch Lightning module for standardized training.

    This base class handles the boilerplate configuration for optimizers
    and learning rate schedulers. Other models in the pipeline should inherit
    from this class and implement their specific `training_step` and
    `validation_step` logic.

    Args:
        optimizer_dict (dict): Configuration dictionary for the optimizer.
            Expected keys include "use" (e.g., "adamw", "sgd", "adam"),
            "learning_rate", and "weight_decay".
        scheduler_dict (dict): Configuration dictionary for the learning rate
            scheduler. Expected keys include "use" (e.g., "cosine", "cosine_warmup",
            "plateau", "exp"), "monitor" (metric to track), and any scheduler-specific
            hyperparameters.
        hyperparam_ignore (list[str], optional): List of parameter names to
            exclude from Lightning's automatic hyperparameter saving. Defaults to [].
        *args: Variable length argument list passed to `pl.LightningModule`.
        **kwargs: Arbitrary keyword arguments passed to `pl.LightningModule`.
    """

    def __init__(
        self,
        optimizer_dict,
        scheduler_dict,
        hyperparam_ignore=[],
        # pass to pl.LightningModule
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.optimizer_dict = optimizer_dict
        self.scheduler_dict = scheduler_dict
        self.save_hyperparameters(ignore=hyperparam_ignore)

    def training_step(self, batch, batch_idx):
        """Perform a single training step.

        Args:
            batch: The training batch data.
            batch_idx: The index of the current batch.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step.

        Args:
            batch: The validation batch data.
            batch_idx: The index of the current batch.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.

        Returns:
            Union[torch.optim.Optimizer, Dict]: Either a single optimizer or a dict
                containing optimizer and lr_scheduler configuration.
        """
        opt_type = self.optimizer_dict.get("use", "adamw")
        lr = self.optimizer_dict.get("learning_rate", 1e-4)
        weight_decay = self.optimizer_dict.get("weight_decay", 0.01)

        lgr_logger.debug(f"Initial/Peak LR: {lr}")
        lgr_logger.debug(f"Weight decay: {weight_decay}")
        match opt_type:
            case "adam":
                optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                )
            case "sgd":
                optimizer = torch.optim.SGD(
                    self.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                )
            case "adamw":
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                )
            case _:
                raise NameError(f"Unknown optimizer {optimizer}")

        sched_use = self.scheduler_dict.get("use", None)
        monitor = self.scheduler_dict.get("monitor", "val_loss")
        hyper_params = self.scheduler_dict.get(sched_use, {})

        # Create scheduler
        if sched_use == "cosine":
            # Cosine annealing from pytorch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **hyper_params
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        elif sched_use == "cosine_warmup":
            warmup_ratio = hyper_params.get("warmup_ratio", 0.1)

            # SAFEGUARD: Calculate steps
            total_steps = self.trainer.estimated_stepping_batches
            lgr_logger.debug(f"Scheduler initialized with TOTAL_STEPS = {total_steps}")
            # Check for edge cases where Lightning returns infinity or valid steps are unknown
            if isinstance(total_steps, (float, int)) and (
                total_steps == float("inf") or total_steps == 0
            ):
                lgr_logger.warning(
                    "Warning: Could not calculate total steps automatically."
                )
                total_steps = hyper_params.get("total_steps", 3000)

            num_warmup_steps = int(total_steps * warmup_ratio)

            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        elif sched_use == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **hyper_params
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor,  # Metric to monitor
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        elif sched_use == "exp":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, **hyper_params
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        else:
            # No scheduler
            return optimizer
