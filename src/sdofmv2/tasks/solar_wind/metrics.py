import torch
from torch import nn
from torchmetrics import (
    Accuracy,
    AUROC,
    CohenKappa,
    F1Score,
    MatthewsCorrCoef,
    MetricCollection,
    Precision,
    Recall,
)


class SolarWindMetrics(nn.Module):
    """Metric collection for Solar Wind classification task.

    This class wraps multiple torchmetrics into a single collection for training,
    validation, and testing stages. It handles the update logic for different
    metric types (discrete predictions vs. continuous logits).

    Args:
        num_classes (int): Number of classes in the classification task.
        stage (str): The stage of the metrics ('train', 'val', or 'test').
    """

    def __init__(self, num_classes: int, stage: str):
        super().__init__()
        self.stage = stage
        self.num_classes = num_classes

        metrics_dict = {
            "f1": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
        }

        if stage == "train":
            metrics_dict.update({
                "acc": Accuracy(task="multiclass", num_classes=num_classes, average="macro"),
            })
        else:
            # Validation and Testing metrics
            metrics_dict.update({
                "acc": Accuracy(
                    task="multiclass", 
                    num_classes=num_classes, 
                    average="micro" if stage == "val" else "macro"
                ),
                "precision": Precision(task="multiclass", num_classes=num_classes, average="macro"),
                "recall": Recall(task="multiclass", num_classes=num_classes, average="macro"),
                "auroc": AUROC(task="multiclass", num_classes=num_classes, average="macro"),
                "mcc": MatthewsCorrCoef(task="multiclass", num_classes=num_classes),
                "kappa": CohenKappa(task="multiclass", num_classes=num_classes),
            })

        self.metrics = MetricCollection(metrics_dict, prefix=f"{stage}_")

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """Updates all metrics in the collection.

        Args:
            logits (torch.Tensor): Model output logits of shape (B, num_classes).
            targets (torch.Tensor): Ground truth labels of shape (B,).
        """
        preds = torch.argmax(logits, dim=1)
        
        # We need to iterate and check types for auroc which needs probabilities or logits
        # but MetricCollection handles multi-input update if we pass the same args.
        # However, some metrics in our list expect preds (discrete) and some expect logits.
        
        for name, metric in self.metrics.items():
            if "auroc" in name:
                metric.update(logits, targets)
            else:
                metric.update(preds, targets)

    def compute(self):
        """Computes all metrics in the collection.

        Returns:
            dict: A dictionary of computed metric values.
        """
        return self.metrics.compute()

    def reset(self):
        """Resets all metrics in the collection."""
        self.metrics.reset()
