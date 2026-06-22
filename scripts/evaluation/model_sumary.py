import os
import inspect

import torch
from omegaconf import OmegaConf
from lightning.pytorch.utilities.model_summary import ModelSummary

from sdofmv2.core import MAE

if __name__ == "__main__":
    cfg = OmegaConf.load("/home/jinsu/project/sdofmv2/configs/pretrain/pretrain_mae_HMI.yaml")

    ckpt_path = os.path.join(cfg.experiment.backbone.ckpt_dir, cfg.experiment.backbone.weight_name)
    print(f"Loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hyper_parameters = ckpt["hyper_parameters"]

    # Get MAE.__init__ argument names (excluding self)
    valid_args = set(inspect.signature(MAE.__init__).parameters.keys()) - {"self"}

    # Keep only parameters accepted by MAE
    model_hparams = {k: v for k, v in hyper_parameters.items() if k in valid_args}

    model = MAE(**model_hparams)
    # model.load_state_dict(ckpt["state_dict"], strict=False)

    summary = ModelSummary(model, max_depth=-1)
    print(summary)
