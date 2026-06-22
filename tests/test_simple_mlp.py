
import torch
from omegaconf import OmegaConf
from sdofmv2.tasks.f107.surya_f107_module import SuryaF107Model

def test_model_forward():
    # Mock config
    config = OmegaConf.create({
        "backbone": {
            "img_size": 128,
            "patch_size": 16,
            "embed_dim": 128,
            "time_embedding": {"type": "linear", "time_dim": 2},
            "depth": 2,
            "n_spectral_blocks": 1,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "drop_rate": 0.0,
            "window_size": 2,
            "dp_rank": 2,
            "path_weights": None
        },
        "data": {
            "channels": ['aia94', 'aia131']
        },
        "model": {
            "mlp_hidden_layer_dims": [64, 32],
            "dropout": 0.1
        },
        "optimizer": {
            "lr": 1e-4,
            "weight_decay": 0.01
        },
        "etc": {
            "output_dir": "./outputs",
            "test_results_filename": "results.csv"
        }
    })

    model = SuryaF107Model(config, max_norm=1.0)
    
    # Mock batch
    batch = {
        "ts": torch.randn(1, 2, 1, 128, 128),
        "time_delta_input": torch.randn(1, 1)
    }

    # Forward pass
    output = model(batch)
    print(f"Output shape: {output.shape}")
    assert output.shape == (1, 1)
    print("Forward pass successful!")

if __name__ == "__main__":
    test_model_forward()
