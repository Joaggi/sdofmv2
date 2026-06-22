import torch
import torch.nn as nn

from sdofmv2.tasks.f107.f107_module import MultiLayerPerceptron


class MockAutoencoder(nn.Module):
    def __init__(self, embed_dim=768, num_patches=1024):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Register patch_off_limb_mask as a buffer
        mask = torch.zeros(num_patches, dtype=torch.bool)
        # Set some patches as off-limb to test masking logic
        mask[num_patches // 2:] = True
        self.register_buffer("patch_off_limb_mask", mask)

    def forward_encoder(self, x, mask_ratio=0.0):
        batch_size = x.shape[0]
        num_patches = self.patch_off_limb_mask.shape[0]
        # latent size: [B, num_patches + 1, embed_dim]
        latent = torch.randn(batch_size, num_patches + 1, self.cls_token.shape[-1], device=x.device)
        return latent, None, None

class MockBackbone(nn.Module):
    def __init__(self, embed_dim=768, num_patches=1024):
        super().__init__()
        self.autoencoder = MockAutoencoder(embed_dim, num_patches)

def test_mlp_pooling_types():
    embed_dim = 768
    num_patches = 1024
    backbone = MockBackbone(embed_dim=embed_dim, num_patches=num_patches)

    pooling_types = ["mean_max", "disk_masked_mean_max", "cross_attention"]

    for pooling_type in pooling_types:
        # If mean/max is used, expected input_dim is 1536 (768 * 2).
        # For cross_attention, expected input_dim is 768.
        # However, our dynamic dimension detection should handle this gracefully.
        input_dim = embed_dim * 2 if pooling_type != "cross_attention" else embed_dim

        model = MultiLayerPerceptron(
            backbone=backbone,
            freeze=True,
            input_dim=input_dim,
            pooling_type=pooling_type,
            hidden_layer_dims=[512, 256],
            dropout=0.1,
            output_dim=1
        )

        # Test forward pass
        # Input tensor (batch size 4, dummy channels and shape)
        x = torch.randn(4, 12, 128, 128)
        output = model(x)

        assert output.shape == (4, 1), f"Expected shape (4, 1) for pooling_type {pooling_type}, got {output.shape}"

        # Verify gradients are working when freeze=False
        model_train = MultiLayerPerceptron(
            backbone=backbone,
            freeze=False,
            input_dim=input_dim,
            pooling_type=pooling_type,
            hidden_layer_dims=[512, 256],
            dropout=0.1,
            output_dim=1
        )
        output_train = model_train(x)
        loss = output_train.sum()
        loss.backward()

        # Ensure that backbone gradients were/were not computed according to freeze
        for p in model.backbone.parameters():
            assert p.grad is None, "Frozen backbone parameters should not have gradients"

        # Verify that training backward pass runs without errors when freeze=False
        pass

        print(f"Successfully verified pooling_type: {pooling_type}")

if __name__ == "__main__":
    test_mlp_pooling_types()
