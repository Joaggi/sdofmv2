import numpy as np
import pytest
import torch

from sdofmv2.utils import hmi_mask, patchify, unpatchify


class TestHmiMask:
    def test_hmi_mask_basic(self):
        data = torch.tensor([[0.0, 1.0, -0.5], [0.0, 0.0, 2.0]])
        mask = hmi_mask(data)
        expected = torch.tensor([[0, 1, 1], [0, 0, 1]], dtype=torch.uint8)
        assert torch.equal(mask, expected)

    def test_hmi_mask_all_zeros(self):
        data = torch.zeros(5, 5)
        mask = hmi_mask(data)
        expected = torch.zeros(5, 5, dtype=torch.uint8)
        assert torch.equal(mask, expected)

    def test_hmi_mask_all_nonzero(self):
        data = torch.ones(5, 5) * 10.0
        mask = hmi_mask(data)
        expected = torch.ones(5, 5, dtype=torch.uint8)
        assert torch.equal(mask, expected)

    def test_hmi_mask_dtype(self):
        data = torch.tensor([1.0, -1.0, 0.0])
        mask = hmi_mask(data)
        assert mask.dtype == torch.uint8


class TestPatchify:
    def test_patchify_shape(self):
        imgs = torch.randn(2, 3, 8, 512, 512)  # B, C, T, H, W
        patches = patchify(imgs, patch_size=16, tubelet_size=1)
        assert patches.shape == (2, 1024, 768)  # B, num_patches, patch_dim

    def test_patchify_with_tubelet(self):
        imgs = torch.randn(2, 3, 4, 512, 512)  # B, C, T, H, W
        patches = patchify(imgs, patch_size=16, tubelet_size=2)
        num_patches_h = 512 // 16
        num_patches_w = 512 // 16
        num_patches_t = 4 // 2
        expected_patches = num_patches_t * num_patches_h * num_patches_w
        patch_dim = 2 * 16 * 16 * 3
        assert patches.shape == (2, expected_patches, patch_dim)

    def test_patchify_values_preserved(self):
        imgs = torch.ones(1, 1, 1, 32, 32)
        patches = patchify(imgs, patch_size=16, tubelet_size=1)
        assert torch.all(patches == 1.0)


class TestUnpatchify:
    def test_unpatchify_shape(self):
        x = torch.randn(2, 1024, 768)  # B, num_patches, patch_dim
        imgs = unpatchify(x, img_size=512, patch_size=16, tubelet_size=1)
        assert imgs.shape == (2, 3, 8, 512, 512)

    def test_unpatchify_with_tubelet(self):
        num_patches_t = 2
        num_patches_h = 32
        num_patches_w = 32
        num_patches = num_patches_t * num_patches_h * num_patches_w
        x = torch.randn(2, num_patches, 768)
        imgs = unpatchify(x, img_size=512, patch_size=16, tubelet_size=2)
        assert imgs.shape == (2, 3, 4, 512, 512)

    def test_unpatchify_roundtrip(self):
        imgs = torch.randn(1, 3, 8, 64, 64)
        patches = patchify(imgs, patch_size=16, tubelet_size=1)
        reconstructed = unpatchify(patches, img_size=64, patch_size=16, tubelet_size=1)
        np.testing.assert_array_almost_equal(imgs.numpy(), reconstructed.numpy(), decimal=5)
