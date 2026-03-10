import joblib
import numpy as np
from sklearn.decomposition import PCA
from .attention_map import patch_id_to_xy


def mapping_dense_to_rgb(
    feature_map,
    visible_patch_ids,
    n_components,
    img_size,
    grid_size,
    patch_size,
    pretrained,
):
    """
    feature_map: torch.Tensor of shape [n_patches, dim]
    visible_patch_ids: masked token ids (which go to encoder network)
    n_components: output dimension after PCA
    img_size: input dimension
    grid_size: size of grid after patchfying
    patch_size: size of patch
    returns: RGB image (H, W, 3) in [0, 1]
    """
    # first check feature map dimension
    # assert feature_map.shape[0] == grid_size**2, \
    #     f"feature_map has {feature_map.shape[0]} patches, but expected {grid_size**2}"

    # Apply PCA to latent
    feature_map = feature_map.detach().cpu().numpy()

    if pretrained:
        pca = joblib.load(pretrained)
        X_pca = pca.transform(feature_map)
    else:
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(feature_map)  # (n_patches, 3)

    # Normalize each component to [0, 1]
    X_pca -= X_pca.min(axis=0, keepdims=True)
    X_pca /= X_pca.max(axis=0, keepdims=True) + 1e-8

    # convert flatten patches to height and width
    full_rgb = np.ones((img_size, img_size, n_components), dtype=np.float32) * 0.5
    for w, patch_id in zip(X_pca, visible_patch_ids):
        x, y = patch_id_to_xy(patch_id, patch_size, grid=grid_size)
        full_rgb[y : y + patch_size, x : x + patch_size] = w

    return full_rgb
