# Assortment of variously useful functions
import collections.abc
import warnings

# Third-party libraries
import numpy as np
import torch
from einops import rearrange

# Astronomy / SunPy libraries
import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.data.sample
import sunpy.map
from sunpy.coordinates.frames import HeliographicStonyhurst


# GENERAL
def days_hours_mins_secs_str(total_seconds):
    """Convert a duration in seconds to a human-readable string.

    Args:
        total_seconds (int or float): The total number of seconds.

    Returns:
        str: A formatted string in the format 'Dd:Hh:Mm:Ss'.
    """
    d, r = divmod(total_seconds, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return "{0}d:{1:02}:{2:02}:{3:02}".format(int(d), int(h), int(m), int(s))


def flatten_dict(d, parent_key="", sep="_"):
    """Flatten a nested dictionary into a single-level dictionary.

    Args:
        d (dict): The input dictionary to flatten.
        parent_key (str, optional): The prefix for nested keys. Defaults to "".
        sep (str, optional): The separator between parent and child keys. Defaults to "_".

    Returns:
        dict: A flattened dictionary with keys joined by the separator.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(dictionary, sep="_", wandb_mode=True):
    """Unflatten a dictionary back into a nested dictionary structure.

    Args:
        dictionary (dict): The flattened dictionary to unflatten.
        sep (str, optional): The separator used to join keys. Defaults to "_".
        wandb_mode (bool, optional): If True, extracts values from 'value' keys
            in nested dicts. Defaults to True.

    Returns:
        AttributeDict: A nested dictionary with keys split by the separator.
    """

    def grab_values(d):
        resultDict = AttributeDict()
        for k, v in d.items():
            if isinstance(v, dict) and "desc" in v.keys() and "value" in v.keys():
                resultDict[k] = v["value"]
            elif isinstance(v, dict):
                resultDict[k] = grab_values(v)
            else:
                resultDict[k] = v
        return resultDict

    resultDict = AttributeDict()
    for key, value in dictionary.items():
        # value = value.value if wandb_mode else value
        parts = key.split(sep)
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = AttributeDict()
            d = d[part]
        d[parts[-1]] = value

    if wandb_mode:
        resultDict = grab_values(resultDict)
    return resultDict


#### MAE FUNCTIONS
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate 1D sine-cosine positional embeddings.

    Args:
        embed_dim (int): The output dimension for each position (must be even).
        pos (ndarray): A list or array of positions to be encoded, shape (M,).

    Returns:
        ndarray: Positional embeddings of shape (M, embed_dim).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    # pos = pos.reshape(-1)  # (M,), should already be this
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Generate 2D sine-cosine positional embeddings.

    Args:
        embed_dim (int): The embedding dimension for each position.
        grid_size (int): The grid height and width (assumed square).
        cls_token (bool, optional): If True, prepends a zero vector for CLS token.
            Defaults to False.

    Returns:
        ndarray: Positional embeddings of shape
            [grid_size*grid_size, embed_dim] or
            [1+grid_size*grid_size, embed_dim] (with cls_token).
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Generate 2D sine-cosine positional embeddings from a grid.

    Args:
        embed_dim (int): The embedding dimension (must be even).
        grid (ndarray): A 2xHxW array containing the 2D grid coordinates.

    Returns:
        ndarray: The positional embeddings of shape (H*W, embed_dim).
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Generate 3D sine-cosine positional embeddings.

    Args:
        embed_dim (int): The embedding dimension (must be divisible by 16).
        grid_size (tuple): A 3-tuple of (t, h, w) representing the grid dimensions.
        cls_token (bool, optional): If True, prepends a zero vector for CLS token.
            Defaults to False.

    Returns:
        ndarray: Positional embeddings of shape (L, embed_dim) where
            L = t * h * w (or L = 1 + t * h * w with cls_token).
    """
    assert embed_dim % 16 == 0

    t_size, h_size, w_size = grid_size

    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4

    w_pos_embed = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
    h_pos_embed = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
    t_pos_embed = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))

    w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
    h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
    t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)

    pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


# HMI MASKING
# From various FDL piror works (sdolatent, solar-vae, etc.)
def hmi_mask(hmi_data):
    """Generate a binary mask for HMI solar disk data.

    Creates a binary mask where 1 indicates pixels within the solar disk
    (non-zero magnetic field values) and 0 indicates pixels outside.

    Args:
        hmi_data (torch.Tensor): The HMI magnetogram data tensor.

    Returns:
        torch.Tensor: A binary mask tensor of the same shape as input.
    """
    return (torch.abs(hmi_data) > 0.0).to(dtype=torch.uint8)


def apply_hmi_mask(data, hmi_mask, value):
    """Apply an HMI mask to solar image data.

    Replaces pixels outside the solar disk (where hmi_mask is 0) with a
    specified scalar value. The mask is applied only to HMI channels;
    AIA channels remain unchanged.

    Args:
        data (torch.Tensor): The input data tensor of shape (B, C, H, W) or (C, H, W).
        hmi_mask (torch.Tensor): A binary mask where 1 represents pixels inside
            the solar disk and 0 represents pixels outside.
        value (float): The scalar value to replace masked pixels with.

    Returns:
        torch.Tensor: The masked data tensor with the same shape as input.
    """
    # hmi mask is a binary mask of 0 and 1 values
    # 1 represents that the pixel is within the solar disk, 0 represents that the pixel is outside the solar disk
    # this function replaces the pixels outside the solar disk with the given scalar value
    if data.ndim == 4:
        hmi = data[:, :3]
        aia = data[:, 3:]

        value_mask = value * (~hmi_mask.to(dtype=torch.bool))
        hmi_mask = hmi_mask.to(device=data.device)
        value_mask = value_mask.to(device=data.device)
        hmi = hmi * hmi_mask
        hmi = hmi + value_mask

        data = torch.cat([hmi, aia], dim=1)
        return data
    elif data.ndim == 3:
        data = data.unsqueeze(0)
        data = apply_hmi_mask(data, hmi_mask, value)
        data = data.squeeze(0)
        return data
    else:
        raise ValueError("Expecting 3d or 4d data")


class AttributeDict(dict):
    """A dictionary subclass that allows attribute-style access to its keys.

    This class lets you use dot notation like obj.key to get and set
    dictionary items. It keeps all standard dictionary methods and uses
    __slots__ to save memory by preventing the creation of an instance
    dictionary.

    Args:
        *args: Positional arguments passed to the dict constructor.
        **kwargs: Keyword arguments passed to the dict constructor.

    Returns:
        A new AttributeDict instance.
    """

    __slots__ = ()
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


aiamap = sunpy.map.Map(
    sunpy.data.sample.AIA_171_IMAGE
)  # example image is loaded at 1024x1024


def stonyhurst_to_patch_index(lat, lon, patch_size, img_w=512, img_h=512):
    """Convert Heliographic Stonyhurst coordinates to patch indices.

    Transforms latitude and longitude coordinates in the Heliographic Stonyhurst
    frame to corresponding patch indices in an image grid.

    Args:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
        patch_size (int): The size of each patch in pixels.
        img_w (int, optional): Image width in pixels. Defaults to 512.
        img_h (int, optional): Image height in pixels. Defaults to 512.

    Returns:
        torch.Tensor: A tensor of shape (2,) containing the patch indices [x, y].

    Warns:
        UserWarning: If image dimensions exceed 1024, indicating potential
            precision loss in coordinate conversion.
    """
    # Heliographic Stonyhurst coordinates to patch index
    coord = SkyCoord(lat * u.deg, lon * u.deg, frame=HeliographicStonyhurst)
    x, y = aiamap.wcs.world_to_pixel(coord)  # (x, y) in pixels
    scale_x = 1024 / img_w
    scale_y = 1024 / img_h
    x, y = x / scale_x // patch_size, y / scale_y // patch_size
    if img_w > 1024 or img_h > 1024:
        warnings.warn(
            "Loss of precision when over 1024 on coordinate converstion, consider upgrading reference image."
        )
    return torch.Tensor([x, y])


def patchify(imgs, patch_size, tubelet_size):
    """Convert image tensors into sequences of patches.

    Takes a 5D image tensor and reorganizes it into a sequence of flattened
    patches suitable for Vision Transformer (ViT) processing.

    Args:
        imgs (torch.Tensor): Input images of shape (B, C, T, H, W).
        patch_size (int): The spatial size of each square patch.
        tubelet_size (int): The temporal size of each tubelet.

    Returns:
        torch.Tensor: Patched tensor of shape (B, L, D) where L is the
            number of patches and D is the flattened patch dimension.
    """
    p = patch_size
    tub = tubelet_size
    x = rearrange(
        imgs, "b c (t tub) (h p) (w q) -> b (t h w) (tub p q c)", tub=tub, p=p, q=p
    )

    return x


def get_zero_patch_mask(imgs, patch_size, tubelet_size):
    """Create binary mask indicating which patches are all-zero.

    Args:
        imgs (torch.Tensor): Images of shape (B, C, T, H, W) BEFORE normalization.
        patch_size (int): Spatial size of each patch.
        tubelet_size (int): Temporal size of each tubelet.

    Returns:
        torch.Tensor: Binary mask of shape (B, L) where 1 = all-zero patch.
    """
    p = patch_size
    tub = tubelet_size

    patches = rearrange(
        imgs, "b c (t tub) (h p) (w q) -> b (t h w) (tub p q c)", tub=tub, p=p, q=p
    )

    patch_is_zero = patches.abs().sum(dim=-1) == 0
    return patch_is_zero


def spatial_to_patch_mask(
    mask_2d: torch.Tensor, patch_size: int, num_frames: int = 1
) -> torch.Tensor:
    """Convert 2D spatial mask to patch-level mask.

    Args:
        mask_2d: 2D binary mask of shape (H, W).
        patch_size: Spatial size of each patch.
        num_frames: Number of frames (temporal). Default: 1.

    Returns:
        torch.Tensor: 1D boolean tensor of shape (L,) where True = off-limb patch.
    """
    H, W = mask_2d.shape
    p = patch_size

    h = H // p
    w = W // p

    mask_3d = mask_2d.unsqueeze(0).unsqueeze(0).expand(num_frames, 1, -1, -1)
    patches = rearrange(mask_3d, "(t c) (h p) (w q) -> (t h w) (p q c)", p=p, q=p)

    patch_is_zero = patches.sum(dim=(-1, -2)) == 0
    return patch_is_zero


def unpatchify(x, img_size, patch_size, tubelet_size):
    """Reconstruct image tensors from sequences of patches.

    Takes a sequence of flattened patches and reorganizes them back into
    a 5D image tensor.

    Args:
        x (torch.Tensor): Patched tensor of shape (B, L, D).
        img_size (int): The spatial size of the original images (assumed square).
        patch_size (int): The spatial size of each patch.
        tubelet_size (int): The temporal size of each tubelet.

    Returns:
        torch.Tensor: Reconstructed images of shape (B, C, T, H, W).
    """
    p = patch_size
    num_p = img_size // p
    tub = tubelet_size
    imgs = rearrange(
        x,
        "b (t h w) (tub p q c) -> b c (t tub) (h p) (w q)",
        h=num_p,
        w=num_p,
        tub=tub,
        p=p,
        q=p,
    )
    return imgs


def norm_target(target) -> torch.Tensor:
    """Normalize target values using z-score normalization.

    Applies z-score normalization to the target tensor along the last dimension,
    computing mean and variance per sample in the batch.

    Args:
        target (torch.Tensor): The input tensor to normalize.

    Returns:
        torch.Tensor: The normalized tensor with the same shape as input.
    """
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True)
    target = (target - mean) / (var + 1.0e-6) ** 0.5

    return target
