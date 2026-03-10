# Assortment of variously useful functions
import collections
import os
import shutil
import warnings

# Third-party libraries
import numpy as np
import torch
import wandb
from einops import rearrange

# Astronomy / SunPy libraries
import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.data.sample
import sunpy.map
from sunpy.coordinates.frames import HeliographicStonyhurst


# GENERAL
def days_hours_mins_secs_str(total_seconds):
    d, r = divmod(total_seconds, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return "{0}d:{1:02}:{2:02}:{3:02}".format(int(d), int(h), int(m), int(s))


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(dictionary, sep="_", wandb_mode=True):

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
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
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
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
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
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: 3d tuple of grid size: t, h, w
    return:
    pos_embed: L, D
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
    return (torch.abs(hmi_data) > 0.0).to(dtype=torch.uint8)


def apply_hmi_mask(data, hmi_mask, value):
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
    __slots__ = ()
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


aiamap = sunpy.map.Map(
    sunpy.data.sample.AIA_171_IMAGE
)  # example image is loaded at 1024x1024


def stonyhurst_to_patch_index(lat, lon, patch_size, img_w=512, img_h=512):
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
    """
    imgs: B, C, T, H, W
    x: B, L, D
    """
    p = patch_size
    tub = tubelet_size
    x = rearrange(
        imgs, "b c (t tub) (h p) (w q) -> b (t h w) (tub p q c)", tub=tub, p=p, q=p
    )

    return x


def unpatchify(x, img_size, patch_size, tubelet_size):
    """
    x: B, L, D
    imgs: B, C, T, H, W
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
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True)
    target = (target - mean) / (var + 1.0e-6) ** 0.5

    return target
