# This model is modified from https://github.com/facebookresearch/mae/blob/main/models_mae.py

import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import to_2tuple
from timm.models.vision_transformer import Block
import torch.nn.functional as F
from ..utils import get_3d_sincos_pos_embed, unpatchify, patchify
from .losses import (
    mae_loss,
    vector_aware_loss,
    pixel_weight_loss,
    patch_weight_loss,
)


class PatchEmbed(nn.Module):
    """Frames of 2D Images to Patch Embedding
    The 3D version of timm.models.vision_transformer.PatchEmbed
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=3,
        tubelet_size=1,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.grid_size = (
            num_frames // tubelet_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1]),
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape  # batch channels frames height width
        # print("input dim", x.shape)
        x = self.proj(x)
        # print("proj dim", x.shape)
        # The output size is (B, L, C), where N=H*W/T/T, C is embid_dim
        if self.flatten:
            x = (
                x.flatten(2).transpose(1, 2).contiguous()
            )  # B,C,T,H,W -> B,C,L=(T*H*W) -> B,L,C
        x = self.norm(x)
        return x


class MaskedAutoencoderViT3D(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=3,
        tubelet_size=1,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer="LayerNorm",
        limb_mask=None,
        ids_limb_mask=None,
        loss_dict={},
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.limb_mask = limb_mask
        self.loss_dict = loss_dict

        # define loss     # define loss
        loss_functions = {
            "mse": F.mse_loss,
            "mae": mae_loss,
            "huber": F.huber_loss,
            "vector_aware_loss": vector_aware_loss,
            "pixel_weight_loss": pixel_weight_loss,
            "patch_weight_loss": patch_weight_loss,
        }
        loss_type = self.loss_dict.get("type", "mse")
        if loss_type not in loss_functions:
            raise ValueError(f"Unknown loss: {loss_type}")

        self.loss_fn = loss_functions[loss_type]

        # define normalization
        match norm_layer:
            case "LayerNorm":
                norm_layer = nn.LayerNorm
            case _:
                raise NotImplementedError(f"Norm layer [{norm_layer}] not implemented.")

        # --------------------------------------------------------------------------
        # Limb masking
        ids_limb_mask = torch.as_tensor(ids_limb_mask, dtype=torch.long)
        ids_limb_mask, _ = torch.sort(ids_limb_mask)
        self.register_buffer(
            "ids_limb_mask", ids_limb_mask
        )  # indices of pixel inside solar disk

        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            img_size, patch_size, num_frames, tubelet_size, in_chans, embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            tubelet_size * patch_size * patch_size * in_chans,
            bias=True,
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_3d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        device = x.device
        # if using a solar limb mask, we're not interested in predicting outside the
        # disk. For that we override these kept ids to include always keeping the mask
        # as we don't want to learn that.
        if self.ids_limb_mask is not None:
            ids_limb = self.ids_limb_mask.to(device).long().flatten()
            # Get indices of patches INSIDE the solar disk
            all_indices = torch.arange(L, device=device)
            mask_limb = torch.zeros(L, dtype=torch.bool, device=device)
            mask_limb[ids_limb] = True

            # Indices inside the disk (where we can do masking)
            ids_inside_disk = all_indices[~mask_limb]
            num_inside = len(ids_inside_disk)

            # Calculate how many inside patches to keep
            len_keep_inside = int(num_inside * (1 - mask_ratio))

            # Generate random noise only for inside patches
            noise_inside = torch.rand(N, num_inside, device=device)
            ids_shuffle_local = torch.argsort(noise_inside, dim=1)

            # Convert local indices back to global indices
            ids_inside_expand = ids_inside_disk.unsqueeze(0).expand(N, -1)
            local_keep = ids_shuffle_local[:, :len_keep_inside]
            ids_keep = torch.gather(ids_inside_expand, dim=1, index=local_keep)

            # Gather the kept patches
            ids_keep_gather = ids_keep.unsqueeze(-1).expand(-1, -1, D)
            x_masked = torch.gather(x, dim=1, index=ids_keep_gather)

            # Create binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, L], device=device, dtype=torch.long)
            mask.scatter_(1, ids_keep, 0)

            # Get the removed indices for all samples at once
            # limb ids repeated per-sample (always removed)
            ids_limb_expand = ids_limb.unsqueeze(0).expand(N, -1)  # [N, num_limb]
            local_removed_inside = ids_shuffle_local[:, len_keep_inside:]
            ids_removed_inside = torch.gather(
                ids_inside_expand, dim=1, index=local_removed_inside
            )  # [N, num_removed_inside]
            ids_removed = torch.cat([ids_limb_expand, ids_removed_inside], dim=1)

            # Concatenate kept and removed for all samples
            # ids_keep: [N, num_kept], ids_removed: [N, num_removed]
            full_order = torch.cat([ids_keep, ids_removed], dim=1)  # [N, L]

            # Apply argsort to all samples at once
            ids_restore = torch.argsort(full_order, dim=1)  # [N, L]

        else:
            len_keep = int(L * (1 - mask_ratio))
            noise = torch.rand(N, L, device=device)  # noise in [0, 1]

            # sort noise for each sample
            ids_shuffle = torch.argsort(
                noise, dim=1
            )  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # keep the first subset
            ids_keep = ids_shuffle[:, :len_keep]

            x_masked = torch.gather(
                x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
            )

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, L], device=device, dtype=torch.long)
            mask.scatter_(1, ids_keep, 0)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        # print("patch_embed dim", x.shape)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward(self, imgs, mask_ratio=0.5):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)

        # Compute patch-level loss
        if self.loss_dict.space == "patch":
            target = patchify(imgs, self.patch_size, self.tubelet_size)

            if self.loss_dict.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target_norm = (target - mean) / (var + 1e-6) ** 0.5
            else:
                target_norm = target

            device = target.device
            batch_size, num_patches, dims = pred.shape
            # Compute loss only on masked patches
            if self.loss_dict.only_masked_patches:
                active_mask = (mask == 1).clone().to(device).bool()
            else:
                active_mask = torch.ones(
                    (batch_size, num_patches), device=device, dtype=torch.bool
                )

            # Exclude the limb patches from the active mask
            # Compute loss only on inner patches
            if self.loss_dict.exclude_limb:
                active_mask[:, self.ids_limb_mask] = False

            # redefine pred and target
            pred_loss = pred[active_mask]
            target_loss = target[active_mask]
            target_norm_loss = target_norm[active_mask]

            if self.loss_dict.type == "pixel_weight_loss":
                loss = self.loss_fn(
                    pred_loss,
                    target_norm_loss,
                    target_loss,
                    self.loss_dict.pixel_weight_loss.base_loss,
                    self.loss_dict.pixel_weight_loss.threshold,
                    self.loss_dict.pixel_weight_loss.ar_weight_ratio,
                )
            # patch weight loss only works when only_masked_patches: false and exclude limb: true/false
            elif self.loss_dict.type == "patch_weight_loss":
                filtered_mask = mask[active_mask]
                loss = self.loss_fn(
                    pred_loss,
                    target_norm_loss,
                    self.loss_dict.patch_weight_loss,
                    filtered_mask,
                )
            else:
                loss = self.loss_fn(
                    pred_loss,
                    target_norm_loss if self.loss_dict.norm_pix_loss else target_loss,
                )

        elif self.loss_dict.space == "pixel":
            # pixel-level reconstruction
            pred_img = unpatchify(
                pred, self.img_size, self.patch_size, self.tubelet_size
            )

            # Apply limb mask if available
            if self.loss_dict.exclude_limb:
                limb_mask = self.limb_mask.to(imgs.device)
                pred_img = pred_img * limb_mask
                imgs = imgs * limb_mask

            # Compute pixel-level loss
            if self.loss_dict.type == "vector_aware_loss":
                loss = self.loss_fn(
                    pred_img, imgs, self.loss_dict.vector_aware_loss.base_loss
                )
            else:
                loss = self.loss_fn(pred_img, imgs)

        else:
            raise ValueError("Unsupported loss space type")

        return loss, pred, mask

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: list[int],
        mask_ratio: float = 0.0,
        reshape: bool = True,
        norm: bool = False,
    ):
        """Modified from timm.VisionTransformer.get_intermediate_layers"""
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, _, _ = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        features = [x]
        for blk in self.blocks:
            x = blk(x)
            features.append(x)

        # Remove cls token from intermediate features
        features = [feat[:, 1:, :] for feat in features]

        if norm:
            features = [self.norm(out) for out in features]

        if reshape:
            grid_size = self.patch_embed.grid_size
            features = [
                out.reshape(x.shape[0], grid_size[1], grid_size[2], -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in features
            ]

        features = [features[i] for i in n]
        return features
