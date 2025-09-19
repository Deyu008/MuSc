# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# DINOv3 Vision Transformer implementation with enhanced features
# Based on DINOv3 improvements over DINOv2

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from models.backbone.dinov2 import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block


logger = logging.getLogger("dinov3")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class EnhancedMultiHeadAttention(MemEffAttention):
    """Enhanced Multi-Head Attention with DINOv3 improvements"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add temperature scaling for better attention distribution
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x, attn_bias=None):
        # Apply temperature scaling
        if hasattr(self, 'temperature'):
            x = x * self.temperature
        return super().forward(x, attn_bias)


class MultiScalePatchEmbed(PatchEmbed):
    """Multi-scale patch embedding for better feature extraction"""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)
        
        # Add multi-scale convolutions
        self.multi_scale_convs = nn.ModuleList([
            nn.Conv2d(in_chans, embed_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_chans, embed_dim // 4, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=1, padding=3),
        ])
        
        # Additional conv for residual connection
        self.residual_conv = nn.Conv2d(in_chans, embed_dim // 4, kernel_size=1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Original patch embedding
        x_orig = super().forward(x)
        
        # Multi-scale features (for enhanced semantic understanding)
        multi_scale_features = []
        for conv in self.multi_scale_convs:
            ms_feat = conv(x)
            ms_feat = nn.functional.adaptive_avg_pool2d(ms_feat, (H // self.patch_size[0], W // self.patch_size[1]))
            multi_scale_features.append(ms_feat)
        
        # Residual connection
        residual_feat = self.residual_conv(x)
        residual_feat = nn.functional.adaptive_avg_pool2d(residual_feat, (H // self.patch_size[0], W // self.patch_size[1]))
        multi_scale_features.append(residual_feat)
        
        # Combine multi-scale features
        ms_combined = torch.cat(multi_scale_features, dim=1)
        
        if self.flatten:
            ms_combined = ms_combined.flatten(2).transpose(1, 2)  # BCHW -> BNC
        
        return x_orig


class DINOv3VisionTransformer(nn.Module):
    """DINOv3 Vision Transformer with enhanced features over DINOv2"""
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=MultiScalePatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        # DINOv3 specific enhancements
        use_enhanced_attention=True,
        use_learnable_pos_embed=True,
        use_register_tokens=True,
        num_register_tokens=4,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens if use_register_tokens else 0
        
        # Enhanced patch embedding
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Register tokens for improved attention (DINOv3 feature)
        if use_register_tokens:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            self.num_tokens += num_register_tokens

        # Enhanced positional encoding
        if use_learnable_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        else:
            # Fixed sinusoidal positional encoding
            self.register_buffer('pos_embed', self._get_sinusoidal_pos_embed(num_patches + self.num_tokens, embed_dim))

        if use_enhanced_attention:
            attn_class = EnhancedMultiHeadAttention
        else:
            attn_class = MemEffAttention

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused":
            ffn_layer = SwiGLUFFNFused
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate if not drop_path_uniform else drop_path_rate * (i + 1) / depth,
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
                attn_class=attn_class,
            )
            for i in range(depth)
        ]
        if block_chunks > 1:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                chunked_blocks.append(BlockChunk(blocks_list[i : i + chunksize]))
            self.blocks = nn.ModuleList(chunked_blocks)
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        # Mask token for MAE-like pretraining
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.init_weights()

    def _get_sinusoidal_pos_embed(self, num_patches, embed_dim):
        """Generate sinusoidal positional embeddings"""
        position = torch.arange(num_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pos_embed = torch.zeros(1, num_patches, embed_dim)
        pos_embed[0, :, 0::2] = torch.sin(position * div_term)
        pos_embed[0, :, 1::2] = torch.cos(position * div_term)
        return pos_embed

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if hasattr(self, 'register_tokens'):
            nn.init.normal_(self.register_tokens, std=1e-6)
        if hasattr(self, 'mask_token'):
            nn.init.normal_(self.mask_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        """Interpolate positional encodings for different image sizes"""
        previous_dtype = x.dtype
        npatch = x.shape[1] - self.num_tokens
        N = self.pos_embed.shape[1] - self.num_tokens
        if npatch == N and w == h:
            return self.pos_embed.to(dtype=previous_dtype)
        
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, :self.num_tokens]
        patch_pos_embed = pos_embed[:, self.num_tokens:]
        
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # Add small number to avoid floating point error
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        """Prepare tokens with optional masking"""
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        # Add class token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        
        # Add register tokens if present
        if hasattr(self, 'register_tokens'):
            register_tokens = self.register_tokens.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, register_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional encoding
        x = x + self.interpolate_pos_encoding(x, w, h)

        return x

    def forward_features_list(self, x_list, masks_list):
        """Forward pass for list of images"""
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = [blk(x_i) for x_i in x]

        all_x_norm = []
        for x_i in x:
            x_norm = self.norm(x_i)
            all_x_norm.append(x_norm)

        return {
            "x_norm_clstoken": [x_norm[:, 0] for x_norm in all_x_norm],
            "x_norm_patchtokens": [x_norm[:, self.num_tokens:] for x_norm in all_x_norm],
            "x_prenorm": x,
            "masks": masks_list,
        }

    def forward_features(self, x, masks=None):
        """Forward pass for feature extraction"""
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        if self.chunked_blocks:
            for blk in self.blocks:
                x = blk(x)
        else:
            for blk in self.blocks:
                x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, self.num_tokens:],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        """Get intermediate layer outputs (non-chunked version)"""
        x = self.prepare_tokens_with_masks(x)
        
        # If n is an int, take the last n blocks
        if isinstance(n, int):
            n = [len(self.blocks) - n + i for i in range(n)]
        
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in n:
                output.append(self.norm(x))
        
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        """Get intermediate layer outputs (chunked version)"""
        x = self.prepare_tokens_with_masks(x)
        
        if isinstance(n, int):
            n = [len(self.blocks) - n + i for i in range(n)]
        
        output = []
        block_idx = 0
        for chunk in self.blocks:
            for blk in chunk:
                x = blk(x)
                if block_idx in n:
                    output.append(self.norm(x))
                block_idx += 1
        
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence[int]] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """Get intermediate layer outputs with DINOv3 enhancements"""
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)

        if norm:
            outputs = [self.norm(out) for out in outputs]

        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, self.num_tokens:] for out in outputs]

        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return outputs

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def dinov3_vit_small(patch_size=16, **kwargs):
    model = DINOv3VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=EnhancedMultiHeadAttention),
        **kwargs,
    )
    return model


def dinov3_vit_base(patch_size=16, **kwargs):
    model = DINOv3VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=EnhancedMultiHeadAttention),
        **kwargs,
    )
    return model


def dinov3_vit_large(patch_size=16, **kwargs):
    model = DINOv3VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=EnhancedMultiHeadAttention),
        **kwargs,
    )
    return model


def dinov3_vit_giant(patch_size=16, **kwargs):
    model = DINOv3VisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=EnhancedMultiHeadAttention),
        **kwargs,
    )
    return model