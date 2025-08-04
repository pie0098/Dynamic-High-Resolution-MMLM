# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from timm.models.layers import DropPath
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutput,
                                            BaseModelOutputWithPooling)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_intern_vit import InternVisionConfig

try:
    from .flash_attention import FlashAttention
    has_flash_attn = True
except:
    print('FlashAttention is not installed.')
    has_flash_attn = False


logger = logging.get_logger(__name__)


class InternRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True) # (1/1024) *  Σ x**2
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


try:
    from apex.normalization import FusedRMSNorm

    InternRMSNorm = FusedRMSNorm  # noqa

    logger.info('Discovered apex.normalization.FusedRMSNorm - will use it instead of InternRMSNorm')
except ImportError:
    # using the normal InternRMSNorm
    pass
except Exception:
    logger.warning('discovered apex but it failed to load, falling back to InternRMSNorm')
    pass


NORM2FN = {
    'rms_norm': InternRMSNorm,
    'layer_norm': nn.LayerNorm,
}


class InternVisionEmbeddings(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size # 1024
        self.image_size = config.image_size # 448
        self.patch_size = config.patch_size # 14
        # 随机初始化class_token的权重，注册为可学习的参数，
        # 维度为1 1 1024，第一个是batch维，第二个num_token维
        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )
        # 动态高分辨率输入 pixel_value.shape = 13 3 448 448, Conv2d: 3 1024 14 14
        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        # (448 // 14)**2 = 32**2 = 1024 ---> num_patches = num_image_tokens
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1 # = 1025 = 1024 + 1, 1 指的是 class token 即分类token
        # 随机初始化位置编码，1 1025 1024
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        # 1 1024 1024 ---> 1 32 32 1024 ---> 1 1024 32 32
        pos_embed = pos_embed.float().reshape(
            1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1).permute(0, 3, 1, 2)
        # 1 1024 32 32 ---> 1 1024 1024
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype   #            nn.Conv2d
        patch_embeds = self.patch_embedding(pixel_values)  # 13 3 448 448 ---> 13, 1024(emb), 32, 32
        batch_size, _, height, width = patch_embeds.shape  #                   b  emb_dim h  w
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2) # 13, 1024(emb), 32, 32 ---> 13 1024(emb) 1024 ---> 13 1024 1024(emb)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype) # 1 1 1024 ---> 13 1 1024
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1) # 13 1025 1024
        position_embedding = torch.cat([
            self.position_embedding[:, :1, :],
            self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)
        ], dim=1) # 1 1025 1024
        embeddings = embeddings + position_embedding.to(target_dtype) # + 号会把位置编码作为计算图的一部分，成为可训练
        return embeddings # 13 1025 1024


class InternAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size # 1024
        self.num_heads = config.num_attention_heads # 16
        self.use_flash_attn = config.use_flash_attn and has_flash_attn
        if config.use_flash_attn and not has_flash_attn:
            print('Warning: Flash Attention is not available, use_flash_attn is set to False.')
        self.head_dim = self.embed_dim // self.num_heads # 64
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).'
            )

        self.scale = self.head_dim ** -0.5 # 1/8
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias) # 1024 3072
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj_drop = nn.Dropout(config.dropout)

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)

        if self.use_flash_attn:
            self.inner_attn = FlashAttention(attention_dropout=config.attention_dropout)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim) # 1024 1024

    def _naive_attn(self, x):
        B, N, C = x.shape # 13 1025 1024
        # 13 1025 3072 ---> 13 1025 3 16 64 ---> 3 13 16 1025 64
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 3 13 16 1025 64 ---> 3 个 13 16 1025 64
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            # 13 16 1025 64 --> 13 1025 16 64 --> 13 1025 1024 --> q_norm() --> 13 1025 16 64 --> 13 16 1025 64
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
        # 13 16 1025 64 @ 13 16 64 1025 --> 13 16 1025 1025
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # 13 16 1025 1025 @ 13 16 1025 64 --> 13 16 1025 64 --> 13 1025 16 64 --> 13 1025 1024
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x) # 13 1025 1024
        x = self.proj_drop(x)
        return x # 13 1025 1024

    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)

        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
            k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)

        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=False
        )
        outs = self.proj(rearrange(context, 'b s h d -> b s (h d)'))
        outs = self.proj_drop(outs)
        return outs

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self._naive_attn(hidden_states) if not self.use_flash_attn else self._flash_attn(hidden_states)
        return x # 13 1025 1024


class InternMLP(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.act = ACT2FN[config.hidden_act] # gelu
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size) # 1024 4096
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size) # 4096 1024

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states) # 13 1025 1024 --> 13 1025 4096
        hidden_states = self.act(hidden_states) # 13 1025 1096
        hidden_states = self.fc2(hidden_states) # 13 1025 4096 --> 13 1025 1024
        return hidden_states


class InternVisionEncoderLayer(nn.Module):
    def __init__(self, config: InternVisionConfig, drop_path_rate: float):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = config.norm_type

        self.attn = InternAttention(config)
        self.mlp = InternMLP(config)
        self.norm1 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)
        # ls1（和同样定义的 ls2）就是实现 LayerScale 的可学习缩放因子
        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim)) # 1*1024
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim)) # 1* 1024
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity() # 是否drop
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity() # 是否drop

    def forward(
            self,
            hidden_states: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        # 13 1025 1024 --> 13 1025 1024
        hidden_states = hidden_states + self.drop_path1(self.attn(self.norm1(hidden_states)) * self.ls1)
        # 13 1025 1024 --> 13 1025 1024
        hidden_states = hidden_states + self.drop_path2(self.mlp(self.norm2(hidden_states)) * self.ls2)

        return hidden_states # 13 1025 1024


class InternVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`InternEncoderLayer`].

    Args:
        config (`InternConfig`):
            The corresponding vision configuration for the `InternEncoder`.
    """

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        # stochastic depth decay rule , torch.linspace(0, 0.1, 24) --> 0.0000 0.0043 ... 0.1000 越往后丢弃越多，直至0.1
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        # 24 * vision encoder
        self.layers = nn.ModuleList([
            InternVisionEncoderLayer(config, dpr[idx]) for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = True

    def forward(
            self,
            inputs_embeds,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds
        # 
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states: # 得到num_hidden_layer-1个hidden_states， len(encoder_layer) = 24, encoder_layer[-1].shape = 13 1025 1024
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    encoder_layer,
                    hidden_states)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                )
            hidden_states = layer_outputs # 13 1025 1024
        # 拿到最后一层
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states
        ) # last_hidden_state 13 1025 1024, hidden_states (24 * 13 1025 1024)


class InternVisionModel(PreTrainedModel):
    main_input_name = 'pixel_values'
    config_class = InternVisionConfig
    _no_split_modules = ['InternVisionEncoderLayer']

    def __init__(self, config: InternVisionConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)

    def resize_pos_embeddings(self, old_size, new_size, patch_size): # 假设 old = 448, new = 224
        pos_emb = self.embeddings.position_embedding # 1 1025 1024
        _, num_positions, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :] # 1 1 1024
        # 1 1024 1024 --> 1 32 32 1024 --> 1 1024 32 32 
        pos_emb = pos_emb[:, 1:, :].reshape(1, old_size // patch_size, old_size // patch_size, -1).permute(0, 3, 1, 2)
        # 1 1024 32 32 --> 1 1024 16 16, new_size // patch_size = 224/14 = 16
        pos_emb = F.interpolate(pos_emb.float(), size=new_size // patch_size, mode='bicubic', align_corners=False)
        # 1 1024 256 --> 1 256 1024
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, embed_dim, -1).permute(0, 2, 1)
        pos_emb = torch.cat([cls_emb, pos_emb], dim=1) # 1 257 1024
        self.embeddings.position_embedding = nn.Parameter(pos_emb) # 1 257 1024
        self.embeddings.image_size = new_size # 224
        logger.info('Resized position embeddings from {} to {}'.format(old_size, new_size))

    def get_input_embeddings(self):
        return self.embeddings # 13 1025 1024

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and pixel_embeds is None:
            raise ValueError('You have to specify pixel_values or pixel_embeds')

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            if len(pixel_values.shape) == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs.last_hidden_state # 13 1025 1024
        pooled_output = last_hidden_state[:, 0, :] # class_embeds 13 1 1024

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state, # 13 1025 1024
            pooler_output=pooled_output, # class_embeds 13 1 1024
            hidden_states=encoder_outputs.hidden_states, # (24 * 13 1025 1024), 24 = num_hidden_layers
            attentions=encoder_outputs.attentions,
        )
