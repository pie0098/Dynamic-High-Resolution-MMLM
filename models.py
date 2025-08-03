import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.activations import ACT2FN
from typing import Optional, Union, List
import logging

from config import VLMConfig

logger = logging.getLogger(__name__)


class InternMultiModalProjector(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.vision_config.hidden_size * int(1 / config.downsample_ratio) ** 2)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * int(1 / config.downsample_ratio) ** 2, config.llm_config.hidden_size
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.llm_config.hidden_size, config.llm_config.hidden_size)

    def forward(self, image_features):
        hidden_states = self.layer_norm(image_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class VLM(PreTrainedModel):
    config_class = VLMConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path, torch_dtype="auto", trust_remote_code=True)
        
        self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path, trust_remote_code=True)
        self.multi_modal_projector = InternMultiModalProjector(self.config)

        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
    def forward(self, input_ids, labels, pixel_values, attention_mask=None):
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        
        image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=self.config.vision_feature_layer,
                vision_feature_select_strategy=self.config.vision_feature_select_strategy
            )
        text_embeds = text_embeds.to(image_features.dtype)
        
        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs[0]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)
    
    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
        num_tiles, num_image_patches, embed_dim = image_features.shape
        batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])
        
        if num_tiles * num_image_patches != image_indices.shape[0]:
            error_msg = f"num_image_tokens in image_features != num_image_tokens in inputs_ids. Expected: {num_tiles*num_image_patches}, Got: {image_indices.shape[0]}"
            logger.error(error_msg)
            logger.error(f"image_features shape: {image_features.shape}")
            logger.error(f"input_ids shape: {input_ids.shape}")
            logger.error(f"batch_indices shape: {batch_indices.shape}")
            logger.error(f"image_indices shape: {image_indices.shape}")
            raise AssertionError(error_msg)
            
        inputs_embeds[batch_indices, image_indices] = image_features.view(-1, embed_dim)
        return inputs_embeds
    
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        **kwargs,
        ):
        pixel_values = pixel_values.to(dtype=self.vision_model.dtype)

        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        downsample_ratio = self.config.downsample_ratio
        if vision_feature_layer == -1:
            vision_features = self.vision_model(pixel_values=pixel_values).last_hidden_state
        else:
            vision_features = self.vision_model(pixel_values=pixel_values).hidden_states[vision_feature_layer]
        if vision_feature_select_strategy == "default":
            vision_features = vision_features[:, 1:, :]

        channels = vision_features.shape[1]
        feature_size = int(channels**0.5)
        batch_size = vision_features.shape[0]

        vision_features = vision_features.reshape(batch_size, feature_size, feature_size, -1)
        vision_features = self.pixel_shuffle(vision_features, scale_factor=downsample_ratio)
        vision_features = vision_features.reshape(batch_size, -1, vision_features.shape[-1])
        vision_features = self.multi_modal_projector(vision_features)
        return vision_features
    
    def pixel_shuffle(self, vision_features: torch.Tensor, scale_factor: float = 0.5):
        batch_size, width, height, channels = vision_features.size()

        if height % scale_factor != 0 or width % scale_factor != 0:
            error_msg = f"Height and width must be divisible by scale_factor for proper downsampling. Height: {height}, Width: {width}, Scale factor: {scale_factor}"
            logger.error(error_msg)
            logger.error(f"vision_features shape: {vision_features.shape}")
            raise ValueError(error_msg)

        vision_features = vision_features.view(
            batch_size, width, int(height * scale_factor), int(channels / scale_factor)
        )
        vision_features = vision_features.permute(0, 2, 1, 3).contiguous()
        vision_features = vision_features.view(
            batch_size, int(height * scale_factor), int(width * scale_factor), int(channels / (scale_factor**2))
        )
        vision_features = vision_features.permute(0, 2, 1, 3).contiguous()

        return vision_features 