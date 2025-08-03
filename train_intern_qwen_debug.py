import os
# os.environ["HF_ENDPOINT"] = "hf-mirror.com"
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM, AutoConfig, CONFIG_MAPPING
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Callable, List, Optional, Tuple, Union
from PIL import Image
import io
import json
import logging
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from transformers.activations import ACT2FN
from typing import List, Dict, Any
from processing_intern_vit import load_image

# 设置logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./output/training.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VLMConfig(PretrainedConfig):
    model_type = "vlm_model"
    has_no_defaults_at_init = True
    def __init__(
        self,
        llm_model_path: Optional[str] = None,
        vision_model_path: Optional[str] = None,
        freeze_vision_model: bool = True,
        downsample_ratio: float = 0.5,
        projector_hidden_act: str = "gelu",
        vision_feature_layer: int = -1,
        vision_feature_select_strategy: str = "default",
        # 额外可直接传入已经加载好的 config 对象：
        llm_config: PretrainedConfig = None,
        vision_config: PretrainedConfig = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        # —— 1. LLM config —— 
        if llm_config is None:
            self.llm_config = AutoConfig.from_pretrained(llm_model_path, trust_remote_code=True)
        else:
            self.llm_config = llm_config
        # —— 2. Vision config —— 
        if vision_config is None:
            self.vision_config = AutoConfig.from_pretrained(vision_model_path, trust_remote_code=True)
        else:
            self.vision_config = vision_config

        self.llm_model_path = llm_model_path
        self.vision_model_path = vision_model_path
        self.freeze_vision_model = freeze_vision_model

        self.vision_feature_layer = vision_feature_layer
        self.downsample_ratio = downsample_ratio
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy

    def to_dict(self):
        # 先拿到父类自己序列化的字典
        output = super().to_dict()
        # 然后把子 config 也 dump 进去
        output['llm_config']    = self.llm_config.to_dict()
        output['vision_config'] = self.vision_config.to_dict()
        return output

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        # 先把字典里分离出两段子 config
        llm_cfg_dict    = config_dict.pop('llm_config')
        vision_cfg_dict = config_dict.pop('vision_config')

        # 用对应类恢复子 config
        llm_config    = PretrainedConfig.from_dict(llm_cfg_dict)
        vision_config = PretrainedConfig.from_dict(vision_cfg_dict)

        # 构造 VLMConfig（kwargs 会把其余字段塞进 __init__）
        return cls(
            llm_config=llm_config,
            vision_config=vision_config,
            **config_dict,
            **kwargs
        )

        
class InternMultiModalProjector(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.vision_config.hidden_size * int(1 / config.downsample_ratio) ** 2) # 4096
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * int(1 / config.downsample_ratio) ** 2, config.llm_config.hidden_size
        ) # 4096, 1536
        self.act = ACT2FN[config.projector_hidden_act] # gelu function
        self.linear_2 = nn.Linear(config.llm_config.hidden_size, config.llm_config.hidden_size) # 1536, 1536

    def forward(self, image_features):
        hidden_states = self.layer_norm(image_features) # 13 256 4096
        hidden_states = self.linear_1(hidden_states) # 13 256 1536
        hidden_states = self.act(hidden_states) # 13 256 1536
        hidden_states = self.linear_2(hidden_states) # 13 256 1536
        return hidden_states # 13 256 1536
        
class VLM(PreTrainedModel):
    config_class = VLMConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path, torch_dtype="auto", trust_remote_code=True)
        
        self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path, trust_remote_code=True)
        self.multi_modal_projector  = InternMultiModalProjector(self.config)

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
        # self.tokenizer('<|image_pad|>') = {'input_ids': [151655], 'attention_mask': [1]}
        batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])
        
        # 检查断言条件，如果失败则记录到logger并抛出异常
        if num_tiles * num_image_patches != image_indices.shape[0]:
            error_msg = f"num_image_tokens in image_features != num_image_tokens in inputs_ids. \
                Expected: {num_tiles*num_image_patches}, Got: {image_indices.shape[0]}"
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
            vision_features = self.vision_model(pixel_values=pixel_values).last_hidden_state # 13 1025 1024
        else:
            vision_features = self.vision_model(pixel_values=pixel_values).hidden_states[vision_feature_layer]
        if vision_feature_select_strategy == "default":
            vision_features = vision_features[:, 1:, :] # 13 1024 1024

        # Calculate dimensions based on vision features
        channels = vision_features.shape[1] # 1024
        feature_size = int(channels**0.5) # 1024**0.5 = 32
        batch_size = vision_features.shape[0] # 13

        # Reshape tensor to spatial dimensions, 13 1024 1024 --> 13 32 32 1024
        vision_features = vision_features.reshape(batch_size, feature_size, feature_size, -1) 

        # Apply downsampling using pixel shuffle, 13 32 32 1024 --> 13 16 16 4096
        vision_features = self.pixel_shuffle(vision_features, scale_factor=downsample_ratio)

        # Reshape tensor to prepare for projection, 13 256 4096
        vision_features = vision_features.reshape(batch_size, -1, vision_features.shape[-1])

        # Project features through multi-modal projector
        vision_features = self.multi_modal_projector(vision_features) # 13 256 1536
        return vision_features # 13 256 1536
    
    def pixel_shuffle(self, vision_features: torch.Tensor, scale_factor: float = 0.5):

        batch_size, width, height, channels = vision_features.size() # 13 32 32 1024

        if height % scale_factor != 0 or width % scale_factor != 0:
            error_msg = f"Height and width must be divisible by scale_factor for proper downsampling.\
                            Height: {height}, Width: {width}, Scale factor: {scale_factor}"
            logger.error(error_msg)
            logger.error(f"vision_features shape: {vision_features.shape}")
            raise ValueError(error_msg)

        # Reshape to allow downsampling, 13 32 32 1024 --> 13 32 16 2048
        vision_features = vision_features.view(
            batch_size, width, int(height * scale_factor), int(channels / scale_factor)
        )
        # Permute dimensions to align downsampled axis correctly, 13 32 16 2048 --> 13 16 32 2048
        vision_features = vision_features.permute(0, 2, 1, 3).contiguous()

        # Reshape to achieve final downsampled dimensions, 13 16 32 2048 --> 13 16(h) 16(w) 4096
        vision_features = vision_features.view(
            batch_size, int(height * scale_factor), int(width * scale_factor), int(channels / (scale_factor**2))
        )

        # Swap height and width back for proper orientation, 13 16(h) 16(w) 4096 --> 13 16(w) 16(h) 4096
        vision_features = vision_features.permute(0, 2, 1, 3).contiguous()

        return vision_features # 13 16(w) 16(h) 4096


    
class MyDataset(Dataset):
    def __init__(self, images_path, data_path, tokenizer, config: VLMConfig):
        super().__init__()
        self.data_path = data_path
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.config = config
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)   
        
            
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        sample = self.datas[index]
        image_name = sample['image']
        image_path = os.path.join(self.images_path, image_name)
        pixel_values = load_image(image_file=image_path)
        num_tiles, _, image_size, _ = pixel_values.shape # 13 3 448 448
        if image_size !=  config.vision_config.image_size:
            error_msg = f"pixel_values image_size != vision_config image_size. \
                        Expected: {self.config.vision_config.image_size}, Got: {image_size}"
            logger.error(error_msg)
            raise AssertionError(error_msg)
        # 13*256 = 13 * 32**2 / 4
        num_image_patches = image_size // self.config.vision_config.patch_size # 448 / 14 = 32
        num_image_tokens = num_tiles * num_image_patches * num_image_patches / (int(1 / config.downsample_ratio) ** 2)
        num_image_tokens = torch.tensor(num_image_tokens, dtype=torch.long)

        conversations = sample['conversations']
        """
        q_text =
        <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
        <|im_start|>user\n提供给定图像的简要描述。\n
        <|image_pad|><|image_pad|><|image_pad|><|im_end|>\n
        <|im_start|>assistant\n
        """
        q_text = self.tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, \
                                                    {"role":"user", "content":conversations[0]['value']}], \
                                                    tokenize=False, \
                                                    add_generation_prompt=True).replace('<image>', '<|image_pad|>'*num_image_tokens)
        # 橄榄油是自由使用的健康成分。<|im_end|>，这里将question和answer拆分，得到要预测的a_input_ids和label
        a_text = conversations[1]['value'] + self.tokenizer.eos_token # image_pad: 151655, \n:198
        q_input_ids = self.tokenizer(q_text)['input_ids'] # im_start: 151644, im_end:151645
        a_input_ids = self.tokenizer(a_text)['input_ids']
        input_ids = q_input_ids + a_input_ids
        labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids # tokenizer.pad_token_id: 151643
        input_ids = input_ids[:-1]
        labels = labels[1:]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        } 

class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        pixel_values = []
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])
            
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long), # B N 
                'labels': torch.tensor(labels, dtype=torch.long), # B l_N
                'pixel_values': torch.cat(pixel_values, dim=0)} 
            # 假设 B = 2, 有两个动态高分辨率输入 13 3 448 448 和 5 3 448 448 --> 18 3 448 448
        
        
if __name__ == '__main__':
    logger.info("Starting training process...")

    try:
        vision_model_path = "/root/autodl-tmp/InternViT-300M-448px-V2_5"
        llm_model_path = "/root/autodl-tmp/qwen25_1_5b"
        config = VLMConfig(llm_model_path=llm_model_path, vision_model_path=vision_model_path)
        logger.info(f"Configuration loaded: vision_model_path={config.vision_model_path}")
        
        model = VLM(config).cuda()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model loaded successfully. Trainable parameters: {trainable_params}")
        print(f'模型参数量为：{trainable_params}')
        
        images_path = '/root/autodl-tmp/datasets/images'
        data_path = '/root/autodl-tmp/datasets/blip_laion_cc_sbu_558k.json'
        
        logger.info(f"Loading tokenizer from {config.llm_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path, trust_remote_code = True)
        
        output_dir = './output'
        logger.info(f"Output directory: {output_dir}")
        
        args = TrainingArguments(
            output_dir=output_dir,
            do_train=True,
            per_device_train_batch_size=10,
            learning_rate=5e-6,
            # lr_scheduler_type='cosine',
            # warmup_steps = 50,
            num_train_epochs=1,
            save_steps=200,
            save_total_limit=2,
            bf16=True,
            gradient_accumulation_steps=16,
            max_grad_norm=5,
            logging_steps=1,
            report_to='tensorboard',
            dataloader_pin_memory=True,
            dataloader_num_workers=4
        )
        args_dict = args.to_dict()
        pretty = json.dumps(args_dict, indent=2)
        logger.info("TrainingArguments:\n" + pretty)
        logger.info("Training arguments configured")
        
        # 10个epoch?，4张a100（40g）两个小时左右，sft5个小时左右 评论： pretrain和sft都训1个epoch差不多了 文本太小后面loss不会掉了
        logger.info("Creating trainer...")
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=MyDataset(images_path, data_path, tokenizer, config),
            data_collator=MyDataCollator(tokenizer)  
        )
        
        logger.info("*****************************************Starting training****************************************")
        trainer.train(resume_from_checkpoint=False)
        
        logger.info("********************************Training completed. Saving model**********************************")
        trainer.save_model(output_dir)
        trainer.save_state()
        logger.info("Model and state saved successfully! 😊")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    

    
    
