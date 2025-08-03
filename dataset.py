import os
import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
from transformers import AutoTokenizer
from config import VLMConfig
from processing_intern_vit import load_image


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
        num_tiles, _, image_size, _ = pixel_values.shape
        if image_size != self.config.vision_config.image_size:
            error_msg = f"pixel_values image_size != vision_config image_size. Expected: {self.config.vision_config.image_size}, Got: {image_size}"
            raise AssertionError(error_msg)
        
        num_image_patches = image_size // self.config.vision_config.patch_size
        num_image_tokens = num_tiles * num_image_patches * num_image_patches / (int(1 / self.config.downsample_ratio) ** 2)
        num_image_tokens = torch.tensor(num_image_tokens, dtype=torch.long)

        conversations = sample['conversations']
        q_text = self.tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, 
                                                    {"role":"user", "content":conversations[0]['value']}], 
                                                    tokenize=False, 
                                                    add_generation_prompt=True).replace('<image>', '<|image_pad|>'*num_image_tokens)
        
        a_text = conversations[1]['value'] + self.tokenizer.eos_token
        q_input_ids = self.tokenizer(q_text)['input_ids']
        a_input_ids = self.tokenizer(a_text)['input_ids']
        input_ids = q_input_ids + a_input_ids
        labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
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
            
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'pixel_values': torch.cat(pixel_values, dim=0)} 