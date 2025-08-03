from transformers import PretrainedConfig, AutoConfig
from typing import Optional, Union, List


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