import logging
import json
from transformers import AutoTokenizer


def setup_logging(log_file='./output/training.log'):
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def print_model_info(model, logger):
    """打印模型信息"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded successfully. Trainable parameters: {trainable_params}")
    logger.info(f"Total parameters: {total_params}")
    print(f'模型参数量为：{trainable_params}')
    return trainable_params


def print_training_args(args, logger):
    """打印训练参数"""
    args_dict = args.to_dict()
    pretty = json.dumps(args_dict, indent=2)
    logger.info("TrainingArguments:\n" + pretty)
    logger.info("Training arguments configured") 