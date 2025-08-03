import os
import json
import logging
from transformers import AutoTokenizer, TrainingArguments, Trainer
from config import VLMConfig
from models import VLM
from dataset import MyDataset, MyDataCollator
from utils import setup_logging, print_model_info, print_training_args


def main():
    logger = setup_logging()
    logger.info("Starting training process...")

    try:
        # 配置路径
        vision_model_path = "/root/autodl-tmp/InternViT-300M-448px-V2_5"
        llm_model_path = "/root/autodl-tmp/qwen25_1_5b"
        images_path = '/root/autodl-tmp/datasets/images'
        data_path = '/root/autodl-tmp/datasets/blip_laion_cc_sbu_558k.json'
        output_dir = './output'
        
        # 创建配置
        config = VLMConfig(llm_model_path=llm_model_path, vision_model_path=vision_model_path)
        logger.info(f"Configuration loaded: vision_model_path={config.vision_model_path}")
        
        # 加载模型
        model = VLM(config).cuda()
        print_model_info(model, logger)
        
        # 加载tokenizer
        logger.info(f"Loading tokenizer from {config.llm_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path, trust_remote_code=True)
        
        logger.info(f"Output directory: {output_dir}")
        
        # 训练参数
        args = TrainingArguments(
            output_dir=output_dir,
            do_train=True,
            per_device_train_batch_size=10,
            learning_rate=5e-6,
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
        print_training_args(args, logger)
        
        # 创建数据集
        logger.info("Creating trainer...")
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=MyDataset(images_path, data_path, tokenizer, config),
            data_collator=MyDataCollator(tokenizer)  
        )
        
        # 开始训练
        logger.info("*****************************************Starting training****************************************")
        trainer.train(resume_from_checkpoint=False)
        
        # 保存模型
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


if __name__ == '__main__':
    main() 