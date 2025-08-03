ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file ./zero3.yaml \
./train_intern_qwen_debug.py \
> ./output/training.log 2>&1
