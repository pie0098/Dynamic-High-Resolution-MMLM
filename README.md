# Dynamic-High-Resolution-MMLM

基于InternViT-300M和Qwen-2.5-1.5B-Instruct训练动态高分辨率的多模态大模型

This project trains a dynamic high-resolution multimodal large model based on InternViT-300M and Qwen-2.5-1.5B-Instruct

## 项目概述 / Project Overview

本项目基于intern-vit-300M和Qwen-2.5-1.5B-Instruct训练动态高分辨率的多模态大模型，图片由processing_intern_vit.py转换为13, 3, 448, 448的pixels_value后与inputs_embeds拼接进入qwen。此处的13表示由12个1, 3, 448, 448的子图和一个1, 3, 448, 448的大图缩略图cat而成，可参照internVL-2.5的文章https://arxiv.org/abs/2412.05271。

This project trains a dynamic high-resolution multimodal large model based on intern-vit-300M and Qwen-2.5-1.5B-Instruct. Images are converted to pixels_value of shape 13, 3, 448, 448 by processing_intern_vit.py and then concatenated with inputs_embeds before entering qwen. The 13 represents 12 sub-images of shape 1, 3, 448, 448 and one thumbnail of shape 1, 3, 448, 448 concatenated together, refer to the internVL-2.5 paper at https://arxiv.org/abs/2412.05271.

## 训练策略 / Training Strategy

本项目只进行预训练1个epoch示例（因为实在很耗时，太贵了），指令微调处理输入、输出的逻辑和预训练一致，只是预训练只进行图片、文本对齐层的训练，如internVL-2.5的文章中的MLP-Warmup；指令微调是在预训练的模型基础上，训练MLP+LLM (optional：+ InternViT)，数据集更小一点。

This project only performs pre-training for 1 epoch as an example (due to high computational cost and time consumption). The instruction fine-tuning logic for input/output processing is consistent with pre-training, but pre-training only trains the image-text alignment layer, such as MLP-Warmup in the internVL-2.5 paper. Instruction fine-tuning is based on the pre-trained model, training MLP+LLM (optional: + InternViT) with a smaller dataset.

### 参数训练控制 / Parameter Training Control

```python
for name, param in model.named_parameters():
    if 'vision_model' in name:
        param.requires_grad = True
```

## 安装与设置 / Installation and Setup

### 下载项目 / Download Project

```bash
git clone https://github.com/pie0098/Dynamic-High-Resolution-MMLM.git
```

### 下载模型 / Download Models

InternViT-300M-448px-V2_5下载自：
InternViT-300M-448px-V2_5 downloaded from:
https://hf-mirror.com/OpenGVLab/InternViT-300M-448px-V2_5

### 安装依赖 / Install Dependencies

```bash
pip install -r requirements.txt
```

## 目录结构 / Directory Structure

- `datasets/`: 用于放置预训练数据集 / for placing pre-training datasets
- `output/`: 用于存放训练日志和checkpoints / for storing training logs and checkpoints
- `train_intern_qwen_debug.py`: 主要训练代码 / main training code
- 其他文件都为该文件拆分 / other files are split from this file

## 训练 / Training

可以直接复制 run.sh中的内容在命令行中，zero3.yaml初始设置为4卡、bf16，4卡H20训练1个epoch用时7小时。

You can directly copy the content from run.sh to the command line. zero3.yaml is initially set for 4 cards, bf16, and training 1 epoch on 4 H20 cards takes 7 hours.

## 数据集 / Dataset

预训练数据集：
Pre-training dataset:
https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain

图片总数为558128，可以用count_files.py查看是否下载所有图片：
Total number of images is 558128, you can use count_files.py to check if all images are downloaded:

```bash
python count_files.py ./datasets/images
```

查看blip_laion_cc_sbu_558k.json中的问答对是否图片数量一致：
Check if the number of Q&A pairs in blip_laion_cc_sbu_558k.json matches the number of images:

```bash
grep -o '"id":' ./datasets/blip_laion_cc_sbu_558k.json | wc -l
```

## 故障排除 / Troubleshooting

在训练时，如果在保存checkpoints遇到huggingface的connection error：xxx set xxx offload...

During training, if you encounter huggingface connection error when saving checkpoints: xxx set xxx offload...

```bash
pip install -U huggingface_hub
vim ~/.bashrc
```

在文件最后一行加入：
Add to the last line of the file:

```bash
export HF_ENDPOINT="hf-mirror.com"
```

最后千万记得在命令行：
Finally, remember to run in command line:

```bash
source ~/.bashrc
```

## 测试 / Testing

测试预训练模型的文件为test.ipynb，预训练参照internVL2.5 MLP Warmup由于只训练MLP projector和1个epoch，没有训练qwen的lm_head和整个LLM，所以qwen回答的不好，本项目意在为训练动态分辨率多模态模型抛砖引玉。

The file for testing the pre-trained model is test.ipynb. The pre-training refers to internVL2.5 MLP Warmup. Since only the MLP projector and 1 epoch were trained, qwen's lm_head and the entire LLM were not trained, so qwen's answer was poor. This project is intended to stimulate discussion for training dynamic resolution multimodal models.

## 致谢 / Acknowledgments

最后，鸣谢以下开源项目，排名不分先后：
Finally, thanks to the following open source projects (in no particular order):

1. https://github.com/wyf3/llm_related/blob/main/train_multimodal_from_scratch
2. https://github.com/jingyaogong/minimind-v
3. https://github.com/yujunhuics/Reyes 
