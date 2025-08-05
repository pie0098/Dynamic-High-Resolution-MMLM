# Easy-Dynamic-High-Resolution-MMLM

## 项目概述
本项目基于intern-vit-300M和Qwen-2.5-1.5B-Instruct训练动态高分辨率的多模态大模型，图片由processing_intern_vit.py转换为13, 3, 448, 448的pixels_value后与inputs_embeds拼接进入qwen。此处的13表示由12个1, 3, 448, 448的子图和一个1, 3, 448, 448的大图缩略图cat而成，可参照internVL-2.5的文章https://arxiv.org/abs/2412.05271。

## 训练策略

本项目只进行预训练1个epoch示例（因为实在很耗时，太贵了），指令微调处理输入、输出的逻辑和预训练一致，只是预训练只进行图片、文本对齐层的训练，如internVL-2.5的文章中的MLP-Warmup；指令微调是在预训练的模型基础上，训练MLP projector+LLM (可选：+ InternViT)，数据集更小一点。
指令微调训练MLP projector+LLM，冻结的InternViT伪代码如下：

```python
for name, param in model.named_parameters():
    if 'vision_model' in name:
        param.requires_grad = False
```
要开启qwen的lm_head训练，伪代码如下：
```python
for _, param in self.llm_model.lm_head.named_parameters():
     param.requires_grad = True
```
## 安装与设置

### 下载项目

```bash
git clone https://github.com/pie0098/Easy-Dynamic-High-Resolution-MMLM.git
```

### 下载模型

InternViT-300M-448px-V2_5下载自：
https://hf-mirror.com/OpenGVLab/InternViT-300M-448px-V2_5

Qwen-2.5-1.5B-Instruct下载自：
https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct

### 安装依赖

```bash
pip install -r requirements.txt
```

## 目录结构

- `datasets/`: 用于放置预训练数据集
- `output/`: 用于存放训练日志和checkpoints
- `train_intern_qwen_debug.py`: 主要训练代码
- 其他文件都为`train_intern_qwen_debug.py`拆分而来

## 训练 
可以直接复制 run.sh中的内容在命令行中，zero3.yaml初始设置为4卡、bf16，4卡H20训练1个epoch用时7小时。

## 数据集 

预训练数据集：

https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain

图片总数为558128，可以用count_files.py查看是否下载所有图片：

```bash
python count_files.py ./datasets/images
```

查看blip_laion_cc_sbu_558k.json中的问答对是否图片数量一致：

```bash
grep -o '"id":' ./datasets/blip_laion_cc_sbu_558k.json | wc -l
```

## 故障排除

在训练时，如果在保存checkpoints遇到huggingface的connection error：
```bash
_main__ - ERROR - Training failed with error: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files. Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
```
#### 执行以下操作：
```bash
pip install -U huggingface_hub
vim ~/.bashrc
```

在文件最后一行加入：

```bash
export HF_ENDPOINT="hf-mirror.com"
```

最后千万记得在命令行：

```bash
source ~/.bashrc
```

## 测试

测试预训练模型的文件为test.ipynb，预训练参照internVL2.5 MLP Warmup由于只训练MLP projector且只有1个epoch，没有训练qwen的lm_head和指令微调MLP projector + LLM，所以qwen回答的不好，本项目意在为训练动态分辨率多模态模型抛砖引玉。


## 致谢

最后，鸣谢以下开源项目，大幅减少了重复造轮子的工作量：
1. https://github.com/OpenGVLab/InternVL
2. https://github.com/wyf3/llm_related/blob/main/train_multimodal_from_scratch
3. https://github.com/jingyaogong/minimind-v
4. https://github.com/yujunhuics/Reyes 
