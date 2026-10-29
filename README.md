---
base_model: Qwen/Qwen3.5-0.8B
library_name: peft
language:
- zh
tags:
- qwen3.5
- lora
- text-classification
- relation-extraction
---

# Qwen3.5-0.8B LoRA：训练、推理与评估

本仓库包含 Qwen3.5-0.8B 的 LoRA 适配器、分词器，以及单 GPU 文本微调代码。任务为中文评论类别识别和 SPO 关系抽取。

仓库根目录的 `adapter_model.safetensors` 是已上传检查点的增量权重，使用时仍需下载基础模型 [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)。它不代表一份独立完整模型，也没有在此声明完成全部预定轮数或达到某项准确率。

## 仓库结构

```text
qwen35-lora/
├── adapter_config.json          已发布适配器配置
├── adapter_model.safetensors    已发布 LoRA 权重
├── tokenizer*.json              已发布分词器
├── chat_template.jinja          对话模板
├── code/
│   ├── train.py                 Python 训练入口
│   ├── inference.py             Python 推理入口
│   ├── evaluate.py              回答 token loss 评估
│   ├── train_config.json        完整数据训练配置
│   ├── example_config.json      小型示例数据配置
│   ├── qwen_config.py           QwenConfig 与相对路径
│   ├── setup_env.py             环境准备与检查
│   ├── requirements.txt         依赖
│   ├── data_handle/             模板、标签掩码和动态 padding
│   ├── utils/                   LoRA、分块 loss 与检查点
│   └── tools/                   数据处理和权重下载工具
├── data/
│   ├── README.md                数据来源与格式说明
│   └── examples/                8 条 train、4 条 dev 合成示例
└── docs/
    ├── training.md              训练流程和参数
    └── inference.md             已发布适配器的使用方式
```

## 环境与基础模型

当前代码在 Windows、RTX 3060 Laptop 6GB 上验证过。项目需要先在基础 Python 中准备可用的 CUDA PyTorch，再运行环境脚本；脚本建立 `.venv`，复用基础 Python 的 CUDA PyTorch，并安装其余依赖。具体依赖以 `code/requirements.txt` 为准。

```text
git clone https://github.com/kiki-sjtu/qwen35-lora.git
cd qwen35-lora
python code/setup_env.py
python code/tools/download_model.py
```

下载工具从 Qwen 官方 ModelScope 仓库获取基础模型，保存到 `models/Qwen3.5-0.8B`，支持断点续传并核验 SHA-256。运行入口会自动切换到项目 `.venv`。

## 使用已发布的 LoRA

在仓库根目录运行，`--adapter .` 指向根目录已发布的适配器：

```text
python code/inference.py --adapter . --prompt "判断评论类别，只输出JSON数组。这台笔记本电脑运行流畅。" --max-new-tokens 64
```

基础模型对照：

```text
python code/inference.py --base-only --prompt "判断评论类别，只输出JSON数组。这台笔记本电脑运行流畅。" --max-new-tokens 64
```

不提供 `--adapter` 时，推理脚本默认寻找本地训练输出 `checkpoints/qwen_lora/best`；该输出目录不随仓库提供。更多说明见 [推理文档](docs/inference.md)。

## 训练

仓库仅附少量格式示例，没有发布完整的 5000/2000 条本地训练/验证集。先用示例检查输入与标签，然后可运行一次短流程：

```text
python code/train.py --config code/example_config.json --check-data
python code/train.py --config code/example_config.json
```

示例训练只用于流程演示，不足以训练出可靠任务能力。正式训练时，准备自己的 `data/train.jsonl`、`data/dev.jsonl`；可选提供 `data/dev_original.jsonl`，修改 `code/train_config.json` 后运行：

```text
python code/train.py
```

当前完整数据配置是 batch size 1、累积 8、rank 8、alpha 32、长度 1024、学习率 1e-4、2 epochs。只更新语言层 LoRA，基础权重、embedding 与视觉模块保持冻结。更多说明见 [训练文档](docs/training.md)。

## 验证范围

本地开发验证覆盖了 loss 掩码、标准 causal loss 与分块 loss 的一致性、最长 965-token 样本反向传播、训练尾部梯度累积、保存/恢复与适配器加载。它们是实现验证，不是任务准确率评测。

原本地数据包含大量规则合成样本，合成验证集得分不能直接代表真实业务泛化；关系文本中的实体为虚构。正式使用前，请用未参与训练的实际业务样本检查分类结果、关系依据和 JSON 输出格式。
