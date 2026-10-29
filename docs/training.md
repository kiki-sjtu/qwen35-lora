# Python LoRA 训练

## 运行顺序

`train.py:main()` 读取 `QwenConfig`，加载分词器并创建 `QwenDataset`。`convert_example_qwen35()` 应用模型模板、分词，并通过字符偏移将提示词位置的标签设为 -100。`QwenCollator` 动态右侧 padding，补齐部分的 attention_mask 为 0，labels 为 -100。

`load_qwen_model()` 在 `model.language_model.layers.*` 中的线性层注入 LoRA，仅这些适配器参数参与 AdamW 更新。`qwen_answer_loss_sum()` 对隐藏状态与下一个 token 标签做因果对齐，只选择答案位置；随后分块计算词表投影和交叉熵。梯度检查点及分块重算减少显存开销。

每次累积组按照所有答案 token 总数归一化 loss，逐微批次反向传播，最后裁剪梯度、更新 LoRA 和学习率。默认每轮打乱训练集，验证时不更新参数。

## 配置

参数读取优先级是运行参数覆盖项、JSON 配置、QwenConfig 默认值。路径字段相对于仓库根目录，显式 `--config` 文件路径相对于当前工作目录。

| 参数 | 完整数据配置 |
| --- | ---: |
| batch_size | 1 |
| gradient_accumulation_steps | 8 |
| epochs | 2 |
| learning_rate | 0.0001 |
| lora_rank / lora_alpha | 8 / 32 |
| lora_dropout | 0.05 |
| max_seq_length | 1024 |
| weight_decay | 0.01 |
| warmup_ratio | 0.03 |
| max_grad_norm | 1.0 |
| save_steps | 200 |

`precision=auto` 在支持 BF16 的 CUDA GPU 上选择 BF16。`max_steps=0` 不额外限制更新步数；`train_limit=0` 和 `eval_limit=0` 使用完整输入文件。5000 条样本、batch 1、累积 8、2 epochs 对应 1250 次更新；示例配置只有 8 条训练样本和 1 epoch，只执行一次更新。

## 开始与续训

```text
python code/train.py --config code/example_config.json --check-data
python code/train.py --config code/example_config.json
```

正式数据准备好后：

```text
python code/train.py
python code/train.py --resume checkpoints/qwen_lora/latest
```

示例运行续训需要同时传入其配置：

```text
python code/train.py --config code/example_config.json --resume checkpoints/example_run/latest
```

非空输出目录不会被当作新实验覆盖。续训需要完整 latest，包括训练状态；配置与数据签名必须匹配。`--stop-after-step N` 可以在指定更新后保存并停止，续训时移除它。强制结束进程不会触发自动保存。

## 验证与保存

每轮结束调用 `evaluate_qwen_model()`，计算答案 token 加权平均 loss 与 perplexity；如果存在原始 dev 文件，会另行评估。它不是生成准确率或 SPO F1。

`best` 按 dev loss 选择，保存适配器；`latest` 另存优化器、调度器、进度和随机状态。每 200 次更新及每轮结束保存 latest，验证安排在每轮结束，并非每次定期保存时都验证。`training_log.jsonl` 记录每次更新，控制台默认每 10 步打印。

Windows 的原生 PyTorch 回退实现可运行，但缺少 causal_conv1d、flash-linear-attention 优化内核时速度会较慢。训练与推理仍需要实际检查 CUDA 使用情况；可运行 `python code/check_env.py`。
