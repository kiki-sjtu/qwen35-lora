# 使用仓库中的适配器

先配置环境并下载基础模型，参见根 README。仓库根目录已有发布的 LoRA 权重，因此在根目录运行时应明确使用 `--adapter .`：

```text
python code/inference.py --adapter . --prompt "判断评论类别，只输出JSON数组。这台笔记本电脑运行流畅。" --max-new-tokens 64
```

`--adapter` 相对路径按当前工作目录解析。从 code 文件夹运行时，根目录适配器对应 `--adapter ..`。不传此参数时，默认寻找本地完整训练输出 `checkpoints/qwen_lora/best`，不会自动使用仓库根目录权重。

## 推理实现

`load_qwen_model(config, adapter_path, training=False)` 加载基础模型，再使用 PEFT 加载适配器并切换 eval 模式。`qwen_inference()` 将 system/user 输入套用官方模板，设置 `enable_thinking=False`，调用 `generate()`，最后只解码新生成部分。

推理默认一次生成完成后才打印，非流式输出；max_new_tokens 是上限，遇 EOS 可以提前停止。分类可先用 32 或 64 做短测试；长关系抽取应留出更充分的输出长度，避免 JSON 被截断。长时间运行的业务程序应只加载一次模型，然后重复调用推理函数。

## Loss 对照

可用附带示例演示验证接口：

```text
python code/evaluate.py --config code/example_config.json --adapter .
python code/evaluate.py --config code/example_config.json --base-only
```

少量示例的 loss 只是接口演示，不是独立效果评测。业务验证应使用未参与训练的真实文本，检查类别、SPO 是否有原文依据、JSON 是否可解析。生成结果验证通过后再进入后续业务流程。
