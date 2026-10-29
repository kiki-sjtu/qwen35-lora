# 数据格式与示例

本仓库只提供 `examples/train.jsonl` 的 8 条和 `examples/dev.jsonl` 的 4 条合成示例。每份都包含评论分类与 SPO 关系抽取，示例正文在两个集合之间不重复。`examples/provenance.json` 按顺序标记来源类型和生成规则。

这些例子从本地规则生成的合成数据中挑选，不是收集到的真实用户评论，也没有上传原始课程样本。SPO 文本使用虚构实体。

## 每行格式

```json
{"messages":[{"role":"system","content":"按要求回答。"},{"role":"user","content":"判断评论类别。\n\n这台电脑运行流畅。"},{"role":"assistant","content":"[\"电脑\"]"}]}
```

每个样本必须为 system/user/assistant 三条消息；`content` 为字符串。答案是可解析 JSON 的字符串。

分类输出是单元素 JSON 数组，类别为水果、洗浴、平板、酒店、衣服、电脑、书籍、蒙牛。SPO 输出为对象数组，每项含 `predicate`、`subject`、`subject_type`、`object`、`object_type` 五个字段。

模型特殊 token 由官方 chat template 添加。数据没有推理过程，因此使用非思考模式，不生成伪思考链。训练只监督 assistant 答案与 EOS。

## 完整本地数据的范围

原本地 train 有 5000 条（902 条原有 + 4098 条合成），dev 有 2000 条（122 条原有 + 1878 条合成）。这些完整文件未随仓库发布。`dev_original` 是其中原有 122 条的副本，与扩充 dev 重叠，不是另一份独立测试集。

若使用自己的完整数据，把文件放到 `data/train.jsonl` 和 `data/dev.jsonl`，按需要提供 `data/dev_original.jsonl`。代码会拒绝空答案、无效 JSON 和超长样本，不会静默截断。

`code/tools/expand_data.py` 是原扩充工具，要求原始 902/122 条输入；不能仅凭这里的 8/4 条示例重建完整课程数据。数据扩充的生成规则族与训练/验证样本共享，即使实体、句式分开，合成验证也不等同于独立真实业务评测。
