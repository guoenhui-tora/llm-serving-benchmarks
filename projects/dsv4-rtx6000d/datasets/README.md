# GovReport 精确输入数据集

用于 DeepSeek-V4-Flash-0731 的 DSpark 性能测试，基于 GovReport 真实报告构建精确 8K、16K、24K 输入。

| 文件 | 输入 tokens | 条数 |
| --- | ---: | ---: |
| [govreport-isl8192-exact-n1024.jsonl](govreport-isl8192-exact-n1024.jsonl) | 8192 | 1024 |
| [govreport-isl16384-exact-n1024.jsonl](govreport-isl16384-exact-n1024.jsonl) | 16384 | 1024 |
| [govreport-isl24576-exact-n512.jsonl](govreport-isl24576-exact-n512.jsonl) | 24576 | 512 |

每行 JSONL 包含 `id`、`prompt`、`input_tokens`，只固定输入长度（ISL），不包含输出长度（OSL）。`input_tokens` 是完整 `prompt` 经 DSV4 tokenizer 编码后的实际 token 数；同一份 16K 输入可以搭配 1024 或 2048 tokens 的输出，无需修改数据文件。

使用 `vllm bench serve` 压测时，通过 `--custom-output-len` 设置输出长度，例如 `--custom-output-len 2048 --ignore-eos`。其中 `--ignore-eos` 用于避免模型提前结束；不加时，实际输出可能短于指定长度。

## 来源

使用 [ccdv/govreport-summarization](https://huggingface.co/datasets/ccdv/govreport-summarization/tree/4e21184e01ae8017e2c036e180fe5e541fef60a0/document) 的两个 train 分片，revision 为 `4e21184e01ae8017e2c036e180fe5e541fef60a0`。取 `report` 的连续前缀，加摘要指令和 DSV4 对话格式，按模型自带 tokenizer 校验完整输入长度；不使用参考 `summary`。压测时使用 `--skip-chat-template`。

原始语料：[GovReport](https://gov-report-data.github.io/)，Huang et al., NAACL 2021，许可 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

## 脚本用法

[prepare_govreport.py](prepare_govreport.py) 依赖 `transformers`、`tokenizers`、`pyarrow`，版本见 manifest。以下命令从本目录执行，替换源数据、模型及输出路径。

生成新数据，使用 `--isl`、`--count`、`--seed` 指定长度、数量和选样顺序：

```bash
python prepare_govreport.py generate \
  --source-dir /path/to/govreport/document \
  --model-dir /path/to/DeepSeek-V4-Flash-0731-NVFP4 \
  --output-dir /path/to/new-dataset-dir \
  --isl 8192 --count 1024 --seed 0
```

脚本会去重、截取并校验长度；合格数量不足时报错，不重复补样本，也不覆盖已有文件。

按 manifest 精确复现现有三档数据：

```bash
python prepare_govreport.py reproduce \
  --source-dir /path/to/govreport/document \
  --model-dir /path/to/DeepSeek-V4-Flash-0731-NVFP4 \
  --manifest govreport-manifest.json \
  --output-dir /path/to/reproduced-jsonl
```

## Manifest

[govreport-manifest.json](govreport-manifest.json) 记录每条样本的来源报告、截取位置和顺序，以及 tokenizer 身份、文件哈希。用于校验和精确复现；`reproduce` 会检查重建文件的 SHA256 是否与记录一致。
