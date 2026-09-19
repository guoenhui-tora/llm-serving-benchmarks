# GovReport 近8K摘要请求

[govreport-near8k.jsonl](govreport-near8k.jsonl) 固定保存256条英文摘要请求，供 DeepSeek V4 Flash NVFP4 的 DSpark OFF/ON 对照使用。最终输入平均8166 tokens，输出预算1024。其他节点直接使用此文件，不需要下载原始数据或重新筛选。

已通过固定 vLLM 0.29.0 客户端的实际tokenizer校验和单/双实例模拟流式检查，并完成前128条请求的单TP4、C32、DSpark off/on性能对照，见[性能报告](../reports/dspark-tp4-20260919.md)。本实验不评估摘要质量。这些请求不是严格等长的8192输入，不能用历史random负载结果计算DSpark加速比。

## 来源与处理

- 数据来自 [ccdv/govreport-summarization](https://huggingface.co/datasets/ccdv/govreport-summarization)，版本 `4e21184e01ae8017e2c036e180fe5e541fef60a0`，分片 `document/train-00000-of-00002.parquet`。
- 来源项目：[GovReport](https://gov-report-data.github.io/)，Huang et al., *Efficient Attentions for Long Document Summarization*, NAACL 2021；原项目标注 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。本子集增加摘要指令和模型角色标记，未截断、拼接或改写报告正文。
- 扫描8759条，排除11条重复正文，无空正文。按DSV4最终prompt长度筛选7168–9216 tokens，得到1364条；按原始行顺序收集后用 Python `random.Random(0).shuffle`，取前256条。不按参考答案长度或模型表现选样。
- `id`末尾的row是原始分片中从0开始的行号。JSONL只包含 `id`、`prompt`、`input_tokens`、`output_tokens`，不含参考摘要。
- 使用 `nvidia/DeepSeek-V4-Flash-0731-NVFP4` 自带的 `encoding/encoding_dsv4.py`，以 `thinking_mode="chat"` 编码用户消息，prompt以 `<｜Assistant｜></think>` 结尾。长度包含BOS、角色标记和指令；固定客户端的默认tokenizer计数与文件标注一致。

用户消息为以下指令、完整报告正文及 `Detailed summary:` 后缀：

> Read the government report below and write a detailed summary in English. Cover its background, central questions, major findings, supporting evidence, and conclusions or recommendations where present. Preserve important facts and figures, organize the summary into coherent paragraphs, and do not add information that is absent from the report.

## 工作量与用法

| 请求集 | 最小输入 | 最大输入 | 平均输入 | 输入token总量 | 固定输出总量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 前128条 | 7168 | 9213 | 8141.79 | 1,042,149 | 131,072 |
| 前256条 | 7168 | 9213 | 8166.11 | 2,090,525 | 262,144 |

文件大小11,324,941 bytes（约10.8 MiB）。输出1024是压测预算，不保证自然生成达到此长度；固定输出对照需 `ignore_eos: true`，自然EOS实验单独统计。

续接项目时，先按[项目工作区规则](../../../README.md#续接项目从该项目的精选配置开始)准备configs，再从仓库根目录复制这两个数据文件：

```bash
mkdir -p experiments/dsv4-rtx6000d/data
cp projects/dsv4-rtx6000d/data/govreport-near8k.{jsonl,md} experiments/dsv4-rtx6000d/data/
```

在工作区新建workload，dataset使用以下字段；保留原random基线文件：

```yaml
dataset:
  name: jsonl
  path: data/govreport-near8k.jsonl
  max_input_tokens: 9216
  output_tokens: 1024
  sha256: 33bdbd36a8b335300974e1b07aa66b2c24c88cc4c4b55a21350aa3564a945d53
```

`path`相对于configs上一级项目目录。每轮按文件顺序取前N条，不重新打乱或循环补样本；双实例交错分片，等分请求数但不保证输入token总量相等。prompt原样通过流式 `/v1/completions` 发送，不再套chat模板。保持前缀缓存关闭，同一对照使用相同文件、tokenizer、请求数、分片和压测协议。并发、重复次数等另在workload配置，见[JSONL配置说明](../../../docs/configuration.md#jsonl-真实文本数据集)。

## 文件身份

- JSONL SHA256：`33bdbd36a8b335300974e1b07aa66b2c24c88cc4c4b55a21350aa3564a945d53`
- tokenizer.json SHA256：`8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf`
- encoding_dsv4.py SHA256：`abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6`
- 原始Parquet SHA256：`8b02abb5691b3fa1ba95f76331cadc8bb3e80195c626b4656b480d36e0c92242`

此数据集只针对上述DSV4编码与tokenizer校验。更换模型时需重新处理角色标记并核对长度，不能直接沿用这些token数。原始Parquet、一次性准备脚本和验证产物不随Git分发。
