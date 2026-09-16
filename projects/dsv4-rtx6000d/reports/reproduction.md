# 基线复现

本页说明如何复现 2026-09-15 的三配置 × C16/C32 × 三重复基线。

## 环境与固定条件

本机 10.90.1.48，8×RTX 6000D / SM120，每卡报告显存 85651 MiB，驱动 580.159.04。双路 Xeon Gold 6530，64 物理核 / 128 逻辑核，四个 NUMA 节点；GPU0–3 靠近 NUMA0，GPU4–7 靠近 NUMA2，无 NVLink。

功耗上限 600W/卡，实验未锁频或修改功耗限制。正式窗口采样温度 44–57℃、SM 频率 2422–2430MHz；未发现外部 GPU 计算进程。采样不能排除更短的干扰。具体互联见 [互联报告](interconnect.html)。

同一 `/data/models/DeepSeek-V4-Flash-0731-NVFP4` 目录和 tokenizer；TP8/PP1/DP1、EP 关闭、上下文 16384、活动容量 32、FP8 E4M3 KV、前缀缓存关闭、prefill 预算 8192、decode-only full Graph。三份 recipe 保留全部实际参数与检查规则，见 [选型报告](image-selection.md)。

## 在新 clone 上准备

安装根 README 中的 Python 依赖，提前准备固定模型和镜像。核对 target 的本机地址、模型路径、缓存路径与权限；不要改成别的权重格式或镜像后仍称为原基线。

```bash
./bench validate projects/dsv4-rtx6000d/configs/campaigns/48-dsv4-aligned-final-c16-c32.yaml
./bench plan projects/dsv4-rtx6000d/configs/campaigns/48-dsv4-aligned-final-c16-c32.yaml

# 必需：将自定义 INFO 日志 JSON 放进两个 SGLang recipe 的缓存挂载目录。
python3 scripts/prepare_logging.py projects/dsv4-rtx6000d/configs/campaigns/48-dsv4-aligned-final-c16-c32.yaml
python3 scripts/prepare_logging.py projects/dsv4-rtx6000d/configs/campaigns/48-dsv4-aligned-final-c16-c32.yaml --check

./bench preflight projects/dsv4-rtx6000d/configs/campaigns/48-dsv4-aligned-final-c16-c32.yaml
./bench run projects/dsv4-rtx6000d/configs/campaigns/48-dsv4-aligned-final-c16-c32.yaml \
  --run-root results/dsv4-rtx6000d/reproduction-01
```

使用新的结果目录；失败尝试也不覆盖。上述 prepare 只写缺失日志文件，不启动容器、不复制或清空 JIT 缓存；已存在但不同的文件会报错，避免悄悄改变日志口径。只测 vLLM 可选 `--case vllm-autotune-off`，它不需要 SGLang 日志文件。

原实验保留已有缓存，并为 SGLang off/on 从同一既有缓存种子复制到独立目录。新机器可能没有这些缓存，启动/预热时间不能与原实验直接比较；必须仍通过既定预热规则，超出上限则记录阻塞，不把冷启动样本塞入表格。历史播种脚本在本机原始产物中，不能在新机器直接执行其中的旧绝对路径。

C16 每次 64 请求、每轮预热 32；C32 每次 128、每轮预热 64。最多五轮预热、两次测量尝试，连续两轮无已知编译事件才进入测量。

## 配置来源与适用性

2026-09-14 获取或读取随 checkpoint 保留的说明，2026-09-15 核对固定镜像 CLI/源码。它们是适配起点，本基线是共同控制配置，不是官方最优参数。

| 来源 | 本轮使用范围 |
| --- | --- |
| [NVIDIA 模型卡](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4) | 读取本地 checkpoint README；在线连接当时失败，未核验最新正文。ModelOpt 0.46.0，NVFP4 routed experts 与高精度其他部分 |
| [DeepSeek inference](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main/inference) / [encoding](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main/encoding) | 模型及 thinking 参数说明；简单中文 probe 关闭 thinking |
| [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash-0731) | 原访问重定向至 DeepSeek-V4-Flash；SM120 profile 不是对该 NVFP4 变体的完整验证。禁用 SM100 专属 FP4 indexer，核对 v0.29.0 |
| [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4) | 滚动文档不能等同固定 v0.5.19；采用 SM12x CUTLASS、关闭共享专家融合，再核对实际支持 |

模型配置/tokenizer 哈希和分片大小一致；没有对全部权重逐字节哈希。完整日志、解析配置和命令保存在原实验机器的 `results/dsv4-aligned-final-20260915/run-01/`，不随 Git 分发。
