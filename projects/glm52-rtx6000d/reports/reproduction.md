# 基线复现

本页复现三套配置：vLLM autotune off、SGLang autotune off/on；每套 C16 / C32 各三次。性能结果来自旧仓库两轮实测，新仓库将它们整理为一个顺序执行的 campaign，未重新测量。

## 环境与固定条件

本机 `gpu-6000d-46`（10.90.1.46），8×RTX6000D / SM120，每卡 85651 MiB，驱动 580.159.04。双路 Xeon Gold 6530、128 逻辑 CPU、4 个 NUMA 节点；GPU0–3 靠近 NUMA0，GPU4–7 靠近 NUMA2，PCIe Gen5×16，无 NVLink。功耗上限 600 W/卡，实验未锁频或改变宿主功耗及 NUMA 全局设置。

target 保留 `SYS_NICE`。SGLang 调度进程的 CPU 允许集合与 GPU 所在 NUMA 对应；未完整验证所有线程及内存页放置，也未单独采集 vLLM 每个 worker 的亲和性。正式采样最高温度不超过 60℃，未观察到外来 GPU 计算任务；低频遥测不能排除瞬时 CPU/I/O 干扰。

固定权重 `/data/models/GLM-5.2-NVFP4` 和同目录 tokenizer；镜像及实际参数见 [选型报告](image-selection.md)。C16 每次正式 64 请求、每轮预热 32；C32 分别为 128、64。每次测量前至少连续两轮无已知编译事件，最多五轮预热、两次测量尝试；这些条件均保留在 workload 中。

## 准备与运行

安装根 README 中的 Python 依赖，提前准备固定模型和镜像。核对 target 的本机地址、模型路径、缓存权限以及 GPU 占用。从新仓库根目录执行：

```bash
./bench validate projects/glm52-rtx6000d/configs/campaigns/46-glm52-aligned-final-c16-c32.yaml
./bench plan projects/glm52-rtx6000d/configs/campaigns/46-glm52-aligned-final-c16-c32.yaml

# 必需：准备两个 SGLang recipe 的容器可见 INFO 日志文件。
python3 scripts/prepare_logging.py projects/glm52-rtx6000d/configs/campaigns/46-glm52-aligned-final-c16-c32.yaml
python3 scripts/prepare_logging.py projects/glm52-rtx6000d/configs/campaigns/46-glm52-aligned-final-c16-c32.yaml --check

./bench preflight projects/glm52-rtx6000d/configs/campaigns/46-glm52-aligned-final-c16-c32.yaml
./bench run projects/glm52-rtx6000d/configs/campaigns/46-glm52-aligned-final-c16-c32.yaml \
  --run-root results/glm52-rtx6000d/reproduction-01
```

`--run-root` 必须是尚不存在的新目录。可用 `--case vllm-autotune-off`、`--case sglang-autotune-off` 或 `--case sglang-autotune-on` 选择单组。runner 顺序启动服务，分别完成 health、models、关闭 thinking 的中文探测、预热及测量。

prepare 只准备缺失的日志 JSON，不启动服务、不清空 JIT 缓存；已有文件内容不一致时会报错。本次迁移仅在临时目录验证 prepare，没有写入正式缓存或启动 preflight 容器。

## 迁移边界与缓存

逐项核对历史 `resolved.json`：target、model、runtime、client、workload、推理 flags/options 及探测参数保持一致。GLM 的 SGLang 未另设 `max-prefill-tokens`；不要直接照抄 DSV4 的额外参数或显存比例。

仅做以下适配：

- recipe/workload 文件路径扁平化，三个 case 改为明确的 autotune 名称，合并为一个 campaign；原 recipe/workload ID 保留。
- SGLang 日志路径从 `/root/.cache/bench-logging/sglang-jit-info.json` 改为新 helper 支持的 `/root/.cache/logging/sglang-jit-info.json`，JSON 内容不变。
- 旧执行器内置的 `\bTileLang begins to compile kernel\b` 在新执行器中通过各 recipe 的 `checks.compilation` 补齐；有效事件正则集合保持一致，未放宽 gate。

recipe 全部内容参与缓存指纹，因此三个新 recipe 会使用新的缓存目录。旧目录完整保留，旧/新目录映射见 [provenance.json](../data/provenance.json)。本次没有复制或删除缓存。

原实验复制兼容的非 autotune JIT 缓存到各 recipe 独立目录，SGLang ON 从 OFF 复制非调优缓存，再生成自己的调优记录。新配置首次运行可能没有这些缓存，启动和预热会更慢；仍须通过既定 gate，不能把冷启动结果混进旧表格。需要复用时，按 provenance 映射核对兼容缓存并复制到独立目录，保留来源，不覆盖已有内容；旧 `bench-logging` 路径不能替代 prepare 步骤。

## 来源与历史证据

官方说明是适配起点，本项目配置是固定镜像上的对照基线，不是官方最优参数。

| 来源 | 访问日期与适用范围 |
| --- | --- |
| [NVIDIA 模型卡](https://huggingface.co/nvidia/GLM-5.2-NVFP4) | 2026-09-14 读取本地 checkpoint README；在线连接失败，未核验最新正文。卡片提及 latest/dev-glm52-nvfp4、transformers≥5.3，不能视为对两个固定镜像的保证 |
| [vLLM GLM-5.2 recipe](https://recipes.vllm.ai/zai-org/GLM-5.2) | 2026-09-14，文档标注 v0.23.0+；另行核对固定 v0.29.0 CLI 和 SM120 实现 |
| [SGLang GLM-5.2 cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2) | 2026-09-14，滚动文档；固定 v0.5.19-cu130 单独核对，使用 `modelopt_fp4`、CUTLASS MoE 并关闭共享专家融合 |

历史 Git commit：`94475c74def011d4dd6d03ffc1812118950d6163`；三组的 runner 源码指纹均为 `9d48bab586a4b45254e1ad10756766ccbe4b4e4a90b1f863a431db3fec643495`。当时有本地事件正则及测试修改，commit 不能单独代表完整执行源码；完整 diff、源码/配置归档和哈希在原始目录中。

以下绝对路径**仅在 46 节点本机可用，不随 Git 分发**，位于新仓库的外层旧目录：

| 内容 | 本机位置 |
| --- | --- |
| vLLM OFF / SGLang OFF 原始产物 | `/home/enhui/llm-serving-benchmarks/results/glm52-jit-fair-repeat-20260915/` |
| SGLang ON 原始产物 | `/home/enhui/llm-serving-benchmarks/results/glm52-sglang-jit-autotune-on-20260916/` |
| OFF 完整分析 | `/home/enhui/llm-serving-benchmarks/reports/glm52-jit-fair-repeat-20260915/` |
| ON 完整分析 | `/home/enhui/llm-serving-benchmarks/reports/glm52-sglang-jit-autotune-on-20260916/` |

各原始目录的 `run-01/cases/<历史 case>/` 保留 `resolved.json`、`argv.json`、`command.sh`、`environment.json`、`host.json`、完整 `server.log` 及独立的预热/测量结果。历史 case、逐次路径和关键文件 SHA256 已写入项目 provenance；模型身份校验来自 metadata、tokenizer 与分片信息，不是全部权重逐字节哈希。

旧运行的容器已清理，结果和缓存保留。本次只迁移项目配置、精选报告和小型数据；重构后的 runner 尚未用这些配置做新的 GPU 实测。
