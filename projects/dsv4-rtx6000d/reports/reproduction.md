# 当前基线与历史复现

**后续单机研究以双TP4为整机基线；单TP4用于四卡独占对照。** 两者服务参数相同，只改变部署数量、GPU/CPU绑定和客户端分流。以下启动命令由当前解析配置导出，仅换了容器名称/所属标签，未手工维护另一套参数。

模型为 `/data/models/DeepSeek-V4-Flash-0731-NVFP4`，同一权重/tokenizer；镜像 `vllm/vllm-openai:v0.29.0`，固定image ID为 `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`，命令直接使用本地ID，不拉取镜像。上下文16384，每实例活动容量32，NVFP4权重、FP8 E4M3 KV；FlashInfer autotune和前缀缓存关闭，V2/async/Graph保留。服务启动命令本身不构成性能结果，必须同时固定下文负载、绑定和验收协议。

## TP4完整启动命令

这是48节点后四卡示例，模型和缓存路径来自target。运行前确认路径、镜像和资源；命令中的既有缓存挂载目录需要存在，不能通过清空缓存复现。单TP4采用端口31249，与双实例后侧统一；历史独占TP4使用31248。

```bash
docker run \
  -d \
  --pull never \
  --name dsv4-tp4-rear \
  --label io.serving-bench.run=manual-dsv4 \
  --network host \
  --ipc host \
  --gpus '"device=4,5,6,7"' \
  -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro \
  -v /home/enhui/.cache/serving-bench/vllm/d19496fcae0f32e092d6:/root/.cache:rw \
  --cpuset-cpus 32-47 \
  --cpuset-mems 2 \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --cap-add SYS_NICE \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_DEBUG=WARN \
  -e TILELANG_CACHE_DIR=/root/.cache/tilelang \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model \
  --served-model-name deepseek-v4-flash \
  --host 127.0.0.1 \
  --port 31249 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --no-enable-prefix-caching \
  --no-enable-expert-parallel \
  --enable-chunked-prefill \
  --jit-monitor-verbose \
  --async-scheduling \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 256 \
  --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' \
  --moe-backend auto \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,12,16,24,32],"max_cudagraph_capture_size":32}' \
  --seed 0 \
  --distributed-executor-backend mp
```

本例缓存目录来自保留的双TP4 recipe；合并前独占TP4的缓存目录仍保留在本机，不复制或删除。新机器按target配置准备自己的目录。正式实验建议使用下方runner入口，它负责功能检查、预算、日志和清理；手工启动示例只解释部署参数。

## 双实例及候选差异

双TP4分别运行上述服务两份，使用独立容器和端口；前侧只需替换下表中的资源参数。两份都使用同一tp4 recipe，不启用内部DP。

| 资源 | 前四卡 | 后四卡／单TP4 |
| --- | --- | --- |
| GPU | 0,1,2,3 | 4,5,6,7 |
| 容器名示例 | dsv4-tp4-front | dsv4-tp4-rear |
| 服务CPU／内存NUMA | 0–15／0 | 32–47／2 |
| 客户端CPU／内存NUMA | 16–19／1 | 48–51／3 |
| API端口 | 31248 | 31249 |

候选保留相同镜像、精度、上下文、绑定规则；相对上述TP4命令的变化如下，完整命令仍由各campaign的 `bench plan` 生成：

| 候选 | 每实例参数变化 |
| --- | --- |
| 双TP2×PP2 | `--tensor-parallel-size 2 --pipeline-parallel-size 2` |
| 双TP2×DP2 EP on | TP=2、DP=2、`--data-parallel-size-local 2`；用 `--enable-expert-parallel` 替换off开关；每DP rank活动容量16，Graph尺寸 `[1,2,4,8,12,16]`、最大16 |

DP候选每rank仍保留8192 prefill预算，整机上限随调度器数量变化，不能将其视为只改变通信。候选历史性能见[项目汇总](../README.md)，新协议尚未实测。

## 压测条件与运行入口

客户端固定vLLM 0.29.0，流式 `/v1/completions`；输入8192、输出1024，seed=0、temperature=0、ignore_eos=true、range_ratio=0、request_rate=inf，不使用跨请求前缀缓存。

| 部署与全局并发 | 每轮总请求 | 每实例并发／请求 | 当前接纳规则 |
| --- | ---: | --- | --- |
| 单TP4 C32 | 128 | 32／128 | 固定3轮，保留事件标记 |
| 双实例 C32 | 128 | 16／64 | 固定3轮，任一侧事件均标记 |
| 双实例 C64 | 256 | 32／128 | 同上 |

当前workload采用quick，每档先1轮2C请求预热，再固定3轮正式测量，全部保留并标记事件；每档预算上限7200秒，达到固定轮数即结束，不等待预算耗尽。预热与正式输入输出长度相同，预热不纳入统计。客户端并发按表切换，服务端容量保持不变。双实例由runner同步并交错分片同一全局请求集，整机吞吐按测量窗口并集计算，不能用两个独立客户端的吞吐相加替代。

```bash
# 仅首次创建工作区时复制；已有configs不要覆盖。
mkdir -p experiments/dsv4-next/results experiments/dsv4-next/reports
cp -a projects/dsv4-rtx6000d/configs experiments/dsv4-next/configs
./bench validate experiments/dsv4-next/configs/campaigns/dual-tp4.yaml
./bench plan experiments/dsv4-next/configs/campaigns/dual-tp4.yaml
./bench preflight experiments/dsv4-next/configs/campaigns/dual-tp4.yaml
./bench run experiments/dsv4-next/configs/campaigns/dual-tp4.yaml \
  --run-root experiments/dsv4-next/results/dual-tp4-01
```

单TP4改用 `tp4.yaml`；两项候选用 `dual-candidates.yaml`，可用 `--case` 只选一项。在其他节点先修改本地target地址并核实CPU/GPU映射。完整客户端命令由plan输出，实际启动命令和每轮客户端命令由runner保存，报告引用实测产物，不手工维护bash和YAML两套标准答案。

## 历史结果边界

2026-09-15的TP8镜像对比、09-17四卡拓扑和09-18八卡矩阵继续保留报告及CSV，不把这些结果改写为新协议实测。旧源码与配置见[归档快照](https://github.com/guoenhui-tora/llm-serving-benchmarks/tree/4642443b1274251b6a47f0d2742e0e59e2dccc95/projects/dsv4-rtx6000d/configs)及各报告记录的执行commit；精确复现还需采用该批补测的预热修改和缓存条件。

TP8原环境：8×RTX6000D/SM120，每卡85651 MiB、驱动580.159.04，Xeon Gold 6530双路64物理核/128线程、4个NUMA节点、PCIe无NVLink。功耗上限600W/卡，未锁频；正式采样温度44–57℃、SM频率2422–2430MHz。八卡拓扑批次的背景负载另见各节点报告，不以原TP8资源状态代替。互联见[测量报告](interconnect.html)。

原TP8验收为每次测量前至少连续两轮无事件、最多五轮预热和两次尝试；C16正式64请求/预热32，C32正式128/预热64。当前配置已迁移，不能直接重跑后声称精确复现旧验收过程。

## 配置来源与适用性

2026-09-14 获取或读取随 checkpoint 保留的说明，2026-09-15 核对固定镜像 CLI/源码。它们是适配起点，本基线是共同控制配置，不是官方最优参数。

| 来源 | 本轮使用范围 |
| --- | --- |
| [NVIDIA 模型卡](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4) | 读取本地 checkpoint README；在线连接当时失败，未核验最新正文。ModelOpt 0.46.0，NVFP4 routed experts 与高精度其他部分 |
| [DeepSeek inference](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main/inference) / [encoding](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main/encoding) | 模型及 thinking 参数说明；简单中文 probe 关闭 thinking |
| [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash-0731) | 原访问重定向至 DeepSeek-V4-Flash；SM120 profile 不是对该 NVFP4 变体的完整验证。禁用 SM100 专属 FP4 indexer，核对 v0.29.0 |
| [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4) | 滚动文档不能等同固定 v0.5.19；采用 SM12x CUTLASS、关闭共享专家融合，再核对实际支持 |

模型配置/tokenizer 哈希和分片大小一致；没有对全部权重逐字节哈希。完整日志、解析配置和命令保存在原实验机器的 `results/dsv4-aligned-final-20260915/run-01/`，不随 Git 分发。
