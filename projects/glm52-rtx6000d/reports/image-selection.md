# GLM-5.2-NVFP4：vLLM / SGLang 镜像选择

**推荐 `lmsysorg/sglang:v0.5.19-cu130`，作为当前 GLM-5.2-NVFP4 在 RTX 6000D ×8 上的部署和调优起点。** 双方关闭 FlashInfer autotune 时，SGLang 在 C16/C32 的输出吞吐分别领先 vLLM **6.94% / 7.27%**，TTFT 和 TPOT 也更低。

偏重吞吐和生成速度时，可开启 SGLang autotune：C16/C32 吞吐进一步提高 **5.08% / 2.77%**，TPOT 降低 **7.39% / 3.10%**。代价是显存约增加 **1.86 GiB/卡**，且 C16 Mean TTFT 从 **14.98 秒增至 16.07 秒**。若更重视 C16 的首 token 响应，可保留 OFF。C32 吞吐更高，但 TPOT 接近 C16 的两倍；两档均未达到 33 ms 的 TPOT 目标。

以下为旧仓库实测结果；迁移后的配置仅做离线验证，未重新测量。测试于 **2026-09-15 至 09-16** 在 `gpu-6000d-46` 完成。结论适用于下面的固定镜像、参数和 8K/1K 负载；vLLM autotune ON 未纳入本次有效对比。

固定镜像如下，digest 来自本机 `docker image inspect` 的 `RepoDigests`；本机 Image ID 的 SHA256 与表中 digest 相同。

| 引擎 | 镜像 | Digest |
|---|---|---|
| vLLM | `vllm/vllm-openai:v0.29.0` | `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1` |
| SGLang | `lmsysorg/sglang:v0.5.19-cu130` | `sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9` |

参数对齐以相同模型、精度、服务容量和负载为准。两引擎使用各自支持的 kernel，不要求底层实现相同。

| 项目 | vLLM | SGLang OFF / ON |
|---|---|---|
| 权重与 tokenizer | `/data/models/GLM-5.2-NVFP4` | 相同 |
| GPU | RTX 6000D ×8，SM120，PCIe，无 NVLink | 相同 |
| 并行方式 | TP=8、PP=1、DP=1；EP 关闭 | TP=8、PP=1、DP=1、EP=1（关闭） |
| 最大上下文 / 活动序列容量 | 16384 / 32 | 16384 / 32 |
| 权重量化 / KV cache | NVFP4 / `fp8_e4m3` | 相同 |
| Prefix cache | `--no-enable-prefix-caching` | `--disable-radix-cache` |
| Prefill | `--enable-chunked-prefill`；`--max-num-batched-tokens 8192` | `--chunked-prefill-size 8192`；`--enable-mixed-chunk`（未另设 `max-prefill-tokens`） |
| CUDA Graph | `FULL_DECODE_ONLY` | decode=`full`，prefill=`disabled` |
| Graph 捕获尺寸 | `[1,2,4,8,12,16,24,32]`，最大 32 | 相同 |
| 显存比例 | `--gpu-memory-utilization 0.90` | `--mem-fraction-static 0.86` |
| 实际 KV token 池 | 321,280 | 310,016（OFF / ON 相同） |
| Attention backend（日志） | `FLASHINFER_MLA_SPARSE_SM120`；MLA prefill 另记录 `FLASH_ATTN` | `flashinfer_sparse_mla` |
| MoE backend（日志） | `FLASHINFER_CUTLASS` | `flashinfer_cutlass` |
| 共享专家融合 | 当前 NVIDIA 对应融合路径未启用 | `--disable-shared-experts-fusion` |
| FlashInfer autotune | `kernel-config.enable_flashinfer_autotune=false` | OFF：`--disable-flashinfer-autotune`；ON：移除此开关 |
| Speculative / CPU offload / enforce-eager | 均未启用 | 相同 |

显存比例的含义不同，因此不强求数值一致。两边 KV 池相差约 3.63%，但均超过 `32 × (8192 + 1024) = 294912` tokens；所有 C32 测量均实际达到 32 个活动请求。SGLang ON 与 OFF 的服务参数仅差 autotune 开关。

对应配置：[vLLM OFF recipe](../configs/recipes/vllm-sm120-aligned-final-cap32.yaml)、[SGLang OFF recipe](../configs/recipes/sglang-sm120-aligned-final-cap32-noautotune.yaml)、[SGLang ON recipe](../configs/recipes/sglang-sm120-aligned-final-cap32-autotune.yaml)。

Workload 使用同一固定客户端 [vllm-bench-0.29.0.yaml](../configs/clients/vllm-bench-0.29.0.yaml)，通过流式 OpenAI `/v1/completions` 接口发送随机 token 请求。

| 条件 | 设置 |
|---|---|
| 输入 / 输出 | 固定 8192 / 1024 tokens |
| 请求调度 | `request_rate=inf`，分别限制最大并发 C16、C32 |
| 采样 | `seed=0`、`temperature=0`、`ignore_eos=true`、`range_ratio=0` |
| C16 | 每次正式 64 请求，每轮预热 32 请求，正式重复 3 次 |
| C32 | 每次正式 128 请求，每轮预热 64 请求，正式重复 3 次 |
| 预热与 gate | 每次测量前至少连续 2 轮无已知编译事件；最多 5 轮预热、2 次测量尝试 |
| JIT 观察 | 保留缓存；`FLASHINFER_JIT_VERBOSE=1`、`FLASHINFER_JIT_DEBUG=0`，TileLang 编译日志开启 |

测量的是预热后的 serving。功能探测独立进行并关闭 thinking，不计入性能结果；三组使用相同执行源码，运行期间未修改配置或源码。

下面比较三组：**vLLM OFF、SGLang OFF、SGLang ON**，OFF/ON 均指 FlashInfer autotune。性能单元格为 **三次均值 ± 样本标准差**；P95 指标也是三次各自 P95 的均值与标准差，**不是合并所有请求后的 P95**。TTFT 为首 token 延迟，TPOT 为每请求平均生成间隔，ITL 为流式 token 间隔，E2E 为端到端延迟。

**C16：每次 64 请求，三次合计 192 请求。**

| 指标 | vLLM OFF | SGLang OFF | SGLang ON |
|---|---:|---:|---:|
| 输出吞吐（tokens/s） | 178.16 ± 0.61 | 190.53 ± 1.85 | 200.20 ± 0.56 |
| 请求吞吐（requests/s） | 0.1740 ± 0.0006 | 0.1861 ± 0.0018 | 0.1955 ± 0.0005 |
| Mean TTFT（秒） | 15.80 ± 0.06 | 14.98 ± 0.47 | 16.07 ± 0.32 |
| P95 TTFT（秒） | 42.58 ± 0.03 | 40.02 ± 0.04 | 40.13 ± 0.05 |
| Mean TPOT（ms） | 74.40 ± 0.27 | 69.36 ± 1.26 | 64.23 ± 0.51 |
| P95 TPOT（ms） | 85.73 ± 0.65 | 77.64 ± 1.41 | 72.50 ± 0.10 |
| Mean ITL（ms） | 74.40 ± 0.27 | 69.36 ± 1.26 | 64.23 ± 0.51 |
| P95 ITL（ms） | 43.61 ± 0.61 | 42.46 ± 0.68 | 38.24 ± 0.49 |
| Mean E2E（秒） | 91.90 ± 0.31 | 85.94 ± 0.84 | 81.78 ± 0.23 |
| P95 E2E（秒） | 118.87 ± 0.35 | 115.49 ± 1.19 | 110.23 ± 0.37 |
| 单次测量耗时（秒） | 367.86 ± 1.25 | 343.99 ± 3.35 | 327.35 ± 0.91 |
| 吞吐变异系数（CV） | 0.34% | 0.97% | 0.28% |
| 成功 / 失败请求（三次合计） | 192 / 0 | 192 / 0 | 192 / 0 |
| 实际输入 tokens（三次合计） | 1,572,864 | 1,572,864 | 1,572,864 |
| 实际输出 tokens（三次合计） | 196,608 | 196,608 | 196,608 |

**C32：每次 128 请求，三次合计 384 请求。**

| 指标 | vLLM OFF | SGLang OFF | SGLang ON |
|---|---:|---:|---:|
| 输出吞吐（tokens/s） | 209.93 ± 0.54 | 225.20 ± 0.79 | 231.44 ± 0.47 |
| 请求吞吐（requests/s） | 0.2050 ± 0.0005 | 0.2199 ± 0.0008 | 0.2260 ± 0.0005 |
| Mean TTFT（秒） | 22.31 ± 0.26 | 20.97 ± 0.65 | 20.92 ± 0.68 |
| P95 TTFT（秒） | 81.65 ± 0.18 | 75.55 ± 0.03 | 75.41 ± 0.01 |
| Mean TPOT（ms） | 130.62 ± 0.35 | 121.57 ± 0.75 | 117.80 ± 0.92 |
| P95 TPOT（ms） | 145.32 ± 1.37 | 134.31 ± 1.13 | 130.41 ± 0.71 |
| Mean ITL（ms） | 130.62 ± 0.35 | 121.57 ± 0.75 | 117.80 ± 0.92 |
| P95 ITL（ms） | 61.69 ± 0.31 | 59.64 ± 0.55 | 55.44 ± 0.32 |
| Mean E2E（秒） | 155.94 ± 0.40 | 145.34 ± 0.51 | 141.43 ± 0.28 |
| P95 E2E（秒） | 225.19 ± 1.99 | 211.04 ± 1.13 | 206.04 ± 0.19 |
| 单次测量耗时（秒） | 624.35 ± 1.60 | 582.02 ± 2.03 | 566.33 ± 1.15 |
| 吞吐变异系数（CV） | 0.26% | 0.35% | 0.20% |
| 成功 / 失败请求（三次合计） | 384 / 0 | 384 / 0 | 384 / 0 |
| 实际输入 tokens（三次合计） | 3,145,728 | 3,145,728 | 3,145,728 |
| 实际输出 tokens（三次合计） | 393,216 | 393,216 | 393,216 |

三组每档均保留全部三次结果，无失败请求、OOM 或测量重试，日志 gate 全部通过。六组吞吐 CV 为 **0.20%–0.97%**，说明本轮重复较稳定；实验按顺序运行，不能据此保证跨日稳定或每个请求都无停顿。ITL 的统计对象与 TPOT 不同，少量长间隔也可能使 Mean ITL 高于 P95 ITL。

保留两项与部署有关的限制：SGLang OFF 首轮预热出现延迟加载 kernel、显存余量低的警告，后续正式窗口未出现；SGLang ON 存在调优未覆盖的 MoE 形状回退，C32 第二次正式测量记录了 16 条相关警告。后者采用默认执行策略，未触发重新调优或运行失败，该次结果完整保留。ON 的实测收益已经包含这些回退，不代表全部 kernel 都经过调优。

逐次指标见 [baseline-samples.csv](../data/baseline-samples.csv)，复现命令与迁移差异见 [基线复现](reproduction.md)，历史路径及文件哈希见 [provenance.json](../data/provenance.json)。这里只保留这 18 次正式测量，未混入早期短负载、未对齐配置或诊断实验。
