# DeepSeek V4：vLLM / SGLang 镜像选型

**选择 `vllm/vllm-openai:v0.29.0`，关闭 FlashInfer autotune，作为后续调优基线。** 在本机 8×RTX 6000D、8192 输入 / 1024 输出的对齐测试中，vLLM 在 C16、C32 的吞吐和延迟均优于两组 SGLang。C16 输出吞吐约 **533 tok/s**，C32 约 **651 tok/s**；更重视请求延迟选 C16，更重视总吞吐选 C32。

SGLang 开启 autotune 后，C16 吞吐提高 4.31%，C32 下降 1.10%，没有改变本轮推荐。此结论适用于下面的硬件、负载与 recipe，不代表所有场景下的引擎排名。

实验日期：2026-09-15。模型：`/data/models/DeepSeek-V4-Flash-0731-NVFP4`。硬件：8×RTX 6000D（SM120，PCIe，无 NVLink），驱动 580.159.04。

## 固定镜像与 recipe

下列 digest 已与本地 `RepoDigests` 核对，本机 image ID 也与其一致。

| 引擎 | 镜像 | Digest |
| --- | --- | --- |
| vLLM | `vllm/vllm-openai:v0.29.0` | `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1` |
| SGLang | `lmsysorg/sglang:v0.5.19-cu130` | `sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9` |

共同条件：同一权重与 tokenizer，TP=8 / PP=1 / DP=1 / EP=1，最大上下文 16384，活动请求容量 32，KV cache 使用 `fp8_e4m3`，内部 seed=0；不启用 speculative decoding 或 CPU offload。

| 关键参数 | vLLM | SGLang |
| --- | --- | --- |
| 关闭 EP | `--no-enable-expert-parallel` | `--ep-size 1` |
| FlashInfer autotune | `--kernel-config '{"enable_flashinfer_autotune":false}'` | off：`--disable-flashinfer-autotune`；on：去掉该开关 |
| 关闭前缀缓存 | `--no-enable-prefix-caching` | `--disable-radix-cache` |
| Prefill 预算 | `--max-num-batched-tokens 8192 --enable-chunked-prefill` | `--chunked-prefill-size 8192 --max-prefill-tokens 8192 --enable-mixed-chunk` |
| Decode CUDA Graph | `cudagraph_mode=FULL_DECODE_ONLY` | `--cuda-graph-backend-decode full` |
| Prefill CUDA Graph | 上述模式不捕获 prefill | `--cuda-graph-backend-prefill disabled` |
| Graph 捕获尺寸 / 最大值 | `[1,2,4,8,12,16,24,32]` / 32 | 相同 |
| 显存比例 | `--gpu-memory-utilization 0.90` | `--mem-fraction-static 0.89` |
| 实际 KV 分配 | 51.75 GiB / 卡 | 51.38 GiB / 卡 |
| Attention | `FLASHINFER_MLA_SPARSE_DSV4` | `DeepseekV4AttnBackend` |
| MoE | `--moe-backend auto`，实际选择 `FLASHINFER_CUTLASS` NVFP4 路径 | `--moe-runner-backend flashinfer_cutlass` |
| KV / 滑窗适配 | `--block-size 256`，`use_fp4_indexer_cache=false` | 自动 page size 256，`--swa-full-tokens-ratio 0.1` |
| 共享专家 | 固定实现未拼接融合 routed/shared 权重 | `--disable-shared-experts-fusion` |

vLLM 的 Graph 配置通过 `--compilation-config` 传入，字段为 `cudagraph_mode`、`cudagraph_capture_sizes`、`max_cudagraph_capture_size`；SGLang 使用 `--cuda-graph-bs-decode` 和 `--cuda-graph-max-bs-decode` 设置捕获尺寸。

**显存按实际 KV 字节预算对齐，相差约 0.7%。** 两个比例参数的语义不同，KV / 滑窗布局和调度实现也不同，因此不要求比例或逻辑 token 池计数相同。两边使用同一 PCIe 拓扑、GPU0–7 和容器 `SYS_NICE` 权限；SGLang 日志未再出现旧的 NUMA 亲和性权限警告。

完整参数以 [vLLM recipe](../configs/recipes/vllm-sm120-aligned-final-cap32.yaml)、[SGLang autotune off recipe](../configs/recipes/sglang-sm120-aligned-final-cap32-noautotune.yaml)、[SGLang autotune on recipe](../configs/recipes/sglang-sm120-aligned-final-cap32-autotune.yaml) 为准。SGLang 两组仅 autotune 服务开关不同。它们是已实测的对照配置，不是官方最优配置。

## 测试负载

固定使用 [vLLM 0.29.0 压测客户端](../configs/clients/vllm-bench-0.29.0.yaml)，通过流式 `/v1/completions` 接口发送 random token 请求。

| 最大并发 | 输入 / 输出 tokens | 正式请求数 / 次 | 有效重复 | 每轮预热请求数 |
| --- | --- | --- | --- | --- |
| C16 | 8192 / 1024 | 64 | 3 | 32 |
| C32 | 8192 / 1024 | 128 | 3 | 64 |

统一 `seed=0`、`temperature=0`、`ignore_eos=true`、`range_ratio=0`、`request_rate=inf`，通过最大并发限制在途请求。保留 JIT 缓存，衡量预热后的 serving。vLLM 开启 `--jit-monitor-verbose`，SGLang 使用自定义 INFO 日志及 JIT 输出；至少连续两轮无已知编译日志事件才测量，最多五轮预热、两次测量尝试。

## 性能结果

下表仅使用本轮 **18 次通过验收的正式测量**，未合并早期不同参数的结果。输出吞吐为三次重复的“均值 ± 样本标准差”；延迟列为三次测量对应指标的均值。**P95 TTFT 是三次各自 P95 的均值，不是合并请求后的 P95。**

| 并发 | 引擎 / autotune | 输出吞吐 tok/s ↑ | Mean TTFT 秒 ↓ | P95 TTFT 秒 ↓ | Mean TPOT ms ↓ | Mean E2E 秒 ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| C16 | vLLM / off | 532.57 ± 1.36 | 7.07 | 13.84 | 23.14 | 30.74 |
| C16 | SGLang / off | 461.74 ± 1.32 | 7.55 | 16.65 | 27.28 | 35.45 |
| C16 | SGLang / on | 481.63 ± 0.67 | 8.31 | 16.61 | 25.10 | 33.99 |
| C32 | vLLM / off | 651.35 ± 1.22 | 8.80 | 26.49 | 40.52 | 50.25 |
| C32 | SGLang / off | 553.36 ± 1.43 | 10.48 | 31.36 | 47.57 | 59.15 |
| C32 | SGLang / on | 547.25 ± 1.58 | 10.29 | 31.36 | 48.41 | 59.81 |

TTFT 为首 token 延迟，TPOT 为首 token 后平均每个输出 token 的耗时，E2E 为单请求端到端延迟。本负载固定输出 1024 tokens，requests/s = 输出 tok/s ÷ 1024，故不另设重复列。

vLLM 相对 SGLang autotune off 的吞吐优势为 **C16 +15.34%、C32 +17.71%**；相对 SGLang autotune on 为 **+10.58%、+19.02%**。六组吞吐重复变异系数均低于 0.3%，差距明显大于本轮吞吐波动。延迟波动需单独看：SGLang on 的 C32 平均 TTFT 为 10.29 ± 0.82 秒，不能据其均值的小幅下降认定 autotune 改善了 TTFT。

vLLM 从 C16 到 C32 吞吐提高 22.30%，平均请求耗时也从 30.74 秒增至 50.25 秒。没有业务延迟目标时，应保留这两档作为吞吐与延迟的取舍。

每个配置的 C16 三次合计 **192 成功 / 0 失败**，实际输入 / 输出为 **1,572,864 / 196,608 tokens**；C32 为 **384 / 0**，实际输入 / 输出为 **3,145,728 / 393,216 tokens**。本轮无 OOM 或超时；一份触发 JIT gate 的 vLLM C32 尝试已剔除并保留证据，未计入表格。

本轮可作为该固定负载的性能基线；未做模型质量评测，也未验证长期线上稳定性。SGLang autotune 日志仍有部分形状未覆盖的 fallback，开启 autotune 不等于所有 kernel 都获得优化。

完整逐次指标见 [CSV](../data/baseline-samples.csv)，复现步骤见 [复现说明](reproduction.md)，已知异常见 [排查记录](lessons.md)。
