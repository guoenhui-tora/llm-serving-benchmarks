# gpu-6000d-47：vLLM / SGLang TP4 对齐比较

**本节点、同机双服务、C32 / 8192 输入 / 1024 输出下，vLLM 输出吞吐比 SGLang 高 15.40%，三次重复稳定。** vLLM 为 **676.05 ± 3.22 tok/s**，SGLang 为 **585.83 ± 2.61 tok/s**；吞吐 CV 分别为 **0.48% / 0.45%**。平均 TTFT、TPOT 和端到端延迟也更低，但本轮没有质量评测或业务 SLA，不能推断所有场景的引擎排名。

实验日期为 **2026-09-17（Asia/Shanghai）**。范围仅为 gpu-6000d-47：GPU0–3 vLLM TP4、GPU4–7 SGLang TP4，同时测量；每实例 C32，整机在途容量 C64。不把它当作独占基线，也不与 48 节点的 TP8 C32 直接推算拓扑收益。跨 GPU 组和跨节点结论仍需其他节点结果支持。

## 性能与稳定性

以下全部来自同一次服务启动、冻结配置和源码的 **6 次有效测量**。每侧每次 128 个成功请求，0 失败；三次共 384 请求、3,145,728 输入 tokens、393,216 输出 tokens。两侧合计 **768 成功、0 失败**。

| 指标 | vLLM TP4 | SGLang TP4 |
| --- | ---: | ---: |
| R1 / R2 / R3 输出 tok/s | 675.62 / 673.06 / 679.46 | 587.62 / 582.83 / 587.04 |
| 输出 tok/s，均值 ± 样本标准差 | **676.05 ± 3.22** | **585.83 ± 2.61** |
| 吞吐 CV / 极差÷均值 | **0.48% / 0.95%** | **0.45% / 0.82%** |
| 平均 TTFT，秒 | 7.172 ± 0.108 | 7.451 ± 0.005 |
| 平均 TTFT CV | 1.50% | 0.06% |
| P95 TTFT，秒 | 22.802 ± 0.027 | 25.721 ± 0.013 |
| 平均 TPOT，ms | 40.310 ± 0.142 | 47.317 ± 0.244 |
| 平均 TPOT CV | 0.35% | 0.52% |
| P95 TPOT，ms | 44.433 ± 0.208 | 51.542 ± 0.230 |
| 平均端到端延迟，秒 | 48.408 ± 0.229 | 55.856 ± 0.250 |
| P95 端到端延迟，秒 | 66.569 ± 0.723 | 78.310 ± 0.178 |

标准差使用三次样本标准差；P95 行是每次 P95 的均值与标准差，不是合并请求分位数。吞吐来自客户端完整计时窗口，包含 prefill 和 decode，不是服务日志瞬时生成速率。ITL 等完整字段见[逐次 CSV](../data/tp4-47-samples.csv)；资源统计与 KV 分项记录保留于下述本机产物。三次重复只反映本次短期稳定性。

运行前的诊断标准为：吞吐 CV > 3% 或极差/均值 > 5% 视为明显波动；平均 TTFT/TPOT CV > 10% 单独标注。本轮全部低于这些标准，没有补跑或挑选最快的正式样本。

## 共同参数与 KV cache

固定同一只读模型目录 `/data/models/DeepSeek-V4-Flash-0731-NVFP4`、tokenizer、配置哈希和分片大小；没有逐字节计算全部权重哈希。固定 vLLM 0.29.0 与 SGLang 0.5.19-cu130 镜像，image ID 与原基线一致；服务和客户端均使用本地镜像，不拉取升级。

| 条件 | vLLM | SGLang |
| --- | --- | --- |
| GPU / 端口 | 0–3 / 31047 | 4–7 / 32047 |
| TP / PP / DP / EP | 4 / 1 / 1 / off | 4 / 1 / 1 / ep-size=1 |
| 最大上下文 / 活动容量 | 16384 / 32 | 16384 / 32 |
| KV 配置 | `fp8_e4m3`；实际 `fp8_ds_mla` | `fp8_e4m3`；DSV4 特殊池 |
| Prefix / radix cache | 显式关闭 | 显式关闭 |
| FlashInfer autotune | 关闭 | 关闭 |
| Prefill 预算 | 8192，chunked prefill | 8192，mixed chunk 开启 |
| Decode Graph | FULL_DECODE_ONLY | full；prefill disabled |
| Graph 捕获尺寸 | [1,2,4,8,12,16,24,32] | 相同 |
| 显存比例 | 0.90 | 0.89 |
| KV 字节预算，每卡 | **32.73 GiB（运行日志）** | **约 32.43 GiB（实际池尺寸重建）** |
| 页面 / indexer | block-size=256，FP4 indexer off | page-size=256，FP4 indexer off |
| Attention / MoE | FLASHINFER_MLA_SPARSE_DSV4 / FLASHINFER_CUTLASS | DeepseekV4AttnBackend / flashinfer_cutlass |

没有 speculative decoding 或 CPU offload。SGLang 关闭 shared-experts fusion；vLLM 沿用固定实现。两侧保留正常 CUDA Graph 和引擎可用优化，调度和 kernel 实现仍不同。

**KV 预算差约 0.93%，满足本轮预先设定的 1% 对齐目标。** 0.90 / 0.89 是从基线继承后经本机 TP4 实际核验的结果，不因比例接近就认定容量相同，也不推广到其他模型或节点。

SGLang 日志没有直接给出这个 DSV4 池的总字节数，因此按固定镜像源码和实际分配尺寸计算，包含主 KV、indexer、FP32 压缩状态及分页 padding，排除小型映射表和分配器余量。实际尺寸为 `swa=443904, c4=1110016, c128=34688, c4_state=27744, c128_state=4224`；前 43 层为 2 个无压缩层、21 个 C4 层、20 个 C128 层。总张量预算 **34,819,194,560 bytes = 32.4279 GiB/卡**，分项字节数保存于本机统计 JSON。主 KV 的 FP8 非旋转部分、BF16 RoPE、scale 与压缩状态不能按单一“1 byte/token”估算。

vLLM 启动提示的逻辑 token 数与 SGLang 的 full/SWA/压缩池不是同一口径，不能直接相除比较容量。正式负载实际达到 C32；vLLM `/metrics` 采样的累计抢占为 0，双方完整日志没有实际抢占/retract、OOM 或 traceback 事件。两次重启后的实际池尺寸均与首次一致。

以上为本轮实测的对照参数，不是官方最优配置。完整 recipe、workload 和实际 CLI 保留于下述本机配置与正式结果目录，不随本次结果归档分发。

## JIT 与预热证据

保留磁盘 JIT 缓存，预热和正式结果独立保存。每轮预热 64 请求，最多 5 轮，连续两轮安静后才测量。最初计划每对最多两次测量尝试；发生观测问题后新建 `jit-audit` workload，将最终每对上限收紧为 **一次**。最终三对都在第一次尝试通过。

前两轮只作诊断，未进入上表：

| 运行 | 证据与处理 |
| --- | --- |
| run-01，INTERRUPTED | SGLang 首轮预热日志事件为 0，但计时窗口内生成 27 个 cubin、3 个 so；vLLM 首次测量有 8 条实际 TileLang compile begin。保存反例，阶段边界停止并清理后，增加持久化编译产物前后快照检查。 |
| run-02，FAIL | vLLM 测量有 8 条 verbose TileLang 告警，但无 compile begin/complete、无编译产物变化。固定镜像源码证实 JITImpl 包装器在查询磁盘缓存前，仅根据进程 `_kernel_cache` miss 就告警。保留 warning，并精确区分该情况；真实编译正例仍被 gate 捕获。 |
| run-03，PASS | 同一次启动完成三对测量。vLLM 首次用 3 轮预热，其余各次及 SGLang 均用 2 轮；全部有效测量的阻塞事件、编译产物变化、进程 cache-miss warning 都为 **0**。 |

因此，**最终 6 个测量窗口没有依靠忽略 cache-miss warning 才通过**。分类仅适用于固定 vLLM 0.29.0 镜像的 verbose cache_key/runtime_shapes 告警格式，并要求显式挂载的 TileLang 缓存无编译产物变化、日志无真实编译开始或完成事件；真实编译仍阻止验收。规则与回归记录保留在本机执行材料中。自定义 SGLang INFO、FlashInfer/TileLang 日志和 vLLM JIT verbose 全程保留；未隐藏 fallback warning。

日志及挂载内编译产物检查不能覆盖所有可能的 JIT 机制或挂载外缓存，因此同时检查了三次吞吐、延迟和资源状态。没有通过放宽 gate 或无限重试获得稳定数据。相关正反例、原始 warning、编译产物时间和固定镜像源码摘录保留于本机工作区。

## 同步、CPU 和硬件状态

服务/客户端使用独立 CPU 集合。vLLM 服务为 0–11,64–75，客户端为 12–15,76–79，内存节点 NUMA0；SGLang 服务为 32–43,96–107，客户端为 44–47,108–111，内存节点 NUMA2。每侧各 12 个物理核服务、4 个物理核客户端，记录 Docker cpuset 和实际进程亲和性。

每次先顺序预热两侧，再让两个客户端在原始 benchmark 计时前通过共享屏障开始。原始请求循环、采样和计时区间不变。

| 重复 | 起点偏差 ms | 重叠秒数 | SGLang 后结束的尾段秒数 |
| --- | ---: | ---: | ---: |
| R1 | 4.72 | 194.00 | 29.06 |
| R2 | 3.72 | 194.74 | 30.15 |
| R3 | 4.00 | 192.90 | 30.37 |

尾段 vLLM 服务保持存活但空闲；未在一侧仍测量时启动另一侧预热或编译。双方完整窗口长度不同，因此不能把两侧 tok/s 简单相加当作本轮整机窗口吞吐。

正式窗口每 2 秒采样：

| 资源 | vLLM GPU0–3 | SGLang GPU4–7 |
| --- | ---: | ---: |
| 温度范围 ℃ | 38–55 | 48–60 |
| SM 频率均值 MHz（范围） | 2413（2287–2422） | 2422（2415–2430） |
| 平均 / 最高采样功耗 W/卡 | 248.32 / 270.29 | 268.51 / 292.32 |
| 最高采样显存 MiB/卡 | 79019 | 79811 |
| 服务 CPU 集合平均忙碌率 | 68.06% | 68.49% |
| 客户端 CPU 集合平均忙碌率 | 59.41% | 55.16% |

GPU 功耗限制保持 600W，未改功耗或锁频。CPU 指标是 `/proc/stat` 中整个 CPU 集合的区间统计，包含该集合上的其他宿主活动；不是仅该进程占用率，也不足以排除单核或短暂瓶颈。两侧给定 CPU 预算相同，未做扩核敏感性测试。2 秒采样不能排除更短的干扰。

保留的限制包括 PCIe 四卡部分 all-reduce / multicast 优化不可用，以及 SGLang FP8 KV scale 默认值警告。统一客户端对 SGLang 可选 `/metrics` 的 404 不属于生成请求失败。结果适用于当前固定引擎、硬件分组和共同资源预算。

## 归档范围与原始产物

本次 Git 归档仅包含项目 README、本报告和[逐次 CSV](../data/tp4-47-samples.csv)，用于审阅参数对齐、性能和稳定性。执行脚本和本轮配置仅本机保留，因此仅凭这三个文件不能直接运行同一套同步实验。原有 TP8 配置未变；基础环境准备可参考[基线复现](reproduction.md)，其中的 TP8 campaign 不能替代本轮 TP4 同步测量。

以下路径均相对于仓库根目录，**仅 gpu-6000d-47 本机存在，不随 Git 分发**：

- 正式结果：`experiments/dsv4-rtx6000d/results/47-tp4-paired-20260917-03/`。
- 两轮诊断：同目录下 `47-tp4-paired-20260917-01/`、`47-tp4-paired-20260917-02/`。
- 本轮完整配置：`experiments/dsv4-rtx6000d/configs/`，campaign 为 `campaigns/47-dsv4-vllm-tp4-c32-jit-audit.yaml` 和 `campaigns/47-dsv4-sglang-tp4-c32-jit-audit.yaml`，CPU 绑定记录为 `paired-bindings-47.json`。
- 配置核验、固定镜像源码摘录、KV 分项重建及测试记录：`experiments/dsv4-rtx6000d/reports/`。
- 执行辅助代码、测试、说明和统计 JSON 备份：`experiments/dsv4-rtx6000d/reports/47-tp4-local-support-20260917/`，按原仓库相对路径保存；统计 JSON 为其中的 `projects/dsv4-rtx6000d/data/tp4-47-summary.json`。
- 持久化编译缓存：`experiments/dsv4-rtx6000d/cache/`。

正式结果保存解析配置、启动/客户端命令、镜像与模型身份、执行前源码快照、Git 状态、完整日志、每轮缓存快照、GPU/CPU 时间序列及计时文件。实验结束后，仅补充从执行前快照重建的 manifest 源码指纹，以兼容标准报告；不改测量数据或验收状态。执行时的源码以该运行目录内的 `source/` 快照为准，本机辅助代码备份包含随后补齐的元数据写入逻辑。

实验完成时，58 项回归测试通过，覆盖同步入口、真实编译/缓存 miss 分类、成对拒绝重试、清理期间持锁和报告兼容性；两份 campaign 已离线 validate/plan。这些验证针对本机执行材料，不表示本次三文件归档新增了公共执行入口。新机器或冷缓存可能超过既定预热预算，应保留失败结果；不能保证任意冷启动都直接完成三次测量。不要清空已有 JIT 缓存。
