# vLLM 与 SGLang 如何公平对比

先声明比较目标：**共同控制参数的基线**用于判断相同主要条件下的差异；**分别调优的部署方案**用于选择实际可用组合。两者都有效，但不能把不同优化条件的结果写成仅由引擎名称造成的差异。

## 先对齐共同条件

同一模型和 tokenizer、硬件数量、上下文、输出长度、KV 精度、客户端、流式接口、采样参数、请求数、并发和重复次数。明确是否启用前缀缓存、投机解码、CPU offload、EP 和异步调度。

固定镜像逐项检查 CLI 与实际日志，不用最新文档的默认值代替旧镜像行为。vLLM 的 V1 engine 日志不代表一定使用 V1 Model Runner；runner 和 async 状态也不能仅从是否显式传参判断。

以下参数对应以本仓库验证过的 vLLM 0.29.0 / SGLang 0.5.19 为例，新版本必须重新核对。

| 目标 | vLLM | SGLang | 需要确认 |
| --- | --- | --- | --- |
| TP / PP / DP | `tensor-parallel-size` 等 | `tp-size`、`pp-size`、`dp-size` | DP 副本与 DP attention 语义不同；EP 不额外乘一次卡数 |
| 关闭 EP | `no-enable-expert-parallel` | `ep-size=1` | 实际专家分片与通信组 |
| 上下文 / 活动容量 | `max-model-len` / `max-num-seqs` | `context-length` / `max-running-requests` | 全服务容量与每 DP rank 容量分开记录 |
| 关闭前缀缓存 | `no-enable-prefix-caching` | `disable-radix-cache` | 关闭跨请求复用，不删除 JIT 缓存 |
| KV 精度 | `kv-cache-dtype` | `kv-cache-dtype` | 实际存储、scale 来源及模型特有布局 |
| FlashInfer autotune | `kernel-config.enable_flashinfer_autotune` | `disable-flashinfer-autotune` | 实际开始/完成、缓存和未覆盖形状 |
| Prefill 预算 | `max-num-batched-tokens` + chunked prefill | `chunked-prefill-size` / `max-prefill-tokens` | 是否包含 decode；mixed chunk 可更接近总 token 预算，但调度仍不同 |
| Decode Graph | compilation-config 中的模式和捕获尺寸 | decode backend 和 bs 列表 | prefill/混合批是否也捕获，是否有 eager fallback |
| 显存预算 | `gpu-memory-utilization` | `mem-fraction-static` | 预算覆盖范围不同，不能要求比例机械相同 |

## 显存按实际分配理解

先从已完成日志估算合适比例，再启动核验，不必为每个猜测都重跑性能。比较实际 KV 字节预算、可承载负载、剩余显存和是否抢占；显存占用总量不等于 KV 大小。

MLA、滑动窗口、压缩 KV、indexer 和分页策略可能不同。逻辑 token 池的计数不能直接相除得出容量优势，也不能强行套同一个 SWA 比例。

DSV4 实测使用 vLLM 0.90 / SGLang 0.89，实际 KV 为 51.75 / 51.38 GiB 每卡；这是该模型和配置的结果，不能推广到 GLM 或其他拓扑。对齐的是容量目标和资源条件，不是所有内存布局。

## 保留硬件可用的正常优化

同一功能不要求同名 kernel。分别记录 attention、MoE、通信 backend 及 fallback。SM120 不支持的 SM100/NVLink 优化不能照搬；共享专家融合也可能受 checkpoint 混合精度布局影响。

某组合因软件问题关闭 autotune 等优化时，应另列配置并说明，不能称其为双方全优化默认配置。关闭 autotune 不等于禁用所有 JIT、CUDA Graph 或高效 kernel。

不同并行布局可能触发 sequence parallel、改变计算形状或 Graph 行为。记录这些联动变化，不把结果简单归因于某一个开关。DP 还会改变整机 prefill 总预算和模型复制开销。

## 怎样报告

同 workload、同并发和容量配置并排展示：成功/失败、实际 tokens、输出 tok/s、requests/s、TTFT、TPOT、ITL、端到端延迟及可用的 P95。报告每次重复或提供逐次数据，均值附标准差。

注明吞吐是客户端完整测量窗口、服务端稳态窗口还是纯 decode 统计，不能直接互换。不混合不同并发求一个平均值；多个 P95 的均值也不叫合并 P95。

结论写清适用并发、吞吐与延迟取舍、重要异常和未验证项。没有业务 SLA 时，不凭最高吞吐宣布全面胜出。完整示例见 [DSV4 镜像选型](../projects/dsv4-rtx6000d/reports/image-selection.md)。
