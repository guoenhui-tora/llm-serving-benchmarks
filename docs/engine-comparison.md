# vLLM 与 SGLang 如何公平对比

先声明比较目标：**共同控制参数的基线**用于判断相同主要条件下的差异；**分别调优的部署方案**用于选择实际可用组合。两者都有效，但不能把不同优化条件的结果写成仅由引擎名称造成的差异。

## 先对齐共同条件

两边使用相同的压测协议、轮数/时间预算及稳定性阈值；快速、累计无JIT和稳定窗口结果不能混为同一种验收。

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

## CPU/GPU 绑定与同机多服务

**先按拓扑确定资源选择规则，再把规则转换为各机器的 GPU/CPU 编号。** 同一轮对照固定规则和资源数量；CPU 编号相同并不保证拓扑相同，编号不同也不代表条件不一致。

| 内容 | 通用选择规则 | RTX6000D 四卡基线示例 |
| --- | --- | --- |
| GPU | 选择满足模型容量、组内互联较近的 GPU 集合，尽量让 TP 组留在同一 NUMA/socket；核实 PCIe 路径，不凭连续编号判断 | 固定使用 GPU4–7，已核实均靠近 NUMA2；“后四卡”不是其他平台的通用优先项 |
| 服务 CPU | 优先使用 GPU 近端 NUMA 的物理核，固定核数，服务进程共享这份预算 | 为整个 TP4 服务分配 16 个物理核，不是每个 worker 各 16 核 |
| 客户端 CPU | 与服务使用不同物理核；有条件时放在同 socket 的另一 NUMA，兼顾资源分离和通信距离 | 分配同 socket 另一 NUMA 的 4 个物理核 |
| SMT / 超线程 | 根据 socket、core 和 thread sibling 信息识别物理核；明确每核用一个还是全部逻辑线程，所有对照一致，服务与客户端不能占用同一核的不同线程 | 每核只选一个逻辑线程，不关闭宿主 SMT |
| 内存节点 | 默认将服务、客户端各自的内存分配限制在其 CPU 所在 NUMA，优先保持服务靠近 GPU；先确认容量和压力 | 服务限制在 NUMA2，客户端限制在 NUMA3 |

16/4 核是本项目的统一起点，不是所有模型和机器的最优值。若同 socket 没有另一 NUMA，可在同一 NUMA 内划分不重叠的物理核；若核数或内存不足，应预先制定一致的替代布局并单独记录，不能在测量中静默放宽绑定。八卡或多副本部署需按 GPU 组规划对称资源，再核对各 worker 的实际亲和性，不能直接套用四卡编号。

使用 `nvidia-smi topo -m`、`lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE`、`numactl --hardware` 和 CPU 的 `thread_siblings_list` 建立映射；启动后核对容器 cpuset、服务 worker 及客户端的实际 CPU/内存允许集合。CPU cpuset 不会隔离后台进程，未分配给本次任务的 SMT 兄弟也可能被其他任务使用；内存节点限制不等于每一页都实际驻留于该节点，需要结合后台负载和内存分布判断。

跨节点对照应保持资源分配策略一致。交错GPU组可以观察位置相关性，但若CPU布局、KV配置也同时改变，就不能单独归因于GPU位置；用同机交换位置或独占对照分离影响。同机两服务结束时间不同时，注明测量重叠和空闲尾段，不直接相加各自完整窗口的吞吐。

## 怎样报告

同 workload、同并发和容量配置并排展示：成功/失败、实际 tokens、输出 tok/s、requests/s、TTFT、TPOT、ITL、端到端延迟及可用的 P95。报告每次重复或提供逐次数据，均值附标准差。

注明吞吐是客户端完整测量窗口、服务端稳态窗口还是纯 decode 统计，不能直接互换。不混合不同并发求一个平均值；多个 P95 的均值也不叫合并 P95。

结论写清适用并发、吞吐与延迟取舍、重要异常和未验证项。没有业务 SLA 时，不凭最高吞吐宣布全面胜出。完整示例见 [DSV4 镜像选型](../projects/dsv4-rtx6000d/reports/image-selection.md)。
