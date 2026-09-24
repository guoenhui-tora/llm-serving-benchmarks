# vLLM 0.30.0：3P1D 的 PP2／DP2 实验结果与待验证方向

本文记录 **2026-09-24 使用 `vllm/vllm-openai:v0.30.0` 镜像完成的 3P1D 实验**：三个 prefill 实例、一个 decode 实例，每实例 4 张 RTX6000D，共 16 卡。模型为 `DeepSeek-V4-Flash-0731-NVFP4`，统一 EP on、DSpark K5、V2 model runner、async scheduling。服务镜像固定为 `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`，客户端沿用已固定的 v0.29 镜像。

**后续实验统一使用 v0.30.0 服务端和压测客户端。** 本文已有结果的客户端版本仍为 v0.29。

**三种拓扑共同需要的模型正确性补丁只有 [v0.30 DSpark MXFP4 草稿分派修复](../patches/v030-dspark-mxfp4/README.md)。**

测量采用 **quick：每档先 1 轮 2C 条完整请求 warmup，再 3 轮各 4C 条正式请求**。

## 当前结论

- Prefill 实例采用 PP2 时，当前负载下的 prefill 阶段耗时较短。
- Decode 实例采用 DP2 时，端到端吞吐和 TPOT 更好。
- Mooncake 混合组吞吐接近 NIXL 全 DP2 基线，尚无证据把全 PP2 的主要损失归因于 KV 传输瓶颈。
- PP2 的批次组织与 stage 负载值得优先优化，但尚未证明能同时提高吞吐并降低 TTFT。

## 待验证清单

- [ ] 测量 decode 两个 PP stage 的计算、草稿生成、通信与空闲时间，判断是否需要非均匀分层。[Stage 负载与 KV 对齐](#stage-负载)
- [ ] 检查相邻 step 是否出现大批／小批或空批交替，再评估限制单步调度量。[批次组织](#批次组织)
- [ ] 若仍有流水空闲，再验证增加在途请求的收益，同时检查容量、Graph 与 TTFT。[请求容量](#请求容量)
- [ ] 若交接开销值得优化，再做同拓扑、同 NIC 的 connector 对照，之后评估逐 layer 传输。[KV 传输](#kv-传输)
- [ ] 固定 decode DP2、connector、NIC 和到达速率，对齐 prefill 容量与预算，验证 prefill PP2 的独立收益。[结论边界](#结论边界)

## 术语与部署差异

沿用 [vLLM v0.30.0 官方说明](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/features/disagg_prefill.md)的 **prefill instance / decode instance**，正文写作“prefill 实例／decode 实例”；只有实验名称保留 3P1D，不再单独用 P、D 表示角色。拓扑中的 PP2 指 TP2 PP2，DP2 指 TP2 DP2。PP stage 是模型层分区：一个 stage 包含多个 transformer layer；“decode stage 1”特指 decode 实例内部第二个 PP stage。

| Prefill 拓扑 | Decode 拓扑 | 除共同 MXFP4 修复外，本次使用的配置与代码 | 复现入口 |
| --- | --- | --- | --- |
| DP2 | DP2 | 原生 NIXL connector、PD 代理及 RDMA／UCX 网络配置；无需额外 PP 兼容或模型 forward 补丁。仍需按实例容量设置 Graph 等参数 | [全 DP2 实测配置与命令](../reports/v030-pd-16k-dp2-20260924.md) |
| PP2 | PP2 | 本次原生 NIXL 路径不支持所需 PP／hybrid KV 组合，改用镜像已有 Mooncake；增加 bootstrap／NIC／GID 配置、独立代理和传输观测 hook，未改模型 forward 或 KV 传输算法 | [Mooncake 配置与代码](../patches/v030-pd-mooncake/README.md)、[全 PP2 实测命令](../reports/v030-pd-16k-pp2-20260924.md) |
| PP2 | DP2 | 使用同一套 Mooncake 代理与观测代码；decode 从远端两个 PP stage 按层映射 KV，两个 DP 引擎均承接请求。没有额外权重转换或另一套模型补丁 | [Mooncake 配置与代码](../patches/v030-pd-mooncake/README.md)、[混合拓扑实测命令](../reports/v030-pd-16k-mixed-20260924.md) |

Mooncake 归档包含相同的 MXFP4 修复及扩展观测 hook，按其说明准备一套运行目录，不叠加加载两套 `sitecustomize.py`。其代理负责唯一 transfer ID、bootstrap／engine 元数据与请求路由；观测 hook 用于核对真实传输和失败计数，不是性能优化补丁。24K 的容量、budget 和 Graph 随负载调整，见 [24K 实测配置](../reports/v030-pd-24k-20260924.md)。

## 实验结果

输出吞吐包含代理、prefill、KV 交接与 decode，为三轮均值 ± 样本标准差；TTFT、TPOT 为三轮均值。不同 connector、网络配置与 prefill 服务容量带来的比较限制见[结论边界](#结论边界)。

### 16K 输入／1K 输出：C128，decode 总容量 128

| Prefill 拓扑 | Decode 拓扑 | Connector | 输出 tok/s | Mean TTFT，s | Mean TPOT，ms |
| --- | --- | --- | ---: | ---: | ---: |
| PP2 | PP2 | Mooncake | 2643.29 ± 19.72 | 8.801 | 36.435 |
| DP2 | DP2 | NIXL | 3091.86 ± 18.69 | 10.254 | 28.391 |
| PP2 | DP2 | Mooncake | 3063.72 ± 9.41 | 9.968 | 29.377 |

保持 prefill PP2 与 Mooncake，decode 从 PP2 改为 DP2 后，吞吐提高 **15.91%**、TPOT 降低 **19.37%**，但 TTFT 增加 **13.26%**。这是优先调查 PP2 decode 的主要依据，尚未隔离通信、调度和负载反馈的贡献。

混合拓扑与全 DP2 基线的吞吐相差约 **0.91%**。这说明 Mooncake 没有阻止本次混合部署达到约 3064 tok/s；不能据此宣布它与 NIXL 等效，或所有负载下都没有传输瓶颈。

### 24K 输入／2K 输出：prefill DP2，decode DP2

使用 NIXL，decode 总容量固定为 80。

| 客户端并发 C | 输出 tok/s | Mean TTFT，s | Mean TPOT，ms |
| ---: | ---: | ---: | ---: |
| 32 | 1607.41 ± 14.60 | 5.167 | 15.983 |
| 48 | 2056.50 ± 21.66 | 6.315 | 18.547 |
| 64 | 2430.03 ± 12.10 | 7.436 | 20.872 |
| 80 | 2759.73 ± 5.28 | 9.107 | 22.412 |

### 24K 输入／2K 输出：prefill PP2，decode PP2

使用 Mooncake，decode 总容量固定为 80。

| 客户端并发 C | 输出 tok/s | Mean TTFT，s | Mean TPOT，ms |
| ---: | ---: | ---: | ---: |
| 32 | 1281.41 ± 18.78 | 5.662 | 20.740 |
| 48 | 1636.04 ± 26.33 | 6.905 | 24.022 |
| 64 | 1933.89 ± 9.07 | 8.071 | 26.832 |
| 80 | 2151.53 ± 19.11 | 9.393 | 29.954 |

### 24K C80：两种拓扑单独对照

两组均为 3P1D、decode 总容量 80、输入 24576／输出 2048 tokens。

| Prefill 拓扑 | Decode 拓扑 | Connector | 输出 tok/s | Mean TTFT，s | Mean TPOT，ms |
| --- | --- | --- | ---: | ---: | ---: |
| DP2 | DP2 | NIXL | 2759.73 ± 5.28 | 9.107 | 22.412 |
| PP2 | PP2 | Mooncake | 2151.53 ± 19.11 | 9.393 | 29.954 |

全 DP2 吞吐高 **28.27%**、Mean TTFT 低 **3.04%**、Mean TPOT 低约 **25.18%**。这档没有出现全 PP2 的端到端优势，但服务端统计仍显示其 prefill 阶段较短，收益被其他等待抵消，见[阶段耗时](#阶段耗时)。

四档整体上，全 DP2 吞吐高约 25%–28%，TTFT 低约 3%–9%。24K 没有测试混合拓扑，不能从 16K 外推。各档两拓扑均未达到 95% 的联合 SLO（逐请求 TTFT ≤ 10 s 且 TPOT ≤ 50 ms）；C80 的达标率分别为全 DP2 78.13%、全 PP2 77.29%。

## 详细证据与假设

每节按 **实测事实 → 源码事实（如适用）→ 推断与待验证项** 展开。源码能证明某机制存在，不能代替性能时间线；所有调参方案都是候选，不是已经验证的最优配置。

### 阶段耗时

**实测事实。** 以下从正式窗口服务端指标重算，按各阶段请求数加权；所有时间单位为秒。

| 负载／并发 | Prefill 拓扑 | Decode 拓扑 | Prefill 阶段耗时 | Prefill 排队 | Decode 排队 | 客户端 TTFT |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 16K C128 | DP2 | DP2 | 4.647 | 3.687 | 0.277 | 10.254 |
| 16K C128 | PP2 | PP2 | 2.595 | 3.704 | 1.226 | 8.801 |
| 16K C128 | PP2 | DP2 | 3.040 | 4.069 | 1.577 | 9.968 |
| 24K C80 | DP2 | DP2 | 3.741 | 3.371 | 0.254 | 9.107 |
| 24K C80 | PP2 | PP2 | 3.059 | 3.609 | 1.230 | 9.393 |

Prefill 阶段耗时是 vLLM 请求阶段的墙钟时间，包含该阶段调度、计算和通信影响，不是纯 kernel 时间；这些指标不是完整、互斥的客户端 TTFT 分解。Decode 排队不能直接当作 RDMA 传输时间。

24K C80 的全 PP2 组，prefill 阶段少约 0.682 s，但 prefill 排队多约 0.238 s、decode 排队多约 0.976 s。**有证据支持 prefill PP2 缩短了当前负载下的阶段耗时；尚不能由此证明其 prefill 吞吐上限更高。**

### Stage 负载

**源码事实。** Decode 使用 PP2 时，非末 stage 在返回中间结果及处理必要同步后，不执行末 stage 的采样与草稿生成路径；末 stage 继续执行 `speculator.propose` 并广播结果。主模型有 43 层，默认按 22／21 分配，末 stage 还承担末端处理和 DSpark 草稿生成。工作内容不对称是确定的，实际耗时差尚未测量。

**待验证假设，优先级高。** 若末 stage 的计算持续更长，可能限制流水吞吐。应先区分主模型计算、draft、通信与依赖等待，再决定是否将主模型层向 stage 0 移动。镜像已有 `VLLM_PP_LAYER_PARTITION`，例如 23／20 是可研究的候选；挪层也可能把瓶颈转移到 stage 0，不能保证收益。

**KV 对齐约束。** 本次 Mooncake 在两端 PP 数相同时优先匹配相同 PP rank。仅将 decode 改为 23／20、prefill 保持 22／21，会产生层集合不一致；应先验证同步改变两端分层，或专门实现并验证跨 stage 映射。同步调整也改变了 prefill 配置，必须分别报告两端变化，不能作为只改 decode 的消融。

### 批次组织

**实际配置与源码事实。** 本次 PP2 启用 `VLLM_USE_V2_MODEL_RUNNER=1`、`--async-scheduling`、`--pipeline-parallel-size 2`，执行队列允许最多 `pp_size + 1 = 3` 个在途批次。同一请求在一次 decode 调度后，隔两个调度 step 才再次符合 decode 调度条件；调度器不会仅因 PP2 自动把 128 条请求均分成两批。

本次 decode 的 `max-num-batched-tokens=16384`。K5 完整 target 验证每请求 6 个 query tokens，128 条仅需 768；该预算不会迫使请求分批。16K PP2 的 target／draft Graph 覆盖为 768／640，DP2 每引擎为 384／320。总请求容量、单步 token 预算、执行队列深度、Graph 覆盖是不同约束。

**待验证假设，优先级高。** 若某一步吸收了大部分可执行请求，下一步可能缺少独立工作，产生流水空闲。但请求陆续到达、KV 陆续就绪也可能自然形成多个批次；不能从大 budget 直接断言某个 stage 总在空闲。

最小证据是相邻 step 的请求数、query tokens、非空批次数和 GPU 重叠时间线。若确有批次集中，可保持 decode 总容量 128，研究限制单步调度量；64 条 K5 对应 target 384 tokens，仅用于说明候选批次规模，不能未经验证直接把整个 decode buffer budget 改为 384。还需核对草稿槽位、首次远端 KV 接入步骤和 Graph 路径，避免缩小 batch 后 kernel 效率下降或调用开销上升。

### 请求容量

**实测事实。** 16K C128 的全 PP2 与混合组 decode 总容量均为 128：

| Decode 拓扑 | 调度引擎数 | 每引擎 max-num-seqs | 总容量 | 正式窗口 running 请求 |
| --- | ---: | ---: | ---: | --- |
| PP2，全 PP2 组 | 1 | 128 | 128 | 均值约 74.6，峰值 127 |
| DP2，混合组 | 2 | 64 | 128 | 两引擎均值约 34.0／33.8，各自峰值 62 |

两组 decode 的 capacity waiting 采样均为 0。Running 是活动请求数量，不是每步 batch 大小；PP2 两个 stage 接力处理同一请求池，总容量不能算成 128×2。24K 的 decode KV 峰值不超过约 22%；PP2 没有 capacity waiting，DP2 仅 C80 短暂观察到 1 条，不支持把差异归为 KV 容量耗尽。

**较弱假设。** 独立请求不足可能让流水线难以填满，但未获执行时间线支持。C128 是全系统在途上限，其中还有请求在 prefill 或 KV 交接阶段；只将 decode 容量改为 256，不会自动增加工作量。应先检查 stage 负载与批次组织，必要时再一起考虑客户端并发、decode 容量和 Graph，并检查 TTFT／SLO 代价。

### KV 传输

**实测事实。** 正式请求命中全部远端输入 KV，decode 整段 prefill 重算为 0，传输／通知失败、过期、抢占均为 0。16K 两个 Mooncake 组每轮 producer 审计字节均为 81.752 GB；混合拓扑未观察到因拓扑不同而增加的审计总字节量。这不等于网络包总量、传输关键路径或每次调用开销相同。

**当前判断。** 混合组吞吐接近 NIXL 基线，支持暂不把 Mooncake 当成已证实的主要吞吐瓶颈；缺少同拓扑、同 NIC 的对照，不能宣布两个 connector 开销相同。Mooncake 组 decode 排队较长，也不能将全部等待算成网络传输时间。先查 decode 流水线，比先假定换 connector 能解决问题更有依据。

**源码事实。** 本次镜像中 MooncakeConnector 与 NixlConnector 的 `save_kv_layer`、`wait_for_layer_load` 均为空实现。本次 Mooncake 代理等 prefill 返回后才提交 decode 请求。两条路径都没有按 transformer layer 就绪即发送、与后续 prefill 计算重叠的实现；这是当前 connector／编排路径的事实，不是 PP 拓扑独有的问题，也不是底层传输库永远不支持该能力。

一个 PP stage 包含多个 transformer layer。“按 layer 名称映射 KV”与“逐 layer 提前传输”不同；每请求经过两个 stage 合起来仍是一次完整模型计算，KV 不必先集中到末 stage 再发送。

**后续探索，证据较弱。** 逐 layer 传输可能缩短交接等待，但应先量出不可重叠的 KV 时间占比，再研究就绪事件、请求编排、KV 生命周期与接收端依赖。该优化未实现，也不能直接解决后续长时间 decode 的流水效率问题。

### 结论边界

**闭环负载影响归因。** 固定 C128 意味着完成一条再补下一条，不是固定到达速率。Decode 加快会提高补发速度，改变 prefill 压力；混合组相对全 PP2 多出的约 1.167 s TTFT，不能全部算成跨拓扑 KV 转换。相同总请求容量也不代表两种拓扑都已达到最优。

**Prefill 服务容量与网络条件不同。** 每个 prefill 引擎容量均为 32，因此 PP2 每实例总容量 32、DP2 为 64；budget 也按引擎配置。NIXL 各服务配置 `mlx5_4–7`，Mooncake 为单 NIC，46 节点前四卡的 prefill 实例用 `mlx5_0`，其他用 `mlx5_4`。代理／观测路径不同，每组只有一次成功部署内三轮，本批不是纯拓扑或纯 connector 消融。

若要证明 prefill PP2 的独立收益，应固定 decode DP2、connector、NIC 与到达速率，对齐 prefill 服务容量和预算口径。若要确认通信开销，应单独做同拓扑 connector 对照。3P1D 配比搜索应在识别瓶颈后再考虑。

**16K 与 24K 不是只改变输入长度。** 全 DP2 从 16K/1K 的 3091.86 tok/s 到 24K/2K C80 的 2759.73 tok/s，输出吞吐下降约 10.7%，但请求速度从约 3.02 降至 1.35 req/s，下降约 55.4%。输出翻倍，可以摊薄每请求的 prefill 与 KV 交接成本；输入／输出 token 比从 16 降至 12，不表示 prefill 算力成本线性减少。并发、decode 容量、prefill budget 和数据内容也变了，不能单独归因为输入长度。

## 证据与复现入口

实际启动命令、节点放置、负载身份、失败尝试和逐轮记录沿用已归档产物；本文不另维护一套命令：

- [16K 全 PP2](../reports/v030-pd-16k-pp2-20260924.md)、[16K 全 DP2](../reports/v030-pd-16k-dp2-20260924.md)、[16K 混合拓扑](../reports/v030-pd-16k-mixed-20260924.md)、[24K 两拓扑](../reports/v030-pd-24k-20260924.md)。
- [逐轮 CSV](../data/v030-pd-20260924-rounds.csv)、[汇总 JSON](../data/v030-pd-20260924-summary.json)、[实测 argv 快照](../data/v030-pd-20260924-commands.json)。失败部署内诊断轮次未并入正式结果。
- [共同 MXFP4 草稿修复](../patches/v030-dspark-mxfp4/README.md)、[Mooncake 代理与观测代码](../patches/v030-pd-mooncake/README.md)。

阶段统计来自本机仓库工作区，未纳入 Git：`experiments/dsv4-v030-pd-20260924/reports/resources-analysis.json` 与 `scheduler-analysis.json`。前者按正式窗口 histogram sum/count 增量求请求均值，再按请求数合并；后者为调度状态采样，running 均值按采样数合并，不能替代逐 step 批次记录或 GPU trace。

源码事实在上方固定镜像中核对，关键定位如下：

| 内容 | 镜像内源码定位 |
| --- | --- |
| 在途批次数与异步执行队列 | `config/vllm.py::max_concurrent_batches`；`v1/engine/core.py` |
| 请求再次 decode 的节拍、单步预算与容量 | `v1/core/sched/async_scheduler.py::_update_after_schedule`；`scheduler.py::schedule` |
| 末 stage 的草稿工作 | `v1/worker/gpu/model_runner.py::sample_tokens` |
| 默认 layer 分配与自定义分层 | `distributed/utils.py::get_pp_indices`；`models/deepseek_v4/nvidia/model.py` 的 `make_layers` |
| 逐层接口、PP rank 与 KV 映射 | `distributed/kv_transfer/kv_connector/v1/` 下 Mooncake、NIXL 的接口；`mooncake_connector.py::receive_kv`、`_align_transfer_regions` |

部分源码快照位于本机 `experiments/dsv4-v030-pd-20260924/reports/image-source/`，其余符号直接从同一镜像读取。上述工作区仅本机保留；异地核验须使用同一镜像，不能用最新上游文件替代。全部候选分析均未启动新 GPU 实验。
