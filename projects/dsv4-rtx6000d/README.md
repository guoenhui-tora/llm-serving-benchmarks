# DeepSeek V4 Flash NVFP4 在 RTX6000D 上的推理优化

本项目使用 NVIDIA 发布的 [nvidia/DeepSeek-V4-Flash-0731-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4)，目标是在 RTX6000D 上优化推理吞吐与延迟，当前以单机为主。权重为 NVFP4 routed experts 与高精度其他部分的混合格式；实验保持同一权重与 tokenizer。

**当前优先调优 vLLM 0.29.0，关闭 FlashInfer autotune。** 已完成的 TP8 基线和四节点 TP4 对照均支持这一选择。下一轮统一运行条件后比较并行拓扑，重点分离 GPU 分组、CPU 绑定和拓扑本身的影响。

## 下一轮：四节点并行拓扑实验

**目标：在 8192 输入 / 1024 输出、全局 C32 下，选出值得继续优化的 vLLM 单机拓扑。** 四台独立执行本机任务，每台同一时间只运行一个推理服务；本轮不测 SGLang、不做跨机推理或 PD 分离。历史结果作为背景，不代替本轮本机参照。

**准备状态：** 公共 TP4 配置和 runner 的绑定支持已完成，四份 campaign 已通过离线 validate/plan；本机已用不申请 GPU、不加载模型的短容器核验绑定。完整模型启动、各节点候选和性能测量仍待执行。

### 1. 确认节点任务

表中每个配置目标为三次有效重复；省略的 PP/DP 为 1，EP 默认关闭。四卡统一先用 GPU4–7，八卡使用 GPU0–7。

| 节点 | 本机参照，按顺序执行 | 候选，按顺序执行 | 比较重点 |
| --- | --- | --- | --- |
| **45：DP 扩展** | TP4 → TP8 | TP4×DP2、EP off → TP4×DP2、EP on | 八卡双副本对 TP8；DP 下开启 EP 的额外收益 |
| **46：PP 优化** | TP4 → TP8 | TP4×PP2 → TP2×PP4 | 将 TP 通信留在近端 GPU 组，是否能抵消 PP 开销 |
| **47：EP 优化** | TP4 → TP8，均 EP off | TP4＋EP4 → TP8＋EP8 | 同卡数、同绑定条件下 EP 的收益 |
| **48：四卡与位置对照** | GPU4–7 的 TP4 → GPU0–3 的 TP4 | GPU4–7 上的 TP2×PP2 | 前后四卡的位置差异；四卡内部 TP/PP 的收益 |

- [x] **45**：部分完成；主矩阵12次gate通过，矿机负载恢复影响11次；结果用于本节点高负载下拓扑筛选，跨节点比较受限，可选实验已停止，见[节点报告](reports/topology-node45.md)。
- [x] **46**：已完成主矩阵12次及prefill初筛2次，TP2×PP4表现最佳；残余后台负载经用户授权保留，见[节点报告](reports/topology-node46.md)。
- [x] **47**：已完成四种拓扑各三次测量及可选 prefill 对照；TP4、EP off 领先，prefill 16384 吞吐小幅提高但平均 TTFT 上升，见[节点报告](reports/topology-node47.md)。
- [ ] **48**：待执行；收尾后更新本行状态并链接节点报告。

所有组合均可运行时，45/46/47 各 12 次、48 共 9 次正式测量，不含预热。各节点完成自己的稳定参照后即可继续，不需要等待其他节点，也不 SSH 操作其他机器。TP8 是同节点八卡参照；TP4 到八卡的结果用于观察整体扩展效率，不能把 GPU 数量和 CPU 绑定策略同时变化后的差异全部归因于拓扑。

### 2. 统一运行条件

| 项目 | 统一要求 |
| --- | --- |
| 模型、镜像、客户端 | 相同 NVFP4 权重与 tokenizer，固定 [vLLM runtime](configs/runtimes/vllm-0.29.0.yaml) 和 [压测客户端](configs/clients/vllm-bench-0.29.0.yaml)；核对完整 image ID，不拉取或更换镜像 |
| 执行机制 | `VLLM_USE_V2_MODEL_RUNNER=1`、`--async-scheduling`、`--distributed-executor-backend mp`；FlashInfer autotune 关闭，保留 verbose JIT 日志与缓存 |
| 精度与容量 | 上下文 16384、FP8 E4M3 KV、活动序列容量合计 32；显存比例统一 0.90 自动分配，记录实际 KV 与抢占；不再固定历史对齐用的 32.21 GiB KV 预算 |
| 调度与 Graph | prefill 预算 8192、chunked prefill；`FULL_DECODE_ONLY`，DP1 捕获尺寸 `[1,2,4,8,12,16,24,32]`；其余 backend、block size、indexer 等沿用公共 recipe |
| 缓存与其他优化 | 关闭跨请求前缀缓存，不开投机解码、CPU offload；不混入其他调优开关 |
| 请求协议 | 同一流式 `/v1/completions`；seed=0、temperature=0、ignore_eos=true、range_ratio=0、request_rate=inf |
| 工作量 | 每次正式测量全服务 C32、128 请求、8192/1024；应完成输入 1,048,576、输出 131,072 tokens，成功 128、失败 0 |

**DP2 的口径：** C32 和 128 请求均为整个服务总量，使用一个客户端压测一个 API。每副本 `max-num-seqs=16`，容量合计 32；每副本 Graph 捕获尺寸为 `[1,2,4,8,12,16]`、上限 16。`max-num-batched-tokens=8192` 按每副本保留，因此整机调度预算上限合计 16384；报告明确这一扩展，不声称与 DP1 的整机调度预算相同。核实本机 DP rank 数、请求路由和实际容量语义，不通过外部脚本分别给两个副本各压 C32。

#### CPU、GPU 与内存绑定

四卡采用统一绑定，八卡保持默认不绑定。选择原则是先确定互联较近的 GPU 组，再为服务分配近端物理核、为客户端分配独立物理核和对应内存；通用方法见[CPU/GPU 绑定规则](../../docs/engine-comparison.md#cpugpu-绑定与同机多服务)。本轮具体分配如下：

| 用途 | 宿主 GPU 索引 | 服务 CPU | 服务内存 NUMA | 客户端 CPU | 客户端内存 NUMA |
| --- | --- | --- | --- | --- | --- |
| 四节点 TP4 基线及后四卡候选 | `4,5,6,7` | `32-47` | `2` | `48-51` | `3` |
| 48 前四卡 TP4 对照 | `0,1,2,3` | `0-15` | `0` | `16-19` | `1` |
| 八卡参照与候选，包括 DP2 | `0,1,2,3,4,5,6,7` | 不显式绑定 | 不显式绑定 | 不显式绑定 | 不显式绑定 |

用户已核对四节点的 GPU/PCIe、CPU 编号和 NUMA 距离一致，本机 48 也已复核。GPU0–3 靠近 socket0 的 NUMA0，GPU4–7 靠近 socket1 的 NUMA2；四卡服务使用近端的 16 个物理核，客户端使用同 socket 另一 NUMA 的 4 个物理核。选择后四卡是为了固定参照，不表示已证明它们更快；48 的前后四卡实验顺序运行。

表中 CPU 为宿主逻辑 CPU ID，每个物理核只选一个线程。SMT 兄弟编号为该 ID 加 64，例如 `32-47` 对应 `96-111`；本次服务和客户端不使用这些兄弟线程，不关闭宿主 SMT。同组四卡候选保持同一资源范围，不增加 CPU 核数或另改线程环境变量；16 个核不等于要求引擎所有线程参数都设为 16。

四卡限制通过 target 的 `binding.server`、`binding.client` 生成 Docker `--cpuset-cpus`、`--cpuset-mems`，不依赖 vLLM `--numa-bind`。只给启动 shell 设置 `taskset` 或 CPU 配额不能代替容器 cpuset。八卡则移除整个 `binding`，不启用 `--numa-bind`，也不从外层 `taskset/numactl` 引入限制；八卡 NUMA 优化留待后续独立实验。

启动后核对 Docker inspect 的 `CpusetCpus`、`CpusetMems`，以及 worker、客户端和线程的 `Cpus_allowed_list`、`Mems_allowed_list`，允许引擎在指定集合内进一步缩小范围。cpuset 不隔离后台任务；内存节点限制也不证明所有页都本地驻留，需结合 `numa_maps` 或 `numastat -p` 观察。字段与自动保存的证据见[配置说明](../../docs/configuration.md#cpu--numa-绑定)。

### 3. 准备本机配置

四台共同使用 [TP4 recipe](configs/recipes/vllm-tp4-baseline.yaml) 和 [C32 workload](configs/workloads/dsv4-8192-1024-c32-n128-repeat3.yaml)。下面的 target 除 ID、地址外相同；模型默认 `/data/models/DeepSeek-V4-Flash-0731-NVFP4`，缓存为执行用户的 `~/.cache/serving-bench`，服务端口为 31248。

| 节点 | Campaign | Target |
| --- | --- | --- |
| 45 | [45-dsv4-vllm-tp4-baseline.yaml](configs/campaigns/45-dsv4-vllm-tp4-baseline.yaml) | [rtx6000d-45-tp4-baseline.yaml](configs/targets/rtx6000d-45-tp4-baseline.yaml) |
| 46 | [46-dsv4-vllm-tp4-baseline.yaml](configs/campaigns/46-dsv4-vllm-tp4-baseline.yaml) | [rtx6000d-46-tp4-baseline.yaml](configs/targets/rtx6000d-46-tp4-baseline.yaml) |
| 47 | [47-dsv4-vllm-tp4-baseline.yaml](configs/campaigns/47-dsv4-vllm-tp4-baseline.yaml) | [rtx6000d-47-tp4-baseline.yaml](configs/targets/rtx6000d-47-tp4-baseline.yaml) |
| 48 | [48-dsv4-vllm-tp4-baseline.yaml](configs/campaigns/48-dsv4-vllm-tp4-baseline.yaml) | [rtx6000d-48-tp4-baseline.yaml](configs/targets/rtx6000d-48-tp4-baseline.yaml) |

先读根 AGENTS/README 和[实验方法](../../docs/benchmark-methodology.md)。将项目 configs 复制到新的 `experiments/dsv4-topology-nodeNN/configs/`，`NN` 为本机节点编号。以下以 48 为例，复制只在首次建立且 `configs/` 不存在时执行；已有工作区直接续用，不覆盖。

```bash
mkdir -p experiments/dsv4-topology-node48/results experiments/dsv4-topology-node48/reports
cp -a projects/dsv4-rtx6000d/configs experiments/dsv4-topology-node48/configs
```

保留公共 TP4 原文件，为本机额外参照和候选新增 recipe、target、campaign。全部从本轮公共配置派生，不直接套用历史 TP8 recipe；引用始终相对于各自 `configs/`。

| 配置 | 相对公共 TP4 的必要变化 |
| --- | --- |
| TP8 参照 | TP=8；GPU0–7；移除整个 target `binding`，服务和客户端均不显式绑定 |
| TP4×DP2 | TP=4、PP=1、DP=2；本机两个 DP rank、一个对外 API；GPU0–7；移除 `binding`；按上述 DP2 口径设置容量与 Graph |
| TP4×DP2、EP on | 在同一 DP2 配置中将 `no-enable-expert-parallel` 替换为 `enable-expert-parallel`，目标专家组覆盖 8 卡 |
| TP4×PP2 / TP2×PP4 | 分别设 TP/PP=4/2、2/4，DP=1；GPU0–7；移除 `binding` |
| TP4＋EP4 / TP8＋EP8 | 在对应 TP4 / TP8 参照上，只将 EP 关闭开关替换为开启；EP4/EP8 指目标专家组大小，需从日志核实 |
| 前四卡 TP4 | 只改 target：GPU0–3；服务 CPU0–15、内存 NUMA0；客户端 CPU16–19、内存 NUMA1 |
| 后四卡 TP2×PP2 | TP=2、PP=2、DP=1；沿用后四卡 TP4 的 GPU 与 CPU/内存绑定 |

PP 的预期 TP 分组为：TP4×PP2 使用 `0–3 / 4–7`；TP2×PP4 使用 `0–1 / 2–3 / 4–5 / 6–7`；后四卡 TP2×PP2 使用 `4–5 / 6–7`。DP2 的两个 TP 组预期为 `0–3 / 4–7`。显式固定 GPU 顺序并检查实际 rank/通信组映射，不能仅凭启动参数推断。EP/PP/DP 组合都是候选，固定镜像能解析参数不代表该模型与 kernel 组合一定可运行。

参照使用公共三次重复 workload；候选初筛与补测可分别准备为 1 次和 2 次重复。除记录 ID、重复数外，服务配置、客户端、单次负载、请求数、预热规则和源码必须一致。runner 按 workload 指纹分组，汇总时核对这些条件后再合并逐次数据，不修改公共分组规则。

运行前准备好本机候选，包括计划执行的可选 prefill 配置，逐一 validate/plan，并核对固定镜像 CLI。记录共同 Git commit、源码指纹、本地差异、解析配置和命令；实验期间不 pull、不修改源码或正在使用的配置。

### 4. 运行、验收与异常处理

**先检查本机资源。** 保存 GPU 索引/UUID/PCI bus ID、`nvidia-smi topo -m`、`lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE`、`numactl --hardware`。GPU 索引均按宿主记录，容器内可能重新编号。确认整机没有其他 GPU 计算任务；有占用则报告阻塞，不终止别人的任务。preflight 只检查选中的 GPU，不能代替整机检查。

记录服务/客户端所用物理核及 SMT 兄弟的后台负载、目标 NUMA 的内存压力。46 曾出现 NUMA2/3 空闲内存很低，需区分可回收文件缓存与实际任务占用，并检查匿名页、共享内存和换页；不清缓存、不静默放宽绑定。明显资源冲突先报告阻塞。测量期间保存 CPU 负载及 GPU 功耗、温度、频率遥测，不修改宿主功耗限制或锁频。

按节点任务表的顺序执行。下面是 48 的公共 TP4 参照入口；其他节点或候选使用对应 campaign，每次指定新的 `--run-root`：

```bash
./bench validate experiments/dsv4-topology-node48/configs/campaigns/48-dsv4-vllm-tp4-baseline.yaml
./bench plan experiments/dsv4-topology-node48/configs/campaigns/48-dsv4-vllm-tp4-baseline.yaml
./bench preflight experiments/dsv4-topology-node48/configs/campaigns/48-dsv4-vllm-tp4-baseline.yaml
./bench run experiments/dsv4-topology-node48/configs/campaigns/48-dsv4-vllm-tp4-baseline.yaml \
  --run-root experiments/dsv4-topology-node48/results/tp4-baseline-01
```

每个新配置都完成 health、models 和关闭 thinking 的简短中文探测，再进入预热与测量。参照直接测三次；候选先一次初筛，正常后补两次，不按快慢挑样本。

| 阶段 | 统一要求 |
| --- | --- |
| 预热 | 每次重复前，每轮 64 请求，正式输入输出长度；至少连续两轮无已知编译事件，最多五轮 |
| 测量 | 每次最多两次尝试；沿用原 gate，预热、拒绝尝试与有效结果分别保存 |
| 日志与工作量验收 | 核对成功/失败及实际 tokens，检查完整日志中的量化、attention/MoE、通信/fallback、Graph/V2、实际 KV、抢占与绑定；保留 warning |
| 稳定性检查 | 三次吞吐 CV（样本标准差/均值）超过 3% 时触发诊断，同时检查 TTFT/TPOT；不删除慢样本，日志安静不等于性能收敛 |
| 异常配置 | OOM、不支持、超时或 gate 阻塞时取证并停止该候选；清理本次容器后可继续独立候选 |
| 诊断上限 | 先查 JIT、抢占、后台任务、CPU/NUMA、温度/功耗/频率；每个异常配置最多追加一次有明确假设的诊断测量，原因不明则保留不稳定结论 |

参照不稳时不据此宣称拓扑提升。不得切换精度、关闭正常优化、回退 V1、隐藏 warning 或反复重跑以取得满意结果；配置声明和日志检查也不等于全部 kernel 或模型数值精度已验证。

**8 小时是进度检查点，不是自动停止时间。** 从本节点首次实验启动计时，到时汇报已完成配置、当前阶段和剩余任务，用本机实测耗时更新结束时间估计。正常推进的主矩阵继续完成，不丢弃正在运行的重复，也不等待新的确认。启动超时沿用 1800 秒、单次客户端阶段沿用 7200 秒；预热、重试与诊断次数不增加，不无限等待卡住的配置。用户明确要求停止或指定硬截止时，以用户要求为准。

若前 8 小时内主矩阵已完成，可在本机较好的拓扑上初筛 prefill **4096、16384**，与已有 8192 对照，其他参数不变，各先一次；有收益且值得复核的补齐三次，差距在波动范围内则报告未分出优劣。到检查点后不再新开可选配置，已开始的按原规则收尾。不扩展到其他并发、C128、PD 或八卡 NUMA 绑定。

### 5. 归档并交接

原始结果放 `experiments/dsv4-topology-nodeNN/results/`，草稿和完整本机汇总放同级 `reports/`；每次使用新的 `--run-root`。结束后清理本次容器，保留模型、镜像、JIT 缓存和历史结果。

**每个节点通常只归档一个 Markdown、一个 CSV，并更新 README 中自己的任务行。** 以下路径均相对于本项目，`NN` 替换为 45、46、47 或 48。

| 内容 | 路径与命名 | 归档要求 |
| --- | --- | --- |
| 节点报告 | `reports/topology-nodeNN.md` | 本节点结论、配置差异、汇总结果和异常 |
| 逐次数据 | `data/topology-nodeNN.csv` | 每次有效测量一行，保留原始数值精度，供统一重算统计 |
| 原始日志、JSON、预热、失败尝试 | 本机 `experiments/` | 完整保留，不上传 GitHub |
| 探索配置、临时脚本 | 本机 `experiments/` | 不批量复制到项目目录；整合后再精选有价值的配置 |

**节点报告统一按下面的顺序写：**

1. **结论与完成范围。** 本节点哪个配置更好、优势是否超过波动；列出完成、部分完成、阻塞和未执行的任务。
2. **运行条件与配置差异。** 引用公共 recipe/workload，用表格完整说明每个候选修改了哪些参数、环境变量和 target 设置。记录执行时的 Git commit、源码指纹、必要的本地改动，以及实际绑定、KV、backend 和兼容性调整。配置声明与日志证据分开表述。
3. **性能汇总。** 同配置给出有效重复数、吞吐均值和样本标准差/CV，以及 requests/s、mean/P95 TTFT、mean/P95 TPOT、可用 ITL/端到端延迟。链接本节点 CSV，不在正文再复制全部逐次数据。按同节点、同卡数参照计算提升，另列四卡到八卡的整体扩展；多个 P95 的均值不是合并请求的 P95。
4. **异常与限制。** 列出 OOM、不支持、超时、gate 拒绝、不稳定和重要 fallback，说明哪些数据不能用于结论；未完成三次的配置明确标注实际重复数，不补造样本。

CSV 至少包含唯一样本标识、配置标签、GPU 组、TP/PP/DP/EP、prefill 预算、并发、重复编号、成功/失败请求数、实际输入输出 tokens 和上述可用指标。延迟统一用毫秒，吞吐用 output tokens/s、requests/s，列名注明单位；不可用值留空，不填零。初筛和补测的重复编号连续且不重复。不混入预热、gate 拒绝或诊断样本；这些记录保留在本地并在报告说明。完全阻塞、没有有效样本时保留 CSV 表头即可。

**报告引用只指向仓库内存在的文件，使用相对链接。** 例如 45 的报告引用数据用 `../data/topology-node45.csv`，引用公共配置用 `../configs/recipes/vllm-tp4-baseline.yaml`。不把本机 `experiments/`、`/tmp` 或绝对路径写成 GitHub 中的复现入口；原始产物位置保留在本地记录。候选差异必须足以从公共配置重建；涉及复杂代码或其他文件时，列明后续需要精选保留的内容，不能省略影响复现的改动。

收尾后，各节点只更新上面属于自己的任务行，添加已经归档的报告链接，其余 README 内容留给整合者。勾选表示“本节点任务已收尾、报告已交付”，不代表所有组合成功；部分完成或阻塞必须在同一行注明。例如：

```markdown
- [x] **45**：已完成，见[节点报告](reports/topology-node45.md)。
- [x] **46**：部分完成，TP2×PP4 不支持，见[节点报告](reports/topology-node46.md)。
```

以上仅为格式示例，不是当前实验状态。各节点不修改别人的报告、CSV 或任务行，也不更新公共汇总表；跨节点整合和优胜配置归档随后统一进行。commit/push 等用户另行指示。

各节点可使用同一条任务指令：

> 阅读 `projects/dsv4-rtx6000d/README.md` 的“下一轮：四节点并行拓扑实验”，确认本机编号，按本节第 1–5 步完成对应任务。仅操作本机，8 小时时汇报进度，正常推进的主矩阵继续完成；异常按文档上限处理，按统一规则归档，不自行扩展矩阵或 push。

## 已完成实验

### 2026-09-15：TP8 基线与镜像选择

2026-09-15的单机八卡、8192/1024对齐实验中，vLLM在C16/C32的输出吞吐为 **532.57 / 651.35 tok/s**。三套配置（vLLM off、SGLang off/on）各三次重复，合计18次测量。这批 TP8 数据独立统计，不与双服务 TP4 样本混合。

| 内容 | 入口 |
| --- | --- |
| TP8镜像、参数与性能表 | [镜像选型](reports/image-selection.md) |
| TP8逐次指标 | [baseline-samples.csv](data/baseline-samples.csv) |
| 环境与TP8运行命令 | [基线复现](reports/reproduction.md) |
| 已知问题 | [排查记录](reports/lessons.md) |
| RTX6000D通信测量 | [互联报告](reports/interconnect.html) |

`configs/` 同时保留三套TP8 recipe及其依赖：完整对比入口为`48-dsv4-aligned-final-c16-c32.yaml`，仅vLLM C32为`48-dsv4-vllm-baseline-c32.yaml`；`48-dsv4-functional.yaml`是已离线检查、尚未实机运行该负载的短验证入口。

### 2026-09-17：四节点双服务 TP4 对照

2026-09-17，四台各部署vLLM TP4和SGLang TP4，同时压测。每实例输入8192、输出1024 tokens，C32、128请求、三次有效重复；整机合计C64。指标由四份逐次CSV重算，运行条件及异常依据各节点提交报告整理。

共同条件为TP4/PP1/DP1、EP off，上下文16384、活动容量32、FP8 E4M3 KV、关闭前缀缓存和autotune；prefill预算8192、chunked/mixed prefill，decode full Graph，捕获尺寸1/2/4/8/12/16/24/32，prefill Graph不启用。vLLM使用FLASHINFER_MLA_SPARSE_DSV4/FLASHINFER_CUTLASS；SGLang使用DeepseekV4AttnBackend/flashinfer_cutlass并关闭共享专家融合。block/page为256，FP4 indexer关闭，SGLang SWA/full ratio为0.1。未启用投机解码或CPU offload。

镜像固定为`vllm/vllm-openai:v0.29.0`、`lmsysorg/sglang:v0.5.19-cu130`，身份见[镜像记录](reports/image-selection.md#固定镜像与-recipe)。客户端统一vLLM 0.29.0、流式`/v1/completions`，seed=0、temperature=0、ignore_eos=true、range_ratio=0、request_rate=inf。

吞吐为三次均值 ± 样本标准差；延迟列为三次对应指标的均值，**P95 TTFT不是合并请求后的P95**。

| 节点 | 引擎 | GPU | 输出 tok/s | CV | Mean TTFT 秒 | P95 TTFT 秒 | Mean TPOT ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 45 | vLLM | 0–3 | 674.06 ± 2.63 | 0.39% | 7.32 | 22.67 | 40.30 |
| 45 | SGLang | 4–7 | 583.47 ± 3.45 | 0.59% | 7.49 | 25.83 | 47.50 |
| 46 | vLLM | 4–7 | 705.14 ± 4.32 | 0.61% | 6.93 | 20.85 | 38.59 |
| 46 | SGLang | 0–3 | 565.00 ± 1.01 | 0.18% | 7.90 | 27.32 | 48.89 |
| 47 | vLLM | 0–3 | 676.05 ± 3.22 | 0.48% | 7.17 | 22.80 | 40.31 |
| 47 | SGLang | 4–7 | 585.83 ± 2.61 | 0.45% | 7.45 | 25.72 | 47.32 |
| 48 | vLLM | 4–7 | 708.47 ± 1.30 | 0.18% | 6.93 | 20.87 | 38.37 |
| 48 | SGLang | 0–3 | 566.78 ± 1.68 | 0.30% | 7.92 | 27.39 | 48.70 |

24次有效测量共 **3072成功、0失败，输入25,165,824、输出3,145,728 tokens**。每个样本均完成128请求和131,072输出tokens。每轮预热64请求、最多五轮，至少连续两轮无已知编译事件；各节点额外检查与重试差异见下文。节点内vLLM吞吐领先约15.4%–25.0%，上表延迟指标也更低；吞吐CV均低于0.62%。有效样本未报告OOM或请求超时。46首对的第一次测量因JIT整对拒绝；47两轮诊断未计入正式数据。

| 节点 | 条件、逐次统计与异常 | 原始精度的逐次指标 |
| --- | --- | --- |
| 45 | [tp4-node45.md](reports/tp4-node45.md) | [tp4-node45.csv](data/tp4-node45.csv) |
| 46 | [tp4-node46.md](reports/tp4-node46.md) | [tp4-node46.csv](data/tp4-node46.csv) |
| 47 | [tp4-node47.md](reports/tp4-node47.md) | [tp4-node47.csv](data/tp4-node47.csv) |
| 48 | [tp4-node48.md](reports/tp4-node48.md) | [tp4-node48.csv](data/tp4-node48.csv) |

#### GPU 分配与性能分组

**45/47是前四卡vLLM、后四卡SGLang；46/48恰好相反。** 按实际绑定分组后，同组两节点的各引擎吞吐均值差均不到0.5%，跨组差异则更明显：

| 分组 | GPU0–3 | GPU4–7 | vLLM节点均值的平均 tok/s | SGLang节点均值的平均 tok/s |
| --- | --- | --- | ---: | ---: |
| 45、47 | vLLM | SGLang | 675.06 | 584.65 |
| 46、48 | SGLang | vLLM | 706.80 | 565.89 |

46/48相对45/47，vLLM约高 **4.70%**，SGLang约低 **3.21%**；也就是两种引擎在GPU4–7上的结果都高于各自在GPU0–3上的结果。组均值仅用于描述这一模式，不合并为新的性能基线。TTFT/TPOT也呈同方向变化。

**这表明存在与绑定分组一致的性能差异，但不能据此确定GPU4–7本身更快。** 每个节点没有交换两种引擎的位置，且CPU布局、内存绑定、KV控制和观测协议也随分组变化：

| 节点 | 每侧服务/客户端物理核 | 内存绑定 | KV条件（vLLM / SGLang，每卡） |
| --- | --- | --- | --- |
| 45 | 14 / 2 | 服务和客户端均在GPU近端NUMA | 32.43 GiB显式预算 / 32.43 GiB张量重建值 |
| 46 | 16 / 8 | 未强绑定内存；客户端在同socket另一NUMA | 32.21 / 32.21 GiB日志分配预算 |
| 47 | 12 / 4 | 服务和客户端均在GPU近端NUMA | 32.73 GiB自动预算 / 32.43 GiB张量重建值 |
| 48 | 16 / 4 | 服务和客户端分别绑定同socket的两个NUMA | 32.21 / 32.21 GiB日志分配预算 |

SGLang张量重建值与日志预算是不同统计口径，不能当作可直接比较的物理显存占用。45/48额外要求末两轮预热吞吐差≤5%/3%；47增加编译产物检查，46增加编译器进程采样。各节点保留了这些协议差异，不能声称四台除GPU位置之外完全一致。

46/48明确记录了`doca_spcx_cc`后台CPU负载；47的CPU集合忙碌率也包含其他宿主活动，45缺少同口径量化。CPU cpuset只限制本次进程，不会排除后台任务。vLLM先结束后，SGLang仍有约29–47秒尾段；两侧完整窗口吞吐不能直接相加为整机吞吐，也不能用于证明独占TP4优于TP8。

有价值的后续验证是在同一节点保持CPU核数、客户端位置策略、内存约束和KV配置一致，交换引擎所在GPU组，或做四卡独占对照。当前结果足以支持优先调优vLLM；GPU位置效应与TP4相对TP8的纯拓扑收益仍待分离。

历史双服务 TP4 实验只保留每节点 CSV 与报告，不提供同步脚本。本轮新配置用于统一绑定的独占基线，不能作为这批双服务数据的复现入口。

## TODO：后续优化

- [ ] **八卡 vLLM NUMA 绑定对照（`--numa-bind`）。** 本轮八卡拓扑筛选统一保持默认不绑定；选出较好的拓扑后，在同一节点、同一拓扑上比较默认不绑定与开启该参数，8K/1K、C32，各三次有效重复，其他参数和客户端资源条件保持一致。观察吞吐、TTFT/TPOT 及重复波动。

固定 vLLM 0.29.0 镜像的 `numa_bind` 默认关闭。开启后按 GPU 归属绑定 worker 的近端 CPU 和内存；核实前四卡 worker 对应 NUMA0、后四卡对应 NUMA2，并检查实际 rank 映射、线程亲和性、内存分布及权限警告。如同时使用 Docker cpuset，两层允许范围必须兼容。当前四卡 Docker 绑定已限制到单个近端 NUMA，不依赖该开关生效；八卡 NUMA 优化的收益尚未实测。
