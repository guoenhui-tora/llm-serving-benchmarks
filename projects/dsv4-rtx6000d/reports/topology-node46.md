# 节点46：TP2×PP4 在本轮 C32 负载下最值得继续优化

## 1. 结论与完成范围

**2026-09-17，gpu-6000d-46 在固定 vLLM 0.29.0、8192输入/1024输出、全服务C32条件下，TP2×PP4、prefill 8192 是本轮四种拓扑中吞吐最高且延迟较低的配置。** 三次有效测量为 **1099.61 ± 19.98 output tokens/s，样本CV 1.82%**；比同节点八卡TP8高 **80.34%**，mean TTFT降低 **64.61%**、mean TPOT降低 **39.90%**。这是本轮实测候选结论，不是所有负载或所有配置的官方最优。

主矩阵按TP4→TP8→TP4×PP2→TP2×PP4顺序完成，各3次，共12个有效样本。随后仅在TP2×PP4上初筛prefill 4096、16384，各1次，共14个有效样本；没有未完成的主矩阵配置、模型不支持或OOM。所有配置通过health、models、关闭thinking的中文探测及最终日志gate。

prefill 4096单次吞吐924.61 tokens/s，比8192均值低15.92%；mean TTFT更低，但TPOT和端到端延迟更高。prefill 16384单次1089.59 tokens/s，比8192均值低0.91%，差距落在已有波动范围内；TTFT更高、TPOT略低。两者均没有证明吞吐提升，未补重复；单次延迟取舍仅作后续候选线索，不声明8192在所有延迟目标下都最优。

本次模型实验从北京时间 **05:58至08:58，约3小时** 完成，未到8小时进度检查点。前期CPU冲突阻塞记录保留在本机；恢复检查仍有残余后台负载，**用户明确授权“继续测量，记录残余后台负载”**后执行。结论适用于这一实际条件，不代表隔离后台任务后的纯拓扑收益。

## 2. 运行条件与配置差异

复现从[公共TP4 recipe](../configs/recipes/vllm-tp4-baseline.yaml)、[节点46 target](../configs/targets/rtx6000d-46-tp4-baseline.yaml)、[公共C32 workload](../configs/workloads/dsv4-8192-1024-c32-n128-repeat3.yaml)及[节点46 campaign](../configs/campaigns/46-dsv4-vllm-tp4-baseline.yaml)派生。公共配置保留原文，探索配置仅留在本机；下表列出全部运行参数差异。

| 配置 | GPU顺序 | TP/PP/DP/EP | 相对公共TP4的改动 | 有效重复 |
| --- | --- | --- | --- | ---: |
| TP4 | 4,5,6,7 | 4/1/1/off | 无 | 3 |
| TP8 | 0,1,2,3,4,5,6,7 | 8/1/1/off | `tensor-parallel-size=8`；八卡target删除整个`binding` | 3 |
| TP4×PP2 | 0,1,2,3,4,5,6,7 | 4/2/1/off | `pipeline-parallel-size=2`；同一八卡target | 1+2 |
| TP2×PP4 | 0,1,2,3,4,5,6,7 | 2/4/1/off | `tensor-parallel-size=2`、`pipeline-parallel-size=4`；同一八卡target | 1+2 |
| TP2×PP4，prefill 4096 | 0,1,2,3,4,5,6,7 | 2/4/1/off | 在TP2×PP4上仅改`max-num-batched-tokens=4096` | 1 |
| TP2×PP4，prefill 16384 | 0,1,2,3,4,5,6,7 | 2/4/1/off | 在TP2×PP4上仅改`max-num-batched-tokens=16384` | 1 |

八卡target只另改ID、说明、GPU列表和删除`binding`；地址仍为10.90.1.46、端口31248，其他字段与公共target一致。recipe另改ID/说明/provenance，所有环境变量、flags和其他options保持一致。初筛和补测workload仅修改ID和`measurement.repetitions=1/2`；campaign引用对应recipe、target及workload，模型/runtime/client依赖不变。合并前逐字段核对服务、客户端、单次负载和源码一致；CSV中的重复编号连续，未修改runner按workload指纹分组的规则。

统一条件：FP8 E4M3 KV、显存比例0.90自动分配、上下文16384、max-num-seqs=32；chunked prefill、FULL_DECODE_ONLY、捕获尺寸[1,2,4,8,12,16,24,32]；V2 runner、async scheduling、mp executor；关闭EP、prefix cache和FlashInfer autotune，无投机解码或CPU offload。attention为FLASHINFER_MLA_SPARSE_DSV4，MoE保留auto，block-size=256、FP4 indexer关闭。prefill预算除表中两个可选配置外均为8192。

每个正式样本为单客户端、流式`/v1/completions`、C32、128请求、8192/1024；seed=0、temperature=0、ignore_eos=true、range_ratio=0、request_rate=inf。每轮预热64请求，同长度和并发，至少连续两轮无已知编译事件、最多五轮；每次测量最多两次尝试。启动1800秒、单客户端阶段7200秒上限不变。

- Git commit：`c54734568603d1ef52f6ccc24ce7673768777c37`。恢复时已有前次本节点阻塞报告、CSV及README状态变更；本轮未改公共源码或归档配置，也未commit/push。
- runner源码指纹：`f5c33cf418ed84629e3786be4f4c7e651d66dccc3827e4226685a42e3361b583`，八次run全程一致；运行前后配置SHA256核对一致。
- 服务和客户端image ID：`sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`；固定[vLLM runtime](../configs/runtimes/vllm-0.29.0.yaml)与[客户端](../configs/clients/vllm-bench-0.29.0.yaml)，未拉取或更换镜像。
- 模型身份：`1e1b7e54442bcf7f15ef55192641b5c5da360bcaa608236aa0d06b273aedf8b4`；48个分片、175,550,788,904字节，配置/tokenizer哈希及分片大小一致，不等同于全部权重内容校验和。
- 节点为8×RTX6000D，每卡85,651 MiB，驱动580.159.04、功耗上限600 W；未修改宿主功耗、频率或SMT设置。

四卡服务实际Docker cpuset为CPU32–47/NUMA2，客户端为CPU48–51/NUMA3；服务16个物理核、客户端4个物理核，各用一个SMT线程，兄弟分别为96–111、112–115。runner逐阶段验证容器及线程允许集合。八卡服务和客户端Docker cpuset为空，未开启`numa-bind`，未从外层引入taskset/numactl限制，另保存了进程/线程亲和性快照。

TP4 worker页面采样主要在NUMA2，同时可见其他节点的文件/共享页；不能把内存允许集合当作全部页面本地驻留的证明。通过实际GPU UUID、worker PID/进程名和日志核对：TP8的TP0–7对应GPU0–7；TP4×PP2的两个TP组为GPU0–3/4–7；TP2×PP4四个TP组为GPU0–1/2–3/4–5/6–7。

所有run日志均出现V2、FLASHINFER_MLA_SPARSE_DSV4、FLASHINFER_CUTLASS、expert_dtype=fp4、fp8_ds_mla、关闭autotune和Graph捕获完成证据。下列为日志中各worker实际KV预算的范围（GiB），不是声明的固定预算；自动分配随PP层分片、激活预算变化。

| 配置 | 实际KV预算范围 GiB/worker |
| --- | ---: |
| TP4 | 32.73–32.73 |
| TP8 | 51.75–51.75 |
| TP4×PP2 | 52.10–52.51 |
| TP2×PP4 | 51.74–53.48 |
| TP2×PP4，prefill 4096 | 52.54–54.40 |
| TP2×PP4，prefill 16384 | 50.14–51.65 |

初筛启动约TP4 318秒、TP8 337秒、TP4×PP2 308秒、TP2×PP4 288秒；相同recipe补测保留缓存，TP4×PP2/TP2×PP4重新启动约124/112秒。启动时间不计入正式吞吐。不同recipe使用框架各自的缓存目录，未手工清空缓存。

## 3. 性能汇总

[逐次CSV](../data/topology-node46.csv)保存14个有效样本原始数值精度。吞吐“±”为样本标准差，CV为样本标准差/均值；延迟表为各次对应指标的算术均值，**多个P95的均值不是合并请求后的P95**。n=1不计算标准差或CV。

| 配置 | n | output tokens/s，均值±标准差 | CV | requests/s | 相对同节点TP8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP4 | 3 | 712.00 ± 0.72 | 0.10% | 0.695309 | — |
| TP8 | 3 | 609.75 ± 0.88 | 0.14% | 0.595463 | +0.00% |
| TP4×PP2 | 3 | 666.70 ± 5.70 | 0.86% | 0.651079 | +9.34% |
| TP2×PP4 | 3 | 1099.61 ± 19.98 | 1.82% | 1.073841 | +80.34% |
| TP2×PP4，prefill 4096 | 1 | 924.61 | — | 0.902942 | — |
| TP2×PP4，prefill 16384 | 1 | 1089.59 | — | 1.064053 | — |

TP4到八卡的整体吞吐扩展分别为：TP8 **0.8564×**，TP4×PP2 **0.9364×**，TP2×PP4 **1.5444×（+54.44%）**。这同时改变GPU数量、CPU/NUMA绑定策略，且保留后台任务，不能视作纯TP/PP或理想线性扩展效率。TP4每GPU吞吐仍高于TP2×PP4；本轮优先选择较高整机吞吐，不宣称卡效全面占优。

| 配置 | mean TTFT ms | P95 TTFT ms | mean TPOT ms | P95 TPOT ms |
| --- | ---: | ---: | ---: | ---: |
| TP4 | 6315.74 | 20307.46 | 38.75 | 42.39 |
| TP8 | 10023.40 | 29473.12 | 42.68 | 48.96 |
| TP4×PP2 | 4396.00 | 12189.72 | 43.70 | 46.48 |
| TP2×PP4 | 3547.74 | 6870.65 | 25.65 | 27.94 |
| TP2×PP4，prefill 4096 | 2329.89 | 6578.17 | 32.33 | 36.20 |
| TP2×PP4，prefill 16384 | 5310.83 | 7929.64 | 24.20 | 27.38 |

| 配置 | mean ITL ms | P95 ITL ms | mean端到端 ms | P95端到端 ms |
| --- | ---: | ---: | ---: | ---: |
| TP4 | 38.75 | 23.53 | 45959.39 | 62562.74 |
| TP8 | 42.68 | 20.35 | 53684.24 | 75823.15 |
| TP4×PP2 | 43.70 | 36.21 | 49102.26 | 55469.86 |
| TP2×PP4 | 25.65 | 23.06 | 29785.77 | 33175.11 |
| TP2×PP4，prefill 4096 | 32.33 | 31.38 | 35404.63 | 38709.36 |
| TP2×PP4，prefill 16384 | 24.21 | 22.29 | 30063.23 | 30740.19 |

四个主配置的吞吐CV均小于3%，未触发追加诊断测量。TP4×PP2的mean TTFT CV为4.20%，单独保留这一波动；其mean TPOT CV为0.66%。TP2×PP4的mean TTFT/TPOT CV分别为2.38%/1.76%。不删除较慢的初筛样本；日志安静不作为数值收敛证明。

有效工作量合计 **1792成功、0失败，输入14,680,064、输出1,835,008 tokens**；每个样本均128成功、0失败，输入1,048,576、输出131,072 tokens。预热、拒绝尝试均未计入CSV或统计。

## 4. 异常与限制

**一次正式尝试被拒绝。** TP4第2个重复的第一次测量完成了工作量，但测量窗口出现24条TileLang编译相关事件，包含实际902-token形状JIT及缓冲输出的旧日志；因此按原gate拒绝，未当作纯日志误判绕过。保留该次630.50 tokens/s的原始结果，并在连续两轮重新预热后使用唯一一次重试取得有效样本。其他正式尝试均无已知编译事件，未放宽gate、增加预热/尝试上限或修改源码。

TP8首个重复及prefill4096初筛分别在第1/3轮预热出现事件，均用满五轮后达到末两轮安静。各PP初筛与补测新进程首轮也有JIT事件；这些属于预热证据，不能计入性能样本。完整服务日志没有匹配到OOM、抢占、CUDA错误或Traceback；这是日志观察结论，不替代更高频追踪或完整模型质量验证。

重要warning与硬件路径均保留：SM120的SymmMem communicator不可用；FlashInfer All Reduce对本次world_size不支持；TP4/TP8组存在超过两张PCIe GPU时custom allreduce禁用的提示；PP路径存在NCCL未批处理P2P的惰性通信器初始化及非可写buffer提示。另有TileLang向量化退为串行循环、FP8 scale精度提示、已弃用indexer配置提示、Graph显存估算提示，以及启动长操作时的共享内存广播等待提示。没有修改参数去隐藏它们；TP2组未出现相同custom allreduce禁用提示也不等于已通过profiler证明所有通信kernel。

恢复前重新采样整机CPU忙碌率约12.7%–13.0%，低于此前约63%，但仍有16个既存doca_spcx_cc进程及目标核/SMT上的动态占用。用户明确接受这一条件后才开测。未终止、迁移或重绑其他任务。NUMA2/3的低MemFree主要对应文件缓存，启动前全机可用内存约954 GiB、无swap，内存压力未见持续stall；保留了匿名页、共享内存、各NUMA内存与CPU压力快照，未清缓存或放宽四卡绑定。

本地监督脚本仅调用原runner，并约每5秒采样CPU和GPU、每30秒保存进程/线程亲和性及NUMA页面证据。下表为主矩阵有效样本近似测量窗口内的汇总；窗口用固定客户端在benchmark返回后写入的UTC秒级`raw.date`减去duration估计，存在秒级端点及返回后开销误差。CPU包含后台及本次活动；GPU功耗为所选GPU总和，不含CPU/整机其他功耗，不据此声称精确能耗。

| 配置 | 全机CPU平均忙碌率 | 所选GPU平均总功耗 W | 平均SM频率 MHz | 观测最高温度 °C |
| --- | ---: | ---: | ---: | ---: |
| TP4 | 16.79% | 1052.3 | 2418.1 | 60 |
| TP8 | 19.95% | 1561.1 | 2422.0 | 56 |
| TP4×PP2 | 19.71% | 1581.7 | 2420.9 | 58 |
| TP2×PP4 | 18.83% | 2067.5 | 2415.2 | 62 |

低频遥测未见持续降频，不能排除短暂干扰或背景负载偏差。顺序执行也不消除时间漂移；本轮未进行重复交错、后台任务隔离、八卡NUMA绑定或其他并发对照。所见PP收益可能同时涉及TP通信范围、PP调度、kernel形状和更大的逻辑KV容量，不能只归因于某一因素。

本次8个run所属服务/客户端及临时CLI容器均已退出，收尾检查Docker运行列表和GPU计算进程为空。保留权重、镜像、全部JIT缓存、原始日志/JSON/预热/失败尝试、GPU映射及CPU/NUMA遥测。**原始产物和探索配置仅节点46本机可用**，具体位置与恢复命令见本机执行记录；本报告的仓库链接仅指向随仓库分发的公共配置与逐次CSV。未批量归档探索配置，也未commit/push。
