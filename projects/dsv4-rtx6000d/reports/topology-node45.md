# 节点45：并行拓扑实测与矿机进程恢复占用CPU的影响

## 1. 结论与完成范围

**主矩阵完成12次 runner gate 通过的测量，仍可用于当前节点在高CPU背景负载下的拓扑筛选；不能直接用作排除该负载后的性能基线或跨节点排名。** 2026-09-17在 `gpu-6000d-45` 按 TP4 → TP8 → TP4×DP2 EP off → EP on 执行。矿机进程在实验期间自动重启、重新占用CPU的情况是在收尾复核时发现的：额外负载从TP4第二次测量期间出现，此后整机CPU忙碌率约60%–64%，并非实验结束后才出现。这里沿用用户对“矿机进程自动重启”的描述；直接证据确认PID5410对应服务恢复消耗CPU，尚未区分进程重新创建、解除冻结或其他自动恢复机制。只有TP4第一次未观察到这次额外负载，其余11个样本均有此资源条件限制。

**本节点八卡对照仍支持优先继续优化DP2 EP on。** TP8、DP2 EP off和EP on的全部测量都在高背景负载出现后完成，EP on三次吞吐均高于另两组。但后台负载对各拓扑的影响未必相同，不能把当前提升比例或排序直接外推到正常背景；TP4三次跨越背景变化，整组比较的可比性更弱。

**8个mlx网卡的 `doca_spcx_cc` 忙线程合计约占用8个宿主逻辑CPU**，属于用户明确接受的固定背景条件，CPU绑定始终保持原方案。另一个异常服务 `tmuxinfoada578b1.service`（cgroup列出PID5410）在收尾采样中单独消耗约64个逻辑CPU。两类负载不可混淆，也没有证据将DP波动全部归因于DOCA。

| 配置/任务 | 执行完成情况 | 可用于的结论 |
| --- | --- | --- |
| TP4，GPU4–7 | 3次gate通过；第2次期间背景改变，第3次受额外负载影响 | 功能、绑定和工作量证据；仅第1次符合已接受背景，不能形成三次基线 |
| TP8，GPU0–7 | 3次gate通过，均有额外CPU负载 | 可作本节点高背景负载下的八卡参照，不作正常背景参照 |
| TP4×DP2，EP off | 初筛1＋补测2；均有额外CPU负载，吞吐CV6.45% | 配置可运行，但性能不稳定 |
| TP4×DP2，EP on | 初筛1＋补测2；均有额外CPU负载，吞吐CV8.07% | 观测均值最高，值得后续复核，不能称为稳定最优 |
| EP on、prefill4096 | 启动中查实资源冲突，主动中断；无测量样本 | 无性能结论 |
| prefill16384 / 其他可选配置 | 已离线validate/plan，未启动 | 无性能结论 |
| 诊断追加测量 | 0次 | 没有通过反复重跑选择快样本 |

主矩阵从北京时间05:58至09:07，约3.16小时；09:19中断可选初筛并清理本次容器，早于8小时检查点。任务按**部分完成、资源阻塞**交接；恢复正常背景后需要独立的新批次复核，不能把本批受影响样本混作干净基线。未操作其他节点、SGLang或宿主功耗/频率。

## 2. 运行条件与配置差异

复现起点为 [TP4 recipe](../configs/recipes/vllm-tp4-baseline.yaml)、[C32 workload](../configs/workloads/dsv4-8192-1024-c32-n128-repeat3.yaml)、[45 target](../configs/targets/rtx6000d-45-tp4-baseline.yaml)、[基线 campaign](../configs/campaigns/45-dsv4-vllm-tp4-baseline.yaml)、[runtime](../configs/runtimes/vllm-0.29.0.yaml)和[客户端](../configs/clients/vllm-bench-0.29.0.yaml)。在工作区复制后派生候选，公共配置保留原样。

| 配置 | 相对公共TP4的recipe变化 | Target与环境 |
| --- | --- | --- |
| TP4 | 无 | GPU4–7；服务CPU32–47/NUMA2；客户端CPU48–51/NUMA3 |
| TP8 | `tensor-parallel-size: 8` | GPU0–7；移除整个 `binding`，不从外层引入亲和性限制 |
| TP4×DP2，EP off | `data-parallel-size: 2`、`data-parallel-size-local: 2`、`max-num-seqs: 16`；Graph尺寸 `[1,2,4,8,12,16]`、上限16 | 同TP8；TP=4、PP=1 |
| TP4×DP2，EP on | 同DP2 off，将 `no-enable-expert-parallel` 替换成 `enable-expert-parallel` | 同TP8 |
| EP on、prefill4096/16384 | 仅改对应 `max-num-batched-tokens`，其余同EP on | 同TP8；4096启动被中断，16384仅离线检查 |

派生配置另改ID/描述；初筛与补测workload仅改ID、重复数为1或2，服务、单次负载、客户端、预热规则和源码已核对一致，CSV重复编号连续为1–3。runner仍按原workload指纹分组，仅离线合并逐次数据。其余环境变量不变：`VLLM_USE_V2_MODEL_RUNNER=1`、`TILELANG_CACHE_DIR=/root/.cache/tilelang`，沿用target/client离线变量和runtime的 `NCCL_DEBUG=WARN`。

统一上下文16384、FP8 E4M3 KV、显存比例0.90自动分配、block-size256、FP4 indexer关闭、chunked prefill、FULL_DECODE_ONLY Graph、FlashInfer autotune关闭、async scheduling、mp executor，无前缀缓存、投机解码或CPU offload。DP1活动容量32，Graph尺寸 `[1,2,4,8,12,16,24,32]`。DP2每rank容量16，**prefill8192也是每rank预算，整机上限16384**；不能称为与DP1相同的整机调度预算。

客户端单API、流式 `/v1/completions`；全局C32、128请求、8192输入/1024输出、seed0、temperature0、ignore_eos、range_ratio0、request_rate无限。每重复预热64请求/轮，连续两轮无已知编译事件，最多五轮；测量最多两次尝试。启动1800秒、客户端阶段7200秒超时，全部沿用原gate。

### 身份、绑定及实际运行证据

- 执行Git commit：`c54734568603d1ef52f6ccc24ce7673768777c37`。
- 所有运行的runner源码指纹：`f5c33cf418ed84629e3786be4f4c7e651d66dccc3827e4226685a42e3361b583`。无公共代码修改；本地新增候选配置、证据包装/分析脚本和节点文档。各次运行前后源码与配置文件哈希一致，包装脚本只负责运行现有runner和记录遥测。
- 服务与客户端镜像：`vllm/vllm-openai:v0.29.0`；完整ID `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`，没有拉取/更换镜像。
- 模型 `nvidia/DeepSeek-V4-Flash-0731-NVFP4`；identity `1e1b7e54442bcf7f15ef55192641b5c5da360bcaa608236aa0d06b273aedf8b4`。保存模型/tokenizer元数据哈希及48个权重分片大小，未对全部权重内容重新计算哈希。
- 硬件8张RTX6000D、每卡85651 MiB、驱动580.159.04；功耗上限600 W，未改变。宿主128逻辑CPU；GPU0–3近NUMA0、GPU4–7近NUMA2。
- TP4的Docker inspect及worker/client线程允许集合核验了上述绑定；SMT兄弟为服务96–111、客户端112–115。本次进程不使用这些兄弟线程，但宿主后台可使用。八卡Docker cpuset字段为空，无显式CPU/内存绑定；已保存 `numa_maps` 页面分布快照，不能据此声称所有页面本地驻留。
- DOCA各忙线程的CPU允许集合为0–127，并非独占保留8个物理核。准备阶段观测到服务CPU33、客户端CPU48，以及与服务CPU32同核的CPU96发生重叠。网卡IRQ/NUMA映射固定不等于DOCA用户态线程独占固定CPU。按用户要求保持原资源绑定，不为避开DOCA更换CPU或关闭SMT。

主矩阵各配置的health/models及关闭thinking的中文探测通过，日志要求与kernel checks通过。实际日志包含V2 Model Runner、`FLASHINFER_MLA_SPARSE_DSV4`、`FLASHINFER_CUTLASS`、expert_dtype fp4、fp8_ds_mla、跳过FlashInfer autotune及Graph捕获。

| 配置 | 日志模型分配 GiB/卡 | 可用KV预算 GiB/卡 | 实际并行语义 |
| --- | ---: | ---: | --- |
| TP4 | 39.84 | 32.73 | GPU4–7，TP4 |
| TP8 | 20.83 | 51.75 | GPU0–7，TP8 |
| DP2 EP off | 21.71 | 49.99 | Attention TP组0–3/4–7；MoE跨DP合并TP分片 |
| DP2 EP on | 22.48 | 49.81 | Attention TP组0–3/4–7；EP rank0–7 |

GPU UUID到worker的映射与日志确认DP0使用GPU0–3、DP1使用GPU4–7。**DP2 EP off并非两个完全独立的完整TP4模型副本**：固定镜像日志明确 `Detected DP deployment with no --enable-expert-parallel. Falling back to AllGather+ReduceScatter dispatch/combine.`；镜像内FusedMoE实现将TP跨DP展开为8路MoE张量分片，源文件证据保存在本机。EP on日志为 `AgRsAll2AllManager`、`MoEPrepareAndFinalizeNaiveDPEPModular`，worker名覆盖EP0–7。

DP2只暴露端口31248、由一个客户端施加全局C32，但vLLM默认启动**2个API worker进程**。抽样 `/metrics` 观察到两个rank同时各16个running请求，EPoff末次测量两个rank各成功64请求；采样抢占计数为0。端口数量不等于API进程数量，DP同步/dispatch及MoE分片均需纳入解释。

## 3. 性能汇总（受资源冲突限制的描述统计）

[逐次CSV](../data/topology-node45.csv)保留12个gate通过样本的原始精度，包含 `runner_gate`、`resource_status` 和 `eligible_for_clean_topology_comparison`。最后一列均为False，表示不能构成已排除异常负载的完整拓扑对照，**不表示样本对本节点高背景负载下的比较没有价值**；即使TP4第一次未见额外负载，单次样本也不能构成完整的正常背景基线。未删除慢样本，未混入预热、拒绝尝试、诊断或中断的可选配置。

每个样本均成功128、失败0，实际输入1,048,576、输出131,072 tokens；12次共**1536成功、0失败，输入12,582,912、输出1,572,864 tokens**。以下保留全部原始统计，用于本节点当前负载条件下的比较；**不能据此宣布正常背景下的提升幅度或跨节点优劣**。

吞吐为三次均值±样本标准差，CV为样本标准差/均值：

| 配置 | n | 输出 tok/s | CV | requests/s均值 | 资源限制 |
| --- | ---: | ---: | ---: | ---: | --- |
| TP4 | 3 | 711.71 ± 5.54 | 0.78% | 0.69503 | 额外负载在第2次期间出现 |
| TP8 | 3 | 649.74 ± 1.02 | 0.16% | 0.63452 | 3次均观察到额外CPU负载 |
| TP4×DP2，EP off | 3 | 683.71 ± 44.11 | 6.45% | 0.66769 | 3次均观察到额外CPU负载 |
| TP4×DP2，EP on | 3 | 833.98 ± 67.27 | 8.07% | 0.81443 | 3次均观察到额外CPU负载 |

以下延迟全部为ms；每列是三次对应指标的均值，**P95的均值不是合并请求的P95**。ITL沿用客户端流式事件统计，不能简单视作逐个输出token的独立间隔。

| 配置 | Mean TTFT | P95 TTFT | Mean TPOT | P95 TPOT | Mean ITL | P95 ITL | Mean E2EL | P95 E2EL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP4 | 6469.98 | 20640.91 | 38.62 | 42.40 | 38.62 | 22.44 | 45981.25 | 62926.33 |
| TP8 | 9116.79 | 26604.19 | 40.33 | 45.94 | 40.33 | 19.60 | 50377.08 | 71165.80 |
| TP4×DP2，EP off | 11085.60 | 24156.72 | 35.24 | 41.17 | 35.24 | 20.55 | 47134.95 | 60139.76 |
| TP4×DP2，EP on | 6408.96 | 14740.37 | 29.90 | 32.97 | 29.90 | 21.62 | 36999.36 | 45634.71 |

| 配置 | Mean TTFT样本标准差ms / CV | Mean TPOT样本标准差ms / CV |
| --- | ---: | ---: |
| TP4 | 77.13 / 1.19% | 0.27 / 0.71% |
| TP8 | 418.59 / 4.59% | 0.43 / 1.07% |
| TP4×DP2，EP off | 660.13 / 5.95% | 1.12 / 3.18% |
| TP4×DP2，EP on | 232.96 / 3.63% | 1.16 / 3.87% |

本节点高CPU背景负载下的八卡比较：EPoff对TP8的均值为 +5.23%，EPon对TP8为 +28.35%，EPon对EPoff为 +21.98%。TP8三次范围648.75–650.79 tok/s，EPoff为632.82–710.94 tok/s，EPon为783.57–910.36 tok/s。**EPon三次均高于另外两组，足以支持将它列为本节点优先复核的候选**；EPoff与TP8的范围重叠，其较小均值优势未分出稳定优劣。上述提升比例是当前条件的描述，尚不代表排除异常CPU负载后的收益。

四卡到八卡的整体变化：TP8相对TP4 -8.71%，DP2 EPoff -3.93%，DP2 EPon +17.18%。GPU数量、CPU/NUMA绑定、DP预算及背景负载均有差异，不能解释成纯拓扑扩展效率。TP4/TP8较低CV也不能抵消资源条件不一致；TP8的Mean TTFT CV另有4.59%。

## 4. 异常与限制

### 实验期间矿机进程自动重启/恢复运行的时间线

准备阶段用户处理PID5410后，曾确认异常cgroup CPU增量为0，整机忙碌率约6.60%，剩余DOCA约8个逻辑CPU，随后获得按原绑定正式运行的授权。回查10秒遥测发现，北京时间**06:26:14–06:26:24**（UTC 2026-09-16 22:26:14–22:26:24），宿主CPU忙碌率从10.88%升至57.19%，随后持续约60%–64%。TP4第二次被接受的测量客户端阶段为06:25:56–06:29:38，覆盖该变化。

北京时间09:17:07–09:17:17只读采样确认 `tmuxinfoada578b1.service/cgroup.events` 为 `frozen 0`、`cgroup.procs` 为5410；约10.001秒内 `usage_usec` 增加640,074,122，等效**64.0007个逻辑CPU**。这是当前异常服务的直接证据；历史遥测未连续采集该cgroup，不能精确断言此前每一秒都由同一PID贡献。

| gate通过样本 | 客户端阶段整机CPU平均忙碌率 | 资格说明 |
| --- | ---: | --- |
| TP4重复1 | 10.1% | 未观察到额外负载；仍含DOCA与服务自身 |
| TP4重复2、3 | 55.6%、60.0% | 第2次期间背景改变，第3次处于高背景 |
| TP8重复1–3 | 62.9%、62.7%、62.9% | 均受额外CPU背景影响 |
| DP2 EPoff重复1–3 | 63.5%、63.5%、63.6% | 均受额外CPU背景影响 |
| DP2 EPon重复1–3 | 63.5%、63.5%、63.1% | 均受额外CPU背景影响 |

忙碌率取 `/proc/stat` 相邻差值，排除idle和iowait，包含服务/客户端自身；这些数值不是后台服务的单独占用率。普通进程快照未显示PID5410，运行期监控没有及时告警，收尾对CPU总量进行交叉核对才发现。原gate检查编译及工作量，不检查这类后台CPU冲突，因此runner的PASS不能代替资源验收。后续恢复需持续监看 `/proc/stat` 和该异常cgroup的CPU增量，不能仅凭 `ps` 看不到PID判断已禁用。

确认冲突后，09:19向本次可选实验runner发送SIGTERM，runner记录INTERRUPTED并完成清理；没有停止/删除外部异常服务，也没有改变网卡线程。可选4096尚在readiness阶段，无正式或预热样本；16384未启动。后台异常持续存在时不继续测量；正式复核应使用新run-root并单独统计。

### 对本节点和跨节点比较的具体影响

| 比较或结论 | 本批数据的用途与限制 |
| --- | --- |
| 本节点TP8、DP2 EP off、DP2 EP on | 三组都在额外高负载恢复后完成，整机CPU平均忙碌率接近，仍能反映当前节点在该背景下的实际拓扑表现。EP on是优先继续优化的候选；DP两组的较高CV需一并保留。 |
| 本节点TP4与八卡拓扑 | TP4重复1在额外负载出现前，重复2跨越恢复时刻，重复3在恢复后；其三次均值混合了不同背景。还同时改变GPU数量、CPU/NUMA绑定和DP调度预算，因此不能据此单独量化拓扑扩展效率。 |
| 与其他节点同配置比较 | 如果其他节点没有同等背景负载，吞吐和延迟差异混入资源干扰，不能直接解释为节点硬件、GPU位置或配置的优劣。数据保留用于审计，不纳入未经条件对齐的跨节点性能排名。 |
| 正常背景下的收益与排序 | 各拓扑使用不同的CPU线程、调度和通信路径，额外CPU负载未必造成相同比例的损失。即使整机忙碌率接近，也不能证明每种拓扑受到相同程度的干扰；当前提升百分比和排序需要在排除异常进程后复核。 |

后台进程可能与服务、客户端及其SMT兄弟争用CPU，也可能影响调度和通信；本批没有同配置、同绑定的正常背景配对实验，**无法量化它具体降低了多少吞吐或增加了多少延迟，也不能将DP波动全部归因于它**。没有删除、校正或按假定比例补偿任何原始指标。8个mlx网卡DOCA忙线程约8个逻辑CPU属于另行记录、用户接受的背景，不与额外约64个逻辑CPU的矿机负载合并归因。

### Gate、波动诊断和保留的warning

测量共3次因编译事件被拒绝：TP4重复2第一次尝试24条，TP8重复1第一次尝试48条，EPoff全局重复2第一次尝试4条TileLang事件。均在原两次尝试上限内重试后通过。预热发生的编译按原连续两轮静默规则处理，无放宽gate。主矩阵未发生OOM、不支持或请求超时；可选中断是资源冲突处置，不是模型不支持。

DP两组CV超过3%，已检查测量窗口JIT、调度计数、CPU/NUMA与GPU遥测。所有被接受窗口均0条已知编译事件，60秒进程快照未见编译器；DPoff活动GPU采样SM频率2422–2430 MHz、最高55°C，DPon为2415–2422 MHz、最高56°C，未见明显热降频证据。GPU/CPU为10秒采样、进程为60秒，不能排除瞬时干扰。`numa_maps`/meminfo已保存，但八卡未绑定，未建立NUMA远端访问的因果结论。

抽样DP请求数与running计数未显示明显持续单rank闲置；不能把chunked prefill期间的排队差异直接认定为路由故障。采样抢占计数0也不能替代完整调度跟踪。异常CPU背景是明确的资源混杂因素，但不足以独立解释全部DP重复差异；未追加诊断测量去挑选更快结果。

保留重要日志：CC12.0不支持SymmMemCommunicator；FlashInfer All Reduce不支持world_size4/8而禁用；超过2张PCIe-only GPU不支持custom allreduce；TileLang向量化fallback；ModelOpt NVFP4实验格式及FP8 scale相关提示；indexer参数弃用警告。EPoff额外有AllGather+ReduceScatter dispatch/combine回退。没有为消除warning增加关闭开关，日志与配置核验不代表全部kernel或数值质量已经验证。

全部原始结果、预热/拒绝窗口、解析配置、命令、镜像内实现摘录、CPU/GPU及NUMA证据保留在**节点45本机实验工作区**，不作为GitHub复现链接。仓库仅归档本报告、逐次CSV及README节点45状态；探索配置可由上表和公共配置重建。结束后本次容器已清理、GPU无计算进程，模型、镜像、缓存和历史结果保留。
