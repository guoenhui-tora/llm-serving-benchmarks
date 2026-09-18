# 节点47：八卡整机部署对照

## 结论与完成范围

**本节点本轮优先保留双 TP4、EP off。** 在固定八卡、8192 输入／1024 输出、统一 CPU/NUMA 与全局活动容量64条件下，完整通过的四种配置中，双 TP4 在 C32、C64 的吞吐均最高，分别为 **1096.13 ± 8.96**、**1429.63 ± 6.19 output tok/s**，吞吐 CV 为0.82%／0.43%。结论仅适用于节点47、本轮固定负载和保留 DOCA 后台任务的条件，不代表其他节点或并发的最优配置。

9月18日凌晨向两个 EP off 候选各追加一个批次：增加本地计时外C64覆盖预热后，TP2×DP4完成C32/C64各三次；TP4×DP2仍出现新的MHC形状和真实编译，未进入正式测量。未修改公共runner、gate或正式recipe。新增结果和比较限制见下文。随后经用户授权，于10:32–12:07仅向TP4×DP2 EP off追加一个扩充预热上限的批次：覆盖预热通过，C32完成三次，但C64两次测量均出现worker首次使用事件而被拒绝；完整日志无TileLang实际编译记录，缓存工件未变。正式CSV仍为24个整机样本，最新部分结果见文末。

TP4×DP2、EP on 的 C32/C64 吞吐较双 TP4 低21.70%／19.45%；TP2×DP4、EP on 低20.97%／13.12%。两个 EP on 配置的 C32 CV 均超过8%，TP4×DP2 的 C64 CV 也超过3%；它们通过 runner gate，但不能称为稳定基线。TP2×DP4、EP on 的 C64 CV 为1.88%，仍未追平双 TP4 吞吐。

首批于2026-09-17 20:15:31 至09-18 00:54:44（北京时间），按分配顺序完成全部五个配置的尝试，历时约4小时39分，未到8小时检查点。没有扩展矩阵、访问其他节点或改动运行参数。

下表保留首批结果；9月18日追加批次见后文“JIT诊断与补测”。当前 TP2×DP4、EP off 已有完整通过的新批次，TP4×DP2、EP off 仍阻塞。

| 配置／case ID | C32 | C64 | 首批最终状态 |
| --- | --- | --- | --- |
| 双 TP4，`dual-tp4` | 三次通过 | 三次通过；第三次使用原上限内第二次尝试 | PASS |
| TP4×DP2，`tp4-dp2-epoff` | 三次部分结果保留 | 首次测量前五轮预热未达标；三次均未测 | FAIL，预热阻塞 |
| TP4×DP2，`tp4-dp2-epon` | 三次通过 | 三次通过 | PASS，波动限制 |
| TP2×DP4，`tp2-dp4-epoff` | 三次部分结果保留 | 首次测量前五轮预热未达标；三次均未测 | FAIL，预热阻塞 |
| TP2×DP4，`tp2-dp4-epon` | 三次通过 | 三次通过 | PASS，C32波动限制 |

[逐次 CSV](../data/node8-node47.csv)由[统一导出器](../../../scripts/export_samples.py)生成，筛选完整 PASS 批次的 `purpose=performance`：首批18个整机样本加补测6个，共24行整机样本、24行 `replica-0`、6行 `replica-1`，合计54行。只对 `scope=node` 求和，共4608成功、0失败，输入37,748,736、输出4,718,592 tokens。首批两份 FAIL 配置的6个 C32 trial 仍单列，不与新批次拼接；追加的诊断预热和验证也不进入CSV。首批共24个已完成且通过 trial gate 的正式测量、1次被拒绝的正式尝试，另外6次 C64 因预热阻塞未执行；补测新增6次正式测量，无拒绝尝试。

## 运行条件与实际参数

复现入口为[节点47公共 campaign](../configs/campaigns/47-dsv4-node8.yaml)，统一规则见[项目任务说明](../README.md#下一轮八卡整机部署对照)。首批先复制公共 `configs/` 到独立工作区，五个配置均通过 validate、plan、固定镜像 CLI preflight；每个配置只启动一次，在同一服务生命周期内顺序处理 C32/C64 和三次重复。没有改公共源码、recipe、workload、gate 或宿主频率／功耗。

- 本机：`gpu-6000d-47`，地址 `10.90.1.47`；8张 NVIDIA RTX 6000D，单卡85,651 MiB，驱动580.159.04，功耗上限600 W。启动前八卡无计算进程，Docker 无已有运行容器。
- 执行 Git commit：`aacdc39f0862b82527753ce3ac5cbf0de94c1c68`，初始工作树干净。五次 run 的源码指纹均为 `edccf507e1246a4525a383ff4c4d3ce45a470378075bf3271922903c10020e34`，结束后重新核对源码及工作区全部配置哈希未变。
- 服务和客户端均为 `vllm/vllm-openai:v0.29.0`，image ID `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`；RepoDigest 为同一 SHA256。全程 `--pull never`，没有下载或替换镜像。
- 权重/tokenizer：`/data/models/DeepSeek-V4-Flash-0731-NVFP4`。48个分片合计175,550,788,904字节；配置、tokenizer及分片大小身份为 `1e1b7e54442bcf7f15ef55192641b5c5da360bcaa608236aa0d06b273aedf8b4`，未重算完整权重内容哈希。
- 全部固定上下文16384、FP8 E4M3 KV、显存比例0.90自动分配、block size256、prefill每 DP rank 8192、chunked prefill、V2 runner、async scheduling、mp executor、`FULL_DECODE_ONLY`。关闭 autotune、prefix cache、投机解码和 CPU offload；保留 JIT 缓存，不启用 `--numa-bind`。

| 配置 | 外部服务数 | 每服务 TP/PP/DP | 每 DP rank 容量 | 整机 prefill 上限 | 实际 KV GiB/卡 | 每 DP rank 日志逻辑 KV tokens |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 双 TP4 | 2 | 4/1/1 | 32 | 16384 | 32.73 | 85,450 |
| TP4×DP2，EP off | 1 | 4/1/2 | 32 | 16384 | 49.95 | 130,404 |
| TP4×DP2，EP on | 1 | 4/1/2 | 32 | 16384 | 49.79 | 129,973 |
| TP2×DP4，EP off | 1 | 2/1/4 | 16 | 32768 | 46.61 | 121,683 |
| TP2×DP4，EP on | 1 | 2/1/4 | 16 | 32768 | 46.47 | 121,298 |

以上实际参数表来自首批，补测两个EP off配置的KV、Graph与绑定保持一致。全局活动容量均为64；容量32的 Graph 尺寸为 `[1,2,4,8,12,16,24,32]`，容量16为 `[1,2,4,8,12,16]`，C32/C64共用同一服务配置。实际日志确认 Graph 捕获、`FLASHINFER_MLA_SPARSE_DSV4`、`FLASHINFER_CUTLASS`、专家 FP4、`fp8_ds_mla`、V2及跳过 autotune。逻辑 KV tokens 与 GiB 是不同口径，0.90不代表等量 KV。

双 TP4 分别使用 GPU0–3／4–7，服务 CPU0–15／32–47、内存 NUMA0／2；客户端 CPU16–19／48–51、内存 NUMA1／3；端口31248／31249。单套八卡 DP 使用上述 CPU/NUMA 集合的并集，端口31248。所有配置服务32个物理核、客户端8个物理核，SMT兄弟（逻辑编号加64）不分配给本次进程。Docker inspect、服务/客户端线程允许集合检查通过；这些绑定不保证后台独占，也不证明页面全部本地驻留。

TP4×DP2 为两个本机 DP rank、TP组0–3／4–7；TP2×DP4 为四个本机 DP rank、TP组0–1／2–3／4–5／6–7，PP均为1。DP使用本地通信、动态初始化端口，未配置外部 rank。最终 TP2×DP4、EP on 的 GPU UUID、宿主 PID 和 worker 标题交叉核对确认 DP0/TP0–1/EP0–1 对应GPU0–1，依次至 DP3/EP6–7 对应GPU6–7。EP on 为八卡专家组，每rank 32/256专家、linear placement；EP off 保留关闭开关。DP内部 MoE 的跨rank通信不等于多份独立 TP 服务。

实际通信日志中，TP4使用 PYNCCL，PCIe超过两卡的 custom allreduce禁用；TP2的 TP组允许 `CUSTOM` 后接 `PYNCCL`，DP/EP相关组为 PYNCCL，DP候选使用 `AgRsAll2AllManager`，NCCL为2.30.7。这里只记录选择路径，没有 profiler 证明耗时归因。DP负载快照中曾观察到 TP4×DP2 的17/15 running、TP2×DP4的15/15/16/14 running；内部路由并非始终严格均分，不能把外部等分协议理解为内部 DP 负载均衡测试。

客户端统一流式 `/v1/completions`、seed0、temperature0、ignore_eos、range_ratio0、request_rate=inf。C32每次128请求、每轮预热64请求；C64每次256请求、预热128请求。双 TP4 按全局 seed0 请求索引交错分片，等分并发/请求；单套 DP 也走相同同步客户端。首批及凌晨补测的正式重复前至少连续两轮无已知编译事件，最多五轮；正式测量最多两次尝试，启动1800秒、客户端阶段7200秒。上午扩充预热上限批次的差异见文末。

## 性能表

下表仅包含完整 PASS 批次（TP2×DP4 EP off使用补测批次），吞吐为三次均值±样本标准差；延迟为三次对应指标的均值，单位均为 ms。P95列不是把三次请求再次合并后的分位数。完整 mean/P95 TTFT、TPOT、ITL、E2EL与各侧窗口见 CSV。

| 配置 | 整机并发 | 输出 tok/s | CV | Mean TTFT | P95 TTFT | Mean TPOT | P95 ITL | P95 E2EL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 双 TP4 | 32 | **1096.13 ± 8.96** | **0.82%** | 4785.93 | 10773.03 | 24.42 | 18.17 | 36424.01 |
| 双 TP4 | 64 | **1429.63 ± 6.19** | **0.43%** | 6396.11 | 20653.55 | 38.41 | 22.84 | 62110.41 |
| TP4×DP2，EP on | 32 | 858.21 ± 74.73 | 8.71% | 6320.32 | 13555.42 | 31.29 | 24.12 | 46495.29 |
| TP4×DP2，EP on | 64 | 1151.59 ± 48.74 | 4.23% | 7876.52 | 26040.84 | 45.77 | 44.15 | 75646.58 |
| TP2×DP4，EP on | 32 | 866.31 ± 70.78 | 8.17% | 7650.69 | 13352.24 | 29.64 | 25.24 | 43852.12 |
| TP2×DP4，EP on | 64 | 1242.03 ± 23.34 | 1.88% | 9248.22 | 23115.01 | 38.67 | 29.02 | 62757.96 |
| TP2×DP4，EP off，补测 | 32 | 786.83 ± 3.74 | 0.47% | 10853.61 | 17838.52 | 30.08 | 24.11 | 48167.82 |
| TP2×DP4，EP off，补测 | 64 | 1029.12 ± 9.54 | 0.93% | 13296.35 | 34180.94 | 46.01 | 27.84 | 82502.48 |

补测在正式C32/C64之前增加计时外C64形状覆盖预热，服务参数、正式负载和gate不变；其预热历程与首批不同，不能视为完全相同的预热协议。补测C32三次为783.95/785.49/791.05，C64为1018.70/1031.25/1037.42 output tok/s，均未超过双TP4；仅验证本次启动内的重复性。

双 TP4 的六次起跑时差为0.073–0.758 ms，结束时差0.35–1.36秒，空闲尾段全部保留。整机吞吐使用两侧官方计时窗口并集，不是两侧自身窗口吞吐直接相加；CSV分列保存自身窗口值与共同窗口贡献。单套 DP 只有一个外部客户端窗口，时差为0。

首批两份失败配置的 C32 **仅作为部分结果**，不混入上表与CSV；C64没有正式结果，不能据此作完整部署排名：

| 配置 | C32三次输出 tok/s | 均值±标准差 | CV | Mean TTFT ms | Mean TPOT ms |
| --- | --- | ---: | ---: | ---: | ---: |
| TP4×DP2，EP off | 745.59 / 672.50 / 646.74 | 688.27 ± 51.28 | 7.45% | 10163.25 | 36.74 |
| TP2×DP4，EP off | 775.45 / 788.83 / 733.49 | 765.93 ± 28.88 | 3.77% | 11081.81 | 31.01 |

首批全部24个已完成 trial 均逐条核对成功状态与8192/1024长度；C32为128成功、0失败、输入1,048,576／输出131,072 tokens，C64为256成功、0失败、输入2,097,152／输出262,144 tokens。副本行没有重复计入整机总量。

## 异常、资源验收与复现证据

**首批两处阻塞均为 C64 预热未达到原门槛，不是 OOM 或模型不支持。** TP4×DP2、EP off 的五轮事件数为64/24/40/24/0；TP2×DP4、EP off 为6/4/2/6/0。均只有最后一轮安静，按原上限退出，未重启补测、改精度、改Graph或更换拓扑。两者完整内核日志 gate 都为 PASS，但 case 为 FAIL；先确认清理成功及资源正常，再继续下一独立候选。

双 TP4 的 C64第三次首次尝试因后四卡24条事件整组拒绝，重新完成两轮安静预热后第二次尝试通过。该次 Docker 时间为13:06:42 UTC的部分记录内嵌12:43:54／12:44:05的 TileLang时间，存在明显缓冲延迟；DP预热也有类似现象，同时存在当轮 JIT monitor 事件。所有反例原样保留，未改正则、隐藏 warning 或将拒绝样本计入CSV。没有额外第三次正式尝试。

首个有效样本前的预热总轮数：双 TP4 的 C32/C64均三轮；TP4×DP2 EP off 的 C32四轮；TP4×DP2 EP on 的 C32三轮、C64四轮；TP2×DP4 EP off 的 C32三轮；TP2×DP4 EP on 的 C32三轮、C64两轮。其他正常重复均两轮；上面的拒绝尝试另行保存。日志无已知事件不证明所有即时编译或调度形状已经收敛。

**所有 CSV 行经逐次审核后标为 `accepted-background`，表示接受约定后台条件，不表示无干扰或性能稳定。** 启动前10秒采样整机 CPU忙碌率9.35%；10个 `doca_spcx_cc` 持续约占10个逻辑CPU，其中6个属于 `roce-init.service`，另外4个在用户会话cgroup，均按约定保留。进程曾运行在服务物理核及SMT兄弟上，因此Docker cpuset并不隔离DOCA。NUMA0–3启动前空闲约157/153/112/39 GiB，整机可用内存约952 GiB、无swap，模型盘与工作区空间充足。

首批运行全程保存5秒 CPU/GPU、约30秒 system-service cgroup 和额外30秒进程增量采样。逐次按官方 monotonic 测量窗口对齐检查：24个已完成 trial 未发现矿工异常或新增持续高负载外部进程；system-service cgroup采样中增量超过0.3 CPU的只有保留的 `roce-init.service`（约6 CPU）。双 TP4及TP4×DP2的窗口平均 CPU约15.86%–16.27%，TP2×DP4约17.21%–17.86%；峰值最高18.75%以内，没有45%资源观察告警。测量采样 GPU温度46–60°C，SM频率2415–2430 MHz，未见持续热降频证据。低频采样不能排除瞬时干扰。

对首批全部CV>3%的组保留慢样本并检查JIT、抢占、CPU、客户端、频率与通信日志：测量gate无已知编译事件；完整日志无抢占记录，诊断时 `/metrics` 各DP rank累计抢占为0；客户端在完全落入测量窗口的30秒进程CPU区间内，单个客户端容器内进程合计峰值约0.133 CPU，未见持续饱和，仍不能排除瞬时停顿。四个DP配置的C32均出现 coordinator `out-of-order step` warning与长ITL，最大ITL分别为10.69、7.79、12.23、9.45秒（按执行顺序），资源变化不足以解释波动，尚未确定根因。没有用不断补测或删除慢样本取得低CV。

保留 SM120 SymmMem不可用、FlashInfer All Reduce限制、TP4 PCIe custom allreduce禁用、TileLang向量化回退、FP8 scale及DP coordinator warning。五个配置原始日志warning匹配数依次为100/450/758/577/1037，并非去重的故障数；全部完整内核检查通过，不等于模型质量或每个kernel数值验证。

完整证据仅在节点47本机 `experiments/dsv4-node8-node47/`，不作为随Git分发的链接：`results/baseline-01` 对应双TP4，`candidates-01` 至 `candidates-04` 对应上述候选顺序。每个run保留解析配置、命令、身份、绑定、完整日志、预热、拒绝尝试、逐请求、官方窗口及遥测；`reports/` 保留初始/最终资源快照、进程时间序列、逐次审查与诊断。五次运行无清理错误；收尾Docker运行容器为空、八卡计算进程为空，本次观测任务已停止。模型、镜像、历史结果和JIT缓存均保留。

从仓库根目录复现时先按公共任务说明建立新的本机工作区，再依次运行以下case；每个run-root必须是未存在的新目录。下面命令仅为首批复现入口；本次补测使用后文的本地诊断campaign，没有重新运行baseline：

```bash
./bench validate experiments/dsv4-node8-node47/configs/campaigns/47-dsv4-node8.yaml
./bench plan experiments/dsv4-node8-node47/configs/campaigns/47-dsv4-node8.yaml
./bench run experiments/dsv4-node8-node47/configs/campaigns/47-dsv4-node8.yaml \
  --case dual-tp4 --run-root experiments/dsv4-node8-node47/results/baseline-new
# 完成资源验收后，按顺序分别选取以下case，各自指定新的run-root：
# tp4-dp2-epoff → tp4-dp2-epon → tp2-dp4-epoff → tp2-dp4-epon
```

归档复核使用公共campaign再次 validate/plan，并核对CSV唯一ID、scope、请求/token总量、单位、报告数字、链接及Git差异；本轮仅提交节点47报告、CSV和自己的完成标记，不改其他节点状态，不push。

## 2026-09-18：JIT诊断与补测

**凌晨批次结论：TP2×DP4、EP off 的本次启动已解决旧形状首次使用造成的阻塞；TP4×DP2、EP off 仍未解决，当时交回统一协调。** 两个候选各只追加一个批次，没有重跑已完成配置、访问其他节点或追加矩阵。TP4×DP2于01:56:09–02:34:24运行；确认清理和资源正常后，TP2×DP4于02:36:26–04:00:23运行，均为北京时间。

### 假设、验证与实际处理

| 诊断假设 | 验证证据 | 结论与处理 |
| --- | --- | --- |
| Docker窗口中的编译日志可能迟到 | 首批完整日志按worker、内核和内部时间关联，TileLang开始/结束记录最大接收延迟约1620.80秒（TP4×DP2）／511.40秒（TP2×DP4）；新TP4×DP2仍有约605.38秒延迟 | 部分事件确实跨窗口迟到，但不能把全部阻塞视为误判；保留所有原事件，未改gate。输出延迟的底层原因未定位 |
| monitor warning不一定代表CPU重新编译 | 固定镜像 `vllm/utils/jit_monitor.py` 的 `_call_with_monitor` 在 `JITImpl._kernel_cache` miss时、调用原函数前记录warning；TileLang `KernelCache.cached` 随后才查磁盘，`JITKernel.__init__` 的 `from_database` 路径跳过编译 | 进程内首次使用、磁盘加载和真实编译需分开。新TP2×DP4完整日志无TileLang begins/completes，60份缓存工件大小/mtime未变，与复用磁盘缓存一致；仍保留warning并按原gate预热 |
| 原C32及两波C64请求不足以覆盖所有worker的MHC形状 | 镜像 `mhc/tilelang_kernels.py::compute_num_split` 按SM数、`ceil(num_tokens/64)`与K维约束选择split；split进入常量缓存键，token维本身动态。首批TP4×DP2后几轮仍出现split13/12/22/26；TP2×DP4的39/52在不同DP worker陆续首次使用 | 在正式C32前增加独立C64覆盖负载，以256请求覆盖与正式C64相同的四波请求和排空阶段；它不能保证所有调度形状都出现，必须用后续完整阶段验证 |

新批次的具体迟到反例：`Worker_DP1_TP1` 的普通MHC编译开始记录，内部时间为18:15:36 UTC，Docker于18:25:41.382759 UTC才接收；见TP4追加批次完整 `server.log` 第1947行及本机 `tp4-log-timeline.json`。该记录不能证明18:25仍在编译，但同阶段其他新形状编译仍需保留识别。

本地诊断负载为 `purpose=calibration`、C64、8192/1024、seed0及原客户端协议，每轮256请求，至少连续2轮安静、最多6轮；达到安静后仅一次256请求诊断验证，出现事件即失败。六轮的依据是首批事件延续至第4轮、第5轮才首次安静，至少还需一轮验证；同时将每轮从128增至256请求检验四波覆盖假设，没有在运行中继续加预算。诊断指标单独保存，不算正式样本。

之后仍使用原正式workload：C32/C64各3次、请求128/256、每轮预热64/128、连续2轮安静、最多5轮、测量最多2次尝试。启动1800秒及客户端阶段7200秒不变。每个配置从诊断到正式测量只启动一次；公共源码、原workload、recipe、服务参数、日志gate、正常Graph优化及JIT缓存均保留。

| 追加批次 | C64诊断预热事件序列 | 诊断验证 | 新正式结果 |
| --- | --- | --- | --- |
| TP4×DP2 EP off | 48/28/20/8/56/12，六轮均非安静 | 未进入 | 无；未启动C32/C64正式测量，失败证据单列，不进CSV |
| TP2×DP4 EP off | 24/6/2/0/0，第4、5轮连续安静 | 256请求，0事件，通过 | C32/C64各三次，六次前均恰好两轮安静，全部第一次测量通过 |

TP4×DP2的172条预热事件中，108条为TileLang monitor、16条为Triton monitor、24条为TileLang编译开始、24条为完成。按内部时间与worker关联，确认6种新的内核/split组合各在4个worker真实编译：普通MHC `with_norm` 的split14/15/31/17，以及broadcast MHC的split14/15；生成6份新内核缓存（共30份新工件，115→145），没有覆盖或删除旧工件。第6轮的部分开始/结束记录来自先前编译的延迟输出，且仍有新的worker首次调用；该轮不能被改判安静。这次结果否定了“只补一轮安静”或“所有事件都只是磁盘加载”的解释，也表明四波HTTP预热仍不能确定性覆盖TP4×DP2所有调度形状。

TP2×DP4首轮24条事件为8条TileLang split52和16条Triton top-k索引内核monitor；第2轮6条为DP0–DP2首次使用split39，第3轮2条为DP3首次使用split39，至此39/52覆盖8个worker。随后两轮、独立诊断验证以及全部正式预热/测量均无已知事件，完整日志没有TileLang实际编译开始/结束。这里确认的是本次启动内通过原gate，不证明任意未来调度形状或冷启动都无JIT。

### 资源、工作量与比较限制

补测执行commit为 `c905e4ff0e915227e63f309afa85f197e7eb28d6`；源码指纹、镜像、模型身份与前文一致。运行前后哈希确认原配置及新诊断配置未在运行期间变化；解析出的recipe、target、runtime、model、client与对应首批逐项相同，缓存仍挂载原目录。两个配置CLI preflight、实际绑定和完整内核日志gate均通过，warning匹配数分别为620/1787，未隐藏warning。

全部30个完成的诊断／预热／测量阶段共5376请求，逐条核验成功与8192/1024长度，输入44,040,192、输出5,505,024 tokens。新增正式6次单独合计1152成功、0失败，输入9,437,184、输出1,179,648 tokens。诊断、正式预热和首批部分结果均不混入CSV；统一导出器产生的14行中，剔除2行 `purpose=calibration` 后，仅将12行正式node/replica样本逐次资源验收后加入原42行。

保留10个DOCA进程。对每个阶段的官方monotonic窗口核验CPU/GPU遥测、进程增量和system-service cgroup增量，未发现矿工异常或新增持续高负载外部进程；超过0.3 CPU的system-service仍只有约定的 `roce-init.service`。全部阶段CPU峰值18.78%；六次正式窗口平均17.48%–17.62%、峰值18.67%，GPU温度47–58°C、SM频率2415–2422 MHz。四个DP rank的诊断指标累计抢占均为0，完整日志无抢占记录；实际GPU/DP/TP映射与原配置一致。

C32/C64吞吐CV分别0.47%/0.93%，均未超过3%的波动诊断门槛；仍保留DP coordinator等warning，正式请求最大ITL约2.92秒，低吞吐波动不代表无长尾。资源结果按 `accepted-background` 记录，不声称完全隔离。首批与补测跨服务启动且计时外预热历程不同，不能把吞吐或波动变化唯一归因于缓存；没有为比较而重跑双TP4或EP on。

### 阻塞交接与复现证据

凌晨批次结束时，TP4×DP2仍无完整正式结果。当时建议统一协调评估**每个worker内对MHC普通/broadcast路径的可达split集合进行确定性预热**，而不是本节点继续随机加轮数；需结合实际SM数和服务调度边界设计，若涉及镜像/启动入口或服务参数，应统一实现和验证，本机未修改。若要精确区分磁盘加载与真实编译，需要在实际编译入口提供独立证据并保留原warning，同时用上述跨窗口日志延迟反例回归验证真实编译仍被捕获；不能仅删除 `JIT compilation` 匹配规则。

本机原始产物仍位于 `experiments/dsv4-node8-node47/`，不随Git分发：

- 新run-root：`results/jit-tp4-dp2-epoff-01`、`results/jit-tp2-dp4-epoff-01`；旧 `baseline-01`、`candidates-01` 至 `candidates-04` 原样保留。
- 诊断配置：`configs/campaigns/47-dsv4-node8-jit-followup.yaml` 仅保留两个EP off case，在原workload列表前加入 `workloads/node8-jit-coverage-c64.yaml`。后者由原C64复制，改id为 `dsv4-node8-jit-coverage-c64`、purpose=calibration、warmup_requests=256、repetitions=1、max_warmup_rounds=6、max_attempts=1，其余字段保持原值。可据此从[公共campaign](../configs/campaigns/47-dsv4-node8.yaml)重建本地入口，原正式配置不覆盖。
- `reports/jit-followup-01/` 保存启动前假设与预算、固定镜像源码副本、旧/新日志时间关联、缓存初始/增量/最终快照、完整进程观测、资源与逐请求审查、导出中间文件，以及初始/最终资源快照。

复现时先按同样方法建立本地诊断配置、validate/plan及CLI preflight，核对资源后依次选择两个case，分别使用未存在的新run-root。不要把本机诊断路径当作随Git分发的正式recipe，也不要重用本次run-root。若需要改变服务参数、公共gate或测量协议，先交统一协调。

两次追加运行均无清理错误；最终Docker运行容器和GPU计算进程为空、31248/31249端口空闲，本次两个观测任务已正常停止，仅保留原有 `minimax-transfer`。模型、镜像、旧结果和所有原有JIT缓存未删除。归档自审覆盖数字、单位、相对链接、CSV唯一ID和旧行不变、原/诊断配置validate/plan、源码/配置哈希与提交范围；只更新节点47报告、CSV和自己的状态，不push。


## 2026-09-18上午：TP4×DP2 EP off 扩充预热上限

**仍未获得完整C32/C64正式批次；这次阻塞为测量期首次使用事件，不能再概括为“预热轮数达到上限”或“仍在真实编译”。** 用户明确授权增加计时外warmup后，仅对此配置追加一次启动，于北京时间10:32:59–12:07:11运行。覆盖预热和三次C32通过，C64首次重复的两次测量均被原gate拒绝，后两次C64未执行。部分C32样本不进入正式CSV，也不与旧批次拼接；其余已完成配置未重跑。

### 固定预算与实际执行

本地campaign `47-dsv4-node8-jit-extended.yaml` 仅含 `tp4-dp2-epoff`，依次引用 `node8-jit-extended-coverage-c64.yaml`、`node8-jit-extended-c32.yaml`、`node8-jit-extended-c64.yaml`，均位于原工作区 `configs/` 对应目录。覆盖负载沿用凌晨C64 calibration的8192/1024、每轮256请求、单次256请求验证：max_warmup_rounds由6增至30，max_attempts由1增至2。两个正式workload只改本地id及max_warmup_rounds（5→30）；C32/C64请求128/256、预热64/128、各三次重复、连续安静2轮及测量尝试2次保持原值。30轮为之前6轮预算的5倍，在启动前固定，未在运行中修改。

**30是最大轮数，不是强制实际轮数。** 现有runner在连续两轮安静后自动进入测量，因此本次覆盖预热实际只执行5轮；该实验验证的是“提高预热容许上限”，没有验证“先强制执行30轮再测量”。提高上限本身不能防止提前进入测量。公共runner、gate、正式recipe、服务参数和正常优化均未修改；用户授权的本地预热差异不回写公共workload。

| 阶段 | 每次尝试的预热事件序列 | 测量／验证事件 | 结果 |
| --- | --- | --- | --- |
| C64覆盖负载 | 76/8/8/0/0 | 0 | 256请求验证通过，仅calibration |
| C32重复1 | 首次0/0；第二次0/0 | 首次4；第二次0 | 第二次通过 |
| C32重复2 | 0/0 | 0 | 通过 |
| C32重复3 | 0/0 | 0 | 通过 |
| C64重复1 | 首次4/0/0；第二次0/0 | 首次4；第二次8 | 两次均拒绝，case FAIL |

新批次C32的三份部分结果为 **678.22 / 680.17 / 680.86 output tok/s**，均值 **679.75 ± 1.37**，CV **0.20%**；mean TTFT为9646.79 ms、mean TPOT为37.66 ms。它们仅说明本次启动的C32通过trial gate，不能作为完整部署对照的新正式结果。统一导出器对本FAIL批次导出0行；归档CSV保持原54行、24个整机样本不变。

### 首次使用事件的定位

完整服务日志共112条monitor事件，其中96条TileLang、16条Triton；**TileLang编译开始／完成均为0**。原有29份TileLang内核、145份工件的大小和mtime从启动前到结束全部一致，15秒缓存观测无增量或删除。96条TileLang事件按worker/kernel/cache_key分组均只出现一次；内部时间与Docker接收时间的最大差值为0.864秒，没有复现前批数百秒的延迟。这些证据结合固定镜像源码，支持本次TileLang事件为进程内首次调用、复用磁盘缓存，不能把monitor的“JIT compilation”文案直接当作CPU重新编译证据。

| 被拒绝阶段 | 具体事件 | 完整server.log定位 |
| --- | --- | --- |
| C32重复1首次尝试 | DP0四个worker首次使用普通MHC split39，token维227 | 第2693行起，03:16:28 UTC |
| C64重复1首次尝试 | DP0四个worker首次使用普通MHC split17，token维553 | 第4573行起，03:54:58 UTC |
| C64重复1第二次尝试 | DP1四个worker首次使用broadcast与普通MHC split13，token维709 | 第5243行起，04:06:26 UTC |

C64正式预热首轮还记录DP0四个worker首次使用普通MHC split22。预热中的split13此前已在DP0出现，但直到第二次C64测量才在DP1触发；磁盘工件存在与每个worker进程内已加载是两种状态。所有事件、warning和被拒绝样本均保留，没有把缓存加载事件事后改判为有效测量。

源码依据仍是上一轮从固定镜像提取的 `vllm/utils/jit_monitor.py`：`_tilelang_cache_miss_key` 检查进程内 `_kernel_cache`，`_call_with_monitor` 在原调用前记录warning；随后 `tilelang/cache/kernel_cache.py::cached` 才尝试磁盘加载，`tilelang/jit/kernel.py` 的 `from_database=True` 路径跳过实际编译。这里没有额外插桩逐调用证明CPU行为，因此保持证据限定，不修改公共观察规则。前批真实编译的开始／完成记录和新增工件继续作为正例保存。

若继续采用HTTP预热，预算需明确“最低实际预热轮数”或更长的连续安静窗口，不能仅调高max_warmup_rounds后期待runner自动多跑；实际并发和调度仍无法保证有限轮数覆盖每个worker的全部形状。若进一步区分首次加载与真实编译，应由统一协调依据这些反例和前批真实编译正例设计精确观察方案，保留warning并验证真实事件识别。本次没有追加第三次测量尝试或再启动一批。

### 工作量、资源与复现入口

执行commit为 `ef41adb`，源码指纹、固定镜像ID、模型/tokenizer身份、服务参数、CPU/NUMA绑定及原JIT缓存目录均与前批一致。实际KV为49.95 GiB/卡、每DP rank 130,404逻辑tokens，Graph捕获和固定backend日志通过；完整内核检查PASS，warning匹配1164条。启动前后配置及源码哈希一致。

全部25个完成阶段共 **3712成功、0失败，输入30,408,704、输出3,801,088 tokens**，逐请求长度与绑定核验通过；包括预热、calibration、被拒绝测量和三份C32部分结果。后台30秒进程采样及runner的5秒CPU/GPU、约30秒system-service cgroup观测覆盖全程；未发现矿工或新增持续高负载外部任务，超过0.3 CPU的system-service仍只有约定保留的 `roce-init.service`，10个DOCA进程保留。全阶段CPU峰值16.94%；三份C32部分结果的窗口平均CPU为15.98%–16.05%、GPU温度48–56°C、SM频率2422–2430 MHz。资源按accepted-background理解，不声称独占或用低CV证明无干扰。

原始产物仅在节点47本机 `experiments/dsv4-node8-node47/`，不随Git分发：

- `results/jit-tp4-dp2-epoff-extended-01/` 保存本次完整run、25阶段、拒绝尝试和完整服务日志。
- `reports/jit-extended-01/` 保存 `budget.md`、本地配置plan/preflight、源码配置固定身份、缓存前后及增量观测、全程进程采样、`log-timeline.json`、`audit-final.json`／`audit-final.txt`、启动前和最终资源快照、旧CSV副本及空的正式导出CSV。
- 从[公共campaign](../configs/campaigns/47-dsv4-node8.yaml)保留TP4×DP2 EP off一个case，按上文重建三份本地workload，可复现预算；正式service recipe和原workload文件保持原样。再次运行必须另选未存在的run-root。

结束后本次服务、客户端和两个观测任务均已停止，Docker运行容器、GPU计算进程为空，31248/31249端口空闲；只保留原有 `minimax-transfer` 会话。模型、镜像、历史结果及全部缓存保留。数字、逐请求、资源窗口、链接、旧CSV不变和清理结果自审后归档；不push。
