# 节点47：八卡整机部署对照

## 结论与完成范围

**本节点本轮优先保留双 TP4、EP off。** 在固定八卡、8192 输入／1024 输出、统一 CPU/NUMA 与全局活动容量64条件下，完整通过的三种配置中，双 TP4 在 C32、C64 的吞吐均最高，分别为 **1096.13 ± 8.96**、**1429.63 ± 6.19 output tok/s**，吞吐 CV 为0.82%／0.43%。结论仅适用于节点47、本轮固定负载和保留 DOCA 后台任务的条件，不代表其他节点或并发的最优配置。

TP4×DP2、EP on 的 C32/C64 吞吐较双 TP4 低21.70%／19.45%；TP2×DP4、EP on 低20.97%／13.12%。两个 EP on 配置的 C32 CV 均超过8%，TP4×DP2 的 C64 CV 也超过3%；它们通过 runner gate，但不能称为稳定基线。TP2×DP4、EP on 的 C64 CV 为1.88%，仍未追平双 TP4 吞吐。

2026-09-17 20:15:31 至09-18 00:54:44（北京时间），按分配顺序完成全部五个配置的尝试，历时约4小时39分，未到8小时检查点。没有扩展矩阵、访问其他节点或改动运行参数。

| 配置／case ID | C32 | C64 | 最终状态 |
| --- | --- | --- | --- |
| 双 TP4，`dual-tp4` | 三次通过 | 三次通过；第三次使用原上限内第二次尝试 | PASS |
| TP4×DP2，`tp4-dp2-epoff` | 三次部分结果保留 | 首次测量前五轮预热未达标；三次均未测 | FAIL，预热阻塞 |
| TP4×DP2，`tp4-dp2-epon` | 三次通过 | 三次通过 | PASS，波动限制 |
| TP2×DP4，`tp2-dp4-epoff` | 三次部分结果保留 | 首次测量前五轮预热未达标；三次均未测 | FAIL，预热阻塞 |
| TP2×DP4，`tp2-dp4-epon` | 三次通过 | 三次通过 | PASS，C32波动限制 |

[逐次 CSV](../data/node8-node47.csv)由[统一导出器](../../../scripts/export_samples.py)生成，仅包含最终 PASS 配置：18行整机样本、18行 `replica-0`、6行 `replica-1`，共42行。只对 `scope=node` 求和，共3456成功、0失败，输入28,311,552、输出3,538,944 tokens。两份最终 FAIL 配置的6个 C32 trial 不进入 CSV，在后文单列。全轮共24个已完成且通过 trial gate 的测量、1次被拒绝的正式尝试；原计划另外6次 C64 因预热阻塞未执行。

## 运行条件与实际参数

复现入口为[节点47公共 campaign](../configs/campaigns/47-dsv4-node8.yaml)，统一规则见[项目任务说明](../README.md#下一轮八卡整机部署对照)。本机先复制公共 `configs/` 到独立工作区，五个配置均通过 validate、plan、固定镜像 CLI preflight；每个配置只启动一次，在同一服务生命周期内顺序处理 C32/C64 和三次重复。没有改公共源码、recipe、workload、gate 或宿主频率／功耗。

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

全局活动容量均为64；容量32的 Graph 尺寸为 `[1,2,4,8,12,16,24,32]`，容量16为 `[1,2,4,8,12,16]`，C32/C64共用同一服务配置。实际日志确认 Graph 捕获、`FLASHINFER_MLA_SPARSE_DSV4`、`FLASHINFER_CUTLASS`、专家 FP4、`fp8_ds_mla`、V2及跳过 autotune。逻辑 KV tokens 与 GiB 是不同口径，0.90不代表等量 KV。

双 TP4 分别使用 GPU0–3／4–7，服务 CPU0–15／32–47、内存 NUMA0／2；客户端 CPU16–19／48–51、内存 NUMA1／3；端口31248／31249。单套八卡 DP 使用上述 CPU/NUMA 集合的并集，端口31248。所有配置服务32个物理核、客户端8个物理核，SMT兄弟（逻辑编号加64）不分配给本次进程。Docker inspect、服务/客户端线程允许集合检查通过；这些绑定不保证后台独占，也不证明页面全部本地驻留。

TP4×DP2 为两个本机 DP rank、TP组0–3／4–7；TP2×DP4 为四个本机 DP rank、TP组0–1／2–3／4–5／6–7，PP均为1。DP使用本地通信、动态初始化端口，未配置外部 rank。最终 TP2×DP4、EP on 的 GPU UUID、宿主 PID 和 worker 标题交叉核对确认 DP0/TP0–1/EP0–1 对应GPU0–1，依次至 DP3/EP6–7 对应GPU6–7。EP on 为八卡专家组，每rank 32/256专家、linear placement；EP off 保留关闭开关。DP内部 MoE 的跨rank通信不等于多份独立 TP 服务。

实际通信日志中，TP4使用 PYNCCL，PCIe超过两卡的 custom allreduce禁用；TP2的 TP组允许 `CUSTOM` 后接 `PYNCCL`，DP/EP相关组为 PYNCCL，DP候选使用 `AgRsAll2AllManager`，NCCL为2.30.7。这里只记录选择路径，没有 profiler 证明耗时归因。DP负载快照中曾观察到 TP4×DP2 的17/15 running、TP2×DP4的15/15/16/14 running；内部路由并非始终严格均分，不能把外部等分协议理解为内部 DP 负载均衡测试。

客户端统一流式 `/v1/completions`、seed0、temperature0、ignore_eos、range_ratio0、request_rate=inf。C32每次128请求、每轮预热64请求；C64每次256请求、预热128请求。双 TP4 按全局 seed0 请求索引交错分片，等分并发/请求；单套 DP 也走相同同步客户端。每次重复前至少连续两轮无已知编译事件，最多五轮；正式测量最多两次尝试，启动1800秒、客户端阶段7200秒，均未放宽。

## 性能表

下表仅包含最终 PASS 配置，吞吐为三次均值±样本标准差；延迟为三次对应指标的均值，单位均为 ms。P95列不是把三次请求再次合并后的分位数。完整 mean/P95 TTFT、TPOT、ITL、E2EL与各侧窗口见 CSV。

| 配置 | 整机并发 | 输出 tok/s | CV | Mean TTFT | P95 TTFT | Mean TPOT | P95 ITL | P95 E2EL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 双 TP4 | 32 | **1096.13 ± 8.96** | **0.82%** | 4785.93 | 10773.03 | 24.42 | 18.17 | 36424.01 |
| 双 TP4 | 64 | **1429.63 ± 6.19** | **0.43%** | 6396.11 | 20653.55 | 38.41 | 22.84 | 62110.41 |
| TP4×DP2，EP on | 32 | 858.21 ± 74.73 | 8.71% | 6320.32 | 13555.42 | 31.29 | 24.12 | 46495.29 |
| TP4×DP2，EP on | 64 | 1151.59 ± 48.74 | 4.23% | 7876.52 | 26040.84 | 45.77 | 44.15 | 75646.58 |
| TP2×DP4，EP on | 32 | 866.31 ± 70.78 | 8.17% | 7650.69 | 13352.24 | 29.64 | 25.24 | 43852.12 |
| TP2×DP4，EP on | 64 | 1242.03 ± 23.34 | 1.88% | 9248.22 | 23115.01 | 38.67 | 29.02 | 62757.96 |

双 TP4 的六次起跑时差为0.073–0.758 ms，结束时差0.35–1.36秒，空闲尾段全部保留。整机吞吐使用两侧官方计时窗口并集，不是两侧自身窗口吞吐直接相加；CSV分列保存自身窗口值与共同窗口贡献。单套 DP 只有一个外部客户端窗口，时差为0。

两份失败配置的 C32 **仅作为部分结果**，不混入上表与CSV；C64没有正式结果，不能据此作完整部署排名：

| 配置 | C32三次输出 tok/s | 均值±标准差 | CV | Mean TTFT ms | Mean TPOT ms |
| --- | --- | ---: | ---: | ---: | ---: |
| TP4×DP2，EP off | 745.59 / 672.50 / 646.74 | 688.27 ± 51.28 | 7.45% | 10163.25 | 36.74 |
| TP2×DP4，EP off | 775.45 / 788.83 / 733.49 | 765.93 ± 28.88 | 3.77% | 11081.81 | 31.01 |

全部24个已完成 trial 均逐条核对成功状态与8192/1024长度；C32为128成功、0失败、输入1,048,576／输出131,072 tokens，C64为256成功、0失败、输入2,097,152／输出262,144 tokens。副本行没有重复计入整机总量。

## 异常、资源验收与复现证据

**两处阻塞均为 C64 预热未达到原门槛，不是 OOM 或模型不支持。** TP4×DP2、EP off 的五轮事件数为64/24/40/24/0；TP2×DP4、EP off 为6/4/2/6/0。均只有最后一轮安静，按原上限退出，未重启补测、改精度、改Graph或更换拓扑。两者完整内核日志 gate 都为 PASS，但 case 为 FAIL；先确认清理成功及资源正常，再继续下一独立候选。

双 TP4 的 C64第三次首次尝试因后四卡24条事件整组拒绝，重新完成两轮安静预热后第二次尝试通过。该次 Docker 时间为13:06:42 UTC的部分记录内嵌12:43:54／12:44:05的 TileLang时间，存在明显缓冲延迟；DP预热也有类似现象，同时存在当轮 JIT monitor 事件。所有反例原样保留，未改正则、隐藏 warning 或将拒绝样本计入CSV。没有额外第三次正式尝试。

首个有效样本前的预热总轮数：双 TP4 的 C32/C64均三轮；TP4×DP2 EP off 的 C32四轮；TP4×DP2 EP on 的 C32三轮、C64四轮；TP2×DP4 EP off 的 C32三轮；TP2×DP4 EP on 的 C32三轮、C64两轮。其他正常重复均两轮；上面的拒绝尝试另行保存。日志无已知事件不证明所有即时编译或调度形状已经收敛。

**所有 CSV 行经逐次审核后标为 `accepted-background`，表示接受约定后台条件，不表示无干扰或性能稳定。** 启动前10秒采样整机 CPU忙碌率9.35%；10个 `doca_spcx_cc` 持续约占10个逻辑CPU，其中6个属于 `roce-init.service`，另外4个在用户会话cgroup，均按约定保留。进程曾运行在服务物理核及SMT兄弟上，因此Docker cpuset并不隔离DOCA。NUMA0–3启动前空闲约157/153/112/39 GiB，整机可用内存约952 GiB、无swap，模型盘与工作区空间充足。

运行全程保存5秒 CPU/GPU、约30秒 system-service cgroup 和额外30秒进程增量采样。逐次按官方 monotonic 测量窗口对齐检查：24个已完成 trial 未发现矿工异常或新增持续高负载外部进程；system-service cgroup采样中增量超过0.3 CPU的只有保留的 `roce-init.service`（约6 CPU）。双 TP4及TP4×DP2的窗口平均 CPU约15.86%–16.27%，TP2×DP4约17.21%–17.86%；峰值最高18.75%以内，没有45%资源观察告警。测量采样 GPU温度46–60°C，SM频率2415–2430 MHz，未见持续热降频证据。低频采样不能排除瞬时干扰。

对全部CV>3%的组保留慢样本并检查JIT、抢占、CPU、客户端、频率与通信日志：测量gate无已知编译事件；完整日志无抢占记录，诊断时 `/metrics` 各DP rank累计抢占为0；客户端在完全落入测量窗口的30秒进程CPU区间内，单个客户端容器内进程合计峰值约0.133 CPU，未见持续饱和，仍不能排除瞬时停顿。四个DP配置的C32均出现 coordinator `out-of-order step` warning与长ITL，最大ITL分别为10.69、7.79、12.23、9.45秒（按执行顺序），资源变化不足以解释波动，尚未确定根因。没有用不断补测或删除慢样本取得低CV。

保留 SM120 SymmMem不可用、FlashInfer All Reduce限制、TP4 PCIe custom allreduce禁用、TileLang向量化回退、FP8 scale及DP coordinator warning。五个配置原始日志warning匹配数依次为100/450/758/577/1037，并非去重的故障数；全部完整内核检查通过，不等于模型质量或每个kernel数值验证。

完整证据仅在节点47本机 `experiments/dsv4-node8-node47/`，不作为随Git分发的链接：`results/baseline-01` 对应双TP4，`candidates-01` 至 `candidates-04` 对应上述候选顺序。每个run保留解析配置、命令、身份、绑定、完整日志、预热、拒绝尝试、逐请求、官方窗口及遥测；`reports/` 保留初始/最终资源快照、进程时间序列、逐次审查与诊断。五次运行无清理错误；收尾Docker运行容器为空、八卡计算进程为空，本次观测任务已停止。模型、镜像、历史结果和JIT缓存均保留。

从仓库根目录复现时先按公共任务说明建立新的本机工作区，再依次运行以下case；每个run-root必须是未存在的新目录。下面命令仅为复现入口，本报告未追加执行：

```bash
./bench validate experiments/dsv4-node8-node47/configs/campaigns/47-dsv4-node8.yaml
./bench plan experiments/dsv4-node8-node47/configs/campaigns/47-dsv4-node8.yaml
./bench run experiments/dsv4-node8-node47/configs/campaigns/47-dsv4-node8.yaml \
  --case dual-tp4 --run-root experiments/dsv4-node8-node47/results/baseline-new
# 完成资源验收后，按顺序分别选取以下case，各自指定新的run-root：
# tp4-dp2-epoff → tp4-dp2-epon → tp2-dp4-epoff → tp2-dp4-epon
```

归档复核使用公共campaign再次 validate/plan，并核对CSV唯一ID、scope、请求/token总量、单位、报告数字、链接及Git差异；本轮仅提交节点47报告、CSV和自己的完成标记，不改其他节点状态，不push。
