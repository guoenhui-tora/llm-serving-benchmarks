# DeepSeek V4 Flash NVFP4 在 RTX6000D 上的推理优化

本项目使用 NVIDIA 发布的 [nvidia/DeepSeek-V4-Flash-0731-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4)，目标是在 RTX6000D 上优化推理吞吐与延迟，当前以单机为主。权重为 NVFP4 routed experts 与高精度其他部分的混合格式；实验保持同一权重与 tokenizer。

**继续选择 vLLM 0.29.0、关闭 FlashInfer autotune。此前 8K 输入 / 1K 输出、C32 的已测候选中，八卡 TP2×PP4、四卡 TP2×PP2表现较好。** 前者在46达到 **1099.61 ± 19.98 tok/s**，后者在48达到 **762.92 ± 17.85 tok/s**，均为三次重复。它们各自优于同节点参照，但尚未在同一节点直接比较，也未验证其他并发或长期稳定性。

## 下一轮：八卡整机部署对照

**目标：固定单机8张GPU、8192输入／1024输出，比较不同部署方式的整机吞吐、延迟与波动。** 本轮不做PD分离。双TP4、EP off是公共整机baseline；不另设单实例扩展验证阶段，历史四卡结果仅作辅助参考。

### 节点任务与完成标记

每台先测双TP4，再按表中顺序测候选。所有配置测整机C32、C64，各三次；省略的TP/PP/DP为1。EP只在DP候选展开off/on，其余固定off。每个配置的全部并发和重复在同一次服务启动内完成。

| 节点 | 方向 | baseline后的候选（case ID） | 含baseline的配置数 / 正式测量数 |
| --- | --- | --- | ---: |
| 45 | 双四卡部署的TP/PP切分 | `dual-tp2-pp2` → `dual-tp1-pp4` | 3 / 18 |
| 46 | 双四卡部署的DP | `dual-tp2-dp2-epoff` → `dual-tp2-dp2-epon` → `dual-tp1-dp4-epoff` → `dual-tp1-dp4-epon` | 5 / 30 |
| 47 | 单套八卡部署的DP | `tp4-dp2-epoff` → `tp4-dp2-epon` → `tp2-dp4-epoff` → `tp2-dp4-epon` | 5 / 30 |
| 48 | 单套八卡部署的TP/PP切分 | `tp8` → `tp4-pp2` → `tp2-pp4` | 4 / 24 |

- [x] **45**：18次正式测量完成；双TP2×PP2吞吐均值领先，PP候选波动需保留。见[本节点报告](reports/node8-node45.md)、[逐次CSV](data/node8-node45.csv)。
- [x] **46**：5个配置共30次正式测量完成；DP4 EP off经独立诊断预热批次补齐，C32/C64波动均需保留。见[节点报告](reports/node8-node46.md)、[逐次CSV](data/node8-node46.csv)。
- [x] **47**：双TP4及两个EP on配置完成18次正式测量；两个EP off候选因C64预热上限阻塞，部分结果保留。见[本节点报告](reports/node8-node47.md)、[逐次CSV](data/node8-node47.csv)。
- [x] **48**：补测双TP4完整通过，连同首批TP8、TP4×PP2共18次正式样本；TP2×PP4仍因C64形状预热上限阻塞，JIT诊断和部分结果保留。见[本节点报告](reports/node8-node48.md)、[逐次CSV](data/node8-node48.csv)。

共10种部署形态、14种EP配置变体；包括四节点各自baseline，总计102次正式测量，不含预热和拒绝尝试。新组合启动失败、OOM或不支持时取证停止该候选；不为了凑满次数切换精度、缩短长度、关闭Graph或改成其他拓扑。对性能明显不稳定的组合先诊断、保留全部样本，不增加无限补测。

### 统一资源与参数

整机使用GPU0–7。双部署按0–3、4–7拆分，两套独立进程组；单套部署联合使用八卡。DP是部署内部的维度，不等于两套外部独立服务；EP不额外乘GPU数。

| 部署形态 | 外部部署数 | 每部署TP/PP/DP | 每DP rank活动容量 | 整机prefill预算上限 |
| --- | ---: | --- | ---: | ---: |
| 双TP4 | 2 | 4/1/1 | 32 | 16384 |
| 双TP2×PP2 | 2 | 2/2/1 | 32 | 16384 |
| 双TP1×PP4 | 2 | 1/4/1 | 32 | 16384 |
| 双TP2×DP2 | 2 | 2/1/2 | 16 | 32768 |
| 双TP1×DP4 | 2 | 1/1/4 | 8 | 65536 |
| TP8 | 1 | 8/1/1 | 64 | 8192 |
| TP4×DP2 | 1 | 4/1/2 | 32 | 16384 |
| TP2×DP4 | 1 | 2/1/4 | 16 | 32768 |
| TP4×PP2 | 1 | 4/2/1 | 64 | 8192 |
| TP2×PP4 | 1 | 2/4/1 | 64 | 8192 |

**整机活动容量固定64，C32与C64共用同一服务配置。** 容量按外部部署数与内部DP数均分，PP阶段不重复计数。每DP rank的prefill预算保持8192，因此整机调度预算随独立调度器数量变化，上表明确列出；这是固定硬件下的完整部署对照，不声称仅改变通信方式或整机prefill预算相同。

固定vLLM 0.29.0及既有image ID、同一NVFP4权重/tokenizer、上下文16384、FP8 E4M3 KV、显存比例0.90自动分配。保留V2、async scheduling、mp executor、chunked prefill、FULL_DECODE_ONLY；关闭FlashInfer autotune、prefix cache、投机解码和CPU offload。Graph尺寸从 `[1,2,4,8,12,16,24,32,48,64]` 截取到本rank容量。记录实际KV、抢占、MoE/attention与通信路径，不把0.90当作相同KV容量。

| 资源 | 双部署前四卡 | 双部署后四卡 | 单套八卡部署 |
| --- | --- | --- | --- |
| 服务CPU | 0–15 | 32–47 | 0–15、32–47 |
| 服务内存NUMA | 0 | 2 | 0、2 |
| 客户端CPU | 16–19 | 48–51 | 16–19、48–51 |
| 客户端内存NUMA | 1 | 3 | 1、3 |
| API端口 | 31248 | 31249 | 31248 |

服务共32个物理核、客户端共8个物理核，每核一个线程，SMT兄弟不分配给本次进程。通过Docker cpuset实现；本轮统一不启用vLLM `--numa-bind`。八卡单套部署使用两侧资源的并集，不能默认使用更多CPU。内存允许集合不保证页面全部本地驻留，cpuset也不排除DOCA等后台任务。

本轮八卡CPU限制与历史不绑定的八卡实验不同，容量也统一为64；历史结果不能直接计入本轮三次重复。两套DP使用本机local DP，固定镜像为本地通信选择IPC和动态DP初始化端口；不配置外部DP rank或把两个服务加入同一个DP组，启动后核实实际rank、GPU及端口。

### 压测与验收

| 整机负载 | 正式请求数 / 重复 | 每轮预热请求数 | 双部署每侧并发 / 正式请求数 / 预热请求数 |
| --- | --- | ---: | --- |
| C32 | 128 / 3次 | 64 | C16 / 64 / 32 |
| C64 | 256 / 3次 | 128 | C32 / 128 / 64 |

同一固定vLLM客户端、流式 `/v1/completions`，seed0、temperature0、ignore_eos=true、range_ratio0、request_rate=inf。全局数据集按seed0生成后按请求索引交错均分，两侧不重复使用同一半数据。等分并发与请求量，**不是动态负载均衡能力测试**；一侧提前结束时保留空闲尾段及负载不均证据。

每次重复前同步执行预热；双方同一轮均无已知编译事件才算一轮安静，至少连续2轮、最多5轮；正式测量最多2次尝试。任一侧测量期出现编译事件，整组尝试拒绝，两侧重新预热。任一侧失败则整组不能进入统计。正常重复复用同一套服务；不要测一次关服，再启动补两次。

所有部署（包括单套八卡）使用同一同步客户端流程。客户端只在官方计时边界插入同步屏障，并在计时外保存逐请求记录；官方生成请求、流式协议和指标计算保持不变。钩子限定已核对的0.29.0版本，未知计时结构直接失败，不静默降级。同步就绪屏障最多等待300秒，启动时差超过0.5秒会使该阶段失败；不能拿两个不同窗口的tokens/s直接相加。

整机吞吐为全部成功输出tokens除以两侧官方计时窗口的并集时长。每侧保留自身窗口吞吐与共同窗口贡献，后者可相加得到整机值。整机TTFT、TPOT、E2EL按两侧请求合并，ITL按流式事件合并后计算P95；三次重复的P95均值仍不是三次合并后的P95。

每次整机C32应成功128、失败0，输入1,048,576、输出131,072 tokens；C64应成功256、失败0，输入2,097,152、输出262,144 tokens。总量、每侧数量及逐请求输入输出均需核对。吞吐CV超过3%先查JIT、抢占、CPU背景、客户端、频率和通信；不删除慢样本，也不把日志安静当作性能收敛。

**资源检查贯穿整个运行。** 开始前确认本机8卡空闲、CPU映射与规则一致、NUMA内存压力可接受。已有DOCA后台负载按用户约定保留并记录；矿工异常若仍存在或重新出现，应停止本机实验并记录资源阻塞，不按正常背景验收。runner保存5秒CPU/GPU遥测、约30秒system service cgroup计数；全机CPU忙碌率超过45%会提示检查，这只是观察提醒，不是自动区分矿工的判据，也不替代人工资源验收。运行任务必须持续核对测量窗口，不能只看进程列表或最终PASS。

### 准备状态与执行入口

**准备检查已完成：** 四份campaign通过离线validate/plan，全部14种recipe变体的参数名通过固定镜像CLI核对，自动化测试覆盖同步计时、请求分片、合并统计、JIT传播、失败/停止清理和配置迁移。

48已用双TP4保留正式优化完成短功能验证：1024输入／32输出、整机C4、8请求、同一对服务连续2次。合计16成功、0失败，输入16,384、输出512 tokens；两次起跑时差约0.15／0.37 ms。首轮预热识别32条编译事件，随后两轮安静；第二次重复两轮安静，两次测量均无已知事件。两份完整服务日志通过原有required/compilation规则；容器已清理、GPU计算进程为空。运行源码指纹为 `edccf507e1246a4525a383ff4c4d3ce45a470378075bf3271922903c10020e34`，原始证据仅保存在48本机准备工作区。

公共配置按本轮协议新增，旧基线和日志gate保留；最终recipe的全部原始日志断言也已对短验证完整日志重新核验。**短验证证明双实例执行链路可用，不证明8K/1K、C32/C64的容量或所有候选拓扑都能运行。** 其余候选只完成离线与CLI检查，是否支持仍需各节点启动核实。

| 节点 | 公共任务入口 |
| --- | --- |
| 45 | [45-dsv4-node8.yaml](configs/campaigns/45-dsv4-node8.yaml) |
| 46 | [46-dsv4-node8.yaml](configs/campaigns/46-dsv4-node8.yaml) |
| 47 | [47-dsv4-node8.yaml](configs/campaigns/47-dsv4-node8.yaml) |
| 48 | [48-dsv4-node8.yaml](configs/campaigns/48-dsv4-node8.yaml) |

先读取根AGENTS/README、[实验方法](../../docs/benchmark-methodology.md)和[多副本配置说明](../../docs/configuration.md#同步多副本部署)。以48为例，首次建立新的工作区时执行；已有目录不要覆盖：

```bash
mkdir -p experiments/dsv4-node8-node48/results experiments/dsv4-node8-node48/reports
cp -a projects/dsv4-rtx6000d/configs experiments/dsv4-node8-node48/configs
./bench validate experiments/dsv4-node8-node48/configs/campaigns/48-dsv4-node8.yaml
./bench plan experiments/dsv4-node8-node48/configs/campaigns/48-dsv4-node8.yaml
./bench run experiments/dsv4-node8-node48/configs/campaigns/48-dsv4-node8.yaml \
  --case dual-tp4 --run-root experiments/dsv4-node8-node48/results/baseline-01
# 检查参照和后台资源后，继续本节点候选；其他节点按任务表换case。
./bench run experiments/dsv4-node8-node48/configs/campaigns/48-dsv4-node8.yaml \
  --case tp8 --case tp4-pp2 --case tp2-pp4 \
  --run-root experiments/dsv4-node8-node48/results/candidates-01
./bench report experiments/dsv4-node8-node48/results/baseline-01 \
  experiments/dsv4-node8-node48/results/candidates-01 \
  > experiments/dsv4-node8-node48/reports/matrix-01.md
python3 scripts/export_samples.py experiments/dsv4-node8-node48/results/baseline-01 \
  experiments/dsv4-node8-node48/results/candidates-01 \
  --output experiments/dsv4-node8-node48/reports/node8-node48.csv
```

runner会在各配置启动前执行preflight，每个服务分别检查health、models和关闭thinking的简短中文生成。可用 `--case dual-tp4` 只运行参照；需要恢复时选未完成的case并指定新run-root，不覆盖历史，也不重跑已完成配置。`./bench stop RUN_ROOT`只停止该run拥有的服务和客户端，保留日志。普通候选失败且清理完成后可继续下一独立候选；资源冲突、清理失败或用户停止则不继续。

四台只操作本机，不SSH其他节点，运行期间不pull、不修改源码/配置。先记录共同Git commit、源码指纹、本地差异、解析配置和命令。每完成一个配置简报结果和剩余任务；8小时为进度检查点，不是强制中断时间，按已测耗时更新预计结束时间。启动1800秒、每次客户端阶段7200秒，预热/重试上限不增加；不额外搜索参数。

### 归档与交接规则

每节点只新增下列两个精选文件，完成后勾选本节自己的任务行并链接报告。不要更新别的节点结果或先行写跨节点胜负结论。

| 归档内容 | 项目内路径 |
| --- | --- |
| 节点报告 | `reports/node8-nodeNN.md` |
| 逐次CSV | `data/node8-nodeNN.csv` |

报告按“结论与完成范围 → 运行条件/实际参数 → 性能表 → 异常与限制”组织，链接公共campaign和本节点CSV。写清哪些候选未运行、失败或受资源干扰，不把runner PASS等同于无干扰；记录实际EP/PP/DP组、KV、Graph、fallback、CPU背景及清理结果。

使用上面的统一导出脚本生成CSV（目前支持vLLM拓扑字段）；默认 `resource_status=review-required`，逐次核对遥测后再标为accepted-background或resource-affected，不能未经检查统一改成通过。脚本只导出最终PASS的case，失败/部分完成的证据在报告单列；不会将拒绝尝试混入CSV。输出文件已存在时拒绝覆盖。

CSV每个正式重复保存一行整机指标，并另存每侧指标行，使用 `scope=node|replica-0|replica-1` 区分；整机统计只筛选node行，不能把副本行再次相加到请求总量。包含唯一sample_id、节点、配置、scope、GPU组、TP/PP/DP/EP、全局容量与并发、重复号、成功/失败、输入/输出tokens、输出tok/s、requests/s、mean/P95 TTFT/TPOT/ITL/E2EL、计时窗口和资源资格说明。各侧自身窗口与共同窗口贡献分列，不能混用。延迟统一ms，未提供的指标留空，不填零。

完整日志、JSON、逐请求记录、预热、拒绝尝试、遥测、探索脚本和临时配置留在本机 `experiments/`。仓库链接只指向随Git存在的文件；不把本地run-root写成GitHub复现链接。不另建重复汇总JSON、不上传各节点相同的公共配置；新发现的必要修复单独说明。保留模型、镜像、缓存与历史结果，只清理本次容器。commit/push按用户在各节点的指示执行。

各节点收到任务后可直接执行：

> 阅读本项目README“下一轮：八卡整机部署对照”，按本机编号完成分配的campaign、资源监控、验收和归档。只操作本机，固定本轮参数与预算；每个配置只启动一次完成C32/C64各三次。资源异常或不支持时取证，按规则停止相关任务，不自行扩展矩阵或push。

## 2026-09-17：四节点单服务拓扑结果

本轮每台同一时间只运行一个 vLLM 服务，全服务 C32。与前期同机双服务、整机 C64 的实验分开统计；本轮未重测 SGLang，也未做跨机推理或 PD 分离。

### 完成范围与资源条件

四台启动前都曾出现用户确认的矿工异常 CPU 负载，处理后重新开测；mlx/DOCA 后台忙线程仍保留，按原 CPU 绑定规则执行。**这些结果反映实际保留后台负载的运行条件，不是完全隔离的硬件性能。** 数据由下列 CSV 重算，运行与日志证据依据各节点归档报告，本次整合未重新执行实验或远程检查日志。

- [x] **45**：主矩阵12次通过 gate；11次受额外 CPU 负载影响，可选 prefill 因资源冲突中断。见[报告](reports/topology-node45.md)、[逐次 CSV](data/topology-node45.csv)。
- [x] **46**：主矩阵12次、prefill 初筛2次完成，TP2×PP4领先。见[报告](reports/topology-node46.md)、[逐次 CSV](data/topology-node46.csv)。
- [x] **47**：主矩阵12次、prefill 对照4次完成，EP 未带来大的收益。见[报告](reports/topology-node47.md)、[逐次 CSV](data/topology-node47.csv)。
- [x] **48**：主矩阵9次、prefill 对照4次完成，前后四卡接近，TP2×PP2更快。见[报告](reports/topology-node48.md)、[逐次 CSV](data/topology-node48.csv)。

合计 **55次通过 gate 的测量，7040成功、0失败，输入57,671,680、输出7,208,960 tokens**。其中45的资源限制单独保留，不把全部55次称为无干扰的有效对照。四台包含预热、启动与补测的总耗时各约2.7–4小时；各节点报告本次容器已清理。

### 参照是否接近

下表均为 prefill 8192、EP off；吞吐为三次均值 ± 样本标准差，括号为 CV。45单列在后文，不参与节点校准。

| 节点 | 后四卡 TP4，tok/s（CV） | 八卡 TP8，tok/s（CV） |
| --- | ---: | ---: |
| 46 | 712.00 ± 0.72（0.10%） | 609.75 ± 0.88（0.14%） |
| 47 | 718.48 ± 2.62（0.37%） | 611.35 ± 1.16（0.19%） |
| 48 | 715.72 ± 1.40（0.20%） | 本轮未测 |

46–48的 TP4 均值最高与最低相差 **0.91%**，46/47的 TP8 相差 **0.26%**，足以支持分节点筛选大幅收益的方案；小幅提升仍需同机判断。46/47的 TP4 吞吐均高于本机 TP8，但四卡绑定、八卡不绑定，不能将差异全部解释为跨 NUMA 通信损失。

### 主矩阵：PP、EP 与 GPU 分组

以下均为 prefill 8192、三次重复。延迟列为各次对应指标的均值，**P95列不是合并全部请求后的P95**；requests/s、P95 TPOT、ITL、端到端延迟及逐次值见节点报告和 CSV。

| 节点 | 卡数 / 配置 | 输出 tok/s，均值±标准差 | CV | Mean TTFT 秒 | P95 TTFT 秒 | Mean TPOT ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 46 | 4 / TP4 | 712.00 ± 0.72 | 0.10% | 6.32 | 20.31 | 38.75 |
| 46 | 8 / TP8 | 609.75 ± 0.88 | 0.14% | 10.02 | 29.47 | 42.68 |
| 46 | 8 / TP4×PP2 | 666.70 ± 5.70 | 0.86% | 4.40 | 12.19 | 43.70 |
| 46 | 8 / **TP2×PP4** | **1099.61 ± 19.98** | **1.82%** | **3.55** | **6.87** | **25.65** |
| 47 | 4 / TP4 | 718.48 ± 2.62 | 0.37% | 6.33 | 20.36 | 38.33 |
| 47 | 8 / TP8 | 611.35 ± 1.16 | 0.19% | 9.84 | 29.38 | 42.72 |
| 47 | 4 / TP4＋EP4 | 706.39 ± 2.70 | 0.38% | 6.29 | 20.06 | 39.14 |
| 47 | 8 / TP8＋EP8 | 634.00 ± 1.74 | 0.28% | 8.59 | 27.21 | 42.07 |
| 48 | 4 / TP4，后四卡 | 715.72 ± 1.40 | 0.20% | 6.38 | 20.34 | 38.45 |
| 48 | 4 / TP4，前四卡 | 715.23 ± 2.31 | 0.32% | 6.32 | 20.35 | 38.54 |
| 48 | 4 / **TP2×PP2，后四卡** | **762.92 ± 17.85** | **2.34%** | **4.15** | **11.25** | **37.91** |

**此前已测的八卡单套部署中，TP2×PP4领先。** 46相对本机 TP8，吞吐提高80.34%，mean TTFT/TPOT分别降低64.61%/39.90%；相对四卡 TP4，整机吞吐提高54.44%，但用卡数翻倍，不能称为卡效也更高。TP4×PP2仅比 TP8快9.34%，mean TPOT还略高，因此不是只要开启 PP 就一定大幅提速。

**四卡 TP2×PP2 有收益，GPU 前后位置未见明显差异。** 48相对同组 TP4，吞吐提高6.60%，mean TTFT降低35.01%，P95端到端延迟降低23.95%；P95 ITL却从23.46升至32.36 ms，并非所有延迟指标都改善。其三次吞吐为782.60、758.38、747.78 tok/s，虽均高于 TP4，仍需留意顺序下降。前后 TP4均值只差0.07%，未复现前期双服务实验的分组差异。

**单独开 EP 优先级较低。** 47在四卡上开启 EP 后吞吐降低1.68%，八卡上提高3.71%；后者超过本轮重复波动，但收益远小于46观察到的 PP 方案，且仍低于本机 TP4。

从实际 rank 映射看，TP2把高频 TP 通信限制在近端两卡组；日志也显示超过两张 PCIe GPU 的 TP 组会禁用 custom allreduce。这与 TP2＋PP 的改善方向一致，但尚无 profiler 证明收益具体来自哪条通信或计算路径。PP 同时改变层分片、调度、kernel形状和逻辑 KV 容量；**V2 runner 与 async 在所有参照和候选中都已开启，本轮不能单独量化它们的收益。**

### 45：DP 结果与矿工恢复的影响

45在北京时间 **06:26:14–06:26:24** 出现额外 CPU 负载，正好落在 TP4第二次测量期间；随后整机 CPU 忙碌率约60%–64%。收尾采样异常服务单独消耗约64个逻辑 CPU，另有 DOCA忙线程约8个逻辑 CPU。这里只确认了异常服务恢复消耗 CPU，未确定其自动恢复机制。

| 配置 | 输出 tok/s，均值±标准差 | CV | Mean TTFT 秒 | P95 TTFT 秒 | Mean TPOT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP4 | 711.71 ± 5.54 | 0.78% | 6.47 | 20.64 | 38.62 |
| TP8 | 649.74 ± 1.02 | 0.16% | 9.12 | 26.60 | 40.33 |
| TP4×DP2，EP off | 683.71 ± 44.11 | 6.45% | 11.09 | 24.16 | 35.24 |
| TP4×DP2，EP on | 833.98 ± 67.27 | 8.07% | 6.41 | 14.74 | 29.90 |

TP8、DP off/on三组均在额外负载出现后完成，整机 CPU 平均忙碌率接近。**这些八卡结果仍有本节点筛选价值：DP＋EP三次均高于另两组，均值比 TP8高28.35%、比 DP off高21.98%。** 但两组 DP 的 CV达6.45%/8.07%，应列为需要复核的候选，不能当作稳定基线。DP off与 TP8的样本范围重叠，尚未分出稳定优劣。

TP4三次跨越了背景变化，不能用其混合均值建立正常背景的三次参照。不同拓扑对后台任务的敏感程度也可能不同，所以本节点八卡排序仍需在排除矿工后复核；45不参与跨节点排名。其 TP8数值高于46/47，恰好说明不能用“CPU更忙，所以吞吐必定按比例更低”解释或修正数据。

**本镜像的 DP2不是简单部署两份独立 TP4。** 日志及镜像实现表明：EP off时 Attention TP分成0–3/4–7，MoE却跨 DP展开为8路张量分片，并使用 AllGather＋ReduceScatter dispatch/combine；EP on则形成8卡专家组。因此不能直接用独立 TP4吞吐乘2预测此配置。DP2的每rank prefill预算8192，整机上限16384，也不同于 DP1的整机8192。

### Prefill 初筛：保留 8192 作为后续起点

以下相对变化均与**同节点、同拓扑**的8192比较；n=1仅为初筛，不计算标准差或稳定性。

| 节点 / 拓扑 | Prefill | n | 输出 tok/s，均值±标准差 | 相对8192 | Mean TTFT 秒 | Mean TPOT ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 46 / TP2×PP4 | 4096 | 1 | 924.61 | -15.92% | 2.33 | 32.33 |
| 46 / TP2×PP4 | 16384 | 1 | 1089.59 | -0.91% | 5.31 | 24.20 |
| 47 / TP4 | 4096 | 1 | 710.98 | -1.04% | 5.28 | 39.78 |
| 47 / TP4 | 16384 | 3 | 727.97 ± 4.44 | +1.32% | 8.15 | 36.00 |
| 48 / TP2×PP2 | 4096 | 3 | 761.14 ± 31.73 | -0.23% | 3.15 | 38.98 |
| 48 / TP2×PP2 | 16384 | 1 | 753.08 | -1.29% | 5.78 | 36.86 |

没有发现适合统一替换8192的预算。47的16384吞吐小幅提高，但 mean TTFT增加28.80%；48的4096降低 mean TTFT，却有4.17%的吞吐 CV及更高 P95端到端延迟，慢样本伴随客户端 CPU高占用，尚未确定因果。46的4096明显损失吞吐；三节点的16384都提高了 mean TTFT。若有明确首token延迟目标，再单独权衡这些方案。

### 日志验收与结论边界

四台共 **10次正式尝试因编译事件被拒绝**（45/46/47/48分别3/1/5/1），在原上限内重新预热、重试；拒绝结果未进入上述 CSV。没有为通过 gate 修改识别规则。主矩阵未报告 OOM、模型不支持或请求超时；45的可选中断是资源问题。各节点报告确认 V2、Graph、NVFP4 MoE、FP8 KV 和关闭 autotune 的日志路径，不等于完整验证模型数值质量。

各节点保留 SM120 SymmMem 不可用、FlashInfer All Reduce限制、PCIe多卡 custom allreduce禁用、TileLang fallback及FP8 scale等 warning。GPU遥测未提供明显持续热降频证据；采样频率不足以排除瞬时干扰。DOCA用户态忙线程可与服务、客户端及SMT兄弟重叠，cpuset不会排除这些后台任务，也不能把它们都归为网卡IRQ。

**gate检查通过、吞吐 CV较小与资源没有干扰是三件不同的事。** 45的异常 CPU负载是在收尾交叉核对整机忙碌率和cgroup计数后发现的；只看进程列表或 JIT日志会漏掉此类问题。后续需要在测量窗口内同时记录整机/逐核负载、相关cgroup增量和本次进程负载，发生变化时单独标记样本，不按假定比例补偿性能。

## 本轮配置与复现

### 固定条件

| 项目 | 实际采用 |
| --- | --- |
| 模型 | 同一 NVFP4权重/tokenizer；默认路径 `/data/models/DeepSeek-V4-Flash-0731-NVFP4` |
| 服务与客户端 | `vllm/vllm-openai:v0.29.0`，image ID `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1` |
| 执行与日志 | `VLLM_USE_V2_MODEL_RUNNER=1`、async scheduling、mp executor；FlashInfer autotune关闭、`jit-monitor-verbose`开启，保留 JIT缓存 |
| 精度与容量 | 上下文16384、FP8 E4M3 KV、总活动容量32；`gpu-memory-utilization=0.90`自动分配，无固定 KV字节预算 |
| 调度与 Graph | 主矩阵prefill8192、chunked prefill、`FULL_DECODE_ONLY`；DP1捕获尺寸 `[1,2,4,8,12,16,24,32]` |
| Backend | `FLASHINFER_MLA_SPARSE_DSV4`；MoE配置auto，日志为 `FLASHINFER_CUTLASS`；block size256、FP4 indexer关闭 |
| 其他 | 前缀缓存关闭，不开投机解码或 CPU offload；EP仅在指定候选开启 |
| 压测 | 流式 `/v1/completions`，全服务C32、128请求、8192/1024；seed0、temperature0、ignore_eos、range_ratio0、request_rate=inf |
| 验收 | 每次成功128、失败0，输入1,048,576、输出131,072 tokens；每轮预热64请求，连续2轮无已知编译事件，最多5轮预热/2次尝试 |

四台报告的执行 commit均为 `c54734568603d1ef52f6ccc24ce7673768777c37`，runner源码指纹均为 `f5c33cf418ed84629e3786be4f4c7e651d66dccc3827e4226685a42e3361b583`；模型元数据/tokenizer及分片大小身份一致，未全量重算权重内容哈希。完整身份、本地差异和运行证据见各节点报告。

0.90相同不意味着各拓扑 KV相同：TP4日志预算约32.73 GiB/卡，TP8约51.75 GiB/卡；PP随分层和prefill预算变化。自动分配属于本轮部署条件，不能把跨拓扑的逻辑token容量当作等量物理显存，也不能把吞吐变化全部归为通信收益。

DP2每rank活动容量16，Graph尺寸 `[1,2,4,8,12,16]`；prefill8192按每rank保留，整机上限16384。全局C32由一个客户端请求一个API端口，实际默认2个API worker；日志/metrics观察到两个DP rank各16个running请求。

### CPU、GPU 与内存绑定

通用原则是先选互联较近的 GPU组，再给服务分配近端物理核，客户端使用同socket另一NUMA的独立物理核及内存。规则见[CPU/GPU绑定](../../docs/engine-comparison.md#cpugpu-绑定与同机多服务)，四台采用以下相同映射：

| 用途 | GPU | 服务 CPU / 内存 NUMA | 客户端 CPU / 内存 NUMA |
| --- | --- | --- | --- |
| 后四卡 TP4、TP2×PP2、EP4 | 4–7 | 32–47 / 2 | 48–51 / 3 |
| 48前四卡 TP4 | 0–3 | 0–15 / 0 | 16–19 / 1 |
| 八卡参照及候选 | 0–7 | 不显式绑定 | 不显式绑定 |

CPU编号为宿主逻辑ID，每物理核只用一个线程；SMT兄弟为该ID加64，本次进程不使用兄弟线程。四卡服务16个物理核、客户端4个物理核，Docker cpuset分别限制 CPU与内存集合；不修改宿主 SMT、频率或功耗。八卡移除整个target `binding`，不启用 `--numa-bind`，也不从外层 taskset/numactl引入限制。

上述规则限制本次进程，**不独占这些CPU**。启动后需检查 Docker inspect、worker/client线程允许集合，并用 `numa_maps` 等观察页面分布；内存允许集合不证明全部页面本地驻留。具体字段见[绑定配置](../../docs/configuration.md#cpu--numa-绑定)。

### 准备本机配置

公共起点为 [TP4 recipe](configs/recipes/vllm-tp4-baseline.yaml) 和 [C32三次 workload](configs/workloads/dsv4-8192-1024-c32-n128-repeat3.yaml)。配置保留实验时原文；recipe中的待验证描述是准备时状态，实际验收以本轮报告为准。各候选的参数差异、GPU顺序和实际rank映射在节点报告中完整记录，探索文件仅留本机。

| 节点 | 公共 Campaign | 候选参数差异 |
| --- | --- | --- |
| 45 | [45-dsv4-vllm-tp4-baseline.yaml](configs/campaigns/45-dsv4-vllm-tp4-baseline.yaml) | [TP8、DP2 EP off/on](reports/topology-node45.md#2-运行条件与配置差异) |
| 46 | [46-dsv4-vllm-tp4-baseline.yaml](configs/campaigns/46-dsv4-vllm-tp4-baseline.yaml) | [TP8、TP4×PP2、TP2×PP4](reports/topology-node46.md#2-运行条件与配置差异) |
| 47 | [47-dsv4-vllm-tp4-baseline.yaml](configs/campaigns/47-dsv4-vllm-tp4-baseline.yaml) | [TP8、EP4/EP8](reports/topology-node47.md#2-运行条件与配置差异) |
| 48 | [48-dsv4-vllm-tp4-baseline.yaml](configs/campaigns/48-dsv4-vllm-tp4-baseline.yaml) | [前四卡TP4、后四卡TP2×PP2](reports/topology-node48.md#2-运行条件与配置差异) |

以48为例，先读根 AGENTS/README及[实验方法](../../docs/benchmark-methodology.md)，在新的本机工作区复制配置；已有工作区不要覆盖。以下只演示公共参照入口，候选另建recipe/target/campaign，保持内部相对引用：

```bash
mkdir -p experiments/dsv4-topology-node48-followup/results experiments/dsv4-topology-node48-followup/reports
cp -a projects/dsv4-rtx6000d/configs experiments/dsv4-topology-node48-followup/configs
./bench validate experiments/dsv4-topology-node48-followup/configs/campaigns/48-dsv4-vllm-tp4-baseline.yaml
./bench plan experiments/dsv4-topology-node48-followup/configs/campaigns/48-dsv4-vllm-tp4-baseline.yaml
./bench preflight experiments/dsv4-topology-node48-followup/configs/campaigns/48-dsv4-vllm-tp4-baseline.yaml
./bench run experiments/dsv4-topology-node48-followup/configs/campaigns/48-dsv4-vllm-tp4-baseline.yaml \
  --run-root experiments/dsv4-topology-node48-followup/results/tp4-baseline-01
```

运行前核实整机 GPU空闲、CPU后台负载、目标NUMA内存压力及固定镜像CLI。每个新服务先做health/models和关闭thinking的简短中文探测；运行期间冻结代码/配置。吞吐CV超过3%时先诊断，不丢慢样本；OOM、不支持、超时或资源冲突取证后停止相关候选，不无限重试。复测需另建run-root，与本轮历史结果分开统计。

### 归档规则

每节点保留一个 `reports/topology-nodeNN.md` 和一个 `data/topology-nodeNN.csv`，报告说明差异、结果、异常并链接CSV；README整合结论。后续新批次使用新的主题文件名，不覆盖本批逐次数据。原始日志、预热、拒绝尝试、探索配置及临时脚本继续放本机 `experiments/`，不批量上传。公共配置和节点差异足以重建本轮配置；最终选定的部署配置再精选归档，保留完整依赖并validate/plan。

## 前期实验：镜像选择与双服务对照

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

### TP4 四节点结果

以下为拓扑实验之前的双服务对照。

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

后续统一绑定的48独占实验已完成：前后四卡vLLM吞吐只差0.07%，未复现本批约4.70%的分组差异。因此保留本批结果作为镜像选择依据，不再用它推断后四卡天然更快；也尚不能将原差异唯一归因于CPU核数或双服务竞争。

历史双服务 TP4 实验只保留每节点 CSV 与报告，不提供同步脚本。公共TP4配置用于统一绑定的独占基线，不能作为这批双服务数据的复现入口。

## 后续方向

先完成上面的八卡整机部署对照，形成同资源、同负载的单机性能表。NUMA绑定优化和多节点PD分离留待本轮结果出来后另行规划；不在当前矩阵中追加。PD阶段再分别筛选P、D拓扑，不把混合推理吞吐直接当作单独P或D能力。
