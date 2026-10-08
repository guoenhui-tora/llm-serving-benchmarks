# 分离式推理：24 卡 16K/1K

本研究使用DeepSeek-V4-Flash-0731-NVFP4和24张RTX6000D；负载为真实GovReport文本，每请求输入精确16384 tokens、输出1024 tokens。本文只比较该负载下已测的PD部署；早期[聚合式推理拓扑](aggregated-serving.md)只提供候选线索，不是同条件的性能基线。

## 结论先读

**本次已测方案的优先工作点为4P2D、P每引擎budget8200、仅P侧PHB、全局C160：输出吞吐4482.2±28.2 tok/s，Mean TTFT 7.995秒。** C176虽有约1.8%的吞吐增量，但TTFT均值已接近10秒，三轮中两轮越线；C160也未达到95%的逐请求联合达标率。完整取舍见[最优配置并发扫描](#6-最优配置并发扫描)。

## 1. 从四卡拓扑到24卡4P2D

### 为什么选择四卡TP2×DP2、EP on、K5

最初借用[聚合式推理四卡／八卡对照](aggregated-serving.md)缩小候选范围。这只能说明四卡部署值得尝试：一个实例同时做prefill和decode时表现较好，不等于它分别作为P或D也最优。更合适的研究方法是分别用prefill-only、decode-only筛选单侧候选，再以真实P→D端到端负载验证；本研究是逐步补齐验证，并非从一开始就完整按此流程进行。选型依据如下：

| 阶段 | 关键观察 | 对后续部署的影响 |
| --- | --- | --- |
| 早期聚合式推理，随机8K/1K | 双四卡TP2×PP2、双四卡TP2×DP2 EP on均优于各自节点的双TP4参照，单八卡并非一律落后 | 将四卡纳入候选，而非直接确定P/D最优拓扑 |
| 四卡真实近8K输入、C32、DSpark | TP2×DP2、EP on的K5约966 tok/s，同机历史off约738；TP4的K3约808，K5未继续提高 | 采用已完成验收的DP2＋EP＋K5；没有证明K5在所有拓扑最优 |

这些早期四卡结果的镜像、负载和测量条件与本文不同，只用于提出候选，不能与后面的24卡结果直接排名。八卡D的部署方式则通过下述decode-only实验单独筛选，最终仍由真实PD负载验证。

### D用两个四卡服务，还是一个八卡服务

在固定v0.30、八卡总资源、全局C128下，先做共享16K前缀的decode-only对照。三组整机名义序列容量均为256：前两组四个DP引擎各S64，最后一组两个引擎各S128。

| D部署 | output tok/s，均值±样本SD |
| --- | ---: |
| **两个独立四卡TP2×DP2，EP on** | **4874.18±139.37** |
| 一个八卡TP2×DP4，EP on | 4241.97±104.06 |
| 一个八卡TP4×DP2，EP on | 3919.88±24.64 |

因此优先使用双四卡D。三组均完成三轮、每轮全局512/512请求成功；每引擎S64的target/draft Graph覆盖384/320，S128覆盖768/640，实际捕获完成。双四卡由两个同步客户端各以C64驱动，单八卡由一个客户端以C128驱动，未使用统一代理。此负载每请求仍本地计算约256个输入tokens，不是零prefill；双四卡和TP2×DP4的部分正式轮有已知JIT，双四卡吞吐也有波动。该结果只用于筛选D部署，不把4874 tok/s当作真实PD吞吐或已证实的稳定优势。

### 24卡布局与测量口径

46、47节点各运行两个四卡P，48节点运行两个四卡D；45仅承担CPU代理与客户端。六服务均为TP2×DP2、EP on、K5、FP8 KV、V2 async，NIXL传输远端KV，P/D本地prefix cache关闭。

| 每DP引擎参数 | P，共8个引擎 | D，共4个引擎 |
| --- | ---: | ---: |
| `max-num-seqs` | 32 | 64 |
| 初始／当前token budget | 16640／8200 | 16384／16384 |
| K5 target／draft Graph覆盖上限 | 192／160 | 384／320 |
| 当前Graph模式 | FULL_DECODE_ONLY | FULL_DECODE_ONLY |

**P与D各有256名义序列容量，不是4P合计128，也不是实际执行batch。** K5每请求target验证形状为6 tokens、draft为5 queries，Graph按每个引擎的容量配置。

后续24卡结果统一使用服务端和客户端vLLM 0.30.0、1024条互异GovReport输入，固定tokenizer、temperature=0、ignore_eos、无限到达率。C始终是一个客户端的全局在途上限。每档先做2C条完整请求预热，再固定三轮各4C条正式请求；相同C取相同请求集，不挑最快轮。吞吐为完整客户端窗口的输出总tokens／时长，包含起跑与排空。

下表及后文的“±”是三轮样本标准差；P95为三次单轮P95的平均。逐请求联合达标要求同一请求TTFT≤10秒且TPOT≤50毫秒，goodput为达标请求数／完整窗口秒数。正常正式窗口要求零失败、输入输出总量正确、远端KV命中、D无整段输入重算，并核对抢占、JIT和Graph；异常不通过删轮或加轮消除。

## 2. 原始4P2D：并发增加，收益在哪里停止

初始P budget=16640。C148、C152在相同配置下于后续测量，其余档位来自同一次服务启动。

| 全局C | output tok/s，均值±SD | Mean／P95 TTFT，s | Mean TPOT，ms | 联合达标率 | goodput，req/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1132.10±34.58 | 2.474／4.839 | 10.675 | 100.00% | 1.106 |
| 32 | 1759.45±5.99 | 3.311／7.296 | 13.936 | 100.00% | 1.718 |
| 48 | 2220.44±14.88 | 4.130／10.516 | 16.159 | 93.75% | 2.033 |
| 64 | 2680.39±14.17 | 4.774／12.669 | 17.462 | 89.06% | 2.331 |
| 80 | 2997.53±12.01 | 5.453／15.852 | 19.392 | 86.25% | 2.525 |
| 96 | 3308.83±27.45 | 6.432／18.843 | 20.441 | 84.38% | 2.726 |
| 112 | 3624.00±34.71 | 7.247／21.081 | 21.297 | 83.18% | 2.944 |
| 128 | 3930.44±18.39 | 8.104／24.284 | 22.431 | 81.71% | 3.136 |
| 144 | 4120.98±8.80 | 9.190／26.510 | 23.516 | 78.70% | 3.167 |
| 148 | 4167.59±12.15 | 9.687／27.712 | 23.549 | 76.52% | 3.114 |
| 152 | 4203.64±5.28 | 10.705／28.071 | 23.093 | 69.08% | 2.836 |
| 160 | 4263.74±35.61 | 11.898／29.558 | 23.355 | 56.20% | 2.339 |

**原配置C144以后主要是在用更多等待换少量吞吐。** C148是最高已测均值SLO达标档，但TTFT余量只有0.313秒；C152已经越线。C160的TTFT进一步升高，因此未继续测C176。

### 同资源的部署对照：4P2D、5P1D与单八卡D

保持24卡、真实16K/1K与C144，又测了两种替代部署：

| 部署 | output tok/s，均值±SD | Mean TTFT，s | Mean TPOT，ms |
| --- | ---: | ---: | ---: |
| 原4P2D，双四卡D | 4120.98±8.80 | 9.190 | 23.516 |
| 5P1D，六个四卡服务 | 3452.64±35.25 | 6.426 | 32.851 |
| 四个四卡P＋单八卡TP2×DP4 D | 3769.06±65.07 | 8.223 | 28.038 |

多一个P确实降低TTFT，但减少D使吞吐下降16.2%；合并八卡D也损失8.5%。因此保留4P2D，转向提高P服务本身的效率。两项均为独立启动的单档比较，不证明4P2D是所有配比中的最优解。

同24卡的`2×(2P1D)`及六个四卡聚合式服务均未配对实测；因此这里比较的是已测PD部署，而不是PD与聚合式推理的收益。

## 3. 第一个有效方向：减小P调度预算

这里始终是16K输入，修改的是每DP引擎`max-num-batched-tokens`。固定C144的各次独立启动结果如下：

| P budget | output tok/s，均值±SD | Mean TTFT，s | 判断 |
| ---: | ---: | ---: | --- |
| 16640 | 4120.98±8.80 | 9.190 | 初始基线 |
| 8208 | 4212.41±14.88 | 8.093 | 相对8200无明确收益 |
| **8200** | **4229.34±32.83** | **7.926** | 采用的候选 |
| 8192 | 4108.76±15.01 | 8.564 | 三轮正式均有JIT匹配 |

其他已测预算没有改变采用8200的判断；8196在启动预热时发生positions对齐错误，没有性能结果。较小budget可能改变chunked prefill的分步与排队节奏，但**不能据此断言8200恰好等于8192加K5所需tokens**，8200与8208的微小差别也未证实可稳定复现。后来在PHB已开启的C160重新测P16640，吞吐4391、TTFT10.119秒，仍不如8200。

## 4. 第二个有效方向：P侧PHB P2P

为避免每次定位都占24卡，使用一个四卡P做真实16K/1输出、C32的prefill-only实验。EP on供给约1.187 req/s，off约0.962，关闭EP损失约19%，因此继续保留EP。

Nsight发现NCCL相关kernel占每卡约47%的观测墙钟时间，其中EP约34.8%、TP约12.6%。这包含同步等待，不能全部解释为有效传输或带宽饱和。后续定位到的是传输路径选择，而不是先更换通信算法。

### PHB与SHM：改的是路径，不是通信协议

```bash
export NCCL_P2P_LEVEL=PHB
```

**PHB（PCI Host Bridge，PCI主桥）是P2P可用的拓扑距离等级，不是NCCL通信协议。** 这个设置允许同一NUMA节点内、路径经过PCI主桥的GPU使用P2P直接访问彼此显存；仍需硬件和驱动支持。经过CPU的PCI桥不等于由CPU核心搬运数据，也不要求主机内存中转。它改变NCCL的选路条件，不改变GPU的物理连接。

**默认不是关闭P2P。** 本次“默认”指不设置`NCCL_P2P_LEVEL`，由NCCL根据拓扑与平台自动选择，不能一概写成默认PXB或默认PHB。四卡微测的实际连接变化如下：

| GPU连接 | 默认路径 | 设置PHB后 |
| --- | --- | --- |
| 4→5、6→7，近端卡对 | P2P/CUMEM | P2P/CUMEM，不变 |
| 5→6、7→4，跨近端卡对 | **SHM/direct** | **P2P/CUMEM** |

本机跨卡对在`nvidia-smi topo`中显示为同NUMA内的NODE，NCCL内部归类为PHB，两种工具的标签不能机械等同。

**SHM确实经过host memory；P2P/CUMEM则不是网卡RDMA。** 按本次NCCL 2.30.7实现，下面区分的是通信数据路径，不包含初始化和控制信息：

| 路径 | 数据经过哪里 | 是否经主机内存中转 | 是否需要RDMA网卡 |
| --- | --- | --- | --- |
| `SHM/direct` | GPU A → 主机共享内存缓冲区 → GPU B | **是** | 否 |
| `P2P/CUMEM`，本次PCIe四卡内 | GPU A通过PCIe peer访问GPU B显存中的通信缓冲区，可采用读或写 | **否** | **否** |
| GPUDirect RDMA，作为概念对照 | 网卡直接访问GPU显存；跨节点可走GPU A → 网卡 → 网络 → 网卡 → GPU B | 数据可绕过主机内存 | 是，针对这里的网络传输场景 |

`SHM/direct`中的`direct`表示GPU直接访问映射的主机共享缓冲区，**不是两张GPU显存直连，也不表示CPU核心逐字节执行拷贝**。CPU参与内存准备和控制，与数据是否经过CPU所连接的DRAM是两个问题。

`CUMEM`指CUDA Driver的`cuMem*`内存分配、共享和映射机制，不是一种通信协议，也不能单凭这个词判断内存位置：SHM也可以用cuMem分配主机内存。这里应看完整标签`P2P/CUMEM`，它表示建立了GPU peer访问路径；具体是对端读取还是写入，不能只凭该标签判断。

因此本次PHB优化应理解为**本机PCIe GPU P2P替代主机内存中转**，不是启用InfiniBand／RoCE RDMA。跨节点P→D的NIXL／UCX是另一条路径；它是否实际使用GPUDirect RDMA，也不能由这条P2P日志推断。

通信还有另外两层：**Ring／Tree是算法，Simple／LL／LL128是协议**，后者由`NCCL_PROTO`控制。本次两组大消息均使用Ring＋Simple，变化是跨卡对的SHM→P2P；三个combine形状耗时下降约22%～28%。这优化的是P服务内部通信，**没有更换跨节点P→D传KV的NIXL连接器**。

### 24卡端到端验证

固定24卡4P2D、真实16K/1K与全局C144。先以原始配置为参照，再在P8200下只改变P侧PHB；D预算、容量与Graph保持不变：

| 配置 | P budget／通信 | Output (tok/s，均值±SD) | Mean／P95 TTFT (s) | Mean TPOT (ms) | goodput (req/s) |
| --- | --- | ---: | ---: | ---: | ---: |
| 原始4P2D | 16640／默认 | 4120.98±8.80 | 9.190／26.510 | 23.516 | 3.167 |
| 小P预算 | 8200／默认 | 4216.56±23.20 | 7.866／26.074 | 24.163 | 3.305 |
| **小P预算＋P侧PHB** | **8200／PHB** | **4286.50±30.79** | **7.010／24.357** | **24.360** | **3.437** |

原始配置到P8200＋PHB的观测变化为吞吐+4.0%、Mean TTFT下降23.7%；原始行来自较早扫描，后两行来自PHB配对实验。**直接比较后两行**，P侧PHB使吞吐提高1.66%、TTFT降低10.9%。四卡微测的约25%收益没有直接变成整机25%，但改善P侧处理为提高并发留下了空间。

固定配置的完整并发曲线见[最优配置并发扫描](#6-最优配置并发扫描)。

## 5. 未采用的优化方向

| 尝试 | 结果与取舍 |
| --- | --- |
| P启用`FULL_AND_PIECEWISE` | C152吞吐约+0.52%，goodput基本不变，Graph内存增加；当时逐步观测缺失，不认定稳定收益 |
| D侧也启用PHB | 新测C160为4500.88±54.13，对照4483.91±20.00，约+0.38%落在波动内 |

四卡P侧对`NCCL_PROTO=Simple`和EP combine算子替换的微测未形成端到端候选，故不列入性能排序。

### D Graph确实变快，为什么仍没有采用

P8200＋PHB保持不变，将D从`FULL_DECODE_ONLY`改为`FULL_AND_PIECEWISE`，实际覆盖了原先NONE分派的步骤。但在闭环负载下，D更快释放请求后，后续请求更早进入P队列，TTFT明显变差：

| D设置／C | output tok/s，均值±SD | Mean TTFT，s | Mean TPOT，ms | goodput，req/s |
| --- | ---: | ---: | ---: | ---: |
| 默认／160，新测对照 | 4483.91±20.00 | 8.072 | 25.411 | 3.537 |
| FULL_AND_PIECEWISE／160 | 4606.97±25.27 | 14.968 | 16.643 | 0.309 |
| FULL_AND_PIECEWISE／128 | 4454.44±13.07 | 9.654 | 16.524 | 3.407 |
| FULL_AND_PIECEWISE／144 | 4571.56±18.88 | 12.304 | 16.591 | 0.930 |

C160的TPOT下降约34.5%，但P阶段平均耗时7.364→14.526秒、P总waiting采样均值16.93→48.15。减少并发到C128后均值SLO通过，吞吐却没有超过原C160。因此它是**真实的decode加速，但不是本目标下的整机改进**。PD分离并不保证每次D调用都适用FULL_DECODE_ONLY；本例也不代表D重新计算整段输入，远端KV和零整段重算均已核对。

### 波动与长窗口，不要误当成优化

C176较早一次独立测量的三轮TTFT为9.557／9.671／11.289秒。请求时间线显示，较快的首批D完成会触发闭环客户端提前补发，而P完成节奏没有同比提前，等待因此转移到P侧；不能仅凭TTFT变大断言P计算突然变慢。后续逐步观测支持这一解释，但尚未确定D早期速度变化的唯一根因。

在C144的P侧PHB对照中，三轮各自后续432条请求全部联合达标，首批144条则各仅41条达标；本轮尾延迟主要集中在突发首批。

另一次连续2640请求得到约5045 tok/s、Mean TTFT7.547秒，是更长窗口摊薄起跑／排空成本，**不是服务优化后突破4500的同协议结果**。本研究仍以固定2C预热＋三轮4C正式请求作主要比较。

## 6. 最优配置并发扫描

四个四卡P部署在46、47节点，两个四卡D部署在48节点；六个服务均为TP2×DP2、EP on、DSpark K5。最终配置为P每引擎S32/budget8200、仅P侧`NCCL_P2P_LEVEL=PHB`，D每引擎S64/budget16384、默认通信，Graph模式均为`FULL_DECODE_ONLY`。服务端和客户端均为vLLM 0.30.0，本地prefix cache关闭。六个服务在本次扫描中只启动一次，逐档只改一个客户端入口的全局并发C。

每档以真实独立的16K/1024请求完成2C条预热，随后正式测三轮、每轮4C条；吞吐按完整客户端窗口计算。表中的吞吐为三轮均值±样本标准差，其他指标为三轮均值；P99 ITL是各轮P99的均值，不是合并请求后的P99。

| 全局并发 | Mean TTFT (ms) | Output (tok/s，均值±SD) | Mean TPOT (ms) | Mean ITL (ms) | P99 ITL (ms) | 均值SLO |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 16 | 2470.0 | 1195.3±16.7 | 10.38 | 34.81 | 220.62 | 通过 |
| 32 | 3170.2 | 1787.6±6.3 | 13.84 | 46.96 | 224.48 | 通过 |
| 48 | 3725.4 | 2310.3±29.5 | 15.92 | 53.76 | 223.33 | 通过 |
| 64 | 4253.8 | 2762.9±11.6 | 17.46 | 59.40 | 216.50 | 通过 |
| 80 | 4675.7 | 3130.5±7.9 | 19.39 | 65.81 | 228.31 | 通过 |
| 96 | 5244.8 | 3457.4±16.1 | 20.83 | 70.62 | 232.19 | 通过 |
| 112 | 5739.6 | 3773.6±28.2 | 21.93 | 74.57 | 236.30 | 通过 |
| 128 | 6311.7 | 4015.8±6.6 | 23.47 | 79.80 | 238.40 | 通过 |
| 144 | 7012.2 | 4289.3±43.8 | 24.36 | 82.88 | 239.89 | 通过 |
| **160** | **7994.7** | **4482.2±28.2** | **25.45** | **86.36** | **242.28** | **通过，优先** |
| 176 | 9939.8 | 4563.2±36.1 | 26.11 | 88.59 | 242.97 | 通过，余量小 |

均值SLO要求三轮Mean TTFT均值≤10000ms且Mean TPOT均值≤50ms，**不是95%的请求同时达标**。C176的三轮TTFT为10068／9546／10205ms：均值虽通过，但两轮越线；相对C160吞吐仅增约1.8%，逐请求联合达标goodput由3.550降至3.428 req/s。因此优先C160，而非把C176的均值通过视为稳定余量。C160的P95 TTFT为26.879秒、逐请求联合达标率约81.1%，仍不能据此承诺线上尾延迟；C161～175未逐点测量。

本次C160的D各引擎running采样峰值36～38/S64，KV峰值约17.4%；C176生成batch峰值约41，Graph padding峰值282，未顶到384。P每引擎running采样峰值2/S32。容量尚未被证实是限制因素，但有容量余量也不等于有算力余量。C16正式窗口有16次已知JIT日志匹配，其余档没有已知模式匹配；所有预定轮次均保留。

[最终配置逐轮 CSV](results/pd-24gpu-16k-1k.csv)保留各档三轮的吞吐、延迟分位数及达标指标。`slo_joint_pass_rate`是同一请求同时满足TTFT和TPOT限值的比例，`slo_goodput_req_s`是达标请求数除以完整窗口时长；与表中的均值SLO不是同一口径。总token吞吐包含输入和输出，生成吞吐使用`output_throughput_tok_s`。

同并发优化收益见上文C144对照；不能把提高并发到C160的收益算成单项优化效果。当前方向是提高P供给效率并控制排队，而不是只增加D的名义容量；若要求严格的逐请求尾延迟，仍需研究P侧调度与准入。

## 7. 启动与复现

<details>
<summary><strong>固定环境、服务启动与压测命令</strong></summary>

### 固定环境和必要前提

服务与客户端均使用`vllm/vllm-openai:v0.30.0`，本次本地image ID为：

```bash
IMAGE_ID=sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
docker image inspect "$IMAGE_ID" --format '{{.Id}}'
```

权重是`nvidia/DeepSeek-V4-Flash-0731-NVFP4`，同一tokenizer。该固定镜像实测加载了**DSpark草稿MoE的MXFP4格式分派修复**：保持target NVFP4及FP8 linear不变，将草稿正确分派到MXFP4，并核验三个草稿MoE模块；没有加载旧DP/mHC修复或改写通信算子。原版镜像标签不包含这个加载hook，不能省略此条件后声称复现了本文。

以下命令按实测参数整理，省去日志采集和实验编排。启动前须准备上述加载修复、具备下述语义的代理和精确数据。

| 运行位置 | 服务 | GPU | CPU／NUMA | HTTP／NIXL／DP RPC端口 |
| --- | --- | --- | --- | --- |
| 46、47各一套 | P，后四卡 | 4,5,6,7 | 32-47／2 | 31449／29300／29600 |
| 46、47各一套 | P，前四卡 | 0,1,2,3 | 0-15／0 | 31450／29320／29620 |
| 48 | D，后四卡 | 4,5,6,7 | 32-47／2 | 31449／29300／29600 |
| 48 | D，前四卡 | 0,1,2,3 | 0-15／0 | 31450／29320／29620 |

宿主GPU0～3对应NUMA0、GPU4～7对应NUMA2。45的代理绑定CPU16～19／NUMA1，客户端CPU20～23／NUMA1。CPU集合不是独占核；实测保留了既有后台进程，未改功耗或锁频。

### P／D共用启动命令

以下以46后四卡P为例；其他服务按表替换变量。各服务使用独立的持久缓存，保留已有JIT缓存；按[PD启动说明](pd-startup.md)先准备MXFP4加载hook。

```bash
ROLE=P
HOST_IP=10.90.1.46
GPUS=4,5,6,7
CPUS=32-47
NUMA=2
HTTP_PORT=31449
NIXL_PORT=29300
RPC_PORT=29600
NAME=pd24-p0
REPO="$HOME/llm-serving-benchmarks"
MODEL_DIR=/data/models/DeepSeek-V4-Flash-0731-NVFP4
CACHE_DIR="$HOME/.cache/dsv4-pd/$NAME"
PATCH_DIR="$CACHE_DIR/dsv4-v030-mxfp4"
IMAGE_ID=sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
test -f "$PATCH_DIR/sitecustomize.py"

P_GRAPH='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[5,6,10,12,20,24,40,48,60,72,80,96,120,144,160,192],"max_cudagraph_capture_size":192}'
D_GRAPH='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[5,6,10,12,20,24,40,48,60,72,80,96,120,144,160,192,200,240,280,288,320,336,384],"max_cudagraph_capture_size":384}'
if [[ "$ROLE" == P ]]; then
  SEQS=32; BUDGET=8200; KV_ROLE=kv_producer; GRAPH="$P_GRAPH"
  ROLE_ENV=(-e NCCL_P2P_LEVEL=PHB -e NCCL_DEBUG=INFO
    -e NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,P2P,SHM,NET)
else
  SEQS=64; BUDGET=16384; KV_ROLE=kv_consumer; GRAPH="$D_GRAPH"
  ROLE_ENV=(-e NCCL_DEBUG=WARN)
fi

docker run -d --pull never --name "$NAME" \
  --network host --ipc host --gpus "\"device=$GPUS\"" \
  --cpuset-cpus "$CPUS" --cpuset-mems "$NUMA" \
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  --cap-add SYS_NICE --device /dev/infiniband \
  -v "${MODEL_DIR:?}:/model:ro" -v "${CACHE_DIR:?}:/root/.cache:rw" \
  -v "${PATCH_DIR:?}:/loader-fix:ro" -e PYTHONPATH=/loader-fix \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e TILELANG_CACHE_DIR=/root/.cache/tilelang \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_NIXL_SIDE_CHANNEL_HOST="$HOST_IP" \
  -e VLLM_NIXL_SIDE_CHANNEL_PORT="$NIXL_PORT" \
  -e UCX_NET_DEVICES=mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1 \
  -e UCX_TLS=rc,cuda_copy,cuda_ipc,sm,self -e UCX_IB_GID_INDEX=3 \
  -e UCX_LOG_LEVEL=info -e NIXL_LOG_LEVEL=INFO "${ROLE_ENV[@]}" \
  --entrypoint vllm "$IMAGE_ID" serve /model \
  --served-model-name deepseek-v4-flash --host 0.0.0.0 --port "$HTTP_PORT" \
  --trust-remote-code --enable-auto-tool-choice \
  --no-enable-prefix-caching --enable-expert-parallel --enable-chunked-prefill \
  --jit-monitor-verbose --async-scheduling \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --data-parallel-size 2 --data-parallel-size-local 2 \
  --data-parallel-rpc-port "$RPC_PORT" --distributed-executor-backend mp \
  --max-model-len 32768 --max-num-seqs "$SEQS" \
  --max-num-batched-tokens "$BUDGET" --gpu-memory-utilization 0.9 \
  --kv-cache-dtype fp8_e4m3 --block-size 256 \
  --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV4","indexer_kv_dtype":"auto"}' \
  --moe-backend auto --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --compilation-config "$GRAPH" --seed 0 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":false}' \
  --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"$KV_ROLE\",\"kv_buffer_device\":\"cuda\",\"kv_load_failure_policy\":\"fail\",\"kv_connector_extra_config\":{\"backends\":[\"UCX\"],\"enforce_handshake_compat\":true}}"
```

还原初始基线时，仅将P的`BUDGET`改回16640、去掉`NCCL_P2P_LEVEL=PHB`及该次通信诊断设置；D容量、预算及Graph不变。前四卡实测也沿用`mlx5_4～7`，不是按NUMA优化过的NIC配置，不能悄悄更换后仍当作同一对照。异机的CPU、GPU、NIC和IP必须按实际拓扑重新核对。

### 代理与压测

代理监听45的`127.0.0.1:31580`，P池为46／47的31449、31450，D池为48的31449、31450。它不是普通HTTP负载均衡：

1. 按least-inflight分别预留P、D；P占用计到prefill完成，D占用计到整个请求结束，相同负载时平衡P→D路径。
2. 向P发送原prompt，设置`stream=false`、`max_tokens=1`、`kv_transfer_params={"do_remote_decode":true,"do_remote_prefill":false}`。
3. 检查P返回的远端engine/request/host/port/block元数据，将其作为`kv_transfer_params`附到原始D请求，保持原1024输出上限并转发D的流。
4. 不自动重试、不退回D本地整段prefill；上游连接不设隐藏的100并发上限。实测每次上游POST使用新连接，超时1200秒。

六个服务全部READY后，在45节点启动代理（CPU16～19／NUMA1）：

```bash
REPO="$HOME/llm-serving-benchmarks"
IMAGE_ID=sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
docker run -d --pull never --name sb-pd24-proxy --network host \
  --cpuset-cpus 16-19 --cpuset-mems 1 \
  -v "$REPO/src:/src:ro" -e PYTHONPATH=/src \
  --entrypoint python3 "$IMAGE_ID" -m serving_bench.pd_proxy \
  --host 127.0.0.1 --port 31580 --timeout 1200 \
  --prefill http://10.90.1.46:31449 http://10.90.1.46:31450 \
            http://10.90.1.47:31449 http://10.90.1.47:31450 \
  --decode http://10.90.1.48:31449 http://10.90.1.48:31450
```

压测数据每行是`{"id":"...","prompt":"...","input_tokens":16384}`。实测数据SHA256为`6166561f98e83a7591eaa5bc36dcb317f20edd42f187a10f4cc6793df839b9b2`；另一份同长度文本只能复现负载规格，不能当作相同请求集。

正式压测须加载仓库的[JSONL适配](../../../src/serving_bench/clients/jsonl_dataset.py)，逐条用tokenizer核对16384输入tokens；原生`custom`不能直接代替这一适配。以下为C160一轮正式测量；运行前为每轮设置不同的绝对路径`RESULT_DIR`，完整预热将`N`改为`2*C`，正式三轮期间服务保持运行：

```bash
REPO="$HOME/llm-serving-benchmarks"
MODEL_DIR=/data/models/DeepSeek-V4-Flash-0731-NVFP4
IMAGE_ID=sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
C=160; N=$((4*C))
RESULT_DIR="${RESULT_DIR:?请先指定本轮独立的结果目录}"
mkdir -p "$RESULT_DIR"
docker run --rm --pull never --network host \
  --cpuset-cpus 20-23 --cpuset-mems 1 \
  -v "$MODEL_DIR:/model:ro" \
  -v "$REPO/projects/dsv4-rtx6000d/datasets/govreport-isl16384-exact-n1024.jsonl:/dataset.jsonl:ro" \
  -v "$REPO/src/serving_bench/clients/jsonl_dataset.py:/jsonl_dataset.py:ro" \
  -v "$RESULT_DIR:/results:rw" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint python3 "$IMAGE_ID" -c '
import runpy
from pathlib import Path
import vllm.benchmarks.serve as serve
runpy.run_path("/jsonl_dataset.py")["install"](serve, {
    "max_input_tokens": 16384, "output_tokens": 1024,
    "sha256": "6166561f98e83a7591eaa5bc36dcb317f20edd42f187a10f4cc6793df839b9b2"
}, Path("/results"), 0, 1)
from vllm.benchmarks.serve import add_cli_args, main
from vllm.utils.argparse_utils import FlexibleArgumentParser
p = FlexibleArgumentParser(); add_cli_args(p); main(p.parse_args())' \
  --backend vllm --base-url http://127.0.0.1:31580 \
  --endpoint /v1/completions --model /model --tokenizer /model \
  --served-model-name deepseek-v4-flash \
  --dataset-name custom --dataset-path /dataset.jsonl \
  --custom-output-len 1024 --skip-chat-template \
  --disable-shuffle --no-oversample --ignore-eos --trust-remote-code \
  --num-prompts "$N" --max-concurrency "$C" --request-rate inf \
  --num-warmups 0 --seed 0 --temperature 0 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 \
  --goodput ttft:10000 tpot:50 --save-result --save-detailed \
  --result-dir /results --result-filename raw.json
```

C160每轮正式应为640/640成功、10,485,760输入tokens、655,360输出tokens；320条完整预热单独计时。原扫描READY／单窗口／每档／campaign上限为2700／900／4200／54000秒，零性能重试。OOM、请求失败或工作量错误停止取证；C144后的追加扫描在Mean TTFT>10秒**或**Mean TPOT>50毫秒时保留该档并停止更高C。

</details>
