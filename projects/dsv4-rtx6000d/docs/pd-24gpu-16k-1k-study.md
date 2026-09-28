# DSV4 / RTX6000D：24卡PD的16K/1K选型与优化

截至2026-09-28，研究对象为DeepSeek-V4-Flash-0731-NVFP4、24张RTX6000D、真实GovReport精确16384-token输入和1024-token输出。

## 结论先读

**当前已测方案中，优先配置是4P2D＋P budget8200＋P侧PHB、全局C160：新一轮同启动扫描为4482.2 output tok/s、Mean TTFT 7.995秒。** 这不是所有配置空间的全局最优；C176虽略增吞吐，但均值TTFT已接近10秒，三轮中两轮越线。

### 2026-09-27～28：优化配置的同启动并发扫描

六个四卡服务只启动一次，固定P budget8200、仅P侧启用PHB；真实独立16K/1024请求、每档2C条预热及三轮各4C条正式测量。以下数值均为三轮正式均值，吞吐包含客户端完整计时窗口；ITL为Mean ITL，P99 ITL为各轮P99的均值，均未参与SLO判定。

| 全局并发 | Mean TTFT (ms) | Output (tok/s) | Mean TPOT (ms) | Mean ITL (ms) | P99 ITL (ms) | 均值SLO |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 16 | 2470.0 | 1195.3 | 10.38 | 34.81 | 220.62 | 通过 |
| 32 | 3170.2 | 1787.6 | 13.84 | 46.96 | 224.48 | 通过 |
| 48 | 3725.4 | 2310.3 | 15.92 | 53.76 | 223.33 | 通过 |
| 64 | 4253.8 | 2762.9 | 17.46 | 59.40 | 216.50 | 通过 |
| 80 | 4675.7 | 3130.5 | 19.39 | 65.81 | 228.31 | 通过 |
| 96 | 5244.8 | 3457.4 | 20.83 | 70.62 | 232.19 | 通过 |
| 112 | 5739.6 | 3773.6 | 21.93 | 74.57 | 236.30 | 通过 |
| 128 | 6311.7 | 4015.8 | 23.47 | 79.80 | 238.40 | 通过 |
| 144 | 7012.2 | 4289.3 | 24.36 | 82.88 | 239.89 | 通过 |
| **160** | **7994.7** | **4482.2** | **25.45** | **86.36** | **242.28** | **通过** |
| 176 | 9939.8 | 4563.2 | 26.11 | 88.59 | 242.97 | 通过 |

均值SLO按三轮Mean TTFT均值≤10000ms且Mean TPOT均值≤50ms判定，**不等于逐请求达标**。C176三轮TTFT为10068／9546／10205ms，均值虽通过但余量很小；吞吐相对C160只增约1.8%，逐请求联合达标goodput由3.550降至3.428 req/s。C16有16次已知JIT日志匹配，其余档正式窗口没有已知模式匹配；不因JIT删轮。此表是后续单次启动的完整扫描，不能与下文较早独立启动批次的C144/C160/C176逐轮拼接。

### 同并发下的优化收益：固定C144

三组均为同一24卡4P2D布局、真实16K/1K负载；D预算16384、容量和Graph不变。“小P”仅指减小P调度预算，不是减少P数量或缩短输入。

| 阶段 | P budget／通信 | output tok/s，均值±SD | Mean／P95 TTFT，s | Mean TPOT，ms | goodput，req/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 原始4P2D | 16640／默认 | 4120.98±8.80 | 9.190／26.510 | 23.516 | 3.167 |
| 小P预算 | 8200／默认 | 4216.56±23.20 | 7.866／26.074 | 24.163 | 3.305 |
| **小P预算＋P侧PHB** | **8200／PHB** | **4286.50±30.79** | **7.010／24.357** | **24.360** | **3.437** |

固定C144，原始配置到最终配置的观测变化为：**吞吐+4.0%，Mean TTFT下降23.7%**。原始行来自较早扫描；后两行取同批PHB配对实验、各自独立启动，小P行不是预算搜索时的4229.34。因而整体变化是跨批次对照，较直接的PHB收益是后两行之间的吞吐+1.66%、Mean TTFT下降10.9%；不能把提高并发到C160的收益混进此表。

### 当前配置与此前独立批次锚点

| 项目 | 当前配置 |
| --- | --- |
| 部署 | 46、47各两个四卡P，48两个四卡D，共24卡；六服务均TP2×DP2、EP on、DSpark K5 |
| 软件与KV | 服务端／客户端vLLM 0.30.0；FP8 KV、V2 async、NIXL，P/D本地prefix cache关闭 |
| P，每DP引擎 | token budget **8200**、S32；仅P设置 **`NCCL_P2P_LEVEL=PHB`** |
| D，每DP引擎 | token budget **16384**、S64；四引擎合计名义容量256，NCCL选路保持默认 |
| Graph与负载 | FULL_DECODE_ONLY；P target/draft覆盖192/160，D覆盖384/320；16K输入／1K输出，推荐全局 **C160** |

**小P预算8200＋P侧PHB的首次三档结果如下。** 每档独立启动，固定三轮4C正式请求；吞吐“±”为三轮样本标准差，不混入后续诊断复测：

| 全局C | output tok/s | Mean／P95 TTFT，s | Mean TPOT，ms | 联合达标率 | goodput，req/s | 均值SLO |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 144 | 4286.50±30.79 | 7.010／24.357 | 24.360 | 82.12% | 3.437 | 通过 |
| **160** | **4501.50±17.16** | **8.219／26.776** | 25.080 | 80.63% | **3.544** | **通过** |
| 176 | 4566.82±5.88 | 10.172／29.589 | 25.818 | 75.19% | 3.353 | TTFT越线 |

**这批独立补测中，C160是吞吐与延迟的折中点；C176吞吐仅再增1.45%，TTFT越线。** 当时尚未完整重扫低并发；后续同启动扫描见开头，C161～175仍未测。

主要改善来自P调度预算和P侧通信路径；换P/D配比、合并D或继续加速D，尚未得到更好的均值SLO内工作点。**均值SLO是三轮Mean TTFT均值≤10秒且Mean TPOT均值≤50毫秒，不是95%请求达标。** 当前P95 TTFT仍约26.8秒，逐请求联合达标率约81%，不能据此承诺线上尾延迟。C144的P侧PHB对照中，三轮各自后续432条请求全部联合达标，首批144条则各仅41条达标，突发首批是这一负载下尾延迟的主要缺口。

## 1. 从四卡拓扑到24卡4P2D

### 为什么选择四卡TP2×DP2、EP on、K5

选型不是一次实验得出的，最初的PP方案也有竞争力：

| 阶段 | 关键观察 | 对后续部署的影响 |
| --- | --- | --- |
| 早期四卡／八卡拓扑筛选，8K/1K | 四卡TP2×PP2比同节点TP4快约6.6%；双四卡PP、双四卡DP均有竞争力，八卡单实例没有形成普遍优势 | 保留四卡作为部署单元，不能仅凭跨NUMA解释全部差异 |
| 四卡真实近8K输入、C32、DSpark | TP2×DP2、EP on的K5约966 tok/s，同机历史off约738；TP4的K3约808，K5未继续提高 | 采用已完成验收的DP2＋EP＋K5；没有证明K5在所有拓扑最优 |
| v0.30、16卡3P1D、16K/1K、C128 | 全PP2为2643，全DP2为3092，PP2→DP2为3064 tok/s | 全DP2成为PD基线；PP使用Mooncake、DP使用NIXL，不能把差值全归因于拓扑 |

这些是选型背景，不与后面的24卡表直接排名：早期镜像、数据、协议不同，v0.30的16卡实验客户端也仍是v0.29。TP2把TP组限制在两卡内，但EP组、调度和服务边界同时变化，不能把“用四卡更好”简化成单一通信结论。

### D用两个四卡服务，还是一个八卡服务

在固定v0.30、八卡总资源、全局C128下，先做共享16K前缀的decode-only对照。三组整机名义序列容量均为256：前两组四个DP引擎各S64，最后一组两个引擎各S128。

| D部署 | output tok/s，均值±样本SD |
| --- | ---: |
| **两个独立四卡TP2×DP2，EP on** | **4874.18±139.37** |
| 一个八卡TP2×DP4，EP on | 4241.97±104.06 |
| 一个八卡TP4×DP2，EP on | 3919.88±24.64 |

因此优先使用双四卡D。此负载每请求仍本地计算约256个输入tokens，不是零prefill；前两组部分正式轮有JIT。它用于筛选D部署，不能拿4874 tok/s与真实PD直接比较。

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

初始P budget=16640。除C148、C152各为独立补测外，其余档位来自同一次六服务启动。

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
| 148，独立补测 | 4167.59±12.15 | 9.687／27.712 | 23.549 | 76.52% | 3.114 |
| 152，独立补测 | 4203.64±5.28 | 10.705／28.071 | 23.093 | 69.08% | 2.836 |
| 160 | 4263.74±35.61 | 11.898／29.558 | 23.355 | 56.20% | 2.339 |

**原配置C144以后主要是在用更多等待换少量吞吐。** C148是最高已测均值SLO达标档，但TTFT余量只有0.313秒；补测C152已经越线。原步长16扫描在C160触线后停止，未继续C176。

### 是不是2:1的P/D配比不适合16K输入

保持24卡、真实16K/1K与C144，又测了两种替代部署：

| 部署 | output tok/s，均值±SD | Mean TTFT，s | Mean TPOT，ms |
| --- | ---: | ---: | ---: |
| 原4P2D，双四卡D | 4120.98±8.80 | 9.190 | 23.516 |
| 5P1D，六个四卡服务 | 3452.64±35.25 | 6.426 | 32.851 |
| 四个四卡P＋单八卡TP2×DP4 D | 3769.06±65.07 | 8.223 | 28.038 |

多一个P确实降低TTFT，但减少D使吞吐下降16.2%；合并八卡D也损失8.5%。因此保留4P2D，转向提高P服务本身的效率。两项均为独立启动的单档比较，不证明4P2D是所有配比中的最优解。

### GPU增加时，吞吐有没有同比增加

应该按相同`全局C/GPU数`比较，而不是固定全局C64：

| 历史部署 → 本次原始4P2D | C/GPU | GPU倍数 | output tok/s变化 | 吞吐倍数 |
| --- | ---: | ---: | ---: | ---: |
| 12卡2P1D/C64 → 24卡4P2D/C128 | 5.33 | 2.00 | 1942.67 → 3930.44 | 2.02 |
| 16卡3P1D/C96 → 24卡4P2D/C144 | 6.00 | 1.50 | 2582.57 → 4120.98 | 1.60 |

历史比例没有显示明显低于线性的扩容，但旧组使用v0.29、不同数据及容量，不能据此宣布2P1D与4P2D性能等价，也不能直接断言3P1D卡效更差。同24卡的`2×(2P1D)`及普通完整推理服务没有配对实测。

## 3. 第一个有效方向：减小P调度预算

这里始终是16K输入，修改的是每DP引擎`max-num-batched-tokens`。固定C144的各次独立启动结果如下：

| P budget | output tok/s，均值±SD | Mean TTFT，s | 判断 |
| ---: | ---: | ---: | --- |
| 33280 | 3297.50±22.91 | 21.228 | 显著退化 |
| 16640 | 4120.98±8.80 | 9.190 | 初始基线 |
| 9216 | 4157.28±20.28 | 8.449 | 两轮正式有JIT |
| 8704 | 4142.92±52.20 | 8.265 | 未优于8200 |
| 8448 | 4172.20±20.46 | 8.171 | 延迟改善 |
| 8208 | 4212.41±14.88 | 8.093 | 相对8200无明确收益 |
| **8200** | **4229.34±32.83** | **7.926** | 采用的候选 |
| 8196 | 无性能结果 | 无 | 启动warmup的positions对齐错误 |
| 8192 | 4108.76±15.01 | 8.564 | 三轮正式均有JIT匹配 |

小budget改变chunked prefill的分步与排队节奏，是合理解释；**没有证明8200恰好等于8192加K5所需tokens，也没有证明8200与8208的小差别可稳定复现。** 后来在PHB已开启的C160重新测P16640，吞吐4391、TTFT10.119秒，仍不如8200。

仅改8200、尚未开启PHB时，又测C152／156／158／160：吞吐分别4284／4252／4286／4261，TTFT为8.957／9.280／9.889／10.223秒。吞吐已经接近平台，当时更适合选C152，而不是继续提高并发。

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

### 回到24卡验证

**最终判断以24卡端到端为准。** 固定P8200、C144，只给四个P启用PHB，同批各一次启动：

| P通信设置 | output tok/s，均值±SD | Mean TTFT，s | goodput，req/s |
| --- | ---: | ---: | ---: |
| 默认 | 4216.56±23.20 | 7.866±0.108 | 3.305 |
| **PHB** | **4286.50±30.79** | **7.010±0.075** | **3.437** |

吞吐提高1.66%，Mean TTFT降低10.9%。微测的约25%收益没有直接变成整机25%，但降低P阶段延迟后，为更高并发留下了空间。

### 优化后的并发选择

C144、C160、C176为独立启动，推理参数相同；档间差值包含并发、请求量与启动批次变化。保留三档对照，便于判断PHB优化后继续增加并发的收益与代价：

| C | output tok/s，均值±SD | Mean TTFT，s | Mean TPOT，ms | goodput，req/s | 选择依据 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 144 | 4286.50±30.79 | 7.010 | 24.360 | 3.437 | 延迟余量更大 |
| **160** | **4501.50±17.16** | **8.219** | **25.080** | **3.544** | 吞吐提高5.02%，均值SLO仍通过 |
| 176 | 4566.82±5.88 | 10.172 | 25.818 | 3.353 | 吞吐仅再增1.45%，TTFT越线、goodput下降 |

C160三轮吞吐为4509.68／4513.04／4481.77，Mean TTFT为8.123／8.600／7.933秒。在这批独立启动结果中，C176只多1.45%吞吐，但Mean TTFT越过10秒，goodput下降，当时停止更高并发。之后的C176诊断复测约10.15～10.33秒；再后来的同启动完整扫描（见开头）测得9.940秒，三轮有两轮越线，仍未推翻优先选择C160的判断。C161～175尚未逐点测量。

容量不是已经确认的限制：C160的D各引擎running采样峰值36～38/S64、KV峰值约17.4%；C176逐步观测生成batch峰值约41，Graph padding峰值282，未顶到384。P每引擎running采样峰值2/S32。增加名义容量并不能自动提高供给，但有容量余量也不等于有算力余量。

## 5. 哪些方向没有带来进一步收益

| 尝试 | 结果与取舍 |
| --- | --- |
| P的`NCCL_PROTO=Simple` | 四卡C16供给仅变化约0.02%，不采用 |
| EP combine改为补齐后的ReduceScatter | 微测慢约11%；模型功能门槛未通过，未测候选正式吞吐。原生输出自身也有差异，不能直接归罪补丁 |
| 等长ReduceScatter通信分解 | 三个形状仍较慢，停止沿该算子替换方向推进 |
| P启用`FULL_AND_PIECEWISE` | C152吞吐约+0.52%，goodput基本不变，Graph内存增加；当时逐步观测缺失，不认定稳定收益 |
| D侧也启用PHB | 新测C160为4500.88±54.13，对照4483.91±20.00，约+0.38%落在波动内 |

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

C176最初三轮TTFT为9.557／9.671／11.289秒。请求时间线显示，较快的首批D完成会触发闭环客户端提前补发，而P完成节奏没有同比提前，等待因此转移到P侧；不能仅凭TTFT变大断言P计算突然变慢。后续保持双API拓扑增加逐步观测，支持这一解释，但尚未确定D早期速度变化的唯一根因。

另一次连续2640请求得到约5045 tok/s、Mean TTFT7.547秒，是更长窗口摊薄起跑／排空成本，**不是服务优化后突破4500的同协议结果**。本研究仍以固定2C预热＋三轮4C正式请求作主要比较。

## 6. 当前配置与启动要点

<details>
<summary>展开固定环境、关键Bash命令和压测条件</summary>

### 固定环境和必要前提

服务与客户端均使用`vllm/vllm-openai:v0.30.0`，本次本地image ID为：

```bash
IMAGE_ID=sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
docker image inspect "$IMAGE_ID" --format '{{.Id}}'
```

权重是`nvidia/DeepSeek-V4-Flash-0731-NVFP4`，同一tokenizer。该固定镜像实测加载了**DSpark草稿MoE的MXFP4格式分派修复**：保持target NVFP4及FP8 linear不变，将草稿正确分派到MXFP4，并核验三个草稿MoE模块；没有加载旧DP/mHC修复或改写通信算子。原版镜像标签不包含这个加载hook，不能省略此条件后声称复现了本文。

下面是从实测argv整理的关键命令，省去日志采集和实验编排；路径用环境变量替换。它们说明如何重建服务设置，**不是仅凭原版镜像即可一键运行的完整PD系统**：仍需上述加载修复、具备下述语义的PD代理及精确数据。参数化排版只做离线核对，没有另起性能实验。

| 运行位置 | 服务 | GPU | CPU／NUMA | HTTP／NIXL／DP RPC端口 |
| --- | --- | --- | --- | --- |
| 46、47各一套 | P，后四卡 | 4,5,6,7 | 32-47／2 | 31449／29300／29600 |
| 46、47各一套 | P，前四卡 | 0,1,2,3 | 0-15／0 | 31450／29320／29620 |
| 48 | D，后四卡 | 4,5,6,7 | 32-47／2 | 31449／29300／29600 |
| 48 | D，前四卡 | 0,1,2,3 | 0-15／0 | 31450／29320／29620 |

宿主GPU0～3对应NUMA0、GPU4～7对应NUMA2。45的代理绑定CPU16～19／NUMA1，客户端CPU20～23／NUMA1。CPU集合不是独占核；实测保留了既有后台进程，未改功耗或锁频。

### P／D共用启动命令

以下以46后四卡P为例；其他服务按表替换变量。`MODEL_DIR`、`CACHE_DIR`、`PATCH_DIR`分别是权重、该服务持久缓存、已准备的MXFP4加载hook目录；各服务独立缓存目录，保留已有JIT缓存。

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

压测数据每行是`{"id":"...","prompt":"...","input_tokens":16384}`。实测数据SHA256为`6166561f98e83a7591eaa5bc36dcb317f20edd42f187a10f4cc6793df839b9b2`；另一份同长度文本只能复现负载规格，不能当作相同请求集。

以下为固定v0.30客户端的**参数部分**，在同镜像内执行，`/model`、`/dataset.jsonl`、`/results`分别挂载tokenizer、数据和本轮输出目录。实际客户端在调用原生采样器前适配了JSONL逐行读取，并逐条核对tokenizer得到16384 tokens；没有更换HTTP发送及计时实现。未经该适配，不能假定原生`custom`直接接受此JSONL格式。

```bash
C=160
# 完整预热用N=$((2*C))；每轮正式用N=$((4*C))，固定三轮。
N=$((4*C))
vllm bench serve \
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

C160每轮正式应为640/640成功、10,485,760输入tokens、655,360输出tokens；320条完整预热单独计时。每轮使用新的输出目录，服务保持运行，只改客户端C，不重启模型来挑结果。原扫描READY／单窗口／每档／campaign上限为2700／900／4200／54000秒；PHB单点组整组上限8400秒，零性能重试。OOM、请求失败或工作量错误停止取证；C144后的追加扫描在Mean TTFT>10秒**或**Mean TPOT>50毫秒时保留该档并停止更高C。

</details>

## 7. 最终判断

**目前更有价值的是提高P供给效率、减少排队，而不是继续增加D名义容量。** P8200＋P侧PHB将可用工作点从原配置C148的约4168 tok/s推进到C160的约4500 tok/s；这是跨配置、跨并发的工作点改善，不是单变量8%的加速证明。最直接的PHB对照仍是C144的吞吐+1.66%、TTFT下降10.9%。

4P2D是本次已测方案中的优先选择，不是证明16K/1K天然适合2:1配比。若目标改成严格逐请求尾延迟，当前约81%的联合达标率仍不够；下一步需要研究P调度／准入和真实到达分布，而不能用更长计时窗口或更快D来代替这个问题。
