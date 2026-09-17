# 节点46：八卡整机部署对照（2026-09-17–18）

## 结论与完成范围

**在本轮固定八卡、8192输入／1024输出、保留既有DOCA背景的条件下，C64的最高已测均值来自双TP2×DP2、EP on：1620.37 ± 17.99 tok/s，比本机双TP4高13.45%。** C32双TP4为1096.87 ± 1.37 tok/s；两种DP2配置的C32波动较大，不能据此确认稳定收益。双TP1×DP4、EP on的C32低于参照，C64的3.77% CV也限制了其结论。

节点46已按分工顺序执行全部5个配置：双TP4、DP2 EP off/on、DP4 EP off/on。4个配置完成C32/C64各三次，共**24次有效正式测量**；DP4 EP off在C32第一次重复的五轮预热内未达到连续两轮安静，停止该候选，**没有正式样本，C64未运行**。没有为凑满计划30次而扩大矩阵、改参数或重启补测。

[逐次CSV](../data/node8-node46.csv)由[统一导出脚本](../../../scripts/export_samples.py)生成，含24行`scope=node`及48行副本指标；整机合计**4608成功、0失败，输入37,748,736、输出4,718,592 tokens**。上述总量不重复计算副本行，不包含预热及2次JIT拒绝尝试。资源逐窗口审核后标为`accepted-background`，表示接受已约定DOCA背景，不代表硬件完全隔离或性能已收敛。

只操作本机`gpu-6000d-46 / 10.90.1.46`，未访问其他节点。本轮运行起止时间为北京时间**09月17日 20:16:53至09月18日 00:36:53，约4.33小时**，未到8小时进度检查点。每个配置只启动一对服务，全部并发与重复在该次启动内执行；失败候选清理完成后才启动下一独立候选。

## 运行条件与实际参数

入口为[46-dsv4-node8.yaml](../configs/campaigns/46-dsv4-node8.yaml)，执行预算与[项目统一协议](../README.md#下一轮八卡整机部署对照)一致。工作区配置从项目configs复制，逐字核对一致；运行期间未修改公共源码、recipe或workload，也未改变日志gate。

| 项目 | 本次固定条件 |
| --- | --- |
| GPU / 驱动 | 8×NVIDIA RTX6000D，单卡85,651 MiB，SM120；驱动580.159.04；功耗上限600 W/卡，未修改功耗或锁频 |
| 服务 / 客户端 | vLLM 0.29.0；`vllm/vllm-openai:v0.29.0`；image ID `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1` |
| 权重 / tokenizer | `/data/models/DeepSeek-V4-Flash-0731-NVFP4`；各run模型身份相同：`1e1b7e54442bcf7f15ef55192641b5c5da360bcaa608236aa0d06b273aedf8b4`；检查元数据/tokenizer哈希及48个分片大小，未全量重算权重内容哈希 |
| 执行Git | `aacdc39f0862b82527753ce3ac5cbf0de94c1c68`；开始前工作树干净，本地差异已保存 |
| runner源码指纹 | `edccf507e1246a4525a383ff4c4d3ce45a470378075bf3271922903c10020e34`，五个run一致 |
| 模型参数 | 上下文16384、FP8 E4M3 KV、显存比例0.90自动分配、block size256、FP4 indexer关闭 |
| 执行与后端 | V2 model runner、async scheduling、mp executor、chunked prefill；`FLASHINFER_MLA_SPARSE_DSV4`、MoE auto实际为`FLASHINFER_CUTLASS`、NVFP4专家；保留`FULL_DECODE_ONLY` |
| 关闭项 | FlashInfer autotune、prefix cache、投机解码、CPU offload；不启用vLLM `--numa-bind` |
| 全局负载 | C32：128请求，预热64请求/轮；C64：256请求，预热128请求/轮；各三次，seed0、temperature0、ignore_eos、range_ratio0、request_rate=inf |
| 测量协议 | 固定vLLM客户端、流式`/v1/completions`；全局seed0数据集按索引交错均分；相同官方计时边界同步起跑；预热2轮连续安静、最多5轮，正式最多2次尝试；启动1800秒、客户端阶段7200秒 |

服务端和客户端镜像ID、实际CLI、权重文件及本机绑定均通过preflight；无需额外自定义日志文件。每侧均通过health/models及关闭thinking的简短中文探测。每个C32正式样本核对128成功、0失败、输入1,048,576、输出131,072 tokens；C64为256成功、0失败、输入2,097,152、输出262,144 tokens，且逐请求长度及每侧数量均正确。

| 资源 | 前侧独立部署 | 后侧独立部署 |
| --- | --- | --- |
| GPU | 0–3 | 4–7 |
| 服务CPU / 内存NUMA | 0–15 / 0 | 32–47 / 2 |
| 客户端CPU / 内存NUMA | 16–19 / 1 | 48–51 / 3 |
| API端口 | 31248 | 31249 |

服务共32物理核、客户端共8物理核，每核只使用一个SMT线程，兄弟编号为对应CPU加64。Docker cpuset及容器内线程允许集合在规定阶段核验通过。cpuset不会排除后台任务；允许的内存集合不证明所有页面均本地驻留，本轮未连续记录实际页面驻留分布。

| 配置 | 每部署TP/PP/DP，EP | 每rank容量 / Graph捕获尺寸 | 整机prefill上限 | KV预算 GiB/卡 | 日志KV tokens/DP rank |
| --- | --- | --- | ---: | ---: | ---: |
| 双TP4 | 4/1/1，off | 32 / 1,2,4,8,12,16,24,32 | 16384 | 32.73 | 85450 |
| 双TP2×DP2 | 2/1/2，off | 16 / 1,2,4,8,12,16 | 32768 | 30.07 | 78506 |
| 双TP2×DP2 | 2/1/2，on | 16 / 1,2,4,8,12,16 | 32768 | 29.89 | 78014 |
| 双TP1×DP4 | 1/1/4，off（预热失败） | 8 / 1,2,4,8 | 65536 | 24.71 | 64515 |
| 双TP1×DP4 | 1/1/4，on | 8 / 1,2,4,8 | 65536 | 24.76 | 64632 |

整机活动容量均为64，每DP rank的prefill预算8192。Graph捕获完成，未用eager替代。KV取自真实启动日志，两侧预算一致；自动显存比例相同不代表KV容量相同，这些逻辑token容量也不等于跨拓扑相同的物理KV存储。

实际GPU进程映射中，双TP4的两侧各有TP0–TP3；DP2每侧的前两卡为DP0/TP0–1、后两卡为DP1/TP0–1；DP4每侧四卡依次为DP0–DP3，每个DP组的TP大小为1、TP rank为0。EP on时每侧四卡形成EP0–EP3，两套服务的DP/EP组独立。DP候选显式使用`data-parallel-size-local=2|4`，未配置外部DP rank；API worker分别为每侧2个或4个，端口及实际GPU进程映射已保存。

DP日志为`AgRsAll2AllManager`与`MoEPrepareAndFinalizeNaiveDPEPModular`；EP off明确回退到AllGather+ReduceScatter dispatch/combine。async下日志提示禁用NCCL的DP同步，这不表示MoE通信也禁用NCCL。双TP4日志显示超过两张PCIe GPU时custom allreduce禁用。这里记录选中路径，未通过profiler分解通信与计算开销，不能将吞吐差异全部归因于EP或通信。

## 性能表

每行n=3。吞吐为三次均值±样本标准差；相对值只比较本节点同并发双TP4。各样本整机吞吐按**全部成功输出tokens / 两侧官方计时窗口并集**计算，不相加两侧自身窗口吞吐。

| 配置 | 整机并发 | 输出 tok/s，均值±标准差 | CV | 相对双TP4 |
| --- | --- | ---: | ---: | ---: |
| 双 TP4 | C32 | 1096.87 ± 1.37 | 0.12% | +0.00% |
| 双 TP4 | C64 | 1428.31 ± 7.21 | 0.51% | +0.00% |
| 双 TP2×DP2，EP off | C32 | 1026.81 ± 59.56 | 5.80% | -6.39% |
| 双 TP2×DP2，EP off | C64 | 1491.49 ± 43.22 | 2.90% | +4.42% |
| 双 TP2×DP2，EP on | C32 | 1078.36 ± 108.98 | 10.11% | -1.69% |
| 双 TP2×DP2，EP on | C64 | 1620.37 ± 17.99 | 1.11% | +13.45% |
| 双 TP1×DP4，EP on | C32 | 956.43 ± 5.31 | 0.55% | -12.80% |
| 双 TP1×DP4，EP on | C64 | 1487.73 ± 56.02 | 3.77% | +4.16% |

延迟统一为ms，表中每个值为三次相应指标的均值；各次整机P95先合并两侧请求或ITL流式事件计算，**三次P95均值不是三次请求合并后的P95**。其余mean/P95指标及每侧指标保存在CSV，缺失字段留空。

| 配置 | 并发 | Mean TTFT | P95 TTFT | Mean TPOT | P95 TPOT | P95 ITL | P95 E2EL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 双 TP4 | C32 | 4705.26 | 10739.86 | 24.40 | 27.67 | 18.36 | 36497.63 |
| 双 TP4 | C64 | 6303.40 | 20629.88 | 38.51 | 42.22 | 22.91 | 62154.46 |
| 双 TP2×DP2，EP off | C32 | 5444.99 | 10659.28 | 25.76 | 30.46 | 23.14 | 37526.08 |
| 双 TP2×DP2，EP off | C64 | 6287.88 | 15681.22 | 33.00 | 36.33 | 25.62 | 50235.69 |
| 双 TP2×DP2，EP on | C32 | 4184.80 | 8691.11 | 24.40 | 30.64 | 24.03 | 37936.31 |
| 双 TP2×DP2，EP on | C64 | 5017.26 | 12305.66 | 31.16 | 33.74 | 26.12 | 45120.74 |
| 双 TP1×DP4，EP on | C32 | 4691.15 | 6775.08 | 28.21 | 31.70 | 28.81 | 35973.20 |
| 双 TP1×DP4，EP on | C64 | 7604.17 | 21527.69 | 30.83 | 37.03 | 27.57 | 51429.49 |

C64下DP2 EP on三次均高于双TP4三次，吞吐均值提高13.45%，mean TTFT/TPOT与P95 E2EL均降低；P95 ITL却高于参照，不能称为全部延迟指标都改善。C32下DP2 EP on均值接近参照但CV为10.11%，不据此推荐其替换稳定baseline。DP4 EP off没有性能结论。

## 异常与限制

### 预热阻塞、拒绝尝试与warning

DP4 EP off五轮C32预热的已知事件数为**12、2、2、0、2**，第五轮前侧DP3的`_compute_global_topk_indices_and_lens_kernel`仍出现两条Triton JIT monitor事件，未达到连续两轮安静。runner报`Warmup did not reach the required consecutive quiet rounds`并停止配置；服务功能及完整kernel日志gate通过，不等于正式测量通过，也不能将此失败写成模型不支持或OOM。C64和全部6次正式测量均未运行。未延长上限、重启该候选或修改识别规则。

两次正式尝试被整组拒绝：双TP4的C64/r3/attempt1，后侧24条已知事件；DP4 EP on的C32/r3/attempt1，前侧DP0/EP0一条TileLang JIT monitor事件。均在原上限内重新同步预热，attempt2通过。部分TileLang行的内嵌时间早于Docker接收时间，同时存在当前JIT monitor警告；保留原始双时间证据，不据此放宽gate或把每条匹配都解释为独立CPU重编译。24个有效样本的测量期均无已知编译事件。

下表为每次重复的预热轮数；`a→b`表示拒绝第一次测量后，为第二次尝试重新预热b轮。所有有效测量前末两轮双方均安静，每次尝试均未超过五轮。

| 配置 | C32 r1 / r2 / r3 | C64 r1 / r2 / r3 |
| --- | --- | --- |
| 双 TP4 | 3 / 2 / 2 | 3 / 2 / 2→2 |
| 双 TP2×DP2，EP off | 3 / 2 / 2 | 4 / 3 / 2 |
| 双 TP2×DP2，EP on | 3 / 2 / 2 | 2 / 2 / 2 |
| 双 TP1×DP4，EP on | 5 / 2 / 4→2 | 3 / 2 / 2 |

保留SM120 SymmMem不可用、FlashInfer All Reduce的world size限制、PCIe TP4 custom allreduce禁用、TileLang向量化回退、NVFP4格式与FP8 scaling提示，以及DP coordinator的`Received stats for out-of-order step` warning。未发现测量期OOM、请求失败或抢占日志；已有DP指标快照的抢占计数为0。warning存在不改变原日志gate，也不等于已证明具体性能原因。

### 波动诊断与资源资格

超过3% CV的组全部保留原样：DP2 EP off/C32三次为958.07、1059.26、1063.09 tok/s；DP2 EP on/C32为1051.77、1198.17、985.14；DP4 EP on/C64为1424.40、1507.98、1530.81。已交叉检查JIT、错误/抢占日志、CPU集合与后台cgroup、客户端、GPU频率以及两侧窗口，没有确定单一原因，未删慢样本或增加补测。

DP2 EP on/C32第三次前侧约105.28秒、后侧133.05秒，空闲尾段约27.77秒；DP4 EP on/C64前两次的慢侧交换，第一轮前/后169.45/184.04秒，第二轮173.84/168.50秒。所有尾段均计入整机吞吐。内部分派/调度、通信和背景线程分布的影响尚不能拆分；本轮固定等分请求，不是动态负载均衡测试。

启动前八卡空闲，整机CPU忙碌率约12.7%–14.8%；已有16个`doca_spcx_cc`忙进程按约定保留，采样合计约16个逻辑CPU。后侧NUMA的低MemFree主要为文件缓存，全机MemAvailable约956 GiB，压力均值为0，未清缓存或放宽绑定。测量期间使用runner约5秒CPU/GPU遥测、约30秒system service cgroup计数，并额外约30秒保存进程CPU增量、允许集合和压力数据；逐次按同宿主monotonic官方计时窗口交叉核对。

| 配置 | 六个窗口CPU均值范围 | 窗口内CPU采样最大值 | GPU最高°C | 测量采样最低SM MHz | 最大两侧尾差 秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 双 TP4 | 20.63–20.97% | 21.87% | 61 | 2415 | 1.87 |
| 双 TP2×DP2，EP off | 22.20–22.44% | 23.42% | 61 | 2415 | 9.99 |
| 双 TP2×DP2，EP on | 21.37–22.70% | 23.37% | 62 | 2407 | 27.77 |
| 双 TP1×DP4，EP on | 24.69–25.68% | 26.48% | 63 | 2407 | 14.59 |

24个有效窗口中DOCA约15.98–16.01个逻辑CPU，相关`roce-init.service`约6核；其他采样system service无异常高CPU增量，未见矿工恢复证据。DOCA会动态占用服务、客户端或SMT兄弟，不能将CPU集合占用全部归因于本次容器；也不能按CPU比例修正吞吐。低频遥测未见持续热降频，不能排除短时干扰。所有样本经人工逐窗口审核后赋予`accepted-background`，这一资源资格与runner PASS、吞吐CV分别报告。

### 复现、产物与清理

使用公共campaign重建一个**新的**工作区，依项目README依次validate/plan、日志准备、preflight，再按本节点case顺序运行；恢复时只选择需要的新任务并使用新的`--run-root`，不得把新批次与本轮三次混合。完整复现协议见[实验方法](../../../docs/benchmark-methodology.md)及[同步多副本配置](../../../docs/configuration.md#同步多副本部署)。本次运行命令均为：

```bash
./bench run experiments/dsv4-node8-node46/configs/campaigns/46-dsv4-node8.yaml \
  --case CASE_ID --run-root experiments/dsv4-node8-node46/results/RUN_NAME
```

| CASE_ID | 本次RUN_NAME | 状态 |
| --- | --- | --- |
| dual-tp4 | baseline-01 | PASS，6次 |
| dual-tp2-dp2-epoff | dp2-epoff-01 | PASS，6次 |
| dual-tp2-dp2-epon | dp2-epon-01 | PASS，6次 |
| dual-tp1-dp4-epoff | dp4-epoff-01 | FAIL，预热上限，0次 |
| dual-tp1-dp4-epon | dp4-epon-01 | PASS，6次 |

**原始证据仅节点46本机可用**：仓库下`experiments/dsv4-node8-node46/results/`保存上述run-root，包含解析配置、实际命令、image/model身份、完整服务日志、逐请求记录、预热、拒绝尝试和遥测；`reports/`保存启动前/收尾快照、监督日志、逐次资源审核及导出草稿。这些路径不是随Git分发的链接。

归档前已重新通过公共campaign的离线validate/plan，确认工作区配置与公共configs逐字一致，并复核CSV唯一性、资源审核对应关系、请求/token总量、共同计时窗口、全部性能表及相对链接。

五次运行均完成本次所属容器清理，无cleanup错误；收尾Docker运行列表和GPU计算进程为空。保留权重、固定镜像、全部JIT缓存与历史产物。仅归档本报告、逐次CSV并更新节点46完成标记；未改其他节点结果、未提交公共代码或配置变更，未push。
