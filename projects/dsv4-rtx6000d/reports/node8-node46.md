# 节点46：八卡整机部署对照（2026-09-17–18）

## 结论与完成范围

**在本轮固定八卡、8192输入／1024输出、保留既有DOCA背景的条件下，C64的最高已测均值来自双TP2×DP2、EP on：1620.37 ± 17.99 tok/s，比本机双TP4高13.45%。** C32双TP4为1096.87 ± 1.37 tok/s；两种DP2配置的C32波动较大，不能据此确认稳定收益。双TP1×DP4、EP on的C32低于参照，C64的3.77% CV也限制了其结论。

补测DP4 EP off为C32 **948.75 ± 36.60 tok/s（CV 3.86%）**、C64 **1500.34 ± 59.23 tok/s（CV 3.95%）**。两组均有波动，且使用独立批次及更长预热；本次解决了该启动内的JIT覆盖阻塞，不代表消除了性能波动或所有形状首次使用。

首批按分工顺序执行全部5个配置，4个配置完成24次有效正式测量；DP4 EP off因五轮预热未连续安静而停止，首批没有正式样本、C64未运行。09月18日按用户追加授权，先诊断JIT机制，再为该候选增加本地计时外校准，在**一次独立启动**内补齐C32/C64各三次。现在5个配置合计**30次有效正式测量**；旧失败、拒绝证据及原24个样本保留，没有重跑其他配置。

[逐次CSV](../data/node8-node46.csv)由[统一导出脚本](../../../scripts/export_samples.py)生成，现含30行`scope=node`及60行副本指标；整机合计**5760成功、0失败，输入47,185,920、输出5,898,240 tokens**。其中`configuration=dual-tp1-dp4-epoff`的18行只来自新批次`dp4-epoff-02`，其余72行保持原样。上述总量不重复计算副本行，不包含预热、补测校准及首批2次JIT拒绝尝试。资源逐窗口审核后标为`accepted-background`，表示接受已约定DOCA背景，不代表硬件完全隔离或性能已收敛。

只操作本机`gpu-6000d-46 / 10.90.1.46`，未访问其他节点。首批运行起止时间为北京时间**09月17日 20:16:53至09月18日 00:36:53，约4.33小时**，未到8小时进度检查点。首批每个配置只启动一对服务，全部并发与重复在该次启动内执行；失败候选清理完成后才启动下一独立候选。追加批次的启动、预热差异与结果见文末。

## 首批运行条件与实际参数

入口为[46-dsv4-node8.yaml](https://github.com/guoenhui-tora/llm-serving-benchmarks/blob/4642443b1274251b6a47f0d2742e0e59e2dccc95/projects/dsv4-rtx6000d/configs/campaigns/46-dsv4-node8.yaml)，首批执行预算与[项目统一协议](../README.md#2026-09-18八卡整机部署结果)一致。首批工作区配置从项目configs复制，逐字核对一致；运行期间未修改公共源码、recipe或workload，也未改变日志gate。

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
| 双TP1×DP4 | 1/1/4，off（首批预热失败） | 8 / 1,2,4,8 | 65536 | 24.71 | 64515 |
| 双TP1×DP4 | 1/1/4，on | 8 / 1,2,4,8 | 65536 | 24.76 | 64632 |

整机活动容量均为64，每DP rank的prefill预算8192。Graph捕获完成，未用eager替代。KV取自真实启动日志，两侧预算一致；自动显存比例相同不代表KV容量相同，这些逻辑token容量也不等于跨拓扑相同的物理KV存储。

实际GPU进程映射中，双TP4的两侧各有TP0–TP3；DP2每侧的前两卡为DP0/TP0–1、后两卡为DP1/TP0–1；DP4每侧四卡依次为DP0–DP3，每个DP组的TP大小为1、TP rank为0。EP on时每侧四卡形成EP0–EP3，两套服务的DP/EP组独立。DP候选显式使用`data-parallel-size-local=2|4`，未配置外部DP rank；API worker分别为每侧2个或4个，端口及实际GPU进程映射已保存。

DP日志为`AgRsAll2AllManager`与`MoEPrepareAndFinalizeNaiveDPEPModular`；EP off明确回退到AllGather+ReduceScatter dispatch/combine。async下日志提示禁用NCCL的DP同步，这不表示MoE通信也禁用NCCL。双TP4日志显示超过两张PCIe GPU时custom allreduce禁用。这里记录选中路径，未通过profiler分解通信与计算开销，不能将吞吐差异全部归因于EP或通信。

## 首批性能表（24次）

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

C64下DP2 EP on三次均高于双TP4三次，吞吐均值提高13.45%，mean TTFT/TPOT与P95 E2EL均降低；P95 ITL却高于参照，不能称为全部延迟指标都改善。C32下DP2 EP on均值接近参照但CV为10.11%，不据此推荐其替换稳定baseline。DP4 EP off首批没有性能结论，独立补测结果见文末。

## 异常与限制

### 预热阻塞、拒绝尝试与warning

首批DP4 EP off五轮C32预热的已知事件数为**12、2、2、0、2**，第五轮前侧DP3的`_compute_global_topk_indices_and_lens_kernel`仍出现两条Triton JIT monitor事件，未达到连续两轮安静。runner报`Warmup did not reach the required consecutive quiet rounds`并停止配置；服务功能及完整kernel日志gate通过，不等于正式测量通过，也不能将此失败写成模型不支持或OOM。C64和全部6次正式测量均未运行。未延长上限、重启该候选或修改识别规则。

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

首批五次运行均完成本次所属容器清理，无cleanup错误；收尾Docker运行列表和GPU计算进程为空。保留权重、固定镜像、全部JIT缓存与历史产物。仅归档本报告、逐次CSV并更新节点46完成标记；未改其他节点结果、未提交公共代码或配置变更，未push。

## 2026-09-18：DP4 EP off诊断与独立补测

本次只补`dual-tp1-dp4-epoff`，新run-root为`dp4-epoff-02`，只启动一对服务，旧`dp4-epoff-01`及全部拒绝证据保留。**采用新增计时外校准预热解决已定位的首次使用覆盖问题，没有修改公共runner、gate、正式recipe、服务参数或正式测量协议。** 原来的24个有效样本不重跑、不替换。本批与首批具有不同预热历史，不能宣称只有拓扑不同的完全等条件比较。

### 诊断假设与验证

| 假设 | 证据与验证结果 |
| --- | --- |
| 原阻塞只是延迟日志 | 不成立。旧TileLang内嵌15:16:14/25的编译起止行到15:19:59才被Docker接收，确有约225/214秒延迟；但原第五轮Triton事件的内嵌15:26:36与Docker接收时间一致，不能将其整体归为延迟输出。以上均为UTC。 |
| 每轮都在无限生成新top-k形状 | 原完整日志显示，两种top-k specialization分别在两侧八个worker各出现一次：第一轮覆盖5个worker，第二轮前侧DP0，第三轮后侧DP1，第五轮前侧DP3。它们是`topk=512/block_size=64`和`topk=128/block_size=2`，而非五轮都在同一worker反复生成新形状。 |
| JIT monitor warning必然代表CPU重编译 | 固定镜像源码否定这一等价关系：Triton `_do_compile`在`compile`返回后调用post hook，而`compile`可能从磁盘metadata直接返回`CompiledKernel`。TileLang wrapper也在进程内`_kernel_cache`缺失时、磁盘查找前报警。保留warning，不由此取消gate。 |
| 混合prefill/decode的指针对齐特化未覆盖 | 本批IR实证支持。启动时已有4份top-k IR；第一轮校准新增2份，其中`token_to_req_indices_ptr`保留16字节对齐属性，`is_valid_token_ptr`没有该属性。FlashInfer sparse prefill源码按`num_decode_tokens`切片两者，int32与bool指针因此可呈不同对齐。前侧新增IR的mtime对应18:02:06首个worker事件；其余worker稍后报警时48个相关缓存文件的路径及mtime均未改变，支持后续为进程首次加载同一磁盘产物。 |
| 增加持续补入请求能覆盖遗漏路径 | 本地校准C32将每阶段请求由64增至256，预计每worker请求由8增至32，使同一次负载包含更多补入/退出和混合批次。首轮即在全部八个worker覆盖上述两种top-k变体，共16条事件；随后未再出现top-k事件。不能据此声称覆盖所有可能调度形状。 |
| TileLang运行形状变化必然新编译 | 本批中文探测先在前侧DP2、后侧DP0触发17-token调用；校准第二轮前侧DP3/后侧DP1触发22/18-token调用，C64首轮另四个worker触发22/29-token调用，均为相同`mhc_pre_big_fuse_with_norm_tilelang`缓存键。校准后检查持久TileLang缓存50个文件均早于本次启动，且无配套编译起止行；结合源码，更支持逐进程首次缓存加载，不能将不同runtime shape都称为独立CPU重编译。所有事件仍计入原gate。 |

另一个缓存边界已实际检查：本镜像Triton使用容器内`/root/.triton/cache`，原runner只持久挂载`/root/.cache`；该挂载内有TileLang、FlashInfer和vLLM缓存，没有此次top-k Triton产物。因此保留宿主缓存不等于重启后保留所有Triton编译产物，也不等于每个worker进程内缓存已就绪。本次未调整挂载或环境变量，而是保持同次服务启动完成预热和测量，并将容器内Triton缓存另存本地证据。若要统一完善持久化或逐worker定向kernel warmup，需另行协调公共方案，本次不实施。

### 本地预热差异与预算

新增本地campaign `configs/campaigns/46-dsv4-node8-dp4off-followup.yaml`，仅含该case，顺序为新增校准workload、原`node8-c32.yaml`、原`node8-c64.yaml`。新增`configs/workloads/node8-dp4off-mixed-coverage-c32.yaml`由C32复制：只将id改为`dsv4-node8-dp4off-mixed-coverage-c32`、purpose改为calibration、每轮预热及检查请求数改为256、repetitions与max_attempts改为1；连续安静2轮、最多5轮、timeout7200秒不变。输入/输出、并发、seed和采样完全不变。这里的configs路径均相对于本机原工作区，不是随Git分发的配置链接。

校准预算上限为5×256预热请求＋1×256检查请求＝1536；实际四轮预热事件数为**16、2、0、0**，随后一次检查0事件，共**1280成功、0失败，输入10,485,760、输出1,310,720 tokens**。每侧和逐请求长度均正确，全部排除于正式CSV。新增校准未放宽任何事件识别规则。

正式负载保持原参数：C32每次128请求、每轮预热64；C64每次256请求、每轮预热128；各三次，连续两轮安静、最多五轮预热、最多两次测量尝试。公共recipe与两份正式workload逐字段等于旧解析计划，运行期间源码和配置冻结。执行Git为`42b9514`，runner源码指纹仍为`edccf507e1246a4525a383ff4c4d3ce45a470378075bf3271922903c10020e34`，固定image/model身份、绑定、正常Graph和通信路径一致。两侧实际KV仍为24.71 GiB/卡、64,515 tokens/DP rank。

### 补测结果、资源与清理

新批次运行时间为北京时间**09月18日01:55:50–03:10:15，约74.41分钟**。校准后C32三次的预热事件数均为`0、0`；C64第一次为`4、0、0`，第二、三次均为`0、0`。六次正式测量均在attempt1通过，**没有正式拒绝尝试**，完整kernel gate为PASS，warning保留1335条。导出的3行校准指标（整机1行、副本2行）已从正式CSV中过滤，失败case的部分样本没有进入CSV。

本批正式样本合计**1152成功、0失败，输入9,437,184、输出1,179,648 tokens**。吞吐为三次均值±样本标准差；延迟为三次对应指标的均值，单位ms。相对首批双TP4的变化仅作本节点描述性参照，预热历史与运行时间不同。

| 并发 | 三次输出 tok/s | 均值±标准差 tok/s | CV | 相对首批双TP4 |
| --- | --- | ---: | ---: | ---: |
| C32 | 968.84、970.89、906.50 | 948.75 ± 36.60 | 3.86% | -13.50% |
| C64 | 1433.91、1519.49、1547.62 | 1500.34 ± 59.23 | 3.95% | +5.04% |

| 并发 | Mean TTFT | P95 TTFT | Mean TPOT | P95 TPOT | P95 ITL | P95 E2EL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C32 | 4910.43 | 9309.76 | 28.48 | 31.66 | 28.13 | 37976.85 |
| C64 | 7645.52 | 29875.55 | 30.19 | 36.30 | 26.89 | 56018.54 |

**C32与C64的CV均超过3%，已诊断并保留全部样本。** C32第三次前/后侧窗口为135.44/144.59秒，尾差9.15秒；C64第一、二、三次为182.82/164.71、172.52/165.00、163.00/169.38秒，最大尾差18.11秒，慢侧会交换。所有尾段均计入整机吞吐。测量日志无已知JIT、OOM、ERROR或抢占；GPU最低采样SM为2407 MHz、最高温度65°C，未见持续热降频。仅凭低频采样不能排除瞬时影响。

六个测量窗口的整机CPU均值为24.66%–25.49%，单次采样最高26.34%；DOCA为15.99–16.01个逻辑CPU，`roce-init.service`约6核，其他system service未见异常增量，无矿工恢复证据。服务/客户端CPU集合含动态分布的背景线程，不能等同于容器占用；慢样本的整机背景与频率未出现足以定位原因的变化。六次双侧DP coordinator乱序stats警告数分别为26/25、28/18、15/16、61/70、62/66、61/62，均保留，未证明其与波动的因果关系。内部分派、通信与背景竞争仍未分解，不增加补测、不删除慢样本。逐请求与资源窗口人工复核后，六次均标为`accepted-background`，不等同于无干扰或性能收敛。

复现追加批次时，应在原公共配置的工作区副本内按上文差异创建校准workload及单case campaign，先validate/plan、日志检查与preflight，再使用**新的**run-root。本次实际入口为：

```bash
./bench run experiments/dsv4-node8-node46/configs/campaigns/46-dsv4-node8-dp4off-followup.yaml \
  --case dual-tp1-dp4-epoff \
  --run-root experiments/dsv4-node8-node46/results/dp4-epoff-02
```

**本机原始证据位置**：原工作区`results/dp4-epoff-02/`保存完整新批次；`reports/dp4-epoff-diagnosis-02/`保存启动前诊断计划、旧事件汇总、固定镜像源码片段、原配置哈希及preflight；`reports/dp4-epoff-02/`保存校准期间缓存IR/时间戳、双侧Triton缓存副本、背景监控、逐测量资源审核、原始21行导出CSV和最终清理快照。这些仅在节点46本机可用。两侧Triton缓存各785个文件，在首次top-k覆盖后至最后一次正式测量前内容一致；原持久缓存及历史产物均保留，未清空或改挂载。

最终两份完整服务日志、绑定和逐请求记录通过核对；本次run及case为PASS，无cleanup错误，Docker运行列表与GPU计算进程均为空。归档只追加完整新批次的18行正式数据、更新本报告和节点46标记。未访问其他节点，未修改公共源码、gate或configs，未push。
