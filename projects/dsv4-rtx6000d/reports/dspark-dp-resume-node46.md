# 节点46：DP＋DSpark补丁后续测（2026-09-20）

**已按顺序执行四卡TP2×DP2、EP on的K5 → K3 → K4；K5 PASS，K3/K4 PARTIAL，本节点任务部分完成。** 适用范围为固定vLLM 0.29.0＋两处加载补丁、GPU4–7、GovReport近8K输入／1024输出、全服务总C32。K5最先三轮clean为第8/9/10轮，输出吞吐 **966.00 ± 15.92 tok/s，CV 1.65%**；相对本机旧off的737.56 ± 22.34 tok/s高 **30.97%**。参照来自不同启动批次，不能解释为独立重启验证或长期稳定收益。

K3用尽12轮，仅第10轮clean（958.26 tok/s）；K4用尽12轮，0轮clean。两组均非启动失败，但没有正式三轮结果，**不能据此判定K3/K4劣于K5或宣布最优K**。按任务停止规则，不追加K2/K1、不重启补满、不补跑off；两项扩展只完成离线准备、补丁校验及CLI预检。

本次共 **34轮、4,352成功、0失败，输入35,433,066／输出4,456,448 tokens**。其中K5的3轮为正式样本，K3另有1轮clean诊断样本，30轮有已知编译观察事件。所有原始轮次保留；[本轮44列CSV](../data/dspark-dp-resume-node46.csv)只记录新批次，不复制历史off。

## off参照与结果边界

旧off来自[上轮46报告](dspark-k-sweep-node46.md)和[原CSV](../data/dspark-k-sweep-node46.csv)，2026-09-20北京时间04:20–05:10，Git `397e9d42dedee00c181c5fb10f5686ace069c111`；本轮为同日12:06–14:23，Git `e8c1da1ce0aafe3331a7ff1dc68cad4980642796`。旧off的第7/10/11轮为三轮clean，吞吐CV 3.03%。两批runner源码指纹相同，模型元数据／tokenizer／分片大小身份、GPU UUID／驱动、服务和客户端镜像完全一致。

实际配置核对：两批TP2、DP2、PP1、EP on、每rank容量16／prefill8192、总C32、同一JSONL前128条、FP8 KV、显存比例0.90、V2＋async、关闭prefix cache和autotune、CPU/内存绑定均一致。on新增DSpark、两处补丁及必需的Graph覆盖；草稿占用额外显存，KV容量变化是部署差异的一部分。DOCA数量与system service CPU增量接近，客户端SMT兄弟核竞争仍有波动，见资源节。因此保留旧off作为有限参照，不把小幅差异当可靠提升，也不把约31%的差异全部归因于单一kernel。

| 配置 / 判定 | clean轮 | 输出 tok/s，均值±样本SD | 吞吐CV | 相对旧off | 加权接受率 | 平均接受草稿长度 / 含补充token |
| --- | --- | --- | --- | --- | --- | --- |
| 旧off / PASS | 7,10,11 | 737.56 ± 22.34 | 3.03% | 参照 | — | — |
| K5 / PASS | 8,9,10 | 966.00 ± 15.92 | 1.65% | +30.97% | 49.196% | 2.4598 / 3.4598 |
| K3 / PARTIAL | 10（仅诊断） | 958.26（n=1） | — | 不计算正式收益 | 65.139% | 1.9542 / 2.9542 |
| K4 / PARTIAL | 无 | — | — | 不计算 | 无clean统计 | 无clean统计 |

K5相对旧off的Mean TTFT下降38.31%、Mean TPOT下降18.76%，但P95 TPOT上升8.54%，不是所有延迟全面改善。下表是各轮指标的均值±样本标准差；**P95重复均值不是合并请求P95**。DSpark的ITL是流式事件间隔，不等于逐token计算耗时。K3的单轮值仅供诊断，不进入正式组均值。

| 指标 | 旧off，n=3 | K5，n=3 | K3，第10轮 |
| --- | --- | --- | --- |
| 请求吞吐 requests/s | 0.7203 ± 0.0218 | 0.9434 ± 0.0155 | 0.9358（n=1） |
| Mean TTFT s | 4.995 ± 0.590 | 3.081 ± 0.207 | 2.854（n=1） |
| P95 TTFT s | 11.619 ± 0.668 | 11.103 ± 0.312 | 10.873（n=1） |
| Mean TPOT ms | 35.127 ± 0.610 | 28.538 ± 0.202 | 29.671（n=1） |
| P95 TPOT ms | 37.805 ± 0.679 | 41.033 ± 0.847 | 40.298（n=1） |
| Mean ITL ms | 35.127 ± 0.610 | 98.581 ± 1.100 | 87.564（n=1） |
| P95 ITL ms | 28.507 ± 0.108 | 630.279 ± 0.264 | 619.092（n=1） |
| Mean端到端 s | 40.929 ± 1.200 | 32.275 ± 0.363 | 33.208（n=1） |
| P95端到端 s | 47.448 ± 1.579 | 45.571 ± 1.607 | 46.341（n=1） |
| 逐轮接受率 % | — | 49.197 ± 0.337 | 65.139（n=1） |
| 逐轮接受长度（含补充token） | — | 3.4599 ± 0.0169 | 2.9542（n=1） |

## 全部轮次、成本与JIT证据

下表按时间顺序列出全部轮次。每行均128成功、0失败，输入1,042,149／输出131,072 tokens；`clean累计`不因后续事件清零。K3的clean诊断样本在CSV中保留`accepted=true`，同时`protocol_status=case_status=PARTIAL`，不能仅按accepted字段建立正式组。完整逐次requests/s、TTFT/TPOT/ITL/端到端Mean和P95、接受counter及KV峰值见CSV。

| 配置 | 轮 | 事件行 | clean累计 | 输出 tok/s | Mean TTFT s | Mean TPOT ms | 接受率 % | 客户端计时 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K5 | 1 | 136 | 0 | 375.61 | 11.245 | 72.274 | 49.178 | 348.96 |
| K5 | 2 | 38 | 0 | 624.28 | 4.634 | 45.165 | 49.870 | 209.96 |
| K5 | 3 | 26 | 0 | 711.24 | 4.429 | 39.409 | 49.907 | 184.29 |
| K5 | 4 | 18 | 0 | 829.29 | 3.052 | 34.349 | 49.156 | 158.05 |
| K5 | 5 | 12 | 0 | 967.49 | 3.109 | 28.576 | 49.198 | 135.48 |
| K5 | 6 | 2 | 0 | 871.96 | 3.594 | 31.514 | 50.004 | 150.32 |
| K5 | 7 | 10 | 0 | 965.32 | 3.326 | 28.229 | 49.453 | 135.78 |
| K5 | 8 | 0 | 1 | 964.23 | 3.132 | 28.756 | 49.362 | 135.93 |
| K5 | 9 | 0 | 2 | 982.73 | 2.853 | 28.359 | 48.809 | 133.38 |
| K5 | 10 | 0 | 3 | 951.03 | 3.258 | 28.499 | 49.420 | 137.82 |
| K3 | 1 | 134 | 0 | 369.43 | 11.276 | 73.475 | 65.793 | 354.80 |
| K3 | 2 | 16 | 0 | 792.80 | 3.656 | 35.224 | 65.994 | 165.33 |
| K3 | 3 | 2 | 0 | 857.93 | 3.597 | 32.270 | 66.672 | 152.78 |
| K3 | 4 | 20 | 0 | 813.83 | 3.743 | 34.177 | 67.447 | 161.06 |
| K3 | 5 | 2 | 0 | 943.21 | 3.063 | 29.207 | 65.440 | 138.96 |
| K3 | 6 | 8 | 0 | 800.54 | 3.757 | 34.797 | 65.602 | 163.73 |
| K3 | 7 | 18 | 0 | 804.85 | 3.662 | 34.765 | 66.182 | 162.85 |
| K3 | 8 | 14 | 0 | 792.80 | 3.463 | 35.392 | 65.474 | 165.33 |
| K3 | 9 | 10 | 0 | 948.39 | 3.261 | 28.805 | 66.921 | 138.20 |
| K3 | 10 | 0 | 1 | 958.26 | 2.854 | 29.671 | 65.139 | 136.78 |
| K3 | 11 | 4 | 1 | 958.58 | 2.720 | 29.775 | 65.445 | 136.74 |
| K3 | 12 | 8 | 1 | 797.27 | 4.042 | 34.640 | 66.651 | 164.40 |
| K4 | 1 | 152 | 0 | 372.31 | 8.476 | 75.211 | 56.236 | 352.05 |
| K4 | 2 | 36 | 0 | 711.09 | 3.434 | 40.583 | 56.980 | 184.32 |
| K4 | 3 | 16 | 0 | 818.41 | 3.527 | 34.291 | 57.385 | 160.16 |
| K4 | 4 | 4 | 0 | 969.71 | 2.904 | 28.734 | 57.905 | 135.17 |
| K4 | 5 | 2 | 0 | 867.84 | 3.375 | 32.020 | 56.226 | 151.03 |
| K4 | 6 | 12 | 0 | 832.97 | 3.048 | 34.303 | 57.177 | 157.35 |
| K4 | 7 | 8 | 0 | 985.61 | 3.035 | 28.387 | 57.336 | 132.99 |
| K4 | 8 | 6 | 0 | 983.04 | 2.781 | 28.726 | 57.147 | 133.33 |
| K4 | 9 | 6 | 0 | 963.85 | 3.307 | 28.667 | 56.733 | 135.99 |
| K4 | 10 | 4 | 0 | 970.67 | 3.392 | 28.229 | 57.167 | 135.03 |
| K4 | 11 | 2 | 0 | 981.31 | 2.771 | 28.673 | 57.670 | 133.57 |
| K4 | 12 | 20 | 0 | 836.25 | 3.384 | 33.888 | 57.643 | 156.74 |

以下全部轮次统计**包含事件样本，仅描述诊断过程和波动**，不能代替上述clean性能结果或用来排名：

| 配置 | 全部轮数 | 输出 tok/s，均值±SD | CV | Mean TTFT s | Mean TPOT ms |
| --- | --- | --- | --- | --- | --- |
| K5 | 10 | 824.32 ± 199.32 | 24.18% | 4.263 ± 2.523 | 36.513 ± 13.801 |
| K3 | 12 | 819.82 ± 158.71 | 19.36% | 4.091 ± 2.297 | 36.017 ± 12.076 |
| K4 | 12 | 857.76 ± 176.78 | 20.61% | 3.619 ± 1.552 | 35.143 ± 13.168 |

同配置只启动一次，无独立预热；累计最先3轮clean，最多12轮／2700秒，单轮超时900秒，启动上限1800秒。K3/K4均因`round_budget`结束，未触发时间中断。下表run总耗时包含内部preflight、启动、功能探测、协议及清理，不含前置prepare／外部preflight和观察脚本收尾；协议成本包含拒绝轮及客户端初始化。

| run_id | 北京时间起止 | 启动至ready s | 协议 s / min | run总 s / min | 判定 |
| --- | --- | --- | --- | --- | --- |
| dpfix-tp2-dp2-epon-k5-01 | 12:06:05–12:47:28 | 437.71 | 1990.25 / 33.17 | 2482.93 / 41.38 | PASS |
| dpfix-tp2-dp2-epon-k3-01 | 12:49:02–13:36:11 | 427.74 | 2351.64 / 39.19 | 2828.68 / 47.14 | PARTIAL |
| dpfix-tp2-dp2-epon-k4-01 | 13:37:40–14:23:35 | 423.65 | 2277.45 / 37.96 | 2755.91 / 45.93 | PARTIAL |

三个run总耗时合计134.46分钟，首个run开始至最后run结束墙钟137.51分钟。没有把累计耗时重复写入CSV每行；CSV的duration_s为该轮官方客户端计时。

事件总数分别为K5 242、K3 236、K4 268，共746条匹配，**不是独立编译次数**。较早轮有明确TileLang begins/completes记录；部分后期轮只有`jit_monitor`标记，没有同窗口编译起止证据。例如K3第11轮4条是DP1两个TP worker的`mhc_pre_big_fuse_*`标记，runtime shape为1804；K4第11轮2条是DP0两个TP worker、runtime shape为2261。不能把这些标记全部解释为真实CPU重编译，也没有证据在本次直接改判为误报。K3第12轮和K4第12轮仍分别有2/4条真实编译开始及对应结束记录。保留原规则、反例窗口和warning，不放宽gate、不清缓存、不重启追求PASS。

## 启动路径、容量与资源背景

三组health/models、关闭thinking的简短中文生成均通过；主模型NVFP4 `FLASHINFER_CUTLASS`，草稿原生MXFP4 `DEEPGEMM_MXFP4`，attention为`FLASHINFER_MLA_SPARSE_DSV4`及`fp8_ds_mla`路径。每组四个worker均记录`LOCAL_DSPARK_FORMAT_FIX`及`UPSTREAM_DSPARK_DP_PROFILE_FIX: PR #54856 facd9a74a1`；目标／草稿Graph捕获完成，V2、autotune-disabled、mixed-tokens=16证据齐全，最终kernel required检查均PASS。日志未出现旧`8192 80`断言、OOM或traceback；这些是可观察的功能／路径证据，不代表完整数值质量验证。

一个API端口31249，由引擎内部DP调度，默认两个API worker；campaign无`replica_targets`。四个worker为DP0/TP0–1/EP0–1、DP1/TP0–1/EP2–3；全服务C32／128请求，不按rank翻倍。各rank活动峰值均16，prefill8192为每rank预算。

| 配置 | KV容量 DP0 / DP1 tokens | KV峰值 DP0 / DP1 % | 活动峰值 DP0 / DP1 | 抢占增量，全部轮次 |
| --- | --- | --- | --- | --- |
| K5 | 64,839 / 64,839 | 28.411 / 28.453 | 16 / 16 | 0 |
| K3 | 65,099 / 65,099 | 28.395 / 28.343 | 16 / 16 | 0 |
| K4 | 64,912 / 64,912 | 28.406 / 28.447 | 16 / 16 | 0 |

CSV容量取两rank最小值、使用率取各rank采样峰值中的最大值，抢占为两rank无重置counter增量之和。off容量为每rank78,014 tokens；本模型压缩缓存的等效token容量不等于普通全注意力KV容量。实际物理可用KV日志在DP0/TP0分别为K5 24.84、K3 24.94、K4 24.87 GiB；其余worker没有单独打印，不推定为每卡完全一致。

GPU4/5、6/7为近端PXB对，位于NUMA2；服务CPU32–47／内存NUMA2，客户端CPU48–51／内存NUMA3，SMT兄弟分别96–111和112–115。服务与每轮客户端的Docker inspect、线程允许集合检查通过。末轮中的worker NUMA页面快照约94.61%–94.68%驻留在NUMA2，其余主要在NUMA0；统计包含共享库和文件页，不等同匿名页远端比例。客户端页面分布未逐轮量化，cpuset不独占这些核。

资源采样约5秒，进程、NUMA页面与本次容器CPU约30–33秒，system service cgroup每6次资源采样一次；均保留实际时间。下表K5取三轮clean窗口，K3取唯一clean窗口，K4取全部12轮诊断窗口，**窗口不同，不能直接据此排名资源效率**。窗口含客户端初始化及收尾。

| 资源指标 | K5 clean 8/9/10 | K3 clean 10（n=1） | K4全部12轮 |
| --- | --- | --- | --- |
| 整机CPU忙碌率 | 13.75% | 13.69% | 13.59% |
| 服务CPU组 | 36.77% | 34.69% | 32.87% |
| 服务SMT兄弟组 | 0.18% | 0.17% | 0.18% |
| 客户端CPU组 | 25.58% | 26.30% | 22.78% |
| 客户端SMT兄弟组 | 17.72% | 0.20% | 0.24% |
| GPU温度 °C | 48.00–61.00 | 48.00–61.00 | 41.00–62.00 |
| GPU功耗 W | 79.86–329.75 | 79.86–326.73 | 77.86–333.94 |
| GPU SM频率 MHz | 2407.00–2430.00 | 2407.00–2430.00 | 2302.00–2430.00 |
| GPU显存 MiB | 77717.00–78321.00 | 77495.00–78355.00 | 75675.00–79813.00 |

全部进程快照的DOCA数量均为12，既有`roce-init.service`平均约6.00逻辑核，其余已采样service低于0.07核；该service集合不覆盖全部DOCA进程。未发现新增矿工等明显异常，未终止后台任务。旧off clean窗口整机13.99%、服务CPU组38.79%、客户端组22.15%、客户端SMT组13.78%；本轮客户端SMT波动说明后台竞争未消除。上述CPU组包含宿主任务，不能都归因于服务或客户端。本次容器CPU采样另外保存，K5 clean窗口服务约491.60%、客户端21.77%（Docker口径100%=一个逻辑核，包含初始化／空闲且采样稀疏），不能与CPU组忙碌率混用。

K4所有轮次中最低SM采样频率2302 MHz，K5/K3 clean窗口为2407–2430 MHz；没有据此按比例补偿性能，低频采样也不足以排除瞬时干扰。未修改宿主功耗、锁频、SMT或NUMA设置。

保留SM120 SymmMem不支持、TP2 FlashInfer All Reduce不可用、NVFP4实验格式、FP4 indexer弃用／FP8尺度、TileLang向量化回退、共享内存等待、API多worker统计提示及DP coordinator out-of-order等warning。最终warning匹配K5/K3/K4分别230/246/242条，含重复行；没有隐藏这些提示取得PASS。

## 接受计数与采样边界

按固定镜像`vllm/v1/spec_decode/metrics.py`的定义，`num_drafts`按request-level draft observation计数，不是整个batch一次。每约1秒保存`/metrics`原文及请求／完成时间，按唯一`engine="0"/"1"`读取后求增量，再累加分子分母，不重复计算TP worker。逐位置counter另按`position`保留。全部34轮每rank的逐位置增量之和等于accepted增量，draft_tokens = K × drafts；全时间序列未见counter下降或重置。

接受率=`100×ΣΔaccepted/ΣΔdraft_tokens`；平均接受草稿长度=`ΣΔaccepted/ΣΔdrafts`。为沿用现有44列CSV约定，`spec_decode_acceptance_length`存**1＋该长度**并设置`acceptance_length_includes_bonus=true`；摘要表同时给出不含／含目标补充token两种值。跨轮摘要使用先累加counter再求比值的加权统计；逐轮比值的均值±SD另列，二者不混称。

边界取每轮客户端初始化阶段、正式请求开始前的空闲快照，以及全部请求完成后的空闲快照。逐轮必须同时满足成功counter增量128、输出counter增量131,072；窗口／rank／重置无法核实时应留空，本次34轮均验证通过。首轮功能探测计数会延迟导出，因此未直接使用protocol started_at前快照，而在探测计数到齐后、正式请求前取基线；K5明确剔除2次草稿、1个接受草稿token的探测增量。完整rank值、窗口时间与偏移保存在本机`round-counter-summary.json`。采样约1秒，末尾计数导出与真实完成存在延迟；KV峰值仅是采样峰值，不能排除间隙更高值。

## 固定身份、缓存与复现入口

先`git pull`从`cd9ee4f`快进至`e8c1da1ce0aafe3331a7ff1dc68cad4980642796`，确认HEAD包含`e8c1da1`及DP修复`d529c37`，初始Git工作区干净。运行中未pull或修改源码／配置；三次收尾冻结校验均`changed_files=[]`。仅使用本机`gpu-6000d-46`／`10.90.1.46`，未SSH其他节点。

| 身份 | 值 |
| --- | --- |
| 运行Git | `e8c1da1ce0aafe3331a7ff1dc68cad4980642796` |
| runner源码指纹（与旧off相同） | `f6e873002b8cf61963bd221d63b0fe0a520e0bc6118f4ca63acfa54fc6726d52` |
| 补丁manifest SHA256 | `2e30906bd35789d8e7cd5940f8e28d5b03eb10aff9a014814a23f780b9a5f368` |
| 服务／客户端image ID和RepoDigest SHA256 | `c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1` |
| 模型身份 | `1e1b7e54442bcf7f15ef55192641b5c5da360bcaa608236aa0d06b273aedf8b4` |
| GovReport JSONL SHA256 | `33bdbd36a8b335300974e1b07aa66b2c24c88cc4c4b55a21350aa3564a945d53` |

服务／客户端tag均为`vllm/vllm-openai:v0.29.0`，驱动580.159.04；权重路径`/data/models/DeepSeek-V4-Flash-0731-NVFP4`。模型身份包含config、tokenizer及48个分片大小，未全量重算权重内容哈希。补丁文件哈希和上游来源见[manifest](../patches/dspark-native-mxfp4/manifest.json)及[兼容性说明](dspark-compatibility.md#dpdspark回移已合并的上游修复)；旧off不加载补丁，旧失败K5不参与本次统计。

每个K使用recipe身份独立缓存；起点均只有prepare生成的5个补丁文件、12,622 bytes，未复制旧kernel缓存，各K不共用可写缓存。因此耗时包含该recipe冷缓存编译，不能作为热缓存独立重启成本。旧缓存及本次新增缓存全部保留。

| 配置 | 缓存目录 |
| --- | --- |
| K5 | `/home/enhui/.cache/serving-bench/vllm/4ace1d8b4976188770f5` |
| K3 | `/home/enhui/.cache/serving-bench/vllm/211df71640be417983b3` |
| K4 | `/home/enhui/.cache/serving-bench/vllm/3391e7346c86de0e136b` |

配置由[公共DP K5 recipe](../configs/recipes/tp2-dp2-dspark-k5.yaml)和[单case正式campaign](../configs/campaigns/dspark-dp-clean.yaml)复制至现有工作区；保留公共model/runtime/client及[正式GovReport workload](../configs/workloads/govreport-c32-jit-clean.yaml)，target改成本机46。K5/K3/K4及条件扩展K2/K1均在第一次启动前完成validate、plan、补丁prepare／check、日志准备及CLI preflight；探索文件只在工作区，不重复归档为新基线。

三个实测配置的EP均on；从K5改K时仅修改recipe id/说明、num_speculative_tokens及Graph尺寸。以下差异表直接从各次实测resolved.json导出，其他服务参数见主命令：

| 配置 | EP | K | Graph尺寸 | Graph上限 |
| --- | --- | --- | --- | --- |
| K5 | on | 5 | 5,6,10,12,20,24,40,48,60,72,80,96 | 96 |
| K3 | on | 3 | 3,4,6,8,12,16,24,32,36,48,64 | 64 |
| K4 | on | 4 | 4,5,8,10,16,20,32,40,48,60,64,80 | 80 |

客户端固定流式`/v1/completions`，JSONL顺序前128条（平均输入8141.789 tokens），每请求输出1024，总C32、request_rate=inf、seed0、temperature0、ignore_eos=true、num_warmups=0；不套chat模板、不shuffle、不oversample。实际tokenizer在每轮计时前复核输入长度，无额外initial test请求。客户端CPU48–51／NUMA3；协议与预算见上文，启动命令本身不能定义性能实验。

以下主命令**由K5实测argv.json导出**，仅增加换行，保留当次容器名、固定image ID、环境及缓存路径。作为当次快照，重新测量应由bench使用新的run-root管理生命周期；K3/K4使用各自表中K/Graph及缓存目录，不共用K5可写缓存。

<details>
<summary>K5实际Docker服务命令</summary>

```bash
docker run -d \
  --pull never \
  --name sb-1f9b0dbdca6948c1-server \
  --label io.serving-bench.run=1f9b0dbdca6948c1 \
  --network host \
  --ipc host \
  --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/vllm/4ace1d8b4976188770f5:/root/.cache:rw \
  --cpuset-cpus 32-47 \
  --cpuset-mems 2 \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --cap-add SYS_NICE -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 -e PYTHONPATH=/root/.cache/dspark-native-mxfp4 -e SERVING_BENCH_DSPARK_DP_PROFILE_FIX=1 \
  --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model \
  --served-model-name deepseek-v4-flash \
  --host 127.0.0.1 \
  --port 31249 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --no-enable-prefix-caching \
  --enable-expert-parallel \
  --enable-chunked-prefill \
  --jit-monitor-verbose \
  --async-scheduling \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 2 \
  --max-model-len 16384 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 256 \
  --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' \
  --moe-backend auto \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[5,6,10,12,20,24,40,48,60,72,80,96],"max_cudagraph_capture_size":96}' \
  --seed 0 \
  --distributed-executor-backend mp \
  --data-parallel-size-local 2 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":false}'
```

</details>

本次实际bench入口由本地只读观察脚本顺序调用：

```bash
./bench run experiments/dspark-k-sweep-node46/configs/campaigns/tp2-dp2-epon-k5.yaml \
  --run-root experiments/dspark-k-sweep-node46/results/dpfix-tp2-dp2-epon-k5-01
./bench run experiments/dspark-k-sweep-node46/configs/campaigns/tp2-dp2-epon-k3.yaml \
  --run-root experiments/dspark-k-sweep-node46/results/dpfix-tp2-dp2-epon-k3-01
./bench run experiments/dspark-k-sweep-node46/configs/campaigns/tp2-dp2-epon-k4.yaml \
  --run-root experiments/dspark-k-sweep-node46/results/dpfix-tp2-dp2-epon-k4-01
```

复现准备顺序为validate → plan → `prepare.py CAMPAIGN` → `prepare.py CAMPAIGN --check` → `scripts/prepare_logging.py CAMPAIGN` → preflight → run，完整规范见[本轮任务说明](../README.md#下一轮dpdspark-补丁后续测)。不得复用已有run-root；仅pull代码不会自动准备当前recipe缓存中的补丁。

## 原始产物与交付

原始目录**仅节点46本机可用，不随Git分发**：`experiments/dspark-k-sweep-node46/`。原`core-off-01`、`core-k5-01`及上轮报告/CSV均保留；新批次为：

```text
results/dpfix-tp2-dp2-epon-k5-01/
results/dpfix-tp2-dp2-epon-k3-01/
results/dpfix-tp2-dp2-epon-k4-01/
reports/dpfix-preparation/                 # Git/资源起点、各候选plan与prepare/preflight
reports/dpfix-tp2-dp2-epon-k*-01/          # metrics原文、CPU/GPU/cgroup、进程/NUMA、缓存起点、冻结校验
reports/dpfix-summary.json                # 本地完整汇总
```

每个results目录保存解析配置、命令、模型／镜像、host、probe、protocol、全部轮次JSON、server-window／compilation和完整server日志；reports对应目录还保存逐rankcounter增量及实际窗口。历史观察脚本未覆盖，本批使用`observe_dpfix.py`；离线`analyze_dpfix.py`、`write_dpfix_report.py`及验证脚本留在工作区。

结束后本次容器均已清理，八卡显存0 MiB，端口31249空闲；模型、镜像、历史结果和缓存未删除。只归档此报告、44列CSV和README中46的“部分完成”进度行，不修改其他节点或公共总表。归档检查从CSV重算、核对解析配置和实际命令、验证相对链接，并执行`git diff --check`；未修改公共代码或归档配置，因此不重跑性能实验或通用单元测试。仅commit，不push。
