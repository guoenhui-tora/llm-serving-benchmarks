# 节点46：DSpark K值实验——off完成，DP草稿路径阻塞

**本节点 TP2×DP2、EP on 的 off 基线通过 `jit_clean`：737.56 ± 22.34 output tok/s，CV 3.03%。K5 在内存 profiling 阶段触发 DP token 数断言 `8192 80`，未就绪、未进行性能测量。** 固定镜像代码显示该错误来自目标与草稿复用不同长度的 DP 元数据，K1–K4共用此路径，按计划停止相关配置；本轮不能给出 DSpark 增益、最佳 K 或接受长度能否抵消草稿开销的结论。

适用范围：2026-09-20，`gpu-6000d-46`／`10.90.1.46`，GPU4–7，vLLM 0.29.0，指定 NVFP4 主模型及原生 MXFP4 草稿，GovReport固定前128条、总C32、每请求输出1024。没有访问45、47、48，也没有运行八卡双部署。完整逐轮数据见 [CSV](../data/dspark-k-sweep-node46.csv)；任务规则见[项目计划](../README.md#下一轮四节点-dspark-拓扑与-k-值实验)。

## 完成情况与可用结果

| 配置 | 状态 | 正式轮／接纳轮 | 输出 tok/s，均值±样本标准差 | 相对同机off | Mean／P95 TTFT，s | Mean／P95 TPOT，ms | 草稿接受率／平均接受长度 |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| tp2-dp2-epon-off | PASS | 11／3 | 737.56 ± 22.34 | 参照 | 4.995／11.619 | 35.127／37.805 | 不适用 |
| tp2-dp2-epon-k5 | 启动FAIL | 0／0 | — | — | — | — | 未测得 |
| tp2-dp2-epon-k1 | 共性阻塞，未启动 | 0／0 | — | — | — | — | 未测得 |
| tp2-dp2-epon-k2 | 共性阻塞，未启动 | 0／0 | — | — | — | — | 未测得 |
| tp2-dp2-epon-k3 | 共性阻塞，未启动 | 0／0 | — | — | — | — | 未测得 |
| tp2-dp2-epon-k4 | 共性阻塞，未启动 | 0／0 | — | — | — | — | 未测得 |

off累计接纳最先通过的第7、10、11轮，分别为754.67、712.30、745.72 tok/s。其余8轮均因已知事件拒绝，未按性能挑样本。PASS只说明满足本次协议，3.03%的CV不代表稳定性能。本节点没有可计算收益的on样本，不以历史TP4、随机负载或其他节点作分母。

六个核心配置及K6/K7候选均已在运行前完成validate、plan、补丁／日志准备和实际CLI preflight；这只证明配置与CLI检查通过。K1–K4未做GPU启动验证，K6/K7的模型运行支持也未验证。核心矩阵因共性阻塞未完成，因此未触发条件扩展或独立重启复测；没有改prefill预算、精度、后端、Graph模式或gate来绕过错误。授权范围内独立可运行的off已完成，无其他独立配置待跑。

## 全部轮次与统计

每轮128成功、0失败、输入1,042,149／输出131,072 tokens。全部11轮合计 **1,408成功、0失败，输入11,463,639／输出1,441,792 tokens**；其中三轮接纳合计384请求、输入3,126,447／输出393,216 tokens。短中文探测另为17输入／3输出tokens，不混入上述计数。没有正式中断轮，K5在正式轮创建前失败，所以CSV只有11行，不为未测配置伪造“第0轮”或零吞吐记录。

| 轮 | 判定 | 事件行数 | output tok/s | Mean TTFT s | P95 TTFT s | Mean TPOT ms | P95 TPOT ms | 客户端计时 s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 拒绝 | 72 | 383.23 | 13.501 | 36.976 | 66.50 | 93.97 | 342.02 |
| 2 | 拒绝 | 60 | 554.29 | 6.351 | 16.014 | 48.27 | 71.32 | 236.47 |
| 3 | 拒绝 | 30 | 580.53 | 5.795 | 22.123 | 46.14 | 60.79 | 225.78 |
| 4 | 拒绝 | 36 | 581.29 | 5.346 | 12.291 | 46.57 | 61.30 | 225.48 |
| 5 | 拒绝 | 22 | 688.23 | 5.301 | 12.344 | 38.02 | 49.35 | 190.45 |
| 6 | 拒绝 | 18 | 714.01 | 4.605 | 10.722 | 40.32 | 58.78 | 183.57 |
| 7 | 接纳 | 0 | 754.67 | 4.607 | 11.235 | 34.57 | 37.23 | 173.68 |
| 8 | 拒绝 | 4 | 728.91 | 5.113 | 11.952 | 35.12 | 37.54 | 179.82 |
| 9 | 拒绝 | 4 | 752.78 | 4.691 | 11.507 | 34.66 | 37.13 | 174.12 |
| 10 | 接纳 | 0 | 712.30 | 5.673 | 12.390 | 35.78 | 38.55 | 184.01 |
| 11 | 接纳 | 0 | 745.72 | 4.703 | 11.231 | 35.03 | 37.64 | 175.76 |

事件总计246条匹配，不等于246次独立编译。日志中既有`jit_monitor`推理shape标记，也有TileLang实际编译起止；部分内嵌时间早于Docker输出时间，存在缓冲／归属边界。沿用原保守gate，未删除延迟日志或改变已发生轮次的判定。

下表只统计接纳的三轮，每项均为各轮值的均值±样本标准差。P95是**各轮P95的平均**，不是合并请求后的P95。

| 指标 | 均值±样本标准差 |
| --- | ---: |
| 输出吞吐，tok/s | 737.56 ± 22.34 |
| requests/s | 0.7203 ± 0.0218 |
| Mean TTFT，s | 4.995 ± 0.590 |
| P95 TTFT，s | 11.619 ± 0.668 |
| Mean TPOT，ms | 35.127 ± 0.610 |
| P95 TPOT，ms | 37.805 ± 0.679 |
| Mean ITL，ms | 35.127 ± 0.610 |
| P95 ITL，ms | 28.507 ± 0.108 |
| Mean E2EL，s | 40.929 ± 1.200 |
| P95 E2EL，s | 47.448 ± 1.579 |

CSV保留原协议CSV的30个字段及原单位，再追加约定的14个字段，共44列。`accepted`仅表示协议实际接纳，`study_phase=core`、TP=2、DP=2、EP=on；off的K为0，所有草稿字段留空。逐轮requests/s、ITL、E2EL和KV峰值均可直接从CSV复核。

## K5失败证据与停止依据

K5于北京时间05:12:05启动服务，05:15:05四个worker在内存profiling路径报错；run于05:15:14结束。没有就绪后的health/models/短生成，也没有完整负载、已分配KV容量或Graph捕获成功证据。

已确认的加载事实：

- 四个worker均输出`LOCAL_DSPARK_FORMAT_FIX: validated 768 native MXFP4 draft experts`；主模型仍为NVFP4，草稿保留group32 E8M0及FP8线性层配置。
- 主模型选择`FLASHINFER_CUTLASS`，草稿选择`DEEPGEMM_MXFP4`及`DeepGemmFP4Experts`，两者均出现`MoEPrepareAndFinalizeNaiveDPEPModular`路径。
- 每worker模型加载约45.22 GiB。失败堆栈为断言，不是CUDA OOM；不能把它解释为K5显存容量不足。

关键堆栈与固定镜像源码位置：

```text
vllm/v1/worker/gpu/model_runner.py:868  profile_run -> _dummy_run(max_num_tokens=8192)
vllm/v1/worker/gpu/model_runner.py:803  self.speculator.propose(..., dp_sync=dp_sync)
vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:336
    num_query_tokens = num_reqs * self.num_query_per_req
vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:365–373
    _generate_draft(..., num_query_tokens,
                    num_tokens_across_dp=dp_sync.num_tokens_across_dp,
                    cudagraph_runtime_mode=NONE)
vllm/forward_context.py:96
    assert num_tokens_across_dp_cpu[dp_rank] == batchsize
AssertionError: 8192 80
```

DSpark继承DFlash的这条profiling分支；本模型采用anchor预测方式，`num_query_per_req=K`。本次`num_reqs=min(8192,16)=16`，草稿实际80 tokens，却继续传入目标侧8192 tokens的DP同步信息。对于相同配置的K1–K4，草稿分别只有16／32／48／64 tokens，改变K不能修正这处长度不匹配；因此将它归为共性DP／草稿路径故障并停止后续K。此推断有固定镜像代码和K5实测堆栈支撑，但**不把K1–K4称为逐项实测失败**。

失败发生时`cudagraph_runtime_mode=NONE`，早于Graph捕获。扩大捕获列表或改成eager不能据此解决该元数据问题；本轮没有尝试这些改动。既有加载补丁只修草稿量化分派，没有修改该DP路径。最终kernel gate仍为FAIL，缺少后续autotune-disabled和mixed-tokens=16预热日志，未删除required规则取得PASS。

## 绑定、Graph、KV与资源背景

GPU4/5及6/7各为近端PXB对，四卡归NUMA2。服务CPU32–47／内存NUMA2，客户端CPU48–51／内存NUMA3，均仅使用每物理核的一个线程；SMT兄弟分别为96–111和112–115。Docker inspect与off逐轮线程允许集合检查通过；cpuset不独占这些核。服务只暴露GPU4–7，日志worker为DP0/TP0–1/EP0–1和DP1/TP0–1/EP2–3，未跨入GPU0–3。

单个API端口31249下默认两个API worker，DP由引擎内部管理，campaign无`replica_targets`。每rank `max-num-seqs=16`，监控观察到两个rank各达到16个running；客户端只发全服务C32／128请求。prefill 8192是每rank预算，DP聚合可达16384，不能声称它与TP4有完全相同的调度语义。

off的`FULL_DECODE_ONLY`按`[1,2,4,8,12,16]`捕获，日志有最终6个尺寸Graph捕获完成及V2 runner证据。所有on列表来自基本批次分别乘K与K+1后合并；K5 CLI和启动参数已确认，其他K只离线／preflight验证。下表目标上限和草稿上限均为每rank token数。

| DSpark | 捕获列表 | 目标上限 | 草稿上限 |
| --- | --- | ---: | ---: |
| off | `1,2,4,8,12,16` | 16 | — |
| k5 | `5,6,10,12,20,24,40,48,60,72,80,96` | 96 | 80 |
| k1 | `1,2,4,8,12,16,24,32` | 32 | 16 |
| k2 | `2,3,4,6,8,12,16,24,32,36,48` | 48 | 32 |
| k3 | `3,4,6,8,12,16,24,32,36,48,64` | 64 | 48 |
| k4 | `4,5,8,10,16,20,32,40,48,60,64,80` | 80 | 64 |
| k6 | `6,7,12,14,24,28,48,56,72,84,96,112` | 112 | 96 |
| k7 | `7,8,14,16,28,32,56,64,84,96,112,128` | 128 | 112 |

off两个DP rank均记录KV容量 **78,014 tokens**：DP0与DP1各78,014。物理预算日志仅DP0/TP0明确输出`Available KV cache memory: 29.89 GiB`，其余worker没有单独打印该值，不把它推定成每卡相同实测值。token数是DSV4压缩缓存的等效值，不是普通全注意力token容量。每轮KV使用峰值约23.52%–23.59%，取两个rank采样最大值；`num_preemptions_total`两rank全程为0，逐轮增量均为0。K5未完成容量分配，相应字段未测得。

资源观察使用每约1秒`/metrics`快照、5秒CPU/GPU及每约30秒进程、服务NUMA页面快照；系统service cgroup CPU计数每6次资源采样保存。采样耗时会使实际间隔略大于设定值。窗口使用runner记录的客户端阶段起止，含客户端初始化和收尾；不等同于仅客户端吞吐计时窗口，也不能排除采样间隙的瞬时峰值。

| off接纳三轮的资源窗口 | 观察值 |
| --- | --- |
| 整机CPU平均忙碌率 | 13.99% |
| 服务32–47／SMT96–111平均忙碌率 | 38.79%／0.30% |
| 客户端48–51／SMT112–115平均忙碌率 | 22.15%／13.78% |
| GPU4–7温度范围 | 48–62 °C |
| GPU4–7功耗范围／样本均值 | 79.86–338.27 W／250.50 W，每卡样本 |
| GPU4–7 SM频率范围 | 2407–2430 MHz |
| GPU4–7显存占用范围 | 79,355–80,091 MiB |

初始及全部97次进程快照均有12个既有`doca_spcx_cc`忙进程，每个累计CPU占用约一个逻辑核；system.slice服务的CPU增量中，`roce-init.service`约消耗6.00个逻辑核，其他单服务均低于0.07个逻辑核，该service集合不覆盖全部DOCA进程。未终止这些后台任务，未发现新增矿工等明显异常。上述CPU组负载包含宿主后台竞争，不能解释为本次服务独占消耗。未改变宿主功耗、锁频或NUMA设置；频率／温度采样不能证明完全无干扰。

第7轮中的一次`numa_maps`快照显示四个worker已驻留页面按映射页大小折算后，NUMA2占94.34%–94.43%，其余主要在NUMA0；共享库／文件页也纳入该统计，不能据此等同于匿名页远端比例。允许内存集合不证明全部页面本地化；没有逐轮测量客户端实际页面分布。

保留的warning包括SM120 SymmMem不可用、world_size=2的FlashInfer All Reduce禁用、NVFP4实验格式、FP4 indexer旧字段弃用、FP8尺度相关提示、TileLang向量化回退、共享内存广播等待及DP coordinator out-of-order统计。off最终保存572条warning匹配，K5为24条，计数包含重复行而非独立问题。K5结束时另有shared_memory清理warning；宿主GPU显存最终归零，本次容器均已删除，未清理其他任务或缓存。

## 接受统计口径与缺失说明

固定镜像`vllm/v1/spec_decode/metrics.py`定义了以下counter，普通标签为`model_name="deepseek-v4-flash"`与`engine="0"/"1"`；逐位置另有`position`标签：

```text
vllm:spec_decode_num_draft_tokens_total
vllm:spec_decode_num_accepted_tokens_total
vllm:spec_decode_num_drafts_total
vllm:spec_decode_num_accepted_tokens_per_pos_total
```

计划按每轮前后快照逐rank求增量，再汇总唯一rank的分子／分母：接受率=`100×ΣΔaccepted/ΣΔdraft_tokens`；平均接受长度=`1+ΣΔaccepted/ΣΔnum_drafts`，**包含目标模型补充token**。`num_drafts`按request-level draft observation累计，不是把整个batch只计一次；逐位置为接受token次数，比例的分母应是相应`ΣΔnum_drafts`。计数重置、标签缺失或窗口不清楚时不计算，不重复累计TP worker或重复导出。

本次off没有草稿，K5在endpoint就绪前退出，无可关联的正式接受计数窗口。因此CSV接受率、平均接受长度、草稿token、验证次数及逐位置数组均留空，不能从K或日志片段推算。K5观察脚本保留了endpoint未就绪的采样记录；没有草稿ITL结果。未来DSpark的ITL仅表示流式事件间隔，不能直接解释为逐token计算时间，接受率也不代替质量评测。

## 协议成本与身份

固定`jit_clean`：每轮完整128请求、累计最先3轮clean、最多12轮／2700秒；单轮超时900秒，启动上限1800秒；无额外安静预热或JIT重启。六小时后不再新增配置、最多两次条件扩展的外层预算未触发，因共性故障提前停止。

| run_id | 北京时间起止 | 启动至ready | 协议耗时 | run总耗时 | 结果 |
| --- | --- | ---: | ---: | ---: | --- |
| core-off-01 | 04:20:55–05:10:05 | 322.17 s | 2576.11 s（42.94 min） | 2949.67 s（49.16 min） | 11轮／3接纳，PASS |
| core-k5-01 | 05:11:21–05:15:14 | 未ready；服务启动到run结束约189.32 s | 未开始 | 233.38 s（3.89 min） | 启动FAIL |

run含内部preflight、服务启动、探测、协议和清理，不含前置配置／预检、缓存扫描与观察脚本最后收尾。启动时间取runner的wait-ready计时；K5单列服务启动到结束近似耗时，不冒充成功ready耗时。两次run耗时合计53.05分钟，不将累计耗时重复加到CSV每一行。

Git首先由`a764cce`经`git pull --ff-only`快进到 **`397e9d42dedee00c181c5fb10f5686ace069c111`**，`git merge-base --is-ancestor 397e9d4 HEAD`成功；初始工作区干净。执行期间未pull、未改源码／配置，两次结束的冻结文件核对均为`changed_files=[]`。两次runner源码指纹同为`f6e873002b8cf61963bd221d63b0fe0a520e0bc6118f4ca63acfa54fc6726d52`。

服务／客户端image ID和RepoDigest均为`sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`，tag为`vllm/vllm-openai:v0.29.0`；驱动580.159.04。模型为`/data/models/DeepSeek-V4-Flash-0731-NVFP4`，模型身份`1e1b7e54442bcf7f15ef55192641b5c5da360bcaa608236aa0d06b273aedf8b4`；48个分片大小已核对，未全量哈希权重内容。

| 身份文件 | SHA256 |
| --- | --- |
| config.json | `bb0d2286d6761439e41d3cef31d16489411b816ed8688922f59730bbd5567cdb` |
| model.safetensors.index.json | `5d2ad3076e04081d6c0728cb4b004dc832850ec5ae732f3adb06cf87c4b437a5` |
| tokenizer_config.json | `6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547` |
| tokenizer.json | `8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf` |
| GovReport JSONL | `33bdbd36a8b335300974e1b07aa66b2c24c88cc4c4b55a21350aa3564a945d53` |

K5使用仓库[本地加载补丁](../patches/dspark-native-mxfp4/manifest.json)，版本随上述commit固定：原模块SHA256为`56f6b81f5712817689e24088b2c5302d5832fd4e6d5b2630d7e28be73c84b298`；覆盖`utils.py`为`e0c060632a5b2a9ce7016edb7e10ac666e3a58464c0216803b9318a083f1b902`；辅助函数为`33df649eec0abfe5350fafc18dc84bcbd492238e65b9652a13fcea9e3bdf4730`。没有修改镜像和原权重。

缓存起点按recipe隔离：off的`ec8114a830aa6f67c655`目录没有文件；K5的`f84953fe7f5a50099c7f`仅有prepare产生的4个覆盖文件、9079 bytes。两者不是复制同一热缓存；未删除旧缓存，运行产生的JIT工件继续保留。这解释了本次成本包含新recipe的冷缓存路径，但不能将全部波动唯一归因于缓存。

## 复现入口与实际命令

公共依赖仍使用[四卡EP on服务参数来源](../configs/recipes/dual-tp2-dp2-epon.yaml)、[DSpark K5来源](../configs/recipes/tp4-dspark-k5.yaml)、[单case结构](../configs/campaigns/dspark-k5-quick.yaml)、[rear target](../configs/targets/rear.yaml)及[GovReport jit_clean](../configs/workloads/govreport-c32-jit-clean.yaml)。数据来源、许可证和构造见[GovReport说明](../data/govreport-near8k.md)，补丁准备见[兼容性说明](dspark-compatibility.md#后续实验如何使用)。

在指定commit的新工作区复制项目configs和JSONL后，target改为本机46；每个campaign只保留一个rear target、一个case、上述jit_clean workload。off采用`dual-tp2-dp2-epon.yaml`的一套四卡服务参数；on保留这些拓扑参数，合并K5来源的environment、checks和speculative-config，仅按上表改变K与Graph列表。`prepare.py`仅对on campaign执行；所有准备在run之前完成。K6/K7只准备、不自动运行。探索配置不归档为已验证候选。

客户端协议：固定镜像的官方vLLM benchmark，`/v1/completions`流式，JSONL顺序前128条；`num-prompts=128`、`max-concurrency=32`、`request-rate=inf`、`num-warmups=0`、seed0、temperature0、ignore_eos；输出1024、关闭chat模板、shuffle和oversample。镜像默认`ready-check-timeout-sec=0`，没有每轮附加的initial test请求。tokenizer在客户端计时前复核长度。off启动功能探测关闭thinking，返回“2。”；探测不属于正式轮。

以下两段**直接由实测`argv.json`导出**，只添加换行，保留实际容器名、缓存和参数；它们是本次启动快照，重放性能测量应使用bench的新run-root管理生命周期。K5这条命令复现的是已知失败，不代表可用性能方案。

<details>
<summary>off实际Docker服务启动命令</summary>

```bash
docker run -d \
  --pull never \
  --name sb-9c0b978fbd3c4f64-server \
  --label io.serving-bench.run=9c0b978fbd3c4f64 \
  --network host \
  --ipc host \
  --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/vllm/ec8114a830aa6f67c655:/root/.cache:rw \
  --cpuset-cpus 32-47 \
  --cpuset-mems 2 \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --cap-add SYS_NICE -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 \
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
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,12,16],"max_cudagraph_capture_size":16}' \
  --seed 0 \
  --distributed-executor-backend mp \
  --data-parallel-size-local 2
```

</details>

<details>
<summary>K5实际Docker服务启动命令</summary>

```bash
docker run -d \
  --pull never \
  --name sb-f36b53c57db94245-server \
  --label io.serving-bench.run=f36b53c57db94245 \
  --network host \
  --ipc host \
  --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/vllm/f84953fe7f5a50099c7f:/root/.cache:rw \
  --cpuset-cpus 32-47 \
  --cpuset-mems 2 \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --cap-add SYS_NICE -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 -e PYTHONPATH=/root/.cache/dspark-native-mxfp4 \
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

本次实际bench入口如下，均由本地`observe_run.py`围绕未修改的runner顺序调用；复跑必须换新的run-root，不能覆盖这些目录：

```bash
./bench run experiments/dspark-k-sweep-node46/configs/campaigns/tp2-dp2-epon-off.yaml \
  --run-root experiments/dspark-k-sweep-node46/results/core-off-01
./bench run experiments/dspark-k-sweep-node46/configs/campaigns/tp2-dp2-epon-k5.yaml \
  --run-root experiments/dspark-k-sweep-node46/results/core-k5-01
```

原始位置 **仅本机可用**：`experiments/dspark-k-sweep-node46/`。其中`configs/`保留全部候选和依赖；`results/core-off-01/`、`results/core-k5-01/`保留plan、解析配置、完整命令、镜像／模型、probe、全部轮次JSON和完整日志；`reports/preparation/`保存各候选validate/plan/preflight。各run的`reports/<run_id>/`保存缓存起点、冻结指纹、`metrics-snapshots.jsonl.gz`、`resource-telemetry.jsonl`、`processes.jsonl`及执行状态；固定镜像DP断言代码在`reports/dp-draft-failure-source.txt`，指标源码快照和本地分析／观察脚本也保留。原始目录不随Git分发。

本次只归档此报告和完整CSV，并更新README中46的完成／阻塞标记。未更新公共基线、其他节点或总表，未push。
