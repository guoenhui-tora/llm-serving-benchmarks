# DeepSeek V4 Flash NVFP4 在 RTX6000D 上的推理优化

本项目使用 NVIDIA 发布的 [nvidia/DeepSeek-V4-Flash-0731-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4)，目标是在 RTX6000D 上优化推理吞吐与延迟，当前以单机为主。权重为 NVFP4 routed experts 与高精度其他部分的混合格式；实验保持同一权重与 tokenizer。

**当前保留 vLLM 0.29.0、FlashInfer autotune off；单机八卡以双 TP4 作为公共基线。** 在8192输入／1024输出下，四节点整机C32约1091–1097、C64约1428–1434 output tok/s，各组三次重复的吞吐CV均低于1%。这是本轮已接受后台负载下的基线，不代表长期无干扰性能。

追求更高吞吐，优先继续研究**双 TP2×PP2**和**双 TP2×DP2＋EP**：C64分别为1663.93 ± 74.85、1620.37 ± 17.99 tok/s，较各自本机双TP4提高16.50%／13.45%。前者CV为4.50%，后者1.11%；两者跨节点、均值只差2.69%，尚未分出可靠胜负。所有结论限定本轮普通推理，不直接代表PD分离或投机解码收益。

## 当前可复用入口

报告和CSV保存研究结论；configs维护下面的普通推理入口，以及DSpark协议与负载对照所需的少量公共配置，不再为历史矩阵或每个节点复制配置。

| 入口 | 用途 |
| --- | --- |
| [tp4.yaml](configs/campaigns/tp4.yaml) | GPU4–7独占TP4，C32 |
| [dual-tp4.yaml](configs/campaigns/dual-tp4.yaml) | 八卡双TP4公共基线，整机C32/C64 |
| [dual-candidates.yaml](configs/campaigns/dual-candidates.yaml) | 双TP2×PP2、双TP2×DP2 EP on，后续重点复核 |
| [dspark-dp-clean.yaml](configs/campaigns/dspark-dp-clean.yaml) | 四卡TP2×DP2、EP on、K5，含DP修复的正式测量入口 |

单TP4与双TP4共用[同一份服务recipe](configs/recipes/tp4.yaml)。三个target只表达整机、前四卡、后四卡；当前地址为48，其他节点在experiments中修改本机地址并核实映射，不另归档四份相同配置。完整启动命令、CPU绑定、负载和历史入口见[基线与复现](reports/reproduction.md)。

上述普通推理入口的workload显式采用 `quick`（1轮2C预热＋3轮正式测量，保留事件标记），这组协议与配置的组合仅做离线验证；历史拓扑结果仍使用当时的预热和验收规则。DSpark的完整负载预热与 `jit_clean` 已实测，见下节。探索配置、全部原始结果留在experiments；只有选定基线和将继续研究的配置进入projects。

### DSpark 对照数据与性能结果

[GovReport](https://gov-report-data.github.io/) 是英文政府报告的长文档摘要数据集，原始数据见 [Hugging Face](https://huggingface.co/datasets/ccdv/govreport-summarization)。本项目筛选了256篇完整报告，加入摘要指令，形成[近8K输入请求集](data/govreport-near8k.jsonl)，用于DSpark off/on性能对照。当前实验固定取前128条，平均输入8141.79 tokens，输出固定1024 tokens；这是摘要任务的推理性能测试，不是摘要质量评测。筛选方法、来源版本、许可证及使用方式见[数据说明](data/govreport-near8k.md)。

2026-09-19已完成同一真实文本负载下的单TP4、C32对照：off为 **663.92 ± 0.53**，修正Graph覆盖的DSpark K5为 **804.65 ± 5.64 tok/s（+21.20%）**，各三轮无已知编译事件。P95 TPOT略高，不能称为所有延迟全面改善。结果、quick/clean时间成本、Graph修正及复现命令见[DSpark性能报告](reports/dspark-tp4-20260919.md)，逐轮数据见[CSV](data/dspark-tp4-20260919.csv)。该负载与历史随机8192/1024不同，不能混用性能基线。

### DSpark 加载兼容性

固定 vLLM 0.29.0 对本模型内置草稿存在 NVFP4/MXFP4 分派问题。已保留原权重并验证本地加载补丁，问题原因、适用范围、使用与回退方法见[DSpark 兼容性说明](reports/dspark-compatibility.md)。补丁不修改镜像；加载兼容性与性能收益分别验证。K5 C32还需扩大Graph捕获范围，不能直接沿用普通TP4的上限32，具体见上述性能报告。

DP＋DSpark另有启动profiling元数据错误；已回移上游已合并PR #54856的最小修复，保留固定镜像。2026-09-20在48完成TP2×DP2、K5、EP on/off功能验证：两组均通过启动、CUDA Graph及完整C32请求，各128成功、0失败。首轮含编译事件，未取得无JIT性能结果；旧失败批次仍保留，后续K值实验在本地recipe中显式启用补丁，并使用新的结果目录。来源、验证范围与启用方式见[DP启动修复](reports/dspark-compatibility.md#dpdspark回移已合并的上游修复)。

## 下一轮：DP＋DSpark 补丁后续测

**本轮只研究四卡TP2×DP2、GovReport近8K／1024、总C32，先测K5、K3、K4。** K是草稿预测token数；off表示关闭DSpark，EP on/off另行标明。TP4完整曲线已由45完成：K3–K5约799–808 tok/s，K1/K2约733／774 tok/s，因此先缩小DP搜索范围；这不代表DP下排序必然相同。

DP启动修复已包含在提交 `d529c37`。48的K5、EP on/off均完成一轮128请求功能验证，但含JIT、协议为PARTIAL，不能作为正式性能样本。本节是**当前执行任务**；上轮结果只作参照，不再执行旧的全K矩阵。

### 节点任务与执行顺序

收到“执行对应节点任务”后，按本机hostname/IP认领下表；只操作本机，不SSH到其他节点。同一时间只运行一套GPU4–7服务。

| 节点 | 核心任务，按顺序执行 | off参照 | 预计耗时 |
| --- | --- | --- | --- |
| **45** | 本轮不追加实验 | 已完成TP4 off、K1–K5及K3复测 | 无 |
| **46** | EP on：K5 → K3 → K4 | 复用本机上轮完整off，737.56 ± 22.34 tok/s | 约1–2小时 |
| **47** | EP off：K5 → K3 → K4 | 复用本机上轮完整off，710.39 ± 20.66 tok/s | 约1–2小时 |
| **48** | EP on下：off → K3；再EP off下：off → K3 | 本次各EP重新取得三轮off | 约2–3小时 |

48每个EP设置都测一组DSpark off和一组K3，合计四组。此前两组off各只有两轮clean，不能拼接进新启动；本次K5功能验证也不计入正式结果。

46/47引用各自[上轮报告](reports/dspark-k-sweep-node46.md)／[47报告](reports/dspark-k-sweep-node47.md)及对应CSV中的三轮off，明确它们与on属于不同启动批次，且CV约3%，不把小幅收益称为可靠提升。先核对模型、服务参数、客户端、负载和资源背景可比；若发生实质变化，只报告on绝对结果并说明参照失效，不强算收益或擅自补跑整套基线。48提供本轮同机off/on与EP交叉核对。

核心完成后的可选扩展有明确上限：

- **46/47：**仅当K3、K4、K5均PASS，且K3吞吐分别高于K4和K5超过 `max(2%, 2×两组较大吞吐CV)`，无明显资源变化或容量问题时补K2；若K2对K3仍满足同一条件，再补K1。最多两次追加启动。阈值只是筛选规则，不是统计显著性证明。
- **48：**核心四组均PASS且核心运行墙钟耗时不足3小时，可再按EP on/K5 → EP off/K5补一对；否则到核心结束即归档，不追求填满时间。最多两次追加启动。
- 不追加K6/K7、其他并发、双部署或独立重启复测。已有PARTIAL或故障时先交付核心结果，不自动扩展。

时间是排期估计，不是3小时内必须成功的承诺。每配置按下文固定预算结束；超过预计总耗时可继续完成核心未测配置，不截断当前轮次、不无限重试。遇到真实共性故障则按停止规则处理。

- [x] **45**：本轮无新增任务，沿用[TP4报告](reports/dspark-k-sweep-node45.md)。
- [ ] **46**：完成EP on的K5/K3/K4或明确记录阻塞，归档本轮报告和CSV。
- [ ] **47**：完成EP off的K5/K3/K4或明确记录阻塞，归档本轮报告和CSV。
- [ ] **48**：完成两组off与K3的同机对照或明确记录阻塞，归档本轮报告和CSV。

### 配置准备：复用公共入口，在工作区修改

先读根README、[压测协议](../../docs/benchmark-methodology.md)和[DP补丁说明](reports/dspark-compatibility.md#dpdspark回移已合并的上游修复)。开始前更新到包含 `d529c37` **及本节公共DP配置**的版本，保存 `git rev-parse HEAD` 与本地差异；运行期间不pull、不修改源码或配置。不要只拿到修复提交就继续照旧任务跑K1–K5。

| 公共文件 | 用法与验证范围 |
| --- | --- |
| [DP K5 recipe](configs/recipes/tp2-dp2-dspark-k5.yaml) | 默认TP2×DP2、EP on，已包含两处补丁入口和required检查；服务参数与48功能验证一致，正式三轮尚未测 |
| [DP正式campaign](configs/campaigns/dspark-dp-clean.yaml) | 单case、单rear target、正式jit_clean，作为各K入口模板；禁止添加 `replica_targets` |
| [GovReport jit_clean](configs/workloads/govreport-c32-jit-clean.yaml) | 128请求，累计三轮clean，最多12轮／2700秒，单轮超时900秒 |
| [rear target](configs/targets/rear.yaml) | GPU4–7，服务CPU32–47／NUMA2，客户端CPU48–51／NUMA3；本地改成本机IP和id |
| [既有普通DP recipe](configs/recipes/dual-tp2-dp2-epon.yaml) | 48的DSpark off来源，只取其中一套四卡服务；不运行双部署campaign |

续用各节点 `experiments/dspark-k-sweep-nodeNN/`。已有工作区时检查并更新所需文件，**不要整体覆盖configs或results**；工作区不存在时复制本项目configs及 `data/govreport-near8k.jsonl`，建立本地results/reports目录。target地址依次为 `10.90.1.46`、`.47`、`.48`，同时核实本机映射和端口31249空闲。runtime继续使用公共的1800秒启动上限与持久化Triton缓存配置。

从公共DP正式campaign复制单case结构，仅替换campaign id、case id和recipe引用，保留公共model、runtime、client及正式workload引用。每个本地recipe/campaign统一命名为 `tp2-dp2-<epon|epoff>-<off|kN>.yaml`，campaign只引用该配置，case id使用同名、不含扩展名。可以直接修改原来失败的配置文件；旧run已保存解析配置、命令和日志，保留旧结果即可，无需再维护一套失败recipe。所有候选须在第一组实验启动前准备并完成validate/plan。

从公共DP K5 recipe准备各K时，只修改id/说明、EP开关、`num_speculative_tokens`和Graph尺寸，其余参数与日志检查保留。EP off移除 `enable-expert-parallel`，换成 `no-enable-expert-parallel`，不能同时保留两个开关。每rank活动容量固定16，Graph配置如下：

| DSpark | `cudagraph_capture_sizes` | `max_cudagraph_capture_size` |
| --- | --- | ---: |
| off | `[1,2,4,8,12,16]` | 16 |
| K5 | `[5,6,10,12,20,24,40,48,60,72,80,96]` | 96 |
| K3 | `[3,4,6,8,12,16,24,32,36,48,64]` | 64 |
| K4 | `[4,5,8,10,16,20,32,40,48,60,64,80]` | 80 |
| 可选K2 | `[2,3,4,6,8,12,16,24,32,36,48]` | 48 |
| 可选K1 | `[1,2,4,8,12,16,24,32]` | 32 |

on尺寸由基本批次 `[1,2,4,8,12,16]` 分别乘K和K+1后取并集，覆盖草稿 `16×K` 与目标 `16×(K+1)`。K1–K4本轮只完成离线准备，仍需实际验证，不能称为已测最优配置。

**DSpark on必须同时启用量化分派和DP修复。** 公共recipe已设置 `PYTHONPATH=/root/.cache/dspark-native-mxfp4`、`SERVING_BENCH_DSPARK_DP_PROFILE_FIX=1`，并要求 `LOCAL_DSPARK_FORMAT_FIX`、`Mxfp4 MoE backend`、`UPSTREAM_DSPARK_DP_PROFILE_FIX: PR #54856 facd9a74a1` 日志。旧recipe需补齐这些项，每个on campaign重新运行prepare与 `--check`；仅pull代码不会更新服务读取的缓存内补丁。

DSpark off使用普通DP recipe，设置对应EP开关和off Graph，不添加 `speculative-config`、补丁环境变量或草稿required检查，也不调用补丁prepare。所有配置都使用同一个正式workload；**不要使用quick campaign或本机一轮smoke验证的workload。**

### 正确启动顺序

下面以46的K5为例，文件须已按上文准备。47替换节点号与epon为epoff；48按任务表选择对应配置。`run_id`每次唯一，目录不能预先创建。

```bash
study=experiments/dspark-k-sweep-node46
case_id=tp2-dp2-epon-k5
campaign="$study/configs/campaigns/$case_id.yaml"
run_id="dpfix-$case_id-01"
mkdir -p "$study/reports" "$study/results"

./bench validate "$campaign"
./bench plan "$campaign" > "$study/reports/$run_id-plan.json"
# 以下两行仅DSpark on执行；off跳过。
python3 projects/dsv4-rtx6000d/patches/dspark-native-mxfp4/prepare.py "$campaign"
python3 projects/dsv4-rtx6000d/patches/dspark-native-mxfp4/prepare.py "$campaign" --check
python3 scripts/prepare_logging.py "$campaign"
./bench preflight "$campaign"
./bench run "$campaign" --run-root "$study/results/$run_id"
```

长任务在tmux等持久会话中运行，复用本地只读观察脚本采集指标和资源，不修改公共runner。prepare、validate、plan或preflight失败时先处理明确阻塞，不能忽略退出码继续run。runner的run还会再次预检，并执行health、models、关闭thinking的中文生成、完整负载和收尾清理。

缓存按完整recipe等身份计算，增加DP开关后通常进入新目录。保留旧JIT缓存，不手写猜测路径或删除缓存。prepare若发现已有不同补丁，更新本地recipe id的版本标识后重新准备，不强行覆盖。若复用兼容的旧kernel缓存，只复制经核对的编译缓存子目录，不能复制旧 `dspark-native-mxfp4` 覆盖新补丁；记录来源与起点，各K不共用可写缓存。

### 共同测量与停止规则

- **固定条件：**服务和客户端均为 `vllm/vllm-openai:v0.29.0`，image ID `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`；权重 `/data/models/DeepSeek-V4-Flash-0731-NVFP4`。不升级镜像，不改变主模型NVFP4／草稿MXFP4、backend或显存比例。
- **服务：**TP2、DP2、PP1、`data-parallel-size-local=2`，每rank容量16、prefill8192，上下文16384，显存比例0.90、FP8 KV、V2＋async、FULL_DECODE_ONLY。关闭FlashInfer autotune与prefix cache，不启用CPU offload。DSpark固定greedy、standard rejection、adaptive verification关闭。本轮只变指定EP和K。
- **资源：**核实GPU空闲、CPU/SMT与NUMA映射和实际绑定；GPU被占用就报告阻塞，不终止其他任务。保留已知DOCA后台负载记录，新增矿工或明显异常先暂停受影响测量。监测服务及客户端CPU、SMT兄弟、GPU温度／功耗／频率；不锁频、不改宿主功耗或全局NUMA。
- **工作量：**同一JSONL、固定前128条、流式 `/v1/completions`，全服务总C32，seed0、temperature0、ignore_eos、request_rate=inf。每轮128成功、0失败，实际输入1,042,149／输出131,072 tokens。DP不能每rank各发C32或128请求。
- **协议：**同配置只启动一次，直接逐轮运行完整128请求，累计最先三轮无已知编译事件且工作量正确的样本。不另外插入预热，不因JIT重启、不挑最快轮。最多12轮／2700秒，单轮900秒，服务启动1800秒；每配置单独计预算。无JIT不等于低波动，仍报告全部轮次和CV。
- **日志：**首次K5或48的K3确认两个补丁标记、主模型NVFP4／草稿MXFP4、两个DP rank、EP状态、Graph覆盖与KV分配；完整首轮同时验证C32容量。记录实际attention/MoE路径、编译、fallback、通信及NUMA警告，不仅看YAML。可观察标记不代表完整验证所有kernel或数值精度。
- **停止：**协议预算用尽记PARTIAL并继续其他独立配置，不重启补满。OOM、请求失败、token错误或异常退出按故障取证；共性DP/草稿路径错误停止其余相关K，单个K容量问题只停止该候选。若原 `8192 80` 复现，先核对运行版本、环境开关、实际缓存补丁及worker标记；不放宽gate、改精度或禁用Graph换取PASS。结束只清理本次容器。

### 指标与归档：每节点一份报告和CSV

本轮在 `projects/dsv4-rtx6000d/` 下归档 `reports/dspark-dp-resume-nodeNN.md` 和 `data/dspark-dp-resume-nodeNN.csv`。原 `dspark-k-sweep-nodeNN` 报告／CSV不覆盖；45不新增空报告。探索配置、脚本、完整日志、全部轮次JSON及资源序列留在本机工作区，不批量提交。

报告先写完成情况与结论，再给off/K值对比表：输出tok/s、相对本机同拓扑同EP off的提升、Mean／P95 TTFT与TPOT、草稿接受率、平均接受长度；报告逐次值、均值、样本标准差及吞吐CV。补充requests/s、ITL、端到端延迟、实际工作量、KV容量／峰值／抢占、各轮JIT、累计clean轮数和时间成本。分开记录启动、协议和run总耗时，拒绝轮也计成本；重复P95的均值不称为合并请求P95。DSpark的ITL是流式事件间隔，不等同逐token计算时间。

复用[现有44列CSV格式](data/dspark-k-sweep-node46.csv)，字段名和单位保持一致，一轮一行，含拒绝和中断轮；未启动的配置只在报告记缺项，不伪造零吞吐。`configuration`为 `tp2-dp2-epon-k3` 等统一名称，`run_id`以 `dpfix-` 开头区分新批次，`study_phase`用core或extension。历史off只在报告链接原CSV并列比较，不复制成新测量、合并样本或改写原判定。报告注明新旧批次、Git commit、runner指纹、补丁manifest哈希、镜像ID、数据哈希及资源差异；启动命令从plan或实测产物导出，一份主命令加EP/K差异表即可复现。

接受统计必须按每轮服务端计数器前后增量计算，功能探测与其他请求不混入。每约1秒保存 `/metrics` 原始快照（记录实际时间），按DP rank保存后累加分子／分母，避免重复计算TP worker。优先核对 `vllm:spec_decode_num_drafts_total`、`spec_decode_num_draft_tokens_total`、`spec_decode_num_accepted_tokens_total` 和逐位置计数的实际定义。接受率=接受草稿tokens／提出草稿tokens，平均接受长度=接受草稿tokens／drafts；若加目标补充token需另注明。窗口、计数器重置或字段无法核实时留空说明，不能用K或轮次百分比冒充加权统计。

KV容量取各DP rank的最小值，KV使用率取各rank采样峰值，抢占为无重置时计数增量之和；同时在报告保存各rank实际值与采样局限。CPU/GPU资源建议每5–10秒采样，额外记录背景进程；复用上轮观察脚本可以，但检查地址、端口、run窗口及此次配置，不能默默使用旧常量。普通单服务runner不会自动提供全部时间序列，各节点自行保存只读采样，不能为此改runner或gate。

完成后只勾选本节自己的节点项，写明“完成／部分完成／阻塞”并链接新报告。提交前从CSV重算汇总、核对单位和相对链接、运行 `git diff --check`；只提交本节点报告、CSV和自己的进度行，不改其他节点或总表，不提交工作区及重复配置。未获push授权时只commit，交由用户安排推送；不得在实验运行中pull或处理合并。

<a id="下一轮四节点-dspark-拓扑与-k-值实验"></a>

### 上轮结果与本轮参照

2026-09-20原K1–K5任务已经结束；旧链接保留在此定位。45完成TP4曲线；46/47只有off取得三轮clean，K5遇到DP启动断言；48的两组off各有两轮clean，K5阻塞。旧结果不因补丁通过而改判。

| 节点 | 原报告 | 原始精度逐轮数据 |
| --- | --- | --- |
| 45 | [TP4曲线](reports/dspark-k-sweep-node45.md) | [CSV](data/dspark-k-sweep-node45.csv) |
| 46 | [EP on基线与阻塞](reports/dspark-k-sweep-node46.md) | [CSV](data/dspark-k-sweep-node46.csv) |
| 47 | [EP off基线与阻塞](reports/dspark-k-sweep-node47.md) | [CSV](data/dspark-k-sweep-node47.csv) |
| 48 | [同机基线与阻塞](reports/dspark-k-sweep-node48.md) | [CSV](data/dspark-k-sweep-node48.csv) |

## 2026-09-20：四节点预热与负载对照

**GovReport下，单TP4、DSpark off的吞吐在四节点均约664 tok/s；一轮完整负载预热后，三轮正式结果接近且无已知编译事件。独立quick与jit_clean的性能、耗时也接近。K5仍有正式轮编译和明显降速，后续DSpark关键对照优先使用jit_clean。**

### 条件与结果

固定vLLM 0.29.0、单实例TP4、GPU4–7、C32、每轮128请求／每请求1024输出；服务CPU32–47／NUMA2，客户端CPU48–51／NUMA3。其余服务条件沿用公共recipe：EP off、FP8 KV、autotune off、prefix cache off。K5预测5个草稿token，使用已验证的加载补丁与扩大后的Graph覆盖。

本轮quick为**128请求预热1轮＋128请求正式3轮**；jit_clean从完整128请求轮次开始，累计接纳最先三轮无已知编译事件的结果。同配置所有轮次在一次启动内完成。GovReport固定使用JSONL前128条，每轮实际输入1,042,149、输出131,072 tokens；random为等长8192输入，每轮输入1,048,576、输出131,072 tokens。全部28轮均128成功、0失败。

下表每行n=3，吞吐为均值±样本标准差，CV为吞吐标准差／均值；延迟为三轮对应均值的平均。quick保留全部正式轮，包括有事件的轮次。逐轮结果、P95和其他指标见节点报告与CSV，不跨节点合并样本。

| 节点 | 负载／DSpark | 协议 | 输出 tok/s | CV | Mean TTFT s | Mean TPOT ms |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 45 | GovReport／off | quick | 664.56 ± 0.28 | 0.04% | 5.650 | 42.61 |
| 45 | GovReport／off | jit_clean | 663.21 ± 0.51 | 0.08% | 5.653 | 42.70 |
| 46 | GovReport／off | quick | 663.40 ± 0.51 | 0.08% | 5.648 | 42.69 |
| 46 | GovReport／K5 | quick | 741.52 ± 116.34 | 15.69% | 4.574 | 38.40 |
| 47 | random／off | quick | 718.42 ± 2.31 | 0.32% | 6.334 | 38.33 |
| 47 | GovReport／off | quick | 664.69 ± 0.37 | 0.06% | 5.649 | 42.60 |
| 48 | GovReport／off | quick | 664.41 ± 0.32 | 0.05% | 5.658 | 42.61 |

### 这轮解决了什么

**off基线可以复现，负载差异需要单独看。** 四节点GovReport off quick均值最大差异仅0.195%，各组三轮CV均低于0.08%。47同机GovReport比random吞吐低7.48%，Mean TPOT高11.12%，Mean TTFT低10.81%；这支持将两种负载分开建立基线。两组为独立启动，random首轮还含8条事件，文本、输入长度分布、运行顺序和后台负载的影响未分别隔离，不能据此认定某一种内部机制是原因。

**本次quick没有明显节约时间。** 45从同一缓存快照的两个独立副本启动：quick与jit_clean吞吐仅差0.20%，Mean TTFT／TPOT差异均低于0.23%。两者实际都执行4轮128请求，协议耗时分别15.79／15.81分钟，整个run均约18.35分钟；run包含内部预检、启动、协议和收尾，不含外部准备及缓存复制。每种协议只独立启动一次，这个结果不代表所有配置的成本都相同。

**K5有提升潜力，但一轮完整预热仍未覆盖后续编译。** 46正式吞吐为806.44／607.21／810.92 tok/s，事件行数为0／40／0；第二轮日志包含实际TileLang编译起止，不能只当作缓存加载。三轮全部保留后均值741.52 tok/s，较本机off高11.78%，但CV达15.69%，尚不能作为稳定收益。编译与慢轮同现，CPU／SMT负载也有差异，未完全隔离因果；事件行数不等于独立编译次数。

此前48的修正K5已取得三轮clean **804.65 ± 5.64 tok/s，较同批off提高21.20%**，可作为已有收益证据，不能与46的两轮快样本拼成一组。该次累计验收共运行5轮，协议耗时15.85分钟，见[夜间报告](reports/dspark-tp4-20260919.md)。46本轮K5 quick协议耗时19.38分钟、run总耗时27.57分钟，但其K5缓存此前不存在，启动与编译历程不同，不能拿这两个批次推断clean比quick更快。

### 后续采用的测量方式

GovReport TP4 off的快速筛选可继续用本轮完整负载quick；**DSpark参数与拓扑的关键收益对照优先使用jit_clean，off/on采用同一协议**。每轮执行完整正式工作量，同一次启动内累计最先三轮无已知编译事件，不再额外要求“两轮安静预热后重新测三轮”。沿用最多12轮／45分钟的协议预算，保留拒绝轮、全部耗时和波动；未完成则明确记录，不能挑最快三轮。

无已知事件仍不等于性能稳定，需同时检查CV、延迟、请求和token数量以及资源干扰。本轮没有运行stable，也没有修改公共默认协议；上述建议限于当前单TP4、C32与这份负载，不能直接推广到双实例或其他K值。

### 节点报告与复现入口

- [x] **45**：独立quick／jit_clean对照已完成，两组PASS；见[节点报告](reports/dspark-protocol-node45.md)与[CSV](data/dspark-protocol-node45.csv)。
- [x] **46**：同机off／K5 quick已完成，保留K5慢轮及JIT事件；见[节点报告](reports/dspark-protocol-node46.md)。
- [x] **47**：两种负载独立启动对照已完成，见[节点报告](reports/dspark-protocol-node47.md)。
- [x] **48**：完整负载预热验证完成，quick PASS；见[节点报告](reports/dspark-protocol-node48.md)。

<details>
<summary>本轮原执行方案、公共配置与归档约定（已完成）</summary>

以下保留本轮执行约定供复现；节点任务均已交付，不是待执行的新一轮实验。

**本轮只做单实例TP4、C32、1024输出，先回答测量是否省时可靠、DSpark收益能否复现、真实文本为何比历史随机负载慢。** 不扩展K值、拓扑或并发，不要求每个节点重新取得三轮clean。四台各自在本机执行，不SSH控制其他节点。

### 节点任务与完成标记

下面每个campaign单独执行一次`bench run`，独立启动并清理服务；同配置的全部轮次在这一次启动中完成。off表示关闭DSpark；K5表示开启DSpark、预测5个草稿token。quick在本轮特指**128请求预热1轮＋128请求正式测量3轮**，不是默认的2C预热。

| 节点 | 按顺序执行的campaign | 本机对照与问题 | 预算，不含准备／取证 |
| --- | --- | --- | --- |
| **45** | `dspark-off-quick` → `dspark-off-clean` | 同一GovReport负载，比较独立启动的quick／jit_clean性能与时间成本；使用同一缓存快照的两个副本 | 两次启动；每段≤45分钟；clean≤12轮 |
| **46** | `dspark-off-quick` → `dspark-k5-quick` | 同机off/on收益；两边均使用一次完整负载预热，保留各自JIT和波动 | 两次启动；每配置固定1＋3轮、≤45分钟 |
| **47** | `tp4-random-quick` → `dspark-off-quick` | off下分别比较随机等长8192与GovReport；两次独立启动，不交替调度；使用同一缓存快照的两个副本 | 两次启动；每负载固定1＋3轮、≤45分钟 |
| **48** | `dspark-off-quick` | 检查一轮128请求预热后是否仍有慢首轮；本机历史clean 663.92 tok/s仅作参照 | 一次启动；固定1＋3轮、≤45分钟 |

45的两种协议只各启动一次，不能据此证明跨启动耗时稳定；47仍有启动和顺序差异，不能把全部差距唯一归因于文本内容。46的收益只用本机off作分母；48历史clean与本轮缓存历程不同，不是严格的同批配对。不得跨节点拼接off/on计算收益。

### 公共配置与本地准备

| 配置 | 用途与状态 |
| --- | --- |
| [tp4 recipe](configs/recipes/tp4.yaml) | 已实测的off服务参数，继续复用 |
| [TP4 DSpark K5 recipe](configs/recipes/tp4-dspark-k5.yaml) | 与夜间实测修正组相同；含原生MXFP4草稿补丁入口、K5和Graph覆盖 |
| [GovReport quick](configs/workloads/govreport-c32-quick-full.yaml) / [random quick](configs/workloads/random-c32-quick-full.yaml) | 本轮已实测：`warmup_rounds: 1`、`warmup_load: full`、`repetitions: 3`；K5仍有正式轮JIT，见上方结果 |
| [GovReport jit_clean](configs/workloads/govreport-c32-jit-clean.yaml) | 每轮128请求，累计最先三轮无已知事件，最多12轮／45分钟 |
| [off quick](configs/campaigns/dspark-off-quick.yaml) / [K5 quick](configs/campaigns/dspark-k5-quick.yaml) / [off clean](configs/campaigns/dspark-off-clean.yaml) / [random quick](configs/campaigns/tp4-random-quick.yaml) | 上表四个执行入口，每个都只有一个case |
| [rear target](configs/targets/rear.yaml) | GPU4–7；服务CPU32–47/NUMA2，客户端CPU48–51/NUMA3；本地改成本机地址 |

先阅读根README、[压测协议](../../docs/benchmark-methodology.md)及本节。四节点使用同一准备版本；运行期间不pull或修改源码／配置。以下以48为例，在仓库根目录执行；其他节点替换工作区名称。工作区已存在时检查续接，不覆盖原文件。

```bash
mkdir -p experiments/dspark-protocol-node48/data experiments/dspark-protocol-node48/results experiments/dspark-protocol-node48/reports
cp -a projects/dsv4-rtx6000d/configs experiments/dspark-protocol-node48/configs
cp projects/dsv4-rtx6000d/data/govreport-near8k.jsonl experiments/dspark-protocol-node48/data/
```

在本地`configs/targets/rear.yaml`中设置本机IP（45/46/47/48对应`10.90.1.45`至`.48`），核实GPU/NUMA映射、SMT兄弟及端口空闲。不要改已归档的target。固定服务与客户端镜像`vllm/vllm-openai:v0.29.0`，ID为`sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`；模型为`/data/models/DeepSeek-V4-Flash-0731-NVFP4`。不拉取镜像、不改权重精度。

46只对K5 campaign运行[补丁准备](reports/dspark-compatibility.md#后续实验如何使用)，不要把off campaign传给这个脚本：

```bash
python3 projects/dsv4-rtx6000d/patches/dspark-native-mxfp4/prepare.py \
  experiments/dspark-protocol-node46/configs/campaigns/dspark-k5-quick.yaml
```

各campaign启动前分别validate、plan、preflight，保存解析配置和命令；run仍会执行health、models和短中文生成。例如48：

```bash
./bench validate experiments/dspark-protocol-node48/configs/campaigns/dspark-off-quick.yaml
./bench plan experiments/dspark-protocol-node48/configs/campaigns/dspark-off-quick.yaml
./bench preflight experiments/dspark-protocol-node48/configs/campaigns/dspark-off-quick.yaml
./bench run experiments/dspark-protocol-node48/configs/campaigns/dspark-off-quick.yaml \
  --run-root experiments/dspark-protocol-node48/results/off-quick-01
```

46第二次使用`dspark-k5-quick.yaml`及新的run-root；45第二次使用`dspark-off-clean.yaml`，47先使用`tp4-random-quick.yaml`。45/47还需先完成下面的缓存准备。不要把两个case塞到同一个campaign共用可写缓存，也不要测一轮就重启。

### 缓存起点与资源规则

**45、47的两次运行必须从内容一致、互不共享写入的缓存副本开始。** 在启动任何一组前，从本机已有off recipe的实际缓存目录制作冻结快照，再复制为两个独立可写目录。使用普通复制或reflink，不用硬链接、不共用可写符号链接；原缓存和快照均保留。不能先跑第一组，再从它已经更新的缓存复制第二组。

缓存路径由image ID、SM架构、model ID、完整recipe决定，不能凭旧路径猜测。可在修改cache_root前，用以下只读命令定位源目录（45示例）：

```bash
PYTHONPATH=src python3 - <<'PY'
from serving_bench.config import resolve
from serving_bench.executors.docker import cache_directory
case = resolve('experiments/dspark-protocol-node45/configs/campaigns/dspark-off-quick.yaml')['cases'][0]
print(cache_directory(case, case['runtime']['image_id']))
PY
```

仅在本地复制两份薄target，分别改`cache_root`，并让两个campaign引用各自target；recipe内容保持不变。仍用`cache_directory`计算两个目标目录，再从同一冻结快照复制。建议缓存根放`~/.cache/serving-bench/dspark-protocol-nodeNN/<批次>/<组名>/`，与results分开。记录快照时间、来源身份、文件相对路径／SHA256／权限清单，两副本开始前内容一致且可读写。遇到不可读文件不得静默漏复制；先解决本次副本访问问题，不能删除原缓存。源不存在或无法形成一致副本时报告阻塞，不擅自清空缓存改成冷启动。克隆目录可能有宿主页缓存、文件路径等剩余差异，要在报告中说明。

46/48正常保留各recipe的持久缓存，记录运行前已有缓存情况。K5与off的缓存、启动历程不要求完全相同，正式指标比较的是各自预热后的serving；如果编译影响不同，报告其影响，不隐去事件。45/47的缓存复制耗时单列，不算服务或协议耗时。

每台同一时间只运行一个推理服务，仅使用本次GPU4–7。所选GPU被其他计算进程占用则停止，不终止他人任务。检查CPU异常负载；已知DOCA负载需记录，新增矿工等异常不能当作正常背景忽略。记录服务/客户端CPU及SMT兄弟忙碌率，低频采样GPU功耗、温度、频率和KV使用；不锁频、不改宿主功耗或全局NUMA配置。结束只清理本次容器，保留缓存及所有结果。

### 统一测量与解释

- TP4/PP1/DP1、EP off、上下文16384、活动容量32、prefill8192、显存比例0.90、FP8 E4M3 KV、V2＋async；关闭autotune和prefix cache。off的Graph上限32；K5保留完整捕获列表，上限192，覆盖目标192／草稿160 tokens。核对启动日志、补丁标记、实际backend和KV分配，不能只看YAML。
- 客户端固定流式`/v1/completions`，C32、128请求、temperature0、seed0、ignore_eos=true、request_rate=inf。GovReport用同一JSONL前128条且顺序固定；每轮实际输入1,042,149、输出131,072 tokens。random为8192/1024、range_ratio0，每轮输入1,048,576、输出131,072。每轮均应128成功、0失败；不把全256条都加入。
- quick只有1轮128请求预热＋3轮正式测量，三轮全部保留，包括JIT与慢轮；本轮先按固定入口完成，不自动加三轮clean或重新启动补测。若首轮明显慢、后两轮接近，明确标记“可能需要更多完整负载预热”，保留逐轮证据，后续再决定诊断。无事件不代表稳定，也不能仅凭首轮慢断言两轮预热必然足够。
- 45的clean是独立服务启动后直接执行完整128请求轮次，不先跑quick。仅该任务要求累计最先三轮无已知事件；最多12轮／2700秒，每客户端阶段900秒。未完成则记录PARTIAL与已花时间，不无限补到PASS。quick同样每段2700秒、单阶段900秒；OOM、请求失败、token数错误属于故障，不当作预热继续。
- 报告每个指标逐次值、均值、样本标准差和CV。若要描述“接近”，先按吞吐、Mean TTFT、Mean TPOT均值相差≤3%观察；各自相对极差≤3%可描述为本组三次接近，不能改称通过stable。记录事件与资源变化，差距落在波动范围内直接说明。48用历史数据只能作旁证，46/47重点用本机参照。
- 分开记录服务启动至health、health至功能检查完成、协议阶段墙钟时间、客户端benchmark计时、清理时间及run总耗时。45重点给出两种协议的总成本和实际请求量；不要拿warmup后的clean计时冒充独立协议成本。不合并不同配置的P95；重复P95的均值须明确标注。

### 统一归档与交接

原始日志、解析配置、计划／命令、环境和哈希、全部轮次JSON、缓存清单及资源时间线留在各自`experiments/dspark-protocol-nodeNN/`；不批量提交探索配置、缓存、脚本或大日志。归档只新增：

- `reports/dspark-protocol-nodeNN.md`：先写结论和任务完成程度，再给同机对照表、逐轮表现、耗时、事件／资源限制和复现入口；记录Git commit、源码指纹、镜像ID、数据哈希、完整实际服务命令及本地target/cache变化。引用仓库内的公共配置与CSV，用相对链接；原始路径仅作为“本机可用”的附注。
- `data/dspark-protocol-nodeNN.csv`：每轮一行，包含预热、全部正式轮及clean拒绝轮，不能仅导出最快或接纳样本。统一字段如下，时间戳使用含时区的ISO8601，数值保留原始精度；未测得字段留空。

```text
node,run_id,configuration,dataset,protocol,phase,round,accepted,protocol_status,case_status,known_event_lines,started_at,finished_at,duration_s,completed,failed,total_input_tokens,total_output_tokens,output_throughput,request_throughput,mean_ttft_ms,p95_ttft_ms,mean_tpot_ms,p95_tpot_ms,mean_itl_ms,p95_itl_ms,mean_e2el_ms,p95_e2el_ms,spec_decode_acceptance_rate,spec_decode_acceptance_length
```

`configuration`固定为`tp4-off`或`tp4-k5`，`dataset`为`govreport`或`random`，`phase`为`warmup`或`measurement`，`round`为该阶段内从1起的原轮次，`accepted`用`true/false`。仅case与protocol均PASS且属于正式接纳轮次时填true；quick接纳不等于无JIT。接受率单位为百分数；`duration_s`仅客户端benchmark计时，阶段时间含初始化／检查；启动、协议总耗时与缓存复制成本另列于报告。现有`export_samples.py`仅导出接纳样本，不能用它代替本轮完整轮次CSV。

各节点完成后**只改本节自己的完成标记**，例如 `- [x] **45**：已完成，见[节点报告](reports/dspark-protocol-node45.md)。` 阻塞也可打勾表示已交付，但必须写“阻塞”或“部分完成”，不伪装成功。不要修改其他节点行或抢先重写四节点总表。

提交前从CSV重算报告数字，检查链接、字段、单位和`git diff --check`；本节点只提交报告、CSV及自己的完成行。运行期间不更新源码；实验完成后如需同步远端再处理文档冲突。除用户另行授权外不push。公共功能确有阻塞时先留证据，单独说明修复范围，不让各节点自行修改gate或扩展矩阵。

</details>

## 2026-09-18：八卡整机部署结果

### 范围与数据来源

本轮固定单机8张GPU，比较两套独立四卡部署与一套八卡部署。共10种部署形态、14种EP配置变体；双TP4在四台分别复现，共17个节点配置、C32/C64各三次，**合计102份已接纳的整机样本**。省略的TP/PP/DP为1，除注明EP on外均为off。

- [x] **45**：双TP4、双TP2×PP2、双TP1×PP4，18次。见[报告](reports/node8-node45.md)、[CSV](data/node8-node45.csv)。
- [x] **46**：双TP4、双TP2×DP2与双TP1×DP4的EP off/on，30次。见[报告](reports/node8-node46.md)、[CSV](data/node8-node46.csv)。
- [x] **47**：双TP4、八卡TP4×DP2与TP2×DP4的EP off/on，30次；TP4×DP2 EP off的两档分别来自上午、下午批次。见[报告](reports/node8-node47.md)、[CSV](data/node8-node47.csv)。
- [x] **48**：双TP4、TP8、TP4×PP2、TP2×PP4，24次。见[报告](reports/node8-node48.md)、[CSV](data/node8-node48.csv)。

四份CSV共264行，统计只取 `scope=node` 的102行，各副本行不重复计数。合计 **19,584成功、0失败，输入160,432,128、输出20,054,016 tokens**；每次C32为128请求、C64为256请求，全部保持8192/1024。预热、诊断和拒绝尝试不计入。下表从CSV重算，运行与JIT、资源证据依据各节点归档报告；此次整合没有重新跑实验或远程检查原始日志。

### 双TP4基线是否接近

吞吐为三次均值±样本标准差，括号为CV。双TP4的整机C32/C64分别对应每实例C16/C32；两档不能混合求均值。

| 节点 | C32 output tok/s（CV） | C64 output tok/s（CV） | 本批DOCA背景，约占逻辑CPU |
| --- | ---: | ---: | ---: |
| 45 | 1091.28 ± 6.06（0.56%） | 1428.22 ± 4.14（0.29%） | 8 |
| 46 | 1096.87 ± 1.37（0.12%） | 1428.31 ± 7.21（0.51%） | 16 |
| 47 | 1096.13 ± 8.96（0.82%） | 1429.63 ± 6.19（0.43%） | 10 |
| 48 | 1095.66 ± 1.99（0.18%） | 1433.82 ± 0.93（0.07%） | 16 |

四节点均值最大差异为C32 **0.51%**、C64 **0.39%**，支持分节点筛选明显收益的方案，但不能消除拓扑与背景负载的交互影响。48的参照来自补测，C64预热请求由128增至256；各节点DOCA负载也不相同，小幅差异仍需同机复核。

45–47双TP4在整机C64时，各侧自身窗口吞吐均值约714–717 tok/s，与此前独占TP4约715 tok/s接近；整机共同窗口仍约1429 tok/s。**本轮实测支持两份TP4接近吞吐相加的预期**，但整机数据仍应同步测量，不能用单实例乘2代替之后的实测。

### 各部署性能对照

所有行n=3，按节点和部署排列，不混合不同并发或跨节点样本。吞吐为均值±样本标准差；延迟列为三次对应指标的均值，**P95列不是三次请求合并后的P95**。TTFT和端到端延迟用秒，TPOT/ITL用毫秒；完整逐次值、requests/s及其余mean/P95指标见节点CSV。

备注：部分结果来自增加计时外预热后的补测；47的TP4×DP2 EP off配置，其C32和C64来自不同启动批次，分别验收后纳入。服务参数和正式工作量保持不变，预热历程的差异见后文“补测批次与JIT处理”。

#### 整机C32

| 节点 | 部署 | 输出 tok/s | CV | Mean TTFT s | P95 TTFT s | Mean TPOT ms | P95 ITL ms | P95端到端 s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 45 | 双TP2×PP2 | 1182.60 ± 35.75 | 3.02% | 3.55 | 7.50 | 22.85 | 21.76 | 32.65 |
| 45 | 双TP1×PP4 | 1015.25 ± 41.01 | 4.04% | 4.11 | 8.86 | 26.77 | 27.01 | 37.42 |
| 46 | 双TP2×DP2 EP off | 1026.81 ± 59.56 | 5.80% | 5.44 | 10.66 | 25.76 | 23.14 | 37.53 |
| 46 | 双TP2×DP2 EP on | 1078.36 ± 108.98 | 10.11% | 4.18 | 8.69 | 24.40 | 24.03 | 37.94 |
| 46 | 双TP1×DP4 EP off | 948.75 ± 36.60 | 3.86% | 4.91 | 9.31 | 28.48 | 28.13 | 37.98 |
| 46 | 双TP1×DP4 EP on | 956.43 ± 5.31 | 0.55% | 4.69 | 6.78 | 28.21 | 28.81 | 35.97 |
| 47 | TP4×DP2 EP off | 679.75 ± 1.37 | 0.20% | 9.65 | 21.75 | 37.66 | 25.56 | 63.12 |
| 47 | TP4×DP2 EP on | 858.21 ± 74.73 | 8.71% | 6.32 | 13.56 | 31.29 | 24.12 | 46.50 |
| 47 | TP2×DP4 EP off | 786.83 ± 3.74 | 0.47% | 10.85 | 17.84 | 30.08 | 24.11 | 48.17 |
| 47 | TP2×DP4 EP on | 866.31 ± 70.78 | 8.17% | 7.65 | 13.35 | 29.64 | 25.24 | 43.85 |
| 48 | TP8 | 611.92 ± 0.91 | 0.15% | 9.26 | 29.32 | 43.24 | 20.34 | 77.07 |
| 48 | TP4×PP2 | 681.99 ± 20.09 | 2.95% | 4.23 | 12.16 | 42.81 | 36.14 | 55.25 |
| 48 | TP2×PP4 | 1105.42 ± 12.86 | 1.16% | 3.49 | 6.85 | 25.54 | 22.96 | 33.26 |

**C32重点关注：**双TP2×PP2吞吐最高；TP2×PP4吞吐稍低，但本批重复波动更小。

#### 整机C64

| 节点 | 部署 | 输出 tok/s | CV | Mean TTFT s | P95 TTFT s | Mean TPOT ms | P95 ITL ms | P95端到端 s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 45 | 双TP2×PP2 | 1663.93 ± 74.85 | 4.50% | 4.16 | 11.40 | 33.80 | 29.44 | 45.70 |
| 45 | 双TP1×PP4 | 1326.79 ± 119.80 | 9.03% | 4.97 | 11.87 | 42.30 | 39.44 | 55.16 |
| 46 | 双TP2×DP2 EP off | 1491.49 ± 43.22 | 2.90% | 6.29 | 15.68 | 33.00 | 25.62 | 50.24 |
| 46 | 双TP2×DP2 EP on | 1620.37 ± 17.99 | 1.11% | 5.02 | 12.31 | 31.16 | 26.12 | 45.12 |
| 46 | 双TP1×DP4 EP off | 1500.34 ± 59.23 | 3.95% | 7.65 | 29.88 | 30.19 | 26.89 | 56.02 |
| 46 | 双TP1×DP4 EP on | 1487.73 ± 56.02 | 3.77% | 7.60 | 21.53 | 30.83 | 27.57 | 51.43 |
| 47 | TP4×DP2 EP off | 878.47 ± 26.48 | 3.01% | 12.27 | 41.02 | 59.12 | 30.86 | 106.30 |
| 47 | TP4×DP2 EP on | 1151.59 ± 48.74 | 4.23% | 7.88 | 26.04 | 45.77 | 44.15 | 75.65 |
| 47 | TP2×DP4 EP off | 1029.12 ± 9.54 | 0.93% | 13.30 | 34.18 | 46.01 | 27.84 | 82.50 |
| 47 | TP2×DP4 EP on | 1242.03 ± 23.34 | 1.88% | 9.25 | 23.12 | 38.67 | 29.02 | 62.76 |
| 48 | TP8 | 681.52 ± 0.69 | 0.10% | 14.44 | 57.59 | 79.75 | 425.04 | 146.18 |
| 48 | TP4×PP2 | 902.79 ± 3.83 | 0.42% | 5.92 | 23.17 | 65.06 | 48.15 | 86.81 |
| 48 | TP2×PP4 | 1472.15 ± 6.18 | 0.42% | 5.11 | 12.45 | 38.48 | 33.52 | 51.17 |

**C64重点关注：**双TP2×PP2与双TP2×DP2 EP on是高吞吐候选，尚未分出可靠胜负；TP2×PP4吞吐较低，但本批重复波动更小。

### 如何选择

**公共基线继续用双TP4。** 四节点重复结果接近；C32的mean TTFT约4.7–4.8秒、TPOT约24.4–24.5 ms，C64约6.3–6.5秒、38.3–38.5 ms。提高并发带来约30%的吞吐增长，也增加等待和逐token延迟，不能只按最高吞吐选择服务并发。

**C32优先研究双TP2×PP2，C64同时保留双TP2×DP2 EP on。** 45的双TP2×PP2相对本机基线，在C32/C64吞吐提高8.37%／16.50%，mean TTFT与TPOT也下降；但CV为3.02%／4.50%，C64三次吞吐1735.11→1670.81→1585.89，仍有待解释的波动。46的双TP2×DP2 EP on在C64提高13.45%、CV 1.11%，是较有希望的候选；其C32却有10.11% CV，不适合直接替代全并发基线。两者C64的P95 ITL均高于本机双TP4，并非所有延迟指标都改善。

**八卡单套部署中，TP2×PP4最有竞争力，但没有明显超过双TP4。** 48的C32/C64吞吐只高0.89%／2.67%；mean TTFT降低约28%／20%，mean TPOT略高，P95 ITL也更高。C64差异超过该批重复波动，但来自不同启动与预热历史，只能作为本轮观测，不能宣称普遍最优。TP8和TP4×PP2吞吐明显落后，双TP1×PP4也未优于同机双TP4。

**DP是否有用取决于通信组范围。** 46把两套DP服务分别限制在四卡内，C64有收益；47的八卡DP候选即使EP on也未追平本机双TP4。不能由此笼统否定DP或EP：本镜像MoE会跨DP展开通信，EP off也不等于独立复制。EP切换还会改变通信/计算路径，且部分off/on来自不同预热批次；目前没有profiler证明差异完全来自跨NUMA通信。

### 补测批次与JIT处理

原始预算为每次至少连续2轮无已知事件、最多5轮预热、2次测量尝试；各节点先按此执行。出现阻塞后，在用户授权下增加计时外预热，未修改公共runner、gate、服务recipe或正式请求量。下表列出最终采用的补测，其他配置沿用首批。

| 节点 / 配置 | 最终采用批次 | 预热差异与验收 |
| --- | --- | --- |
| 46 / 双TP1×DP4 EP off | `dp4-epoff-02`，C32/C64各三次 | 正式流程前加入C32、256请求/轮的校准；四轮事件16/2/0/0，随后验证无事件；正式预算保持原值 |
| 47 / TP2×DP4 EP off | `jit-tp2-dp4-epoff-01`，C32/C64各三次 | 加入C64、256请求/轮的覆盖预热，最多6轮，实际第4/5轮安静；验证及后续正式测量无事件 |
| 47 / TP4×DP2 EP off | 上午 `jit-tp4-dp2-epoff-extended-01` 的C32三次；下午 `jit-tp4-dp2-epoff-c64-resume-01` 的C64三次 | 上限增至30轮；下午先完成11轮覆盖预热，再接续同一服务测C64；正式尝试前仍要求两轮安静，最多2次尝试 |
| 48 / 双TP4 | `baseline-02`，C32/C64各三次 | C64每轮预热128→256请求，上限仍5轮 |
| 48 / TP2×PP4 | `tp2-pp4-03`，C32/C64各三次 | C64每轮256预热请求、上限10轮，首次实际第7/8轮连续安静；六次测量均一次通过 |

47的C32三次为678.222104／680.173244／680.862191 tok/s，取上午批次实际通过的尝试（r1第二次、r2/r3第一次），并未从不同批次挑选。它们的测量日志、工作量、资源和完整kernel检查通过；后续C64失败不自动否定已验收的C32。**本次按用户授权独立纳入这组三次，原case仍为FAIL**；CSV保留来源、原case状态和接纳依据，默认导出规则不变。下午C64为独立启动，其本地控制器接续了已预热的worker，沿用公共trial及计时/日志检查；不能称为同一次启动完成两档。旧部分结果和拒绝尝试继续保留，不进入本表。

本轮JIT诊断区分了三种情况：新MHC split或top-k特化的真实编译、各worker首次从磁盘加载已有内核，以及日志延迟输出。缓存工件变化、固定镜像源码和双时间戳提供了相互印证；`jit_monitor` warning本身不等于CPU重新编译。没有因这一区别事后放行带事件测量，最终采用的窗口均无已知事件。

46还发现Triton产物位于容器 `/root/.triton/cache`，现有持久挂载为 `/root/.cache`，未覆盖该路径。保留JIT缓存不能理解为所有编译产物都跨容器持久化；历史实验当时未改挂载。当前runtime已显式设置 `TRITON_CACHE_DIR=/root/.cache/triton`，纳入已有持久挂载；两份固定镜像的小kernel跨容器复用已验证，未重跑模型性能，历史统计不变。验证范围见[缓存说明](../../docs/benchmark-methodology.md#jit观察边界)。

### 无事件测量仍有波动

**通过JIT事件检查，不代表性能稳定。** 47的TP4×DP2 EP off在下午C64补测前完成11轮覆盖预热，之后仍执行原验收流程；最终三次无已知事件测量的吞吐为878.47 ± 26.48 tok/s，CV为3.01%。46的双TP2×DP2 EP on在C32同样通过事件检查，CV仍为10.11%。详见[47节点报告](reports/node8-node47.md)和[46节点报告](reports/node8-node46.md)。额外预热解决了事件验收阻塞，但没有消除重复波动，根因尚未确定。

后续默认用quick控制筛选成本；需要稳定窗口时用stable，明确要求排除已知编译事件时才用jit_clean，不要求逐级执行。选择依据和限制统一见[通用压测协议](../../docs/benchmark-methodology.md#为什么默认用quick)。本批历史数据和验收结论保持原样，不改标为新协议结果。

### 比较边界与运行条件

四节点报告均未发现本批矿工恢复，按约定保留DOCA背景；CPU绑定不会排除背景线程及其SMT竞争。GPU低频采样未见持续热降频，不能排除瞬时干扰。部分PP/DP配置在测量无已知JIT、无明确抢占的情况下仍有较大波动、双实例尾段或DP coordinator warning，根因尚未唯一定位；所有慢样本均保留。

本轮未通过降低精度、缩短负载或关闭Graph解决阻塞；没有报告OOM或模型不支持。SM120 SymmMem、PCIe custom allreduce限制、TileLang fallback及FP8 scale等warning保留。日志路径通过不代表模型数值精度或全部kernel已完整验证。`accepted-background`、日志验收通过与性能稳定分别解释。

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

服务共32个物理核、客户端共8个物理核，每核一个线程，SMT兄弟不分配给本次进程。通过Docker cpuset实现；本轮统一不启用vLLM `--numa-bind`。八卡单套部署使用两侧资源的并集，未默认使用更多CPU。内存允许集合不保证页面全部本地驻留，cpuset也不排除DOCA等后台任务。

本轮八卡CPU限制与历史不绑定的八卡实验不同，容量也统一为64；历史结果不能直接计入本轮三次重复。两套DP使用本机local DP，固定镜像为本地通信选择IPC和动态DP初始化端口；不配置外部DP rank或把两个服务加入同一个DP组，启动后核实实际rank、GPU及端口。

### 正式负载、身份与复现

| 整机并发 | 正式请求数 / 重复 | 双部署每侧并发 / 请求数 | 每次整机输入 / 输出tokens |
| --- | --- | --- | --- |
| C32 | 128 / 3次 | C16 / 64 | 1,048,576 / 131,072 |
| C64 | 256 / 3次 | C32 / 128 | 2,097,152 / 262,144 |

同一固定vLLM客户端、流式 `/v1/completions`，seed0、temperature0、ignore_eos=true、range_ratio0、request_rate=inf。双服务将同一全局随机请求集按索引交错均分，是固定分流，不是动态负载均衡。所有部署使用相同的官方计时边界；整机吞吐按总输出tokens除以窗口并集，尾段保留，各侧共同窗口贡献可相加。每次整机延迟先合并请求/流式事件计算，再对三次指标求均值。

服务与客户端镜像均为 `vllm/vllm-openai:v0.29.0`，image ID为 `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`。首批执行commit为 `aacdc39f0862b82527753ce3ac5cbf0de94c1c68`，补测commit见节点报告；公共runner源码指纹始终为 `edccf507e1246a4525a383ff4c4d3ce45a470378075bf3271922903c10020e34`。模型元数据/tokenizer及分片大小身份一致，未全量重算权重内容哈希。47的本地接续控制器另存指纹，不将其等同于原campaign直接运行。

本批已结束，四节点完整矩阵配置见[固定版本快照](https://github.com/guoenhui-tora/llm-serving-benchmarks/tree/4642443b1274251b6a47f0d2742e0e59e2dccc95/projects/dsv4-rtx6000d/configs/campaigns)。各节点报告保留参数差异、补测预热修改和执行commit；精确复现时使用对应源码及配置，不将当前quick入口当作历史验收协议。

当前继续研究使用上方三个入口，按[工作区说明](../../README.md#先建立本地实验工作区)复制到experiments，再核对资源、镜像和绑定。原始日志、预热、拒绝尝试及遥测仍在各节点本机保留；未删除历史结果或JIT缓存。

每节点仍只归档一份报告和一份CSV。本批归档时，report/export默认只导出最终PASS case；47的C32独立接纳是本次显式复核，没有修改原始状态或当时的gate。合并不同CSV按列名读取：47新增来源、资格和更多延迟列，其他文件未提供的字段留空，不补零。用于本页对照的主要指标四份均齐全。

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

## 2026-09-17：单服务配置与复现

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

### 配置入口

本节描述2026-09-17历史实验。各候选的参数差异、GPU顺序和rank映射见节点报告，完整配置见[历史快照](https://github.com/guoenhui-tora/llm-serving-benchmarks/tree/4642443b1274251b6a47f0d2742e0e59e2dccc95/projects/dsv4-rtx6000d/configs)。当前继续研究的TP4命令及配置入口统一放在[基线与复现](reports/reproduction.md)，不再保留四份节点campaign。

### 归档规则

每节点保留一个 `reports/topology-nodeNN.md` 和一个 `data/topology-nodeNN.csv`，报告说明差异、结果、异常并链接CSV；README整合结论。后续新批次使用新的主题文件名，不覆盖本批逐次数据。原始日志、预热、拒绝尝试、探索配置及临时脚本继续放本机 `experiments/`，不批量上传。报告写明实际命令、负载和验收规则；configs只保留当前基线及仍需复用的候选，并携带完整依赖。历史候选不逐项复制配置到当前目录。

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

TP8对比已经完成，参数与结果保留在报告，原三套recipe及campaign见[固定版本配置](https://github.com/guoenhui-tora/llm-serving-benchmarks/blob/4642443b1274251b6a47f0d2742e0e59e2dccc95/projects/dsv4-rtx6000d/configs/campaigns/48-dsv4-aligned-final-c16-c32.yaml)。它们不再作为当前基线重复维护。

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

当前八卡整机性能表已完成。后续先围绕双TP4基线及双TP2×PP2、双TP2×DP2 EP on候选处理波动，再考虑以下独立对照；具体预算另行确定，不自动追加实验。

- [ ] 核实并完善Triton缓存持久化、逐worker预热及日志时间归属，保持真实编译可观察。
- [ ] 比较vLLM `--numa-bind` 与现有Docker cpuset，记录worker映射和实际页面分布。
- [ ] 在同一引擎、同一拓扑下检查DSpark支持并比较开关收益，草稿模型计入整机资源。
- [ ] 多节点PD阶段分别筛选P、D拓扑，不把混合推理吞吐直接当作单独P或D能力。
