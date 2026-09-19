# 48节点：DSpark 拓扑与 K 值实验（2026-09-20）

**固定 vLLM 0.29.0 的 TP2×DP2＋DSpark 在启动 profiling 阶段阻塞，当前不能得出 K4/K5 收益、接受率或最佳 EP＋K 组合。** EP on／K5 已实测失败：草稿 forward 的 token 数为80，但收到目标 profiling 的DP token元数据8192，触发 `DPMetadata.make` 断言。补丁已加载，原生MXFP4草稿分派成功；失败发生在服务就绪、Graph捕获及正式请求之前，不是已证实的显存容量问题。

EP on／DSpark off 在45分钟预算内取得第9、11轮两个clean样本，协议为 **PARTIAL**；不能称为完整的三轮基线。EP off／DSpark off 同样在45分钟内仅取得第6、10轮两个clean样本，协议也为 **PARTIAL**。范围限定本机gpu-6000d-48、GPU4–7、GovReport前128条、总C32以及本次持久缓存和DOCA后台负载。

## 结果与完整逐轮数据

完整精度见[逐轮CSV](../data/dspark-k-sweep-node48.csv)：23行、44字段，保留21个完整轮和2个预算中断轮。K5启动失败及三个跳过配置均没有正式轮次，仅在报告中记录。中断轮没有raw/metrics，性能、成功/失败和token字段留空，`known_event_lines`留空表示该轮未完成事件检查；不能把它们解释为0。`accepted=true`仅表示协议已选中的轮次，case/protocol仍为PARTIAL。

以下每组n=2，仅为保留clean样本的描述性统计，**不是完成的基线，也不进入正式收益对照**。标准差为样本标准差。各P95列是轮次P95的平均，不能称为合并全部请求的P95。两个EP状态独立顺序启动，缓存和客户端CPU背景不同，不据此排出EP优胜者。

| 配置 | 最终状态 / clean n | output tok/s，均值±SD | CV | 相对同EP off收益 | Mean/P95 TTFT ms | Mean/P95 TPOT ms | 草稿接受率 / 平均接受长度 |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| TP2×DP2 EP on / off | PARTIAL / 2 | 737.70 ± 13.09 | 1.77% | —（基线未完成） | 5044.10 / 11824.28 | 34.972 / 37.431 | — / —（off不适用） |
| TP2×DP2 EP off / off | PARTIAL / 2 | 754.78 ± 37.40 | 4.95% | —（基线未完成） | 5468.00 / 13856.27 | 35.738 / 38.996 | — / —（off不适用） |
| TP2×DP2 EP on / K5 | 启动FAIL / 0 | — | — | — | — | — | — / —（未进入测量） |
| TP2×DP2 EP off / K5 | 共性阻塞，未启动 | — | — | — | — | — | — / — |
| TP2×DP2 EP on / K4 | 共性阻塞，未启动 | — | — | — | — | — | — / — |
| TP2×DP2 EP off / K4 | 共性阻塞，未启动 | — | — | — | — | — | — / — |

保留clean样本各指标的均值±样本标准差（两列均n=2）：

| 指标 | EP on / off | EP off / off |
| --- | ---: | ---: |
| 输出 tok/s | 737.699 ± 13.092 | 754.781 ± 37.398 |
| 请求/s | 0.720 ± 0.013 | 0.737 ± 0.037 |
| Mean TTFT ms | 5044.098 ± 395.971 | 5468.004 ± 48.270 |
| P95 TTFT ms | 11824.281 ± 463.893 | 13856.275 ± 372.940 |
| Mean TPOT ms | 34.972 ± 0.116 | 35.738 ± 0.121 |
| P95 TPOT ms | 37.431 ± 0.194 | 38.996 ± 0.026 |
| Mean ITL ms | 34.972 ± 0.116 | 35.738 ± 0.121 |
| P95 ITL ms | 28.434 ± 0.127 | 27.068 ± 0.029 |
| Mean E2EL ms | 40820.297 ± 514.257 | 42027.833 ± 172.073 |
| P95 E2EL ms | 47971.610 ± 583.074 | 50495.356 ± 554.375 |
| 客户端计时 s | 177.705 ± 3.154 | 173.869 ± 8.615 |

四个保留clean样本均为128成功、0失败，输入1,042,149、输出131,072 tokens；这些固定计数的样本标准差为0。两组所有21个完整轮合计 **2688成功、0失败，输入21,885,129、输出2,752,512 tokens**。这个合计不含功能探测和两个未完成轮，不能据此宣称全部已发起请求无失败/取消。中断轮的部分请求只存在客户端进度和服务日志，未拼成完整样本。

| run_id | 轮次 | 输出 tok/s | 已知事件行数 | 接纳 | 说明 |
| --- | ---: | ---: | ---: | --- | --- |
| epon-off-01 | 1 | 382.89 | 86 | 否 | 事件轮，拒绝 |
| epon-off-01 | 2 | 579.00 | 30 | 否 | 事件轮，拒绝 |
| epon-off-01 | 3 | 529.81 | 40 | 否 | 事件轮，拒绝 |
| epon-off-01 | 4 | 584.64 | 24 | 否 | 事件轮，拒绝 |
| epon-off-01 | 5 | 751.86 | 18 | 否 | 事件轮，拒绝 |
| epon-off-01 | 6 | 761.51 | 2 | 否 | 事件轮，拒绝 |
| epon-off-01 | 7 | 660.16 | 8 | 否 | 事件轮，拒绝 |
| epon-off-01 | 8 | 726.95 | 16 | 否 | 事件轮，拒绝 |
| epon-off-01 | 9 | 746.96 | 0 | 是 | clean，但整组PARTIAL |
| epon-off-01 | 10 | 755.47 | 12 | 否 | 事件轮，拒绝 |
| epon-off-01 | 11 | 728.44 | 0 | 是 | clean，但整组PARTIAL |
| epon-off-01 | 12 | — | — | 否 | 预算中断，无完整metrics |
| epoff-off-01 | 1 | 342.15 | 78 | 否 | 事件轮，拒绝 |
| epoff-off-01 | 2 | 502.28 | 56 | 否 | 事件轮，拒绝 |
| epoff-off-01 | 3 | 705.89 | 4 | 否 | 事件轮，拒绝 |
| epoff-off-01 | 4 | 641.34 | 16 | 否 | 事件轮，拒绝 |
| epoff-off-01 | 5 | 515.56 | 38 | 否 | 事件轮，拒绝 |
| epoff-off-01 | 6 | 728.34 | 0 | 是 | clean，但整组PARTIAL |
| epoff-off-01 | 7 | 638.25 | 12 | 否 | 事件轮，拒绝 |
| epoff-off-01 | 8 | 685.06 | 6 | 否 | 事件轮，拒绝 |
| epoff-off-01 | 9 | 640.80 | 12 | 否 | 事件轮，拒绝 |
| epoff-off-01 | 10 | 781.23 | 0 | 是 | clean，但整组PARTIAL |
| epoff-off-01 | 11 | — | — | 否 | 预算中断，无完整metrics |

事件数为匹配日志行数，不等于独立编译次数。前三轮EP on均存在TileLang实际编译起止；EP off第2轮亦有14组起止。即使中途出现clean轮，后续仍可能出现新编译，不能把单轮无事件当作已经收敛。所有拒绝及中断轮留在CSV和本机原始结果，未修改gate。

## 协议时间成本

| run_id | run内预检至启动 s | wait_ready至就绪 s | 协议墙钟 s | 启动后至失败 s | run总耗时 s / min |
| --- | ---: | ---: | ---: | ---: | ---: |
| epon-off-01 | 44.745 | 314.194 | 2700.331 | — | 3067.158 / 51.119 |
| epon-k5-01 | 44.624 | —（未就绪） | — | 187.330 | 231.954 / 3.866 |
| epoff-off-01 | 44.664 | 332.460 | 2700.369 | — | 3081.920 / 51.365 |

run总耗时包含内部预检、启动、功能探测、协议与取证清理；外部validate/plan/preflight、准备补丁和离线取证不计入。协议每组2700秒预算包含客户端初始化和检查，不只benchmark计时；超时清理允许少量尾差。EP on执行12轮（11完整＋1中断），EP off执行11轮（10完整＋1中断），均因`time_budget`停止，没有取得规定的三轮。启动耗时与协议总耗时只按run列出，没有重复计入每一行CSV。

## 资源与日志验收

EP on/off两个普通推理run的kernel检查均PASS，分别保留599/511条warning匹配；K5启动FAIL，保留24条warning，缺少后续autotune关闭及mixed tokens=16预热必需日志，不改判通过。保留SM120 SymmMem不可用、world_size=2的FlashInfer All Reduce禁用、FP8 KV scale风险、NVFP4实验性、FP4 indexer参数弃用、TileLang循环串行回退和共享内存等待提示。K5退出还保留共享内存清理警告。没有OOM证据；不等同于完成数值质量验证。

实际off decode Graph捕获完成，列表为[1,2,4,8,12,16]。EP on每rank KV等效容量78,014 tokens，日志可用显存29.89 GiB/卡；EP off为78,506 tokens、30.07 GiB/卡。容量是DSV4压缩缓存的等效token，不是普通全注意力容量。每秒左右采样的各rank KV观测峰值分别23.688%和23.447%，不是连续精确峰值；两组完整采样中各rank累计preemptions始终0，逐轮增量0。实际可观察到两rank各16个活动请求，总容量32，prefill8192是每rank预算，不能声称与TP4的聚合调度预算相同。

资源约每10秒采样，以下CPU忙碌率按每个阶段内首末`/proc/stat`增量计算，包含客户端初始化和检查，并非纯benchmark窗口。集合包含后台任务，不代表本次进程独占率。

| run / clean轮 | 整机 % | 服务32–47 % | SMT96–111 % | 客户端48–51 % | SMT112–115 % | DOCA逻辑CPU当量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| epon-off-01 / 9 | 14.01 | 36.21 | 0.19 | 33.02 | 14.77 | 12.00 |
| epon-off-01 / 11 | 14.07 | 36.44 | 0.23 | 30.08 | 11.61 | 12.00 |
| epoff-off-01 / 6 | 13.99 | 36.56 | 0.23 | 60.24 | 0.12 | 12.00 |
| epoff-off-01 / 10 | 14.12 | 36.71 | 0.22 | 16.66 | 0.19 | 12.00 |

12个`doca_spcx_cc`持续各约1个逻辑CPU，未见新增矿工式后台任务；cgroup与逐进程增量均保留。客户端及SMT集合负载差异明显（EP off clean两轮客户端忙碌率60.24%/16.66%），不能称为无干扰、等背景对照，也不按假设比例补偿吞吐。cpuset只约束本次进程，不排除后台线程迁移及SMT竞争。

四个clean阶段的GPU4–7采样温度47–62°C、SM频率2407–2430 MHz、功耗80.52–311.86 W，含轮次准备/空闲区间；未见持续热降频证据，低频采样不能排除瞬时影响。EP on显存79,355–80,091 MiB，EP off为79,899–80,531 MiB。未修改宿主锁频、功耗或NUMA设置。

Docker inspect和服务/客户端线程允许集合检查通过。宿主观察器无法读取root worker的`numa_maps`，EP on缺少实际页分布证据；EP off另在启动期从本次容器内部读取四个worker页面快照，各worker约79.6%–79.9%的已映射驻留页在NUMA2，约19.3%–19.6%在NUMA3，另有少量NUMA0/1页。该口径含文件/共享映射，不能跨worker相加视为独占物理页，也不证明测量期持续分布。允许集合为NUMA2不等于所有映射页均驻留NUMA2。

缓存起点：两个off recipe目录均为空，K5缓存仅4个补丁文件（9079字节）；这是新recipe隔离目录的状态，没有删除、复制覆盖或清空其他持久JIT缓存。新启动的进程内加载与首次磁盘编译成本均包含在本次预算中，不与历史热缓存数据混合解释。观察器在K5启动及EP off期间各留下一次Docker snapshotter `lstat`统计提示；采样继续、服务状态由runner单独验证，未将提示隐藏或当作模型失败原因。

## 完成范围与停止决定

六份本地配置全部validate、plan与实际CLI preflight通过。EP on／K5是唯一实际启动的DSpark配置。固定镜像源码中，`set_forward_context` 的DP元数据分支条件为DP>1、MoE模型且存在token数，不检查EP开关；DSpark经DFlash基类传递未按草稿token数重建的DP元数据。因此按本轮“共性DP／草稿路径故障停止其余相关K”的规则，跳过EP off／K5和EP on/off／K4。K4的草稿上限64与8192也不同属于源码推断，**没有把跳过配置写成实测失败**。

保留主模型NVFP4、草稿原生MXFP4与所有warning，未修改镜像、公共runner或gate，未关闭Graph、更换采样方式或缩减prefill预算规避断言。随后完成独立可运行的EP off基线预算内测量。核心矩阵没有全部完成，且不存在有效的EP＋K候选，故不满足条件扩展前提，没有K6/K7或独立重启补测。

## DSpark接受统计边界

固定镜像指标源码已保存：`v1/spec_decode/metrics.py` 的计数器为 `vllm:spec_decode_num_draft_tokens_total`、`vllm:spec_decode_num_accepted_tokens_total`、`vllm:spec_decode_num_drafts_total`，逐位置为 `vllm:spec_decode_num_accepted_tokens_per_pos_total`。服务端标签为`model_name`和`engine`（DP rank）；逐位置另含`position`。

预定口径是每轮计数器末值减初值，先按DP rank检查连续性，再累加分子分母：接受率为100×Σ接受草稿token／Σ提出草稿token；平均接受长度为1＋Σ接受草稿token／Σ验证步数，含目标补充token；逐位置保存计数而非百分比，若计算比例分母为对应验证步数。每秒保存原始`/metrics`快照，窗口不得混入功能探测，缺失或重置不得跨窗口相减。

**本轮DSpark没有进入请求测量，所有接受率、接受长度、草稿计数和逐位置字段均留空。** off没有草稿，这些字段也不适用；不能把缺失写成0，不能由K或启动日志猜测接受率。因此本轮无法回答接受长度能否抵消草稿与验证开销，也不是模型质量评测。DSpark的ITL即使获得，也应解释为流式事件间隔而非逐token计算时间。

## 失败证据

实测调用链为`determine_available_memory → profile_run → _dummy_run → speculator.propose → DSpark._generate_draft → DFlash._run_model → set_forward_context → DPMetadata.make`。最后一层为固定镜像`forward_context.py:96`：

```python
assert num_tokens_across_dp_cpu[dp_rank] == batchsize
# AssertionError: 8192 80
```

四个worker均出现同一断言，两DP engine启动失败。日志在此之前确认四个worker的`LOCAL_DSPARK_FORMAT_FIX: validated 768 native MXFP4 draft experts`，草稿后端为`DEEPGEMM_MXFP4`，目标NVFP4路径保留。Graph计划覆盖正确不等于实际捕获完成：K5在profiling阶段已失败，因此不能声称其Graph已验证。

本机原始工作区为`experiments/dspark-k-sweep-node48/`，**仅本机可用**。`evidence/dp-draft-failure-source.json`保存固定镜像相关源码，`evidence/blocked-configurations.json`保存停止范围与理由；失败run的完整日志、镜像/模型身份、实际argv及解析配置均保留。没有访问或操作其他节点。

## 固定条件与复现

执行前工作区干净，`git pull --ff-only`成功，`git merge-base --is-ancestor 397e9d4 HEAD`退出码0；执行HEAD为`397e9d42dedee00c181c5fb10f5686ace069c111`。runner源码指纹`f6e873002b8cf61963bd221d63b0fe0a520e0bc6118f4ca63acfa54fc6726d52`。服务与客户端固定镜像ID为`sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`（vLLM 0.29.0），不拉取镜像。

模型路径`/data/models/DeepSeek-V4-Flash-0731-NVFP4`，元数据/tokenizer及分片大小身份`1e1b7e54442bcf7f15ef55192641b5c5da360bcaa608236aa0d06b273aedf8b4`；未全量哈希权重。数据使用[GovReport JSONL](../data/govreport-near8k.jsonl)固定前128条，SHA256为`33bdbd36a8b335300974e1b07aa66b2c24c88cc4c4b55a21350aa3564a945d53`。补丁固定于同一Git提交的[manifest](../patches/dspark-native-mxfp4/manifest.json)，`dspark_native_mxfp4.py` SHA256为`33df649eec0abfe5350fafc18dc84bcbd492238e65b9652a13fcea9e3bdf4730`；没有新增DP修复。

TP2×DP2、PP1，`data-parallel-size-local=2`；每rank活动容量16、prefill预算8192，全服务目标容量32，DP聚合prefill上限16384。上下文16384、显存比例0.90、FP8 E4M3 KV、block size256、V2＋async、FULL_DECODE_ONLY，关闭prefix cache和FlashInfer autotune，保留JIT缓存。DP由一个服务内部管理，不使用runner的`replica_targets`。固定GPU4–7；服务CPU32–47/NUMA2，客户端CPU48–51/NUMA3，SMT兄弟为96–111和112–115。

所有测量流式调用`/v1/completions`，全服务C32，每轮128请求、固定输出1024，seed0、temperature0、ignore_eos=true、request_rate=inf。每完整轮要求输入1,042,149、输出131,072 tokens，128成功/0失败；实际tokenizer重新计数。短中文chat探测关闭thinking，独立于测量；客户端不额外执行warmup请求。`jit_clean`同次启动累计最先三轮clean，最多12轮/2700秒，每客户端阶段900秒且受剩余预算约束，启动上限1800秒；无独立预热或因JIT重启。

本地配置由[单套TP2×DP2 recipe](../configs/recipes/dual-tp2-dp2-epon.yaml)、[DSpark recipe](../configs/recipes/tp4-dspark-k5.yaml)、[rear target](../configs/targets/rear.yaml)、[jit_clean workload](../configs/workloads/govreport-c32-jit-clean.yaml)和[单case campaign结构](../configs/campaigns/dspark-k5-quick.yaml)派生，完整副本在工作区。六个campaign各一个case，保留原归档recipe；on合并DSpark环境和必需日志规则，并分别运行[prepare.py](../patches/dspark-native-mxfp4/prepare.py)准备独立缓存。以下参数表直接取各自冻结plan，缓存路径按固定image ID解析（preview plan中的tag缓存路径不是实测路径）。

| configuration | EP开关 | K | Graph捕获列表 | 上限 | 固定镜像对应缓存目录尾段 |
| --- | --- | ---: | --- | ---: | --- |
| tp2-dp2-epon-off | `enable-expert-parallel` | 0 | `[1, 2, 4, 8, 12, 16]` | 16 | `c5463738c5fe22b6114d` |
| tp2-dp2-epon-k5 | `enable-expert-parallel` | 5 | `[5, 6, 10, 12, 20, 24, 40, 48, 60, 72, 80, 96]` | 96 | `3b36c466644ca37f1a6d` |
| tp2-dp2-epoff-off | `no-enable-expert-parallel` | 0 | `[1, 2, 4, 8, 12, 16]` | 16 | `196119661d88c1c7ce91` |
| tp2-dp2-epoff-k5 | `no-enable-expert-parallel` | 5 | `[5, 6, 10, 12, 20, 24, 40, 48, 60, 72, 80, 96]` | 96 | `d1cae908bd412c2762e8` |
| tp2-dp2-epon-k4 | `enable-expert-parallel` | 4 | `[4, 5, 8, 10, 16, 20, 32, 40, 48, 60, 64, 80]` | 80 | `c5119d9d126480164e54` |
| tp2-dp2-epoff-k4 | `no-enable-expert-parallel` | 4 | `[4, 5, 8, 10, 16, 20, 32, 40, 48, 60, 64, 80]` | 80 | `e78954517d593cec1e60` |

缓存根为`/home/enhui/.cache/serving-bench/vllm/`。on额外设置`PYTHONPATH=/root/.cache/dspark-native-mxfp4`；`speculative-config`固定`method=dspark`、`draft_sample_method=greedy`、`rejection_sample_method=standard`、`enable_adaptive_verification=false`，仅`num_speculative_tokens`取表中K。off不含草稿补丁。Graph列表由B=[1,2,4,8,12,16]分别乘K及K+1后合并；K5目标/草稿覆盖96/80，K4为80/64。on的计划和CLI均检查过，因前述阻塞没有实际Graph完成证据。

下面是从本次EP on／DSpark off基线`command.sh`原样导出的完整服务命令。其他配置按上表替换EP、Graph、缓存和DSpark参数；重新运行应由对应campaign生成新容器名，不复用历史owner。实际服务使用image ID。

<details><summary>EP on／DSpark off实际启动命令</summary>

```bash
docker run -d --pull never --name sb-3f6d0f649a0a45a7-server --label io.serving-bench.run=3f6d0f649a0a45a7 --network host --ipc host --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/vllm/c5463738c5fe22b6114d:/root/.cache:rw --cpuset-cpus 32-47 --cpuset-mems 2 --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --cap-add SYS_NICE -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model --served-model-name deepseek-v4-flash --host 127.0.0.1 --port 31249 --trust-remote-code --enable-auto-tool-choice --no-enable-prefix-caching --enable-expert-parallel --enable-chunked-prefill --jit-monitor-verbose --async-scheduling --tensor-parallel-size 2 --pipeline-parallel-size 1 --data-parallel-size 2 --max-model-len 16384 --max-num-seqs 16 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8_e4m3 --block-size 256 --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' --moe-backend auto --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' --kernel-config '{"enable_flashinfer_autotune":false}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,12,16],"max_cudagraph_capture_size":16}' --seed 0 --distributed-executor-backend mp --data-parallel-size-local 2
```

</details>

## 本机原始产物与执行入口

实际独立run顺序为`epon-off-01 → epon-k5-01 → epoff-off-01`，北京时间2026-09-20 04:19:50–06:09:39；间隔包含取证，没有其他服务并行测量。每次均在tmux内通过本地观察包装器调用未修改的bench，显式使用新的run-root，例如：

```bash
./bench run experiments/dspark-k-sweep-node48/configs/campaigns/tp2-dp2-epon-off.yaml \
  --run-root experiments/dspark-k-sweep-node48/results/epon-off-01
./bench run experiments/dspark-k-sweep-node48/configs/campaigns/tp2-dp2-epon-k5.yaml \
  --run-root experiments/dspark-k-sweep-node48/results/epon-k5-01
./bench run experiments/dspark-k-sweep-node48/configs/campaigns/tp2-dp2-epoff-off.yaml \
  --run-root experiments/dspark-k-sweep-node48/results/epoff-off-01
```

以上是历史路径快照，重跑必须换新run-root，不复用已有目录。各campaign运行前已执行validate、plan、prepare_logging和preflight；on还执行补丁prepare。其他三个候选只完成准备和预检，没有bench run。复现相同配置应从本报告固定Git版本及前述公共文件重新派生，保留完整依赖和数据；这些探索配置没有复制进精选configs。

原始目录**仅本机可用**：`experiments/dspark-k-sweep-node48/`。`configs/`与`data/`保留冻结输入；`results/<run>/`包含解析配置、完整argv/command、模型/镜像身份、GPU拓扑、亲和性、probe、完整服务日志、每轮raw/metrics/dataset manifest及protocol；`results/<run>-launch/`保存每秒原始metrics、资源/cgroup/逐进程时间线、缓存起点、源码哈希和runner日志。`evidence/`保留全部六组plan/preflight、失败源码快照、停止决定、NUMA快照与结束状态；`tools/`保留观察和归档脚本，`reports/analysis.json`保存从逐轮原始结果重算的统计。

结束后源码/配置冻结哈希全部一致，本次容器全部清理、八卡恢复空闲，模型、镜像、历史结果和缓存保留。归档仅本报告、完整CSV及README中48的完成标记；不修改其他节点或总表，不将探索脚本/配置批量提交。
