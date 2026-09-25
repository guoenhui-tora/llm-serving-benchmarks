# DSV4 / RTX 6000D: 24 卡四 P＋单八卡 TP2×DP4 D 的 C144 对照

## 结论与范围

固定 46/47/48 各八卡、四套四卡 TP2×DP2 P、全局 C144 和真实独立 GovReport 16K/1024 输入，只将 48 上**两个独立四卡 TP2×DP2 D**改为**单个八卡 TP2×DP4 D**，保持每 D 引擎 S64、FP8 KV、DSpark K5 和整机名义 D 序列容量 256。在相同完整客户端窗口，单八卡 D 三轮 **3769.06±65.07 output tok/s**，低于此前 [4P2D/C144](pd-24gpu-16k-4p2d-20260925.md) 的 **4120.98±8.80（−8.54%）**；Mean TTFT **8.223±0.078 s**，优于双四卡的 **9.190±0.064 s（−10.52%）**，但 Mean TPOT **28.038±0.603 ms** 对 **23.516±0.051（+19.23%）**，goodput **2.991±0.052** 对 **3.167±0.037 req/s（−5.58%）**。**未同时改善吞吐和 Mean TTFT**，不调整既有 4P2D 结论或 C144 工作点。

此前[八卡 decode-only](../docs/decode-only-8gpu-20260925.md)获胜的也是**双四卡 D**，不是单八卡 TP2×DP4；不过其共享前缀和客户端拓扑不同，不能代替此处真实 PD 检验。本次单八卡 D 的四个 DP 引擎均通过 P→D NIXL 传输、Graph 与完整三轮；其吞吐/TPOT 劣势是真实 PD 端到端测出的，不是因连接器不兼容、容量缩小或整段输入重算造成的。服务边界、D 内 DP/EP 组和代理 D 路由同时变化；相同 P 布置下 TTFT 变化也不能孤立归因于跨 NUMA 通信。

## 三种已测部署在 C144

三种部署均为同一 24 GPU、同 v0.30 image ID、同独立 16K 输入数据、同单客户端代理入口、全局 C144 和一轮 288 完整请求预热＋固定三轮各 576 条正式请求，**但分别属于独立启动批次**。表中每组 `±` 是三轮指标样本标准差；P95 是逐轮 P95 的均值；逐请求联合达标为 TTFT≤10 s 且 TPOT≤50 ms，同轮达标请求数除以完整客户端窗口时长得到 goodput。[5P1D/C144 原报告](pd-24gpu-16k-5p1d-c144-20260925.md)保留该组差异和启动问题。

| 24 卡拓扑 | output tok/s | Mean / P95 TTFT s | Mean / P95 TPOT ms | Mean / P95 E2EL s | 联合达标率 | goodput req/s |
|---|---:|---:|---:|---:|---:|---:|
| 四 P＋双四卡 D | **4120.98±8.80** | 9.190 / 26.510 | **23.516 / 28.448** | **33.247 / 52.232** | 78.70% | **3.167±0.037** |
| 四 P＋单八卡 TP2×DP4 D（本批） | 3769.06±65.07 | **8.223 / 26.459** | 28.038 / 34.382 | 36.906 / 56.215 | 81.25% | 2.991±0.052 |
| 五 P＋单四卡 D | 3452.64±35.25 | **6.426 / 21.857** | 32.851 / 39.643 | 40.033 / 55.280 | **82.58%** | 2.784±0.035 |

此处单八卡在吞吐、TTFT、TPOT 上都介于另两者之间，但**P95 TTFT 几乎等于双四卡**（26.459 vs 26.510 s）；“Mean TTFT 较低”不等于首 token 尾延迟已解决。5P1D 多一套 P 且只保留四卡 D，不能将它与另两组当作只改变 D 服务边界的消融。所有部署的逐请求联合达标率在 C144 均不到 95%；本批按事前 Mean TTFT/TPOT 停扫规则执行，95% 不是此次停止或补跑门槛。

## 本批逐轮结果与工作量

| 单八卡 D 正式轮 | output tok/s | Mean/P95 TTFT s | Mean/P95 TPOT ms | Mean/P95 E2EL s | 达标数 / 576 | goodput req/s | 完整窗口 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3733.68 | 8.236/26.474 | 28.383/34.441 | 37.272/56.652 | 469 | 2.969 | 157.974 |
| 2 | 3844.15 | 8.140/26.449 | 27.342/34.234 | 36.110/54.581 | 468 | 3.050 | 153.434 |
| 3 | 3729.34 | 8.294/26.455 | 28.389/34.469 | 37.336/57.410 | 467 | 2.953 | 158.158 |

[三轮全部原始指标 CSV](../data/pd24-single8d-c144-20260925-rounds.csv)不删中间较快轮：每轮 **576/576 成功、0 失败**，P 输入计算 **9,437,184** tokens、客户端输出 **589,824** tokens；正式合计 **1728/1728 成功、1,769,472 输出 tokens**。8 条 16-output-token 功能请求 **8/8**，288 条完整请求预热 **288/288**，预热输出 294,912 tokens；原始指标、逐请求记录和完整代理/服务日志仅在本机[独立 run-root](../../../experiments/dsv4-pd-24gpu-single-d-tp2dp4-c144-20260925/results/pd24-single-d-tp2dp4-c144-01/state.json)。[离线独立重算](../../../experiments/dsv4-pd-24gpu-single-d-tp2dp4-c144-20260925/results/pd24-single-d-tp2dp4-c144-01/analysis.json)验收三轮；正式 JIT/TileLang 日志匹配 **0/0/0**，功能窗口 P 各 8、D 24，完整预热 0。日志匹配 0 不证明没有任何未被观察到的编译或跨批次稳定性。

代理事件按 ID 和各自正式窗口逐轮严格配对 **576/576**：双四卡的 `start→prefill_done` 平均 **8.460/8.422/8.540 s**，本批单八卡 **7.460/7.305/7.488 s**；`prefill_done→done` 分别 **24.692/24.770/24.733 s** 和 **29.767/28.758/29.803 s**。P 保持原布置，前段缩短约 1.06 s，但后段延长约 4.71 s；代理时间包括等待与通信，**不是 P 排队时长或 GPU 纯计算计时**，不以此独立证明具体内部瓶颈。

四条 P→D 路由每正式轮分别有 **141–147** 条完成请求，[全路径核验](../../../experiments/dsv4-pd-24gpu-single-d-tp2dp4-c144-20260925/results/pd24-single-d-tp2dp4-c144-01/route-check.json)通过。D 四引擎每轮分别完成 `[147,141,145,143]`、`[142,142,148,144]`、`[141,145,145,145]`，均通过远端 KV 命中；每轮 D `external_prefix_cache_hits` **9,437,184 tokens**，`request_prefill_kv_computed_tokens` **0**，两个 TP rank 合计 **1152 次/约 99.29 GB** 成功传输，K5 draft/accepted 非零；失败传输/通知、KV 过期和抢占的增量均为 0。[每引擎计数样例](../../../experiments/dsv4-pd-24gpu-single-d-tp2dp4-c144-20260925/results/pd24-single-d-tp2dp4-c144-01/c144/round-1/engine-counter-deltas.json)及逐轮快照保留原值。

四 P 实际 target/draft Graph 均捕获 **13/13、14/14**；[八卡 D 启动日志快照](../../../experiments/dsv4-pd-24gpu-single-d-tp2dp4-c144-20260925/results/pd24-single-d-tp2dp4-c144-01/d0/startup-graph.json)为 **20/20、20/20**，按每引擎 S64 覆盖 target/draft **384/320**。正式阶段约 30 s 一次采样，21 个样本其中 15 个有活动：P 各引擎最高 waiting **13–15**；D 四引擎最高 running **36/35/35/35**、waiting **2/1/1/1**、KV 使用最高 **8.53%**。旧双四卡 D 四引擎 sampled running **33/35/30/31**、KV 最高 15.94%；新 D 采样没有达到 S64，但序列槽和 KV 有余**不能推出逐 token GPU 计算有余**。P 节点显存约 80,555–84,489 MiB/卡、D 节点约 70,873 MiB/卡；最高 GPU 利用率抽样 100%、温度 62°C。三个节点 `doca_spcx_cc` 后台 CPU 持续存在，旧批也有；启动日志的 out-of-order stats、SymmMem/FlashInfer/KVConnector warning 保留，未放宽 gate。

## 部署、实际命令与复现

四 P 和原 4P2D 的目标节点、GPU、CPU/NUMA、HTTP/NIXL/DP RPC 端口、recipe flags/options/env 结构化比对**一致**；D 在 48 用 GPU0–7、CPU0–15/32–47、NUMA0/2，TP2×DP4、本地 DP4、每引擎 S64、token budget16384、FP8 KV、EP on、V2 async、DSpark K5，Graph384/320，NIXL consumer；D 相对原 D0 的 recipe options **只改** `data-parallel-size` 和 `data-parallel-size-local` 从 2 到 4，仍用原 NIC 配置 `mlx5_4..7`。前四卡因此不使用最本地的 NIC，**没有做网卡选择消融**。代理在 45 的 `127.0.0.1:31580` 以 least-inflight 路由到四 P/单 D，客户端同用 CPU20–23/NUMA1。服务端/客户端固定 vLLM 0.30.0 image ID `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`，NVFP4 权重路径 `/data/models/DeepSeek-V4-Flash-0731-NVFP4`；启动前 image ID、服务端/客户端实际 CLI、模型 config/index/tokenizer 元数据、三节点 GPU/NUMA/NIC/端口已核对，**没有全量哈希权重内容**。

归档的是真实 `docker run` argv JSON：[P0](../data/pd24-single8d-c144-20260925-commands/p0.json)、[P1](../data/pd24-single8d-c144-20260925-commands/p1.json)、[P2](../data/pd24-single8d-c144-20260925-commands/p2.json)、[P3](../data/pd24-single8d-c144-20260925-commands/p3.json)、[八卡 D0](../data/pd24-single8d-c144-20260925-commands/d0.json)、[代理](../data/pd24-single8d-c144-20260925-commands/proxy.json)、[正式客户端第 1 轮](../data/pd24-single8d-c144-20260925-commands/client-c144-round1.json)，可用 `python3 -c 'import json,shlex,sys; print(shlex.join(json.load(open(sys.argv[1]))))' <command.json>` 展示 shell 命令；其他轮 argv 仍在本机 run-root。[冻结计划](../data/pd24-single8d-c144-20260925-commands/plan.json)与本机[运行前协议](../../../experiments/dsv4-pd-24gpu-single-d-tp2dp4-c144-20260925/reports/protocol.md)固定**只测全局 C144**：8 短功能请求、1 轮 2C=288 条完整预热、3 轮各 4C=576 条正式请求，最长 2016 条完整请求＋8 条短请求，READY/客户端窗口/整档/整个 campaign 上限分别 **2700/900/4200/8400 s**，性能重试 **0**；未加并发/轮次。dataset 使用[精确 GovReport 16K 文件](../datasets/govreport-isl16384-exact-n1024.jsonl)，SHA256 `6166561f98e83a7591eaa5bc36dcb317f20edd42f187a10f4cc6793df839b9b2`，每条由固定 tokenizer 校验为输入 16384 tokens、输出 1024、temperature=0、`ignore_eos`、`--skip-chat-template`、无限到达率，本地 prefix cache 关闭。

本机被 Git 忽略的[编排脚本](../../../experiments/dsv4-pd-24gpu-single-d-tp2dp4-c144-20260925/scripts/run.py)与[验收脚本](../../../experiments/dsv4-pd-24gpu-single-d-tp2dp4-c144-20260925/scripts/analyze.py)保留原始计划与验收入口。此处的命令是**本次执行记录，不是重跑建议**；计划和 run-root 不可覆盖，异机复现需准备原 G2 resolved.json、对应节点的 MXFP4 补丁/日志配置与持久 JIT 缓存，重新建立 owner/工作区并核对端口/NIC/镜像/模型。尤其不能照用已结束批次的 owner 或结果目录：

```bash
PYTHONPATH=src python3 experiments/dsv4-pd-24gpu-single-d-tp2dp4-c144-20260925/scripts/run.py prepare
PYTHONPATH=src python3 experiments/dsv4-pd-24gpu-single-d-tp2dp4-c144-20260925/scripts/run.py offline
PYTHONPATH=src:experiments/dsv4-pd-24gpu-single-d-tp2dp4-c144-20260925/scripts python3 -m unittest discover -s experiments/dsv4-pd-24gpu-single-d-tp2dp4-c144-20260925/scripts -p test_run.py
PYTHONPATH=src python3 experiments/dsv4-pd-24gpu-single-d-tp2dp4-c144-20260925/scripts/run.py preflight
PYTHONPATH=src python3 experiments/dsv4-pd-24gpu-single-d-tp2dp4-c144-20260925/scripts/run.py run --run-root experiments/dsv4-pd-24gpu-single-d-tp2dp4-c144-20260925/results/pd24-single-d-tp2dp4-c144-01
PYTHONPATH=src:experiments/dsv4-pd-24gpu-single-d-tp2dp4-c144-20260925/scripts python3 experiments/dsv4-pd-24gpu-single-d-tp2dp4-c144-20260925/scripts/analyze.py
```

最终[状态和清理证据](../../../experiments/dsv4-pd-24gpu-single-d-tp2dp4-c144-20260925/results/pd24-single-d-tp2dp4-c144-01/state.json)为 `COMPLETE`、`changed_files=[]`、`cleanup_errors=[]`，46/47/48 的 24 卡回到 0 MiB，只清理本批 owner 的五个服务及代理/客户端，没有删除模型、镜像、缓存或历史结果。

**决策：** 本机同 C144 真实 PD 对照中，双四卡 D 的整机吞吐、goodput、TPOT、E2EL 均较优；单八卡 D 只在 **Mean TTFT** 明确占优（P95 TTFT 几乎持平），不能作为“一步同时改善吞吐与 TTFT”的替换。若继续攻当前 4P2D 的双目标，下一步更值得**单独**验证保持双 D、只改善 P 供给/路由或 NUMA/NIC 选择的候选，而不是假定八卡单实例 D 会提速；这些后续优化**本次没有实测**，不能称为推荐配置。只完成一档三轮 quick，服务批次不同、背景 CPU 未消除、未测其他并发及跨启动稳定性，结论仅限当前固定 16K/1K/C144 条件。
