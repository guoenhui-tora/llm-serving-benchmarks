# 2026-09-22 夜间 PD 计划执行记录

## 结论

首个 N22O5 候选确实因缓存缺少 DP profiling overlay 而失败；修正缓存准备逻辑后，N22O6/N22O7 普通三服务和 N22P6/N22P7 2P1D 均完成 16K、C64/C32 正式配对。四组都使用同一真实数据、12 卡 TP2×DP2 EP on、DSpark K5 和三轮 clean quick。

## 尝试的候选

本轮使用 `experiments/dsv4-pd-dspark16k-c32c64/` 的独立 workspace，固定真实 GovReport 16,384 输入／1,024 输出、TP2×DP2、DP2、EP on、DSpark K5、FP8 KV、固定 vLLM `v0.29.0` 镜像和既定 warmup 组合。普通候选 N22O5 为 46/47/48 各一套四卡服务，C64、每轮 N=256；计划中的 PD 候选为同 GPU 规模的 2P1D。

N22O5 的 46 和 48 服务完成了预热并进入 API 启动阶段；47 的一个 DP worker 在固定镜像启动 profiling 中重复报错：

```text
vllm/forward_context.py:96, DPMetadata.make
AssertionError: 16384 160
```

这里 `16384` 是目标侧传入的跨 DP token 数，`160` 是 DSpark K5、32 个请求对应的草稿 token 数（32×5）。根因不是补丁逻辑本身，而是 N22O5 的 47 节点 seed 缓存 `638faf7b431b04ccaa` 没有 `dspark-native-mxfp4` 目录；仅设置 `PYTHONPATH` 并不会生成缺失文件。日志同时确认：

- 46/48 的 worker 日志出现 `UPSTREAM_DSPARK_DP_PROFILE_FIX: PR #54856`，并完成输入 kernel、mHC 和 DSpark warmup；47 的失败进程没有记录该标记，因为缓存里根本没有 overlay；
- 断言发生在 `gpu_worker.profile_run → model_runner._dummy_run → speculator.propose`，不是正式请求或 NIXL 传输阶段；
- N22O6/N22P6 的每个新缓存都显式写入 12 个覆盖文件，三节点均出现补丁标记并通过启动 gate。
- 该证据只说明“16K budget + max-num-seqs=32 + DP2/K5 的组合在本次 47 启动路径未通过”，尚不足以断言所有节点或所有 budget 都必然失败。

N22O5 失败发生在 READY 和任何 HTTP 性能请求之前，没有性能轮、KV 传输轮或吞吐样本。`nvidia-smi` 只读核查显示 46/47/48 八卡显存占用均为 0；本轮 owner 容器未残留。其他节点上的服务没有停止或修改。

## 修复后正式配对结果

| 部署 | 正式输出吞吐 tok/s | Mean TTFT，s | Mean TPOT，ms | Mean E2EL，s | goodput，req/s | 联合 SLO 达标率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通三服务，共12卡 | **1579.45 ± 23.34** | 4.940 | 34.310 | 40.040 | 1.233 | 79.95% |
| 2P1D，共12卡 | **1942.67 ± 26.41** | 9.870 | 20.796 | 31.144 | 1.468 | 77.34% |
| 2P1D 相对普通 | **+23.00%** | +99.78% | −39.39% | −22.22% | +19.01% | −2.60 个百分点 |

两组均为一轮 256 条完整预热＋三轮 256 条正式请求，C64、真实 GovReport 精确 16,384 输入／1,024 输出。正式轮均 0 已知 JIT 事件、256/256 成功。PD 的吞吐和 goodput 明显高于普通，但 TTFT 约翻倍，联合 SLO 略低；因此当前结论是“吞吐/完整请求效率领先，交互首 token 质量仍需权衡”，不能称为无条件最优。

PD 的真实传输验收如下：每个正式轮两个 P→D API 路由各完成 128 条；D 的 `external_prefix_cache_hits` 为 4,194,304，等于 256×16,384；D 的 `request_prefill_kv_computed_tokens` 为 0；NIXL 每轮 512 次 rank 传输、约 44.13 GB、失败传输/通知/过期/抢占均为 0。由此排除了 D 重算整段输入。

结果产物：

- 普通：[N22O6 结果](../../../experiments/dsv4-pd-dspark16k-c32c64/results/N22O6-01/)
- PD：[N22P6 结果](../../../experiments/dsv4-pd-dspark16k-c32c64/results/N22P6-01/)
- 普通正式协议：[N22O6 protocol.json](../../../experiments/dsv4-pd-dspark16k-c32c64/results/N22O6-01/benchmark/c64/protocol.json)
- PD 正式协议：[N22P6 protocol.json](../../../experiments/dsv4-pd-dspark16k-c32c64/results/N22P6-01/benchmark/c64/protocol.json)

## C32 配对结果

N22O7/N22P7 保持 N22O6/N22P6 的 12 卡部署、真实 GovReport 精确 16,384 输入／1,024 输出、每 DP engine `max-num-seqs=32`、`max-num-batched-tokens=16384` 和 DSpark K5，只将全系统客户端并发从 C64 改为 C32；每组仍为一轮 256 条完整预热＋三轮 256 条正式请求。

| 部署 | 正式输出吞吐 tok/s | Mean TTFT，s | Mean TPOT，ms | Mean E2EL，s | goodput，req/s | 联合 SLO 达标率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通三服务，共12卡 | **1194.11 ± 8.53** | 2.642 | 23.541 | 26.724 | 1.139 | 97.66% |
| 2P1D，共12卡 | **1391.24 ± 6.51** | 5.022 | 17.222 | 22.640 | 1.261 | 92.84% |
| 2P1D 相对普通 | **+16.51%** | +90.13% | −26.84% | −15.28% | +10.75% | −4.82 个百分点 |

三轮正式轮全部 0 已知 JIT 事件、256/256 成功。2P1D 的每轮 P→D 路由均覆盖两条边；D 每轮 `external_prefix_cache_hits=4,194,304`，`request_prefill_kv_computed_tokens=0`，NIXL 512 次传输、约 44.13 GB，失败传输／通知／KV 过期／抢占均为 0。因此这组吞吐领先建立在真实 KV 传输上，不能解释为 D 重算输入。

结果产物：

- 普通：[N22O7 结果](../../../experiments/dsv4-pd-dspark16k-c32c64/results/N22O7-01/)
- PD：[N22P7 结果](../../../experiments/dsv4-pd-dspark16k-c32c64/results/N22P7-01/)
- 普通正式协议：[N22O7 protocol.json](../../../experiments/dsv4-pd-dspark16k-c32c64/results/N22O7-01/benchmark/c32/protocol.json)
- PD 正式协议：[N22P7 protocol.json](../../../experiments/dsv4-pd-dspark16k-c32c64/results/N22P7-01/benchmark/c32/protocol.json)

## 16 卡比例扩展

在 C32/C64 两档 12 卡配对均显示 2P1D 吞吐领先后，继续测了 16K/C64、同 TP2×DP2 EP on、DSpark K5、每 DP engine `max-num-seqs=32` 和 `max-num-batched-tokens=16384` 的 16 卡组。普通为 4 个完整服务；3P1D 为 3 个 P＋1 个 D；2P2D 为 2 个 P＋2 个 D。三组均一轮 256 条完整预热＋三轮 256 条正式请求。

| 部署 | 输出吞吐 tok/s | Mean TTFT，s | Mean TPOT，ms | Mean E2EL，s | goodput，req/s | 联合 SLO 达标率 | 相对普通吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通四服务，共16卡 | **1920.85 ± 10.87** | 4.223 | 27.931 | 32.796 | 1.639 | 87.37% | — |
| 3P1D，共16卡 | **2048.31 ± 12.65** | 6.701 | 22.319 | 29.533 | 1.649 | 82.42% | **+6.64%** |
| 2P2D，共16卡 | **2137.25 ± 1.93** | 12.438 | 15.439 | 28.232 | 0.579 | 27.73% | **+11.27%** |

3P1D 的吞吐只比普通高 6.64%，goodput 仅高 0.60%，TTFT 增加 58.70%，联合 SLO 下降 4.95 个百分点；2P2D 的输出吞吐再高 11.27%，但 TTFT 增至 12.44 秒，goodput 反而比普通低约 64.68%，联合 SLO 仅 27.73%。因此在当前 16K/C64/SLO 条件下，增加 D 可以继续提高完整窗口吞吐，却不能按吞吐单指标选择比例；3P1D 已是较弱的交互折中，2P2D 不具备当前 goodput 价值。

两组 PD 正式轮均 0 已知 JIT 事件、256/256 成功。3P1D 每轮三条 P→D 路由均有完成请求；2P2D 每轮四条 P→D 路由均有完成请求。两组 D 均 `request_prefill_kv_computed_tokens=0`，远端命中覆盖 4,194,304 输入 tokens；每个 D 服务的 NIXL 传输、失败计数和 KV 计数均通过验收。

第一次 N22P8 启动未进入 READY，原因是 46 节点两套 DP2 服务的 NIXL 基端口配置重叠：DP2 会占用基端口及下一个端口，`26300` 与 `26301` 相交并报 `Address already in use`。该尝试无功能／性能请求，owner 容器已清理；修正第二服务基端口为 `26302` 后以 N22P8-02 完成正式测量。

结果产物：

- 普通四服务：[N22O8 结果](../../../experiments/dsv4-pd-dspark16k-c32c64/results/N22O8-01/)
- 3P1D：[N22P8-02 结果](../../../experiments/dsv4-pd-dspark16k-c32c64/results/N22P8-02/)
- 2P2D：[N22P9 结果](../../../experiments/dsv4-pd-dspark16k-c32c64/results/N22P9-01/)
- 端口冲突日志：[N22P8-01 p0 server.log](../../../experiments/dsv4-pd-dspark16k-c32c64/results/N22P8-01/p0/server.log)

## 今晚停止条件与后续分支

继续把 16384 盲目改成其他值会混淆“16K budget”研究问题，也可能把旧的 8192 兼容路径误当成同一配置。N22O5 已停止取证；N22O6/N22P6 的 C64 和 N22O7/N22P7 的 C32 配对均已完成。

C32 与 C64 的 12 卡配对、以及 16 卡的 3P1D／2P2D 均已完成。结果共同说明：PD 可以提高完整窗口 output tok/s，但 TTFT、goodput 和联合 SLO 可能同步恶化；当前不再扩展 4P2D 或更多比例矩阵，先以 2P1D／3P1D 的吞吐—交互折中作为候选结论。

所有后续实验直接复用已通过的完整缓存准备方式；不要从没有 overlay 的旧 seed 复制。任何新的 `AssertionError`、服务未 READY、功能请求失败、NIXL 失败或工作量错误立即停止。

## 复现入口

- 原计划：[pd-night-20260922-plan.md](pd-night-20260922-plan.md)
- 失败 workspace：`experiments/dsv4-pd-dspark16k-c32c64/`（仅本机原始日志）
- 最新失败日志：[N22O5 r1 server.log](../../../experiments/dsv4-pd-dspark16k-c32c64/results/N22O5-01/r1/server.log)
- 控制日志：[N22O5 controller log](../../../experiments/dsv4-pd-dspark16k-c32c64/reports/N22O5-01-controller.log)
