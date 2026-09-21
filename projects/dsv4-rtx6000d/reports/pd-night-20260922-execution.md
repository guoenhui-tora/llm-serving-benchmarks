# 2026-09-22 夜间 PD 计划执行记录

## 结论

首个 N22O5 候选确实因缓存缺少 DP profiling overlay 而失败；修正缓存准备逻辑后，N22O6 普通三服务和 N22P6 2P1D 均完成 16K/C64 正式配对。两组都使用同一真实数据、12 卡 TP2×DP2 EP on、DSpark K5 和三轮 clean quick。

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

## 今晚停止条件与后续分支

继续把 16384 盲目改成其他值会混淆“16K budget”研究问题，也可能把旧的 8192 兼容路径误当成同一配置。因此今晚不直接重试 N22O5，也不启动 2P1D、C32/C64 或 16 卡扩展。

后续实验应直接复用 N22O6/N22P6 的完整缓存准备方式；不要再从没有 overlay 的旧 seed 复制。下一步优先做同一拓扑的 C32 配对，确认 C64 的吞吐收益是否以并发为条件：

1. 固定 N22O6/N22P6 的 `max-model-len=32768`、每 DP engine `max-num-seqs=32`、`max-num-batched-tokens=16384` 和 K5 Graph 覆盖；
2. 先做普通三服务 C32，再做 2P1D C32，仍各一轮完整预热＋三轮 quick；
3. 若 C32 仍保持吞吐领先，再根据 TTFT/SLO 决定是否测 16 卡 3P1D；若 C32 优势消失，优先分析 P 供给与并发，而不是扩大比例矩阵。

C32 配对预算约 25–35 分钟每组；任何新的 `AssertionError`、服务未 READY、功能请求失败、NIXL 失败或工作量错误立即停止。

## 复现入口

- 原计划：[pd-night-20260922-plan.md](pd-night-20260922-plan.md)
- 失败 workspace：`experiments/dsv4-pd-dspark16k-c32c64/`（仅本机原始日志）
- 最新失败日志：[N22O5 r1 server.log](../../../experiments/dsv4-pd-dspark16k-c32c64/results/N22O5-01/r1/server.log)
- 控制日志：[N22O5 controller log](../../../experiments/dsv4-pd-dspark16k-c32c64/reports/N22O5-01-controller.log)
