# 2026-09-22 夜间 PD 计划执行记录

## 结论

本轮没有产生新的性能数据。12 卡 TP2×DP2 EP on、DSpark K5 的首个普通基线在启动阶段未能让三台服务同时通过 READY，因此按停止规则没有发送压测请求，也没有启动 2P1D。已有的 8 卡 1P1D C32/C64 结果仍是历史配对结果，不能冒充本轮 12 卡结果。

## 尝试的候选

本轮使用 `experiments/dsv4-pd-dspark16k-c32c64/` 的独立 workspace，固定真实 GovReport 16,384 输入／1,024 输出、TP2×DP2、DP2、EP on、DSpark K5、FP8 KV、固定 vLLM `v0.29.0` 镜像和既定 warmup 组合。普通候选 N22O5 为 46/47/48 各一套四卡服务，C64、每轮 N=256；计划中的 PD 候选为同 GPU 规模的 2P1D。

N22O5 的 46 和 48 服务完成了预热并进入 API 启动阶段；47 的一个 DP worker 在固定镜像启动 profiling 中重复报错：

```text
vllm/forward_context.py:96, DPMetadata.make
AssertionError: 16384 160
```

这里 `16384` 是目标侧传入的跨 DP token 数，`160` 是 DSpark K5、32 个请求对应的草稿 token 数（32×5）。日志同时确认：

- 46/48 的 worker 日志出现 `UPSTREAM_DSPARK_DP_PROFILE_FIX: PR #54856`，并完成输入 kernel、mHC 和 DSpark warmup；47 的失败进程在这些 warmup 标记之前退出，虽然容器环境挂载了同一 overlay，但该进程没有记录补丁加载标记；
- 断言发生在 `gpu_worker.profile_run → model_runner._dummy_run → speculator.propose`，不是正式请求或 NIXL 传输阶段；
- 因此不能把本次失败归因于显存 OOM 或 NIXL 传输，也不能声称 DP profiling 修复已经覆盖了 47 的失败路径；
- 该证据只说明“16K budget + max-num-seqs=32 + DP2/K5 的组合在本次 47 启动路径未通过”，尚不足以断言所有节点或所有 budget 都必然失败。

失败发生在 READY 和任何 HTTP 性能请求之前，没有性能轮、KV 传输轮或吞吐样本。`nvidia-smi` 只读核查显示 46/47/48 八卡显存占用均为 0；本轮 owner 容器未残留。其他节点上的服务没有停止或修改。

## 今晚停止条件与后续分支

继续把 16384 盲目改成其他值会混淆“16K budget”研究问题，也可能把旧的 8192 兼容路径误当成同一配置。因此今晚不直接重试 N22O5，也不启动 2P1D、C32/C64 或 16 卡扩展。

下一次实施前先做一个独立的启动诊断，保持同一镜像、权重和补丁，只改变一个启动约束，并且只要求单服务 READY：

1. 复用已经通过的 TP2×DP2 K5 组合，逐项核对 `max-model-len`、每 DP engine 的 `max-num-seqs`、`max-num-batched-tokens` 和 DSpark Graph 上限；
2. 首先验证 `max-num-seqs=16`、`max-num-batched-tokens=8192` 的已知兼容路径，再单独增加 `max-num-seqs=32` 或 budget 16384；每个候选只做启动、日志和短功能请求，不计性能；
3. 只有单服务在目标 16K 上通过 profiling、Graph、真实请求和无整段重算验收，才恢复三服务普通基线；普通三服务通过后再启动 2P1D；
4. 若 16K budget 仍只在 47 失败，保留节点差异和完整调用栈，暂停跨节点性能扩展，不把 8K/8192 结果与 16K/16384 结果合并。

这条诊断预算为每候选约 20–30 分钟，最多两个候选；任何新的 `AssertionError`、服务未 READY、功能请求失败或工作量错误立即停止。性能阶段仍沿用原计划：先 12 卡普通三服务 C64，再 12 卡 2P1D C64；只有 P 队列证据支持时才补 C32 或 16 卡分支。

## 复现入口

- 原计划：[pd-night-20260922-plan.md](pd-night-20260922-plan.md)
- 失败 workspace：`experiments/dsv4-pd-dspark16k-c32c64/`（仅本机原始日志）
- 最新失败日志：[N22O5 r1 server.log](../../../experiments/dsv4-pd-dspark16k-c32c64/results/N22O5-01/r1/server.log)
- 控制日志：[N22O5 controller log](../../../experiments/dsv4-pd-dspark16k-c32c64/reports/N22O5-01-controller.log)
