# 16K C96 3P1D：D 容量对照结果

## 结论

> 2026-09-23修正：本报告D96使用的Graph上限仍为192，未覆盖每引擎48条K5请求的288-token目标验证与240-token草稿形状。本地8K/C80补齐Graph后TPOT由35.86降至10.06ms，见[Graph核查与验证](decode-graph-coverage-20260923.md)。以下历史数字保留，但容量建议仅适用于当时Graph配置，不能认定D96有固有性能劣势；修正后的16K PD现已完成[同协议Graph及容量复测](pd-16k-graph-capacity-20260923.md)，本报告仍保留原测量协议和数据。

在真实 GovReport 16,384 输入／1,024 输出、TP2×DP2、EP on、DSpark K5、16 卡 3P1D、客户端 C96 下，D 每个 DP engine 的 `max-num-seqs=32`（服务合计约 64）是一个真实容量限制：两个 D engine 都达到 running=32，并出现 `capacity` waiting。

把 D 上限提高到每个 DP engine 48（服务合计约 96）后，capacity waiting 基本消失，但完整窗口 output throughput 从 **2441.70±3.82** 降到 **1567.52±23.21 output tok/s**；Mean TPOT 从 **24.23±0.06 ms** 升到 **54.67±0.86 ms**，联合 TTFT≤10 s、TPOT≤50 ms 的 goodput 也没有改善。因此在当时Graph配置下，D=64 的吞吐与 TPOT 更好；D=96 存在未扩展 Graph 覆盖这一混杂因素，不能单独归因于容量。

这两组都完成 768/768 请求、三轮正式测量均无事件，NIXL 每轮 1,536 次传输全部成功，D 没有 prefill 重算，结果描述当时容量与Graph配置的共同效果。

## 实验条件

| 组 | P | D | 总资源 | 工作量 |
|---|---|---|---:|---|
| N22P12-01 | 3 个四卡服务；每 DP engine `S=32`, `T=16384` | 1 个四卡服务；每 DP engine `S=32`, `T=16384`，合计约64 | 16 GPU | C96，768 warmup + 3×768 quick |
| N22P13-01 | 同上 | 每 DP engine `S=48`, `T=16384`，合计约96 | 16 GPU | 同上 |

所有服务使用固定 vLLM 0.29.0 镜像、NVFP4 权重、FP8 KV、NIXL connector、DSpark K5 和同一 1,024 条数据池前 768 条。实际启动命令保存在各组结果的 `*/command.json`，配置入口为 `experiments/dsv4-pd-dspark16k-c32c64/N22P12.json` 和 `N22P13.json`。

`max-num-seqs` 是每个 DP engine 的上限。DP2 服务均衡时两个 engine 的服务级上限约为 `2×S`，所以 S=32 对应约64，S=48 对应约96；C96 是代理覆盖 P、传输、D 和排队的全系统在途请求数，不要求每个 engine 都设置为96。第一组故意保留约64的 D 容量，是为了在 C96 下把容量等待压出来；第二组才将服务级容量提高到约96。

## 正式结果

在相同 D=64 配置下补测 C80，使用768条完整预热和三轮768条正式请求。已有 C64 结果来自同拓扑的先前 C64/N256 批次，因此这里把 C80 作为中点验证，C64 的严格同协议复测暂不重复。

| C | D 每个 DP engine `S` | output tok/s | req/s | Mean TTFT | P95 TTFT | Mean TPOT | P95 TPOT | Mean E2EL | request goodput | 联合SLO通过率 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | 32 | **2447.57±4.93** | 2.3902±0.0048 | **7.788±0.005 s** | 15.635±0.017 s | 24.06±0.02 ms | 28.38±0.49 ms | **32.400±0.023 s** | **2.103** | **87.97%** |
| 96 | 32 | 2441.70±3.82 | 2.3845±0.0037 | 13.812±0.030 s | 20.377±0.110 s | 24.23±0.06 ms | 28.80±0.46 ms | 38.604±0.094 s | 0.202 | 8.46% |
| 96 | 48 | 1567.52±23.21 | 1.5308±0.0227 | 5.284±0.059 s | 20.361±0.114 s | 54.67±0.86 ms | 66.96±1.14 ms | 61.209±0.860 s | 0.292 | 19.40% |

C80 与 C96/D64 的吞吐几乎相同（+0.24%），但 Mean TTFT 降低约43.6%，联合SLO达标率从8.46%提高到87.97%；D 的 capacity waiting 也从约3–4条/engine降到约1条。C80 是当前负载下的并发拐点候选。D=96 的 TTFT 均值进一步下降，但 TPOT 超过50 ms门槛，且端到端完成时间和 output throughput 变差。这说明只看 TTFT 会误判容量扩展；必须同时看完整窗口吞吐、TPOT 和 goodput。

## 队列与传输证据

正式窗口每秒 Prometheus 采样的均值／最大值如下；D 的数值按单个 DP engine 展示。

| C / D S | D running | D capacity waiting | D deferred waiting | D KV 使用率峰值 |
|---|---:|---:|---:|---:|
| C80 / 32 | 23.78–24.50 / 32 | 0.69–1.52 / 4–6 | 0.78–1.13 / 4–7 | 14–15% |
| C96 / 32 | 23.93–24.61 / 32 | 2.76–4.38 / 11–14 | 2.82–4.53 / 11–13 | 16–17% |
| C96 / 48 | 36.67–37.73 / 48 | 0.00–0.06 / 0–1 | 0.31–0.42 / 2–3 | 19% |
P 三个服务的 running 仍只有约 0.7–1.0（每 engine），但都有少量 capacity waiting；这是 16K 请求受 `max-num-batched-tokens=16384` 约束时的 token 调度现象。D=96 后 P 的供给约束没有消失，D 的序列容量等待减少；更大batch超出旧Graph覆盖的影响需单独验证。

每个正式轮 D 均报告 1,536 次 NIXL KV 传输、0 次失败；C80 的传输耗时总和约124–146 s，C96/D64约130–152 s，C96/D96约280–291 s。各组远端 KV 命中均为 12,582,912 tokens，D 端 `request_prefill_kv_computed_tokens_sum=0`，没有证据表明 D 重算了输入。DSpark 每轮约 554k accepted tokens，未出现正式事件。

## 后续使用边界

本结果回答“在 16K/16卡/3P1D/TP2×DP2/K5 下，C80/C96 与 D 约64/约96如何取舍”。它不能直接推出所有并发或输入长度下的最优 D 容量。后续应优先复测补齐 Graph 的 D96/C96，再决定是否继续调整并发、容量或 P/D 配比。不要把 D=96 的较低 TTFT 单独作为部署优势。

原始结果：

- [N22P14-01（C80，D=64）](../../../experiments/dsv4-pd-dspark16k-c32c64/results/N22P14-01/benchmark/c80/protocol.json)
- [N22P12-01（C96，D=64）](../../../experiments/dsv4-pd-dspark16k-c32c64/results/N22P12-01/benchmark/c96/protocol.json)
- [N22P13-01（D=96）](../../../experiments/dsv4-pd-dspark16k-c32c64/results/N22P13-01/benchmark/c96/protocol.json)
