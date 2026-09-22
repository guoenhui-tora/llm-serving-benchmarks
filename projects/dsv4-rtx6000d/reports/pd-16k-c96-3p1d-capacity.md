# 16K C96 3P1D：D 容量对照结果

## 结论

在真实 GovReport 16,384 输入／1,024 输出、TP2×DP2、EP on、DSpark K5、16 卡 3P1D、客户端 C96 下，D 每个 DP engine 的 `max-num-seqs=32`（服务合计约 64）是一个真实容量限制：两个 D engine 都达到 running=32，并出现 `capacity` waiting。

把 D 上限提高到每个 DP engine 48（服务合计约 96）后，capacity waiting 基本消失，但完整窗口 output throughput 从 **2441.70±3.82** 降到 **1567.52±23.21 output tok/s**；Mean TPOT 从 **24.23±0.06 ms** 升到 **54.67±0.86 ms**，联合 TTFT≤10 s、TPOT≤50 ms 的 goodput 也没有改善。因此当前负载下不能把 D=96 称为更优容量；D=64 是吞吐和 TPOT 更好的工作点，D=96 只证明了扩大容量会以更大的实际 decode batch 和更高 TPOT 为代价。

这两组都完成 768/768 请求、三轮正式测量均无事件，NIXL 每轮 1,536 次传输全部成功，D 没有 prefill 重算，结果可用于容量判断。

## 实验条件

| 组 | P | D | 总资源 | 工作量 |
|---|---|---|---:|---|
| N22P12-01 | 3 个四卡服务；每 DP engine `S=32`, `T=16384` | 1 个四卡服务；每 DP engine `S=32`, `T=16384`，合计约64 | 16 GPU | C96，768 warmup + 3×768 quick |
| N22P13-01 | 同上 | 每 DP engine `S=48`, `T=16384`，合计约96 | 16 GPU | 同上 |

所有服务使用固定 vLLM 0.29.0 镜像、NVFP4 权重、FP8 KV、NIXL connector、DSpark K5 和同一 1,024 条数据池前 768 条。实际启动命令保存在各组结果的 `*/command.json`，配置入口为 `experiments/dsv4-pd-dspark16k-c32c64/N22P12.json` 和 `N22P13.json`。

`max-num-seqs` 是每个 DP engine 的上限。DP2 服务均衡时两个 engine 的服务级上限约为 `2×S`，所以 S=32 对应约64，S=48 对应约96；C96 是代理覆盖 P、传输、D 和排队的全系统在途请求数，不要求每个 engine 都设置为96。第一组故意保留约64的 D 容量，是为了在 C96 下把容量等待压出来；第二组才将服务级容量提高到约96。

## 正式结果

| 组 | output tok/s | req/s | Mean TTFT | P95 TTFT | Mean TPOT | P95 TPOT | Mean E2EL | request goodput | 联合 SLO通过率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D=64 | 2441.70±3.82 | 2.3845±0.0037 | 13.812±0.030 s | 20.377±0.110 s | 24.23±0.06 ms | 28.80±0.46 ms | 38.604±0.094 s | 0.202 | 8.46% |
| D=96 | 1567.52±23.21 | 1.5308±0.0227 | 5.284±0.059 s | 20.361±0.114 s | 54.67±0.86 ms | 66.96±1.14 ms | 61.209±0.860 s | 0.292 | 19.40% |

D=96 的 TTFT 均值下降，但 TPOT 超过 50 ms 门槛，且端到端完成时间和 output throughput 变差。这说明只看 TTFT 会误判容量扩展；必须同时看完整窗口吞吐、TPOT 和 goodput。

## 队列与传输证据

正式窗口每秒 Prometheus 采样的均值／最大值如下；D 的数值按单个 DP engine 展示。

| 组 | D running | D capacity waiting | D deferred waiting | D KV 使用率峰值 |
|---|---:|---:|---:|---:|
| D=64 | 23.93–24.61 / 32 | 2.76–4.38 / 11–14 | 2.82–4.53 / 11–13 | 16–17% |
| D=96 | 36.67–37.73 / 48 | 0.00–0.06 / 0–1 | 0.31–0.42 / 2–3 | 19% |

P 三个服务的 running 仍只有约 0.7–1.0（每 engine），但都有少量 capacity waiting；这是 16K 请求受 `max-num-batched-tokens=16384` 约束时的 token 调度现象。D=96 后 P 的供给约束没有消失，D 只是从序列容量等待转成更大的 decode batch 运行代价。

每个正式轮 D 均报告 1,536 次 NIXL KV 传输、0 次失败；传输耗时总和 D=64 为约130–152 s，D=96 为约280–291 s。两组的远端 KV 命中均为 12,582,912 tokens，D 端 `request_prefill_kv_computed_tokens_sum=0`，没有证据表明 D 重算了输入。DSpark 每轮约 554k accepted tokens，未出现正式事件。

## 后续使用边界

本结果只回答“在 16K/C96/16卡/3P1D/TP2×DP2/K5 下，D 的运行容量从约64提升到约96是否有效”。它不能直接推出所有并发或输入长度下的最优 D 容量。若继续研究，应先固定 D=64，单独扫描较低 C 或比较 P/D 配比；只有在 TPOT 与 goodput 约束下仍有容量等待，才值得尝试其他 D 上限。不要把 D=96 的较低 TTFT 单独作为部署优势。

原始结果：

- [N22P12-01（D=64）](../../../experiments/dsv4-pd-dspark16k-c32c64/results/N22P12-01/benchmark/c96/protocol.json)
- [N22P13-01（D=96）](../../../experiments/dsv4-pd-dspark16k-c32c64/results/N22P13-01/benchmark/c96/protocol.json)
- [本轮计划](pd-16k-c96-3p1d-plan.md)
