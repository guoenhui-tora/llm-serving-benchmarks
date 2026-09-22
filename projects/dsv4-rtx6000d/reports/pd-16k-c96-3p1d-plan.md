# 16K C96：16卡 3P1D 与 D 容量验证

## 目的

固定已经验证的 16K TP2×DP2 EP on、DSpark K5、真实 GovReport 链路，把客户端全系统并发固定为 C96，并先用 D 的每个 DP engine `max-num-seqs=32` 建立容量受限基线，再用 `48` 做容量对照。这样可以把“3P 是否能承接 C96”和“D 的运行序列上限是否压低吞吐”分开观察。

本轮以现有 [N22P8-02 16卡 3P1D C64](../../../experiments/dsv4-pd-dspark16k-c32c64/results/N22P8-02/) 为基线，不修改服务的 `max-num-seqs` 或 token budget。C96 会主动把一个 D 服务的代理在途数推到约 96，而当前 TP2×DP2 D 服务每个 DP engine 的 `max-num-seqs=32`，服务合计容量约 64；因此本轮可以直接观察多出的请求是否在 D 排队。

## 固定条件

- 16张 GPU，4个四卡服务：3P＋1D；每服务 TP2×DP2、EP on、DSpark K5。
- 每个 P DP engine：`max-num-seqs=32`、`max-num-batched-tokens=16384`。
- A 组 D 每个 DP engine：`max-num-seqs=32`（服务合计约64）；B 组 D 为 `48`（服务合计约96）。两组 D 的 `max-num-batched-tokens=16384` 不变。
- 真实 GovReport 前缀，精确 16,384 input tokens、1,024 output tokens，使用已准备的 1024 条池；每轮固定前 768 条，避免 C96 下请求量过少造成填充/排空偏差。
- 客户端 `max-concurrency=96`、`request_rate=inf`；代理使用已修复的无限上游连接池。
- 固定 vLLM `v0.29.0` 镜像、NVFP4 权重、FP8 KV、NIXL connector 和现有完整 warmup/cache；不清 JIT 缓存。
- 一轮完整 768 条 HTTP 预热，三轮各 768 条正式 quick；不重试、不挑选最快轮。

## 必须保存的证据

- output tok/s、request throughput、Mean/P95 TTFT、TPOT、E2EL、goodput、联合 SLO；
- P0/P1/P2/D 的每秒 `num_requests_running`、`num_requests_waiting` 和 `capacity/deferred`；
- D 的 KV cache 使用率、NIXL bytes/耗时/失败计数、远端命中和 `request_prefill_kv_computed_tokens`；
- DSpark 草稿/接受 token、JIT 事件、实际 argv、GPU/CPU/NIC 背景。

PD 端到端吞吐仍按完整窗口计时，包含代理、P、KV 传输、D 排队和生成。单独记录的 running/waiting 指标用于定位瓶颈，不能替代吞吐定义。

## 判定

- A 组 D 出现 `capacity` waiting，B 组应减少该等待：这是确认容量变量确实生效的条件；
- B 组吞吐上升且 TPOT 不恶化：才支持扩大 D 容量；
- B 组吞吐不升或下降、TPOT 上升：D 的更大实际批量代价超过了容量收益，应保留 D=32 作为该负载下的容量点；
- 两组 D capacity waiting 都很低而 P 仍排队：瓶颈在 P 供给、token 调度或 P→D 交接；
- 吞吐上升但 TTFT/goodput/SLO 明显恶化：记录为吞吐扩展，不能称为更好的交互部署。

出现 OOM、服务未 READY、请求失败、NIXL 错误、D 整段重算、工作量错误或持续 JIT 时停止该组并保留证据。若本轮结果无法区分 P 与 D，再补同样 C96 的 12卡 2P1D；不在首轮同时铺普通 C96 或其他拓扑。
