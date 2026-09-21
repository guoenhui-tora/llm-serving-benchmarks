# 16K K5 输入预算与四槽验证

## 结论

在真实 GovReport 精确 16384 输入、1024 输出、TP2×DP2 EP on、DSpark K5、跨节点 1P1D、C32 条件下，P 的 `max-num-batched-tokens=16384` 确实因为 K5 预留 4 个输入槽而把单条输入拆成了 `16380+4` 两步；A 组仍稳定得到 **1145.96±2.50 output tok/s**（三轮 clean）。

原计划用 P budget 16640（B/B2）验证是否能把 16384 输入放进一次调度，但固定镜像配套的另一层 mHC 预热模块仍硬限制 `max_tokens <= 16384`。B-01 和 B2-01 都在发请求前退出，没有性能数据。因此本轮只能确认“4 槽造成了调度分块”，不能确认提高预算后的性能收益，也不能把启动失败解释为吞吐下降。

## A 组有效测量

固定条件：

- P：`max-num-batched-tokens=16384`，D：`16384`；双方 `max-num-seqs=32`（每个 DP engine）。
- TP2×DP2、EP on、DSpark K5；P 为 NIXL producer，D 为 consumer；46 和 48 各使用 GPU 4–7。
- GovReport 固定前 256 条，输入精确 16384、输出 1024；客户端全系统 C32；一轮 256 请求完整预热，三轮 256 请求 quick。
- 固定镜像 ID：`sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`。未清理 JIT 缓存、未修改镜像或权重。

| 指标 | 三轮均值±SD |
| --- | ---: |
| Output throughput | **1145.96±2.50 tok/s** |
| Mean TTFT | 11.939±0.154 s |
| Mean TPOT | 15.131±0.149 ms |
| Mean E2EL | 27.418±0.014 s |
| 成功／失败 | 768／0 |
| 已知编译事件 | 0 |

功能追踪在两个 DP rank 均完成，记录到首条输入的 `16380` 后 `4` 个 token 补步；P 端总 prefill token 数仍为 16384。D 端远端 KV 命中与 NIXL 传输验收沿用同组 PD gate，未见整段输入重算。

原始证据在本机忽略目录 [`experiments/dsv4-pd-dspark-budget/results/A-01/`](../../../experiments/dsv4-pd-dspark-budget/results/A-01/)，协议摘要见 `benchmark/c32/protocol.json`，调度追踪见 `functional-prefill-trace.json`。这些原始产物默认仅本机可用。

## B/B2 启动失败

| 单元 | P budget | 结果 | 根因 |
| --- | ---: | --- | --- |
| B-01 | 16640 | 发请求前失败 | 输入预热补丁主动范围检查仍为 `<=16384`；没有性能请求 |
| B2-01 | 16640 | 发请求前失败 | 修正输入预热检查后，`dsv4-mhc` 的 `warmup_plan.py` 仍报 `This warmup is scoped to DSV4 Flash, max_tokens <= 16384` |

B2 仅复制缓存并扩展了输入预热检查到 16640，其余服务参数、镜像、模型、数据和观测器均冻结。第二次失败说明要测 16640 还需扩展另一层固定预热覆盖；本轮到此停止，避免把预热补丁验证扩大成新的代码兼容性工作。完整失败日志见本机 [`experiments/dsv4-pd-dspark-budget/results/B2-01/p0/server-final.stdout.log`](../../../experiments/dsv4-pd-dspark-budget/results/B2-01/p0/server-final.stdout.log)。

## 对后续实验的含义

这组结果足以说明 K5 的 4 槽会改变 16K 输入的调度形态，但没有证据说明该补步是 16K PD 吞吐的主要瓶颈：A 的吞吐与此前同预算 1P1D C32 结果约 1146 tok/s 一致。若以后要测 16640，必须先单独完成 mHC 预热模块的范围审计和离线回归，再重新做完整功能验证；在此之前不应把 16384 与 16640 当作性能对照，也不应继续追加预算矩阵。
