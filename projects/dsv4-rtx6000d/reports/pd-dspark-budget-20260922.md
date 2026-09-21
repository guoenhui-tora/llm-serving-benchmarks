# 16K K5 输入预算与四槽验证

## 结论

在真实 GovReport 精确 16384 输入、1024 输出、TP2×DP2 EP on、DSpark K5、跨节点 1P1D、C32 条件下，P 的 max-num-batched-tokens=16384 确实因为 K5 预留 4 个槽而把单条输入拆成 16380+4 两步。把 P budget 提高到 16640 后，单请求可以在一次 prefill 调度中放下 16384 输入，但三轮正式吞吐为 1143.79±0.95 tok/s，与 16384 budget 的 1145.96±2.50 tok/s 相差 -0.19%，没有显示可测的吞吐收益。

因此，4 个 DSpark 槽会改变调度形态，但在这组 16K/C32/1P1D 负载中不是主要性能瓶颈。后续 PD 优化应继续关注 P 供给、并发和 P/D 配比；没有理由仅为消除这 4 个槽而扩大预算矩阵。

## 对照条件与结果

两组均使用：

- P/D：TP2×DP2、EP on、DSpark K5；P 为 NIXL producer，D 为 consumer；46 和 48 各使用 GPU 4–7。
- GovReport 固定前 256 条，输入精确 16384、输出 1024；客户端全系统 C32；一轮 256 请求完整预热，三轮 256 请求 quick。
- 双方 max-num-seqs=32（每个 DP engine），D budget=16384；只改变 P budget 和对应的 mHC 预热覆盖。
- 固定镜像 ID：sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1。未清理 JIT 缓存、未修改镜像或权重。

| 指标 | A：P budget 16384 | B4：P budget 16640 | B4 相对 A |
| --- | ---: | ---: | ---: |
| Output throughput | 1145.96±2.50 tok/s | 1143.79±0.95 tok/s | -0.19% |
| Mean TTFT | 11.939±0.154 s | 12.014±0.058 s | +0.63% |
| Mean TPOT | 15.131±0.149 ms | 15.082±0.083 ms | -0.32% |
| Mean E2EL | 27.418±0.014 s | 27.442±0.033 s | +0.09% |
| 正式成功／失败 | 768／0 | 768／0 | — |
| 正式已知编译事件 | 0 | 0 | — |

A 的功能追踪记录首条输入为 16380+4。B4 的追踪中，单请求在空余预算足够时可一次调度 16384；并发批次仍会按剩余总 budget 分块，这是正常的 chunked prefill 行为。B4 的 mHC 预热记录在全部 4 个 P worker 上包含 max token size=16640，输入预热的 K5 draft span 也包含 16640，DSpark 启动预热覆盖 token_max=192。

两组正式轮均完成真实 KV 传输：D 每轮 512 次 TP-rank 传输、远端命中 4194304 个输入 token、D 端 computed-prefill 为 0，NIXL 传输和通知失败均为 0。

原始证据在本机忽略目录 [A-01](../../../experiments/dsv4-pd-dspark-budget/results/A-01/) 和 [B4-01](../../../experiments/dsv4-pd-dspark-budget/results/B4-01/)。协议摘要分别在 benchmark/c32/protocol.json，调度追踪在 functional-prefill-trace.json；这些原始产物默认仅本机可用。

## 预热边界审计

本次逐层检查了当前实验补丁、缓存中实际挂载的预热模块，以及固定镜像内的 DSV4 相关代码：

- mHC 预热规划器是唯一发现的明确边界。B2 使用的缓存版本拒绝 max_tokens > 16384，其测试也把 16385 定义为非法。离线扩展到 16640 后，在 RTX6000D 的 156 SM dispatch 计算中只增加原有 [1, large, 1, 1] 分组的末端，没有新增 split 类别；B4 的 4 个 P worker 均实际完成该覆盖，第二遍没有新增 cache key。
- 输入 kernel 预热的 draft_spans() 已覆盖到 16640，K3/K5 两种 query token 均通过边界计算；top-k 几何覆盖由实际 DSV4 C4/C128 builder 生成，不依赖 16384 常量。
- DSpark 预热仍覆盖草稿 token 1–192；它与 prefill budget 是两个不同维度，16640 不要求把 DSpark 草稿范围扩大到 16640。
- 固定镜像本身的 scheduler、DSV4 MegaMoE、DSpark、FlashMLA、sparse MLA 代码均按 max_num_batched_tokens 动态分配或使用 chunked prefill；没有发现 16384/16640 硬编码上限。scheduler 只要求 chunked prefill 开启时 budget 可以小于 max model length，并把 budget 纳入编译缓存 key。

这说明 B2 的退出是预热覆盖不完整，不是模型或固定镜像拒绝 16640。B4 只增加了 mHC 预热 planner 的边界覆盖，没有修改 forward、kernel 实现、权重、精度或公共 runner。

## 启动失败记录

| 单元 | P budget | 结果 | 根因 |
| --- | ---: | --- | --- |
| B-01 | 16640 | 发请求前失败 | 输入预热补丁主动范围检查仍为 <=16384 |
| B2-01 | 16640 | 发请求前失败 | 修正输入预热后，缓存中的 mHC warmup_plan.py 仍拒绝 >16384 |
| B3-01 | 16640 | 未启动服务测量 | 旧实验工作区的全局截止时间已过，未触碰 kernel |
| B4-01 | 16640 | PASS | 修正截止时间并加入 mHC 16640 边界覆盖；完整协议通过 |

B3 的截止时间错误只增加了一次无请求的准备重跑；B4 与 B3 配置、数据和请求预算相同，没有增加正式性能单元。

## 后续含义

如果以后改变上下文长度、拓扑、K 值或并发，仍需重新验收相应预热组合；但对当前 16K/C32/TP2×DP2/K5 PD 路径，16640 budget 已完成一次完整验证，继续围绕 16384/16640 做矩阵没有信息价值。后续应回到 P 供给和 P/D 配比实验，并保持本报告中的 mHC 16640 预热覆盖作为可复用候选。
