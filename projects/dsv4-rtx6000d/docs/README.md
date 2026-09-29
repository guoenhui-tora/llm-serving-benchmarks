# DSV4 / RTX6000D 研究导航

术语参照 [NVIDIA Dynamo 的部署示例](https://docs.nvidia.com/dynamo/v1.0.2/backends/v-llm/examples)：**聚合式推理**（aggregated serving）指每个实例同时完成 prefill 和 decode；即使同一节点运行多个完整实例，仍属于聚合式推理。**分离式推理**（disaggregated serving，本文简称 PD）由独立的 P、D 实例分别执行 prefill、decode，并将 KV cache 从 P 传给 D。两者区分的是实例职责，不是实例数量或是否经过代理。

结果按**三个独立问题**阅读，不将早期 v0.29 的 8K 聚合式推理与 v0.30 的 16K/24K PD 数值直接排名：

| 研究问题 | 结论与范围 |
| --- | --- |
| [聚合式推理：四卡和八卡](aggregated-serving.md) | 历史 v0.29：固定 8192/1024 的双四卡扩容与单八卡对照；附一组不同输入负载的 K1–K5 接受率观察，不参与拓扑排名。 |
| [24 卡 PD：16K/1K](pd-24gpu-16k-1k-study.md) | v0.30 的 4P2D 选型、P 预算/通信优化、端到端并发；同资源八卡 D 的 [decode-only 筛选](decode-only-8gpu-20260925.md)是本负载的单侧证据。 |
| [16 卡 PD：24K/2K](pd-16gpu-24k-2k-study.md) | v0.30 的 3P1D 选型、P 预算与 PHB、端到端并发；[PP/DP 真实 PD 对照](pp-dp-3p1d-analysis-20260924.md)是其拓扑依据。 |

两项 PD 的单侧诊断只缩小候选范围，最终判断来自各自的完整客户端窗口；当前均未与同版本、同负载和同 GPU 的聚合式推理配对，因此不能声称 PD 优于聚合式推理。[PD 研究策略](pd-research-strategy.md)归纳下一次如何分别筛 P/D、再与聚合式推理比较，**不冒充本次实验顺序**。

[逐轮精选数据](results/README.md)是上述两篇最终配置并发扫描的完整已归档轮次，不依赖本机 `experiments/`。CSV 中 `slo_ttft_limit_s=10`、`slo_tpot_limit_s=0.05`；`slo_joint_pass_rate` 是**同一请求**同时达标的比例，`slo_goodput_req_s` 是达标请求数除以完整客户端窗口秒数。它们不同于按三轮均值判断的 SLO，也不同于所有请求的 output tok/s。三轮原始数值应分别读取，不把逐轮 P95 当作合并请求的 P95。

[当前 PD 启动说明](pd-startup.md)给出固定镜像、权重、端口、NIXL/UCX、代理与 v0.30 MXFP4 修复；两篇 PD 报告保留各自实际负载和参数。[历史 JIT 排障经验](historical-jit.md)不是当前安装步骤。探索性结果已提炼进研究正文，`results/` 只保留少量关键对照与最终扫描的 CSV；[输入数据](../datasets/README.md)独立于这些输出结果。
