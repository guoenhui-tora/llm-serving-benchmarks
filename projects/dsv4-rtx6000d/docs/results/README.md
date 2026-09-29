# 精选逐轮结果

本目录保留最终配置的完整并发扫描，以及直接决定拓扑选型的少数逐轮对照；不归档探索阶段的全部 CSV。下面两份最终扫描均使用固定 vLLM 0.30.0 服务及客户端、NVFP4 权重、真实互异 GovReport 输入、全局并发 C；每档 2C 条完整请求预热后，三轮各 4C 条正式测量，CSV **只含正式轮**。字段及数值未重算或筛掉轮次。

| 场景 | 完整已归档逐轮指标 | 对应报告 |
| --- | --- | --- |
| 24 卡 4P2D，16K 输入/1K 输出，P8200+仅P侧PHB，C16～176 | [33 轮 CSV](pd-24gpu-16k-1k-optimized-rounds.csv) | [选型与扫描](../pd-24gpu-16k-1k-study.md) |
| 16 卡 3P1D，24K 输入/2K 输出，P8200+仅P侧PHB，C32～128 | [24 轮 CSV](pd-16gpu-24k-2k-optimized-rounds.csv) | [选型与扫描](../pd-16gpu-24k-2k-study.md) |

[16 卡 PP/DP 拓扑配对 CSV](v030-pd-20260924-rounds.csv)是早期 v0.30 服务端、**v0.29 客户端**的独立诊断数据，包含 16K/1K C128 和 24K/2K C32～80 两组负载；有失败批次的诊断行，**只以 `eligible=True` 的完成批次比较性能**。它支撑 [PP/DP 拓扑分析](../pp-dp-3p1d-analysis-20260924.md)，不能与上述新客户端最优扫描混作同一批次。

上表两份最终扫描每一行是一个并发档的一轮：`duration_s` 是完整客户端窗口秒数；`output_throughput_tok_s` 是生成 tokens/s，`total_token_throughput_tok_s` 还包括输入。`*_ttft_ms`、`*_tpot_ms`、`*_itl_ms`、`*_e2el_ms` 分别是首 token、每请求输出 token、流式事件间隔与端到端延迟（毫秒）；分位数按轮计算。`slo_ttft_limit_s=10`、`slo_tpot_limit_s=0.05` 是逐请求阈值；`slo_joint_pass_rate` 是同一请求同时满足两者的比例（0～1），`slo_goodput_req_s` 是达标请求数/完整窗口秒数。报告的“均值 SLO”只按三轮 Mean TTFT/TPOT 均值判断，**不等于**联合达标率；报告的“±”是三轮吞吐样本标准差，其他表列为三轮算术均值。输入数据在 [`../../datasets/`](../../datasets/README.md)，不是本目录的输出 CSV。
