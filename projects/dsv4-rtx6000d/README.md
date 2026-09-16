# DeepSeek V4 / RTX6000D

**当前推荐 vLLM 0.29.0、FlashInfer autotune 关闭，作为后续调优起点。** 2026-09-15 的对齐实验使用本机八卡、8192/1024 负载：vLLM 在 C16/C32 的输出吞吐为 532.57 / 651.35 tok/s，均优于同轮 SGLang off/on。该结果不代表硬件性能上限。

## 已确认的结果

| 内容 | 入口 |
| --- | --- |
| 镜像 digest、recipe 对齐和性能表 | [镜像选型](reports/image-selection.md) |
| 18 次有效测量及全部指标 | [CSV](data/baseline-samples.csv) / [JSON](data/baseline-samples.json) / [重复统计](data/baseline-statistics.json) |
| 环境、来源与复现步骤 | [复现说明](reports/reproduction.md) |
| 已知问题和取证边界 | [排查记录](reports/lessons.md) |
| RTX6000D 互联测量 | [交互 HTML](reports/interconnect.html) / [阅读摘要](reports/interconnect.md) |
| 原始产物与版本身份 | [数据说明](data/README.md) / [来源记录](data/provenance.json) |

正式配置只有三套：vLLM off、SGLang off、SGLang on。它们各有 C16/C32 三次有效重复，合计 1728 成功、0 失败。历史 smoke 和未对齐的探索数据不混入基线。

## 配置入口

以下命令从仓库根目录执行：

```bash
# 原三配置、C16/C32、各三次的完整基线
./bench validate projects/dsv4-rtx6000d/configs/campaigns/48-dsv4-aligned-final-c16-c32.yaml
./bench plan projects/dsv4-rtx6000d/configs/campaigns/48-dsv4-aligned-final-c16-c32.yaml
```

| Campaign | 用途与验证状态 |
| --- | --- |
| `48-dsv4-aligned-final-c16-c32.yaml` | 已实测的完整矩阵，迁移前后解析配置和命令等价 |
| `48-dsv4-vllm-baseline-c32.yaml` | 从原矩阵选出 vLLM C32；新增组合入口已离线校验，本次未重跑 |
| `48-dsv4-functional.yaml` | 两引擎性能 recipe 的短功能验证；新增 workload 已离线校验，本次未实机运行 |

复现前先按 [复现说明](reports/reproduction.md) 准备 SGLang 日志文件，再 preflight/run。target 固定 48 节点；在其他机器使用时新建或修改明确的 target，不能跳过本机身份检查。日志准备与短负载都不会免除模型加载和编译时间。

## 后续研究

暂定以 8192/1024、全局 C32 为单机拓扑筛选负载。讨论中的候选是 TP4×PP2、TP8+EP8、TP4×DP2（EP off/on），尚未形成性能结论。四台机器可先各自校准 baseline，再分配候选并同机复核优胜方案。本次仓库整理没有启动这些实验。

后续 PD 还需明确 P/D 实例与卡数分配。3P1D 是实例比例，不直接等于 24 卡；当前执行器没有跨节点 PD 编排能力。

[外部阶段报告](references/pro6000d-deepseek-v4-flash-stage-report-2026-08-23.pdf) 仅作参考。其第 3 页说明表格由实测锚点和参考曲线归一化形成，且未给出完整量化/镜像/测量窗口；不能直接当作与本基线同口径的逐点实测。
