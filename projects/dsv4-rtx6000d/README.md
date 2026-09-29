# DeepSeek V4 Flash NVFP4 / RTX6000D

本项目研究 NVIDIA `DeepSeek-V4-Flash-0731-NVFP4` 在 RTX6000D 上的拓扑、DSpark 和跨节点 prefill/decode (PD) 部署。**当前主线是固定 vLLM 0.30.0 的实测条件**；早期 v0.29 的四卡结果仅作当时条件下的拓扑线索，不与 v0.30 数值直接排名，旧预热 worker 不是当前启动依赖。

当前已测的两个负载分别是 24 卡、真实独立 GovReport **16K 输入/1K 输出**的 4P2D，以及 16 卡、真实独立 GovReport **24K 输入/2K 输出**的 3P1D；各 P/D 是独立四卡 TP2×DP2、EP on、DSpark K5 服务。这些是固定负载与均值 SLO 下的优先工作点，不是所有拓扑的全局最优，也不满足 95% 逐请求联合 SLO。共享前缀 decode-only 测量不代表完整 PD 性能。

[研究导航](docs/README.md)按[聚合式推理四卡/八卡参照](docs/aggregated-serving.md)、[24 卡 16K/1K PD](docs/pd-24gpu-16k-1k-study.md)、[16 卡 24K/2K PD](docs/pd-16gpu-24k-2k-study.md)组织结果；[PD 研究策略](docs/pd-research-strategy.md)单独总结从单侧筛选到端到端配对的方法。[PD 启动说明](docs/pd-startup.md)记录固定镜像、权重、NIXL/UCX、代理及 v0.30 MXFP4 草稿修复；[最终配置逐轮 CSV](docs/results/README.md)提供表外指标。早期聚合式推理结果不充当当前 PD 的同负载基线。

[GovReport 输入数据集](datasets/README.md)提供精确 8K、16K、24K JSONL、来源、tokenizer 核验及制备脚本。服务研究的精选逐轮结果在 `docs/results/`，不要与输入目录 `datasets/` 混淆。原始运行日志和探索脚本保存在各节点本机的忽略工作区 `experiments/`，不随 Git 分发；提交的研究结论以 `docs/` 中的表格、启动参数和少量精选 CSV 为准。
