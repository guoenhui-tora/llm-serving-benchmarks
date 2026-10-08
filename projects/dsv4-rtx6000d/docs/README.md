# 文档一览

参照 [NVIDIA Dynamo 的术语](https://docs.nvidia.com/dynamo/v1.0.2/backends/v-llm/examples)：**聚合式推理**（aggregated serving）的每个实例同时处理 prefill 和 decode；**分离式推理**（disaggregated serving，本文简称 PD）由独立实例分别处理两阶段并交接 KV cache。区别在于实例职责，不在于实例数量。

```text
docs/
├── aggregated-serving.md     聚合式推理：历史四卡、八卡拓扑与 DSpark 对照
├── pd-24gpu-16k-1k.md        24 卡 16K/1K：4P2D 选型、优化和并发扫描
├── pd-16gpu-24k-2k.md        16 卡 24K/2K：3P1D 选型、优化和并发扫描
├── pd-pp2-dp2-topology.md    16 卡 3P1D：PP2/DP2 拓扑与混合实例对照
├── pd-startup.md             当前 PD 部署的服务、传输与代理启动
├── pd-research-strategy.md   分离式推理的单侧筛选与整机对照方法
├── historical-jit.md        JIT 排障方法和历史案例
└── results/
    ├── README.md             逐轮 CSV 的范围与指标说明
    ├── pd-24gpu-16k-1k.csv
    ├── pd-16gpu-24k-2k.csv
    └── pd-pp2-dp2-topology.csv
```

结果：[聚合式推理](aggregated-serving.md) · [24 卡 PD](pd-24gpu-16k-1k.md) · [16 卡 PD](pd-16gpu-24k-2k.md) · [PP2/DP2 对照](pd-pp2-dp2-topology.md)。复现部署见 [PD 启动说明](pd-startup.md)，表外逐轮指标见 [CSV 说明](results/README.md)。

聚合式推理的早期 v0.29 数值不与 v0.30 PD 结果直接排名；两项 PD 负载也尚未与同版本、同负载、同 GPU 的聚合式部署配对测量。
