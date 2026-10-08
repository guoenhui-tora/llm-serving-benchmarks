# DeepSeek V4 Flash NVFP4 / RTX6000D

本项目研究 [nvidia/DeepSeek-V4-Flash-0731-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4) 在 RTX6000D 上的聚合式推理拓扑、投机解码，以及跨节点 prefill/decode 分离式推理。当前 PD 结果基于 vLLM 0.30.0；早期 v0.29 结果保留为独立的历史参照。

```text
dsv4-rtx6000d/
├── docs/       实验报告、部署方法和精选逐轮 CSV
├── datasets/   GovReport 输入数据及制备说明
├── patches/    按镜像版本归档的模型兼容修复与 PD 组件
└── configs/    早期 v0.29 实验配置，不是当前 PD 的启动入口
```

从[文档一览](docs/README.md)查看报告与部署入口；输入数据见[数据集说明](datasets/README.md)。
