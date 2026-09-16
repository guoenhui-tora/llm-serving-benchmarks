# GLM-5.2 / RTX6000D 历史配置

这里保留仓库最初的 46 节点 vLLM 配置，避免目录整理丢失已有工作。**仅 smoke 有本仓库的实机验证记录；calibration/performance 是当时的候选，不是稳定性能基线。** 不包含其他机器或其他会话后来完成的 GLM 对比数据。

```bash
./bench validate projects/glm52-rtx6000d/configs/campaigns/46-glm52-vllm-smoke.yaml
./bench plan projects/glm52-rtx6000d/configs/campaigns/46-glm52-vllm-smoke.yaml
```

target 固定 10.90.1.46，执行器不会 SSH，也不要在 48 上直接运行。部署前核对本机 target、模型路径和镜像。原始功能记录见 [初始 smoke 验证](../dsv4-rtx6000d/reports/initial-smoke-validation.md)。

新研究请使用新的 recipe/campaign，并阅读 [实验方法](../../docs/benchmark-methodology.md)。不能用 GLM 与 DSV4 两个不同模型的 smoke 结果比较引擎速度。
