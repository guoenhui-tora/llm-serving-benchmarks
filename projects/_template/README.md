# 项目模板：RTX6000D + DSV4 参考

这是从 DSV4 已验证配置整理的最小完整项目，不是任意模型/硬件都能直接运行的默认配置。参考版本为 vLLM 0.29.0、SGLang 0.5.19-cu130；来源见 [DSV4 项目](../dsv4-rtx6000d/README.md)。模板没有自己的性能结果。

## 创建项目

从仓库根目录复制，例如：

```bash
cp -a projects/_template projects/my-study
```

随后更新项目 README 和配置身份。运行前逐项核对：

- target：本机地址、GPU 列表和型号、模型/缓存路径、端口及容器权限。
- model/runtime：模型架构、量化、tokenizer、固定镜像 ID 和实际版本支持。
- recipe：TP/PP/DP/EP、上下文、容量、KV、attention/MoE backend、Graph 和 autotune。不能沿用 DSV4 参数猜测其他模型支持。
- workload：长度、全局并发、请求量和预热/重复上限。
- probe/checks：模型的 thinking 参数、生成预期和实际日志证据。

`functional.yaml` 使用两引擎的正式 serving recipe 做 128/32、C1 短验证，仍会完整加载模型并编译；它不是关闭优化的快速启动 recipe。`c32-comparison.yaml` 使用三套 recipe 测 8192/1024、C32、128 请求、三次重复。

## 验证和执行

```bash
./bench validate projects/my-study/configs/campaigns/functional.yaml
./bench plan projects/my-study/configs/campaigns/functional.yaml
python3 scripts/prepare_logging.py projects/my-study/configs/campaigns/functional.yaml
./bench preflight projects/my-study/configs/campaigns/functional.yaml
./bench run projects/my-study/configs/campaigns/functional.yaml \
  --run-root results/my-study/functional-01
```

确认功能后再使用 `c32-comparison.yaml`；同样先 prepare_logging、preflight，再 run 到新目录。日志准备只复制 JSON 到 recipe 对应的挂载缓存目录，不启动 Docker，也不清空缓存。仅 vLLM 的 case 无需 SGLang 日志文件。

项目内配置引用均相对于自己的 `configs/`。不要复制其他项目的性能结论。精选报告放本项目 reports，小型逐次数据放 data；原始产物放根 results。

通用要求见 [实验方法](../../docs/benchmark-methodology.md)、[引擎对比](../../docs/engine-comparison.md) 和 [仓库管理](../../docs/repository-management.md)。
