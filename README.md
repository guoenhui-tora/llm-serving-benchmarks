# LLM Serving Benchmarks

用同一套脚本在不同模型、镜像和硬件上运行可追溯的 serving 实验。框架负责启动、检查、压测和保存证据；每个项目独立固定配置与结论。

当前支持 Linux 本机、NVIDIA GPU、Docker、vLLM/SGLang，以及统一的 OpenAI `/v1/completions` 流式压测。多台机器可以分别运行同一版本；尚未实现跨节点 SSH 编排和 PD 部署。

## 从哪里开始

| 入口 | 用途 |
| --- | --- |
| [DSV4 / RTX6000D](projects/dsv4-rtx6000d/README.md) | 已验证的三套基线、镜像选型和互联报告 |
| [项目模板](projects/_template/README.md) | 基于真实配置创建新项目 |
| [GLM-5.2 / RTX6000D](projects/glm52-rtx6000d/README.md) | 保留初始 smoke 和未验证的性能候选 |
| [实验方法](docs/benchmark-methodology.md) | 预热、JIT、稳定性诊断及结果验收 |
| [引擎对比](docs/engine-comparison.md) | 参数、显存与测量口径如何对齐 |
| [仓库管理](docs/repository-management.md) | 项目、版本、报告与原始数据怎么保存 |
| [配置格式](docs/configuration.md) / [实现边界](docs/architecture.md) | 配置字段和代码职责 |

## 安装与检查

需要 Python 3.10+。模型和固定镜像应提前准备；执行器使用 `--pull never`，不会下载或升级镜像。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./bench --help

./bench validate projects/dsv4-rtx6000d/configs/campaigns/48-dsv4-aligned-final-c16-c32.yaml
./bench plan projects/dsv4-rtx6000d/configs/campaigns/48-dsv4-aligned-final-c16-c32.yaml
```

`validate` 和 `plan` 不访问 Docker、GPU 或网络。程序会从 campaign 路径找到所属 `configs/`；也可显式传入 `--config-root`。配置引用均相对于这个项目的配置根。

实际运行前先阅读项目 README，核对 target 地址、模型路径、镜像 ID 和缓存权限。DSV4 的 SGLang recipe 还需要准备自定义日志文件，具体命令见项目入口。

```bash
./bench preflight projects/dsv4-rtx6000d/configs/campaigns/48-dsv4-aligned-final-c16-c32.yaml
./bench run projects/dsv4-rtx6000d/configs/campaigns/48-dsv4-aligned-final-c16-c32.yaml \
  --run-root results/dsv4-rtx6000d/my-new-run
```

`preflight` 会启动临时 CLI 检查容器，服务端帮助可见一张 GPU，但不加载模型。`run` 会再次 preflight，再顺序启动各 case，完成 health、models、中文探测、预热和测量。`--case vllm-autotune-off` 可只选一个 case。已有结果目录拒绝覆盖。

## 实验基本原则

- 固定权重、tokenizer、镜像身份、客户端和负载；运行期间冻结代码及配置。
- smoke 用于验证功能。正式测量保留 CUDA Graph 等正常优化，不能沿用仅为加快启动而禁用优化的设置。
- 保留 JIT 缓存，预热与测量分开保存；日志静默只表示没有识别到已知事件，不证明性能已经收敛。
- 禁用前缀复用的实验必须显式关闭 prefix/radix cache；检查实际成功数、失败数及输出 token 总量。
- 跨引擎先对齐语义，再检查实际日志。显存比例相同不等于 KV 容量相同。
- gate 未通过的尝试不进入性能结论；保留 fallback 和 warning，不为通过验收而隐藏问题。
- GPU 被其他任务占用时停止，不终止别人的任务；只清理本次所属容器，保留结果和缓存。

执行器会保存解析配置、命令、镜像/模型/GPU 信息、源码指纹、完整服务日志和原始指标。额外的 CPU/NUMA、拓扑、功耗/频率时间序列以及 Git 本地差异需要实验负责人另行保存，框架不会自动完成所有取证。

## 目录与结果

```text
src/、tests/、scripts/     通用实现、测试与辅助工具
docs/                     通用方法和配置说明
projects/_template/       项目模板
projects/<项目>/configs/  项目独立固定的配置
projects/<项目>/reports/  精选报告
projects/<项目>/data/     小型逐次数据与统计
results/、reports/        本机原始产物与草稿，不进 Git
artifacts/                本机归档，不进 Git
```

项目中的报告和数据正常纳入版本管理。根目录的结果、草稿、缓存、模型和运行环境不随 Git 分发。旧根目录 `configs/` 已迁移，回退方式见 [迁移记录](docs/repository-migration-20260916.md)。

## 验证与打包

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 scripts/package.py
```

测试使用模拟进程和回环 HTTP，不启动真实推理服务。源码包包含框架、项目配置、精选报告及模板；排除本地运行产物和缓存。打包不是原始结果备份。
