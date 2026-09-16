# LLM Serving Benchmarks

用同一套脚本在不同模型、镜像和硬件上运行可追溯的 serving 实验。框架负责启动、检查、压测和保存证据；每个项目独立固定配置与结论。

当前支持 Linux 本机、NVIDIA GPU、Docker、vLLM/SGLang，以及统一的 OpenAI `/v1/completions` 流式压测。多台机器可以分别运行同一版本；尚未实现跨节点 SSH 编排和 PD 部署。

## 从哪里开始

| 入口 | 用途 |
| --- | --- |
| [DSV4 / RTX6000D](projects/dsv4-rtx6000d/README.md) | TP8基线、四节点TP4对照与互联报告 |
| [GLM-5.2 / RTX6000D](projects/glm52-rtx6000d/README.md) | 三套最终配置、C16/C32 重复结果与镜像选型 |
| [项目模板](projects/_template/README.md) | 基于真实配置创建新项目 |
| [实验方法](docs/benchmark-methodology.md) | 预热、JIT、稳定性诊断及结果验收 |
| [引擎对比](docs/engine-comparison.md) | 参数、显存与测量口径如何对齐 |
| [仓库管理](docs/repository-management.md) | 项目、版本、报告与原始数据怎么保存 |
| [配置格式](docs/configuration.md) / [实现边界](docs/architecture.md) | 配置字段和代码职责 |

## 安装

需要 Python 3.10+、Docker 和 NVIDIA GPU。模型与固定镜像应提前准备；执行器使用 `--pull never`，不会下载或升级镜像。以下命令都从仓库根目录执行。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./bench --help
```

## 先建立本地实验工作区

**拿到项目后，先在 `experiments/<项目>/` 中工作。** 配置尝试、原始结果和分析草稿都放在这里；阶段实验完成后，再将值得保留的配置和成果精选到 `projects/<项目>/`。

```text
experiments/<项目>/         本地工作区，整体不进 Git、不进源码包
├── configs/                实验配置，包含候选和失败尝试
├── results/                各次运行的日志、预热和原始指标
└── reports/                实验记录、草稿和临时汇总

projects/<项目>/            精选成果，提交 Git
├── README.md               项目目标、当前结论与后续计划
├── configs/                可复现结论的配置及完整依赖
├── reports/                精选报告
└── data/                   小型逐次数据与统计

projects/_template/         配置模板
src/、tests/、scripts/      通用实现、测试与辅助工具
docs/                      通用方法和配置说明
```

### 新项目：从模板开始

```bash
mkdir -p experiments/my-study/results experiments/my-study/reports
cp -a projects/_template/configs experiments/my-study/configs
```

按 [模板说明](projects/_template/README.md) 修改本地配置：核对模型、硬件、镜像、路径、recipe 和 workload。模板是已知平台的参考，不能直接当作新模型的已验证配置。

### 续接项目：从该项目的精选配置开始

例如继续优化 DSV4：

```bash
mkdir -p experiments/dsv4-rtx6000d/results experiments/dsv4-rtx6000d/reports
cp -a projects/dsv4-rtx6000d/configs experiments/dsv4-rtx6000d/configs
```

以上复制步骤只在工作区首次建立时执行；若 `configs/` 已存在，直接续用或另建工作区，不重复覆盖。先阅读该项目 README，再在工作区新增候选 recipe/campaign，保留已验证的基线。此时不必向 `projects/` 添加探索文件。

## 在工作区运行实验

以下以新项目模板的功能验证为例；续接项目时换成工作区里的实际 campaign：

```bash
./bench validate experiments/my-study/configs/campaigns/functional.yaml
./bench plan experiments/my-study/configs/campaigns/functional.yaml
python3 scripts/prepare_logging.py experiments/my-study/configs/campaigns/functional.yaml
./bench preflight experiments/my-study/configs/campaigns/functional.yaml
./bench run experiments/my-study/configs/campaigns/functional.yaml \
  --run-root experiments/my-study/results/functional-01

./bench report experiments/my-study/results/functional-01 \
  > experiments/my-study/reports/functional-01.md
```

`validate`、`plan` 不访问 Docker、GPU 或网络。`prepare_logging.py` 准备 recipe 需要的日志文件；`preflight` 会启动临时 CLI 检查容器，不加载模型。`run` 再次 preflight，并顺序执行 health、models、简短生成、预热和测量；`--case` 可只选择一个 case。

每次运行显式指定新的 `--run-root`，不要提前创建具体运行目录，执行器会自动创建并拒绝覆盖已有目录。省略该参数仍会写入根 `results/`，不会自动跟随 campaign 位置。根目录已有的 `results/`、`reports/` 继续本地保留和忽略，后续实验使用工作区。

## 把精选成果归档到 projects

实验完成并核对数据后，再做以下整理：

1. 将选定 campaign 及其引用的 recipe、workload、target、model、runtime、client 和所需 logging 文件复制到 `projects/<项目>/configs/`，保留它们在 `configs/` 内的相对路径。已有项目只补充本轮选定配置，不整体覆盖旧基线。
2. 将结论整理到项目 `reports/`，逐次小型数据放 `data/`，更新项目 README。报告链接应指向项目内随 Git 分发的文件；仅本机存在的原始日志路径要明确标注。
3. 对归档后的 campaign 再运行 `validate`、`plan`，核对参数和报告链接后提交。完整原始结果继续在工作区保留并另行备份，不整个复制进 Git。

**只改变外层目录不会破坏配置引用。** 例如 campaign 中的 `recipe: recipes/tp4.yaml` 始终相对于所属 `configs/`，不是相对于仓库根目录或当前 shell。内部结构和文件内容不变时，从 `experiments/` 复制到 `projects/` 后，解析配置与缓存路径保持一致。

模型目录、target 地址和缓存根路径仍取自配置，换机器需要另行核对；改变内部文件名需同步修改引用，Markdown 链接也需检查。归档时不要依赖指向工作区的软链接。详细边界见 [仓库管理](docs/repository-management.md)。

## 实验基本原则

- 固定权重、tokenizer、镜像身份、客户端和负载；运行期间冻结代码及配置。
- smoke 用于验证功能。正式测量保留 CUDA Graph 等正常优化，不能沿用仅为加快启动而禁用优化的设置。
- 保留 JIT 缓存，预热与测量分开保存；日志静默只表示没有识别到已知事件，不证明性能已经收敛。
- 禁用前缀复用的实验必须显式关闭 prefix/radix cache；检查实际成功数、失败数及输出 token 总量。
- 跨引擎先对齐语义，再检查实际日志。显存比例相同不等于 KV 容量相同。
- gate 未通过的尝试不进入性能结论；保留 fallback 和 warning，不为通过验收而隐藏问题。
- GPU 被其他任务占用时停止，不终止别人的任务；只清理本次所属容器，保留结果和缓存。

执行器会保存解析配置、命令、镜像/模型信息、源码指纹、GPU/CPU/拓扑静态快照、完整服务日志和原始指标。实际 CPU/NUMA 绑定、功耗/频率时间序列及 Git 本地差异需要另行保存。

## 验证与打包

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 scripts/package.py
```

`tests/fixtures/` 存放自动化测试使用的固定配置，与实际项目配置分开维护。测试使用模拟进程和回环 HTTP，不启动真实推理服务。`.github/workflows/tests.yml` 在 push 或 PR 时自动用 Python 3.10、3.12 运行测试并检查打包。

源码包包含框架、项目配置、精选报告及模板；排除本地运行产物和缓存。打包不是原始结果备份。
