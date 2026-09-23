# LLM Serving Benchmarks

用同一套脚本在不同模型、镜像和硬件上运行可追溯的 serving 实验。框架负责启动、检查、压测和保存证据；每个项目独立固定配置与结论。

当前支持 Linux 本机、NVIDIA GPU、Docker、vLLM/SGLang，以及统一的 OpenAI `/v1/completions` 流式压测。可显式配置本机一或两套服务同步测量，保存各侧和整机指标，见[同步多副本部署](docs/configuration.md#同步多副本部署)。常规runner面向本机；另有有界的实验性SSH多服务PD工具，已用于DSV4跨节点KV传输、有限功能检查及同资源quick对照，完整数值等价与稳态收益尚未确认，见[部署与协同流程](projects/dsv4-rtx6000d/reports/pd-cross-node-flow.md)。随机负载和本地 [JSONL 真实文本负载](docs/configuration.md#jsonl-真实文本数据集)共用压测流程，后者支持实际 tokenizer 长度校验和双实例固定分片。

## 从哪里开始

| 入口 | 用途 |
| --- | --- |
| [DSV4 / RTX6000D](projects/dsv4-rtx6000d/README.md) | 镜像选择、四节点整机性能对照与复现入口 |
| [GLM-5.2 / RTX6000D](projects/glm52-rtx6000d/README.md) | 三套最终配置、C16/C32 重复结果与镜像选型 |
| [项目模板](projects/_template/README.md) | 基于真实配置创建新项目 |
| [实验方法](docs/benchmark-methodology.md) | 预热、JIT、稳定性诊断及结果验收 |
| [引擎对比](docs/engine-comparison.md) | 参数、显存与测量口径如何对齐 |
| [仓库管理](docs/repository-management.md) | 项目、版本、报告与原始数据怎么保存 |
| [配置格式](docs/configuration.md) / [实现边界](docs/architecture.md) | 配置字段和代码职责 |

## 压测协议

默认研究目标是预热后的持续 serving 性能。新实验默认推荐 `quick`，先用固定成本筛选；需要排除已知编译事件或确认稳定性时再选择其他协议：

| 协议 | 运行和接纳规则 | 用途 |
| --- | --- | --- |
| `quick` | 默认1轮2C请求预热，可配置完整请求量；再默认3轮正式测量（可显式单轮初筛），保留JIT标记 | 默认的配置、拓扑初筛，不保证无JIT |
| `jit_clean` | 每轮完整正式负载，累计最先通过检查的3轮 | 重点候选的无已知编译事件对比 |
| `stable` | 每轮完整负载，首次连续3轮无已知事件，且吞吐、Mean TTFT、Mean TPOT相对极差均≤2% | 重点候选稳定性确认 |

**无已知编译事件不等于性能稳定。** DSV4中部分通过事件检查的配置，吞吐CV仍约3%–10%。因此默认用quick筛选，需要稳定窗口时用stable；jit_clean仅用于明确要求排除已知编译事件的对照，不是必经步骤。依据和边界见[协议选择经验](docs/benchmark-methodology.md#为什么默认用quick)。

一次启动完成同配置的所有轮次；JIT事件不触发重启。后两种最多12轮，所有新协议必须显式设置每档负载的时间预算。OOM、请求失败或工作量错误单独按故障处理。三个协议的PASS含义不同，不能混合统计；资源背景仍需人工审阅。

完整规则、统计边界及产物见[压测协议](docs/benchmark-methodology.md)，字段见[配置说明](docs/configuration.md#workload)。模板提供三种C32 workload。当前配置必须显式指定 `protocol`，不再保留旧预热/重试执行分支。历史数据按当时协议解释，精确复现使用报告记录的Git版本。

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
├── datasets/               可选的本地JSONL数据集
├── results/                各次运行的日志、预热和原始指标
└── reports/                实验记录、草稿和临时汇总

projects/<项目>/            精选成果，提交 Git
├── README.md               项目目标、当前结论与后续计划
├── configs/                当前基线、仍需复用的候选及公共依赖
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

使用 JSONL 时，将数据放入工作区 `datasets/`，workload 的 `dataset.path` 相对于工作区目录；归档时与 configs 一起复制。

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

**projects以报告和精选数据为主，configs不是实验历史仓库。**

1. 报告直接写结论、完整启动命令、镜像身份、绑定与环境、客户端负载、压测协议和结果。读者不必翻多个YAML才能理解实验；启动命令本身不能确定性能。
2. configs只保留当前推荐基线和仍需复用的少量候选，携带完整依赖。相同服务参数共用recipe，节点差异在本地target填写；已结束矩阵、失败尝试和补测配置留在experiments或Git历史，不全部上传。
3. 启动命令从 `bench plan` 或实测产物导出，报告保存当次快照，不手工维护bash和YAML两套标准答案。命令示例与实际测量、当前协议与历史协议分别注明。
4. 归档后执行validate/plan，核对命令、数字和相对链接。更新基线或精简旧配置时保留历史报告/CSV，原始结果另行备份；精确复现历史用报告记录的commit和运行配置。

**只改变外层目录不会破坏配置引用。** 例如 campaign 中的 `recipe: recipes/tp4.yaml` 始终相对于所属 `configs/`，不是相对于仓库根目录或当前 shell。内部结构和文件内容不变时，从 `experiments/` 复制到 `projects/` 后，服务参数、负载身份和缓存路径保持一致；本机JSONL挂载路径随项目位置更新。

模型目录、target 地址和缓存根路径仍取自配置，换机器需要另行核对；改变内部文件名需同步修改引用，Markdown 链接也需检查。归档时不要依赖指向工作区的软链接。详细边界见 [仓库管理](docs/repository-management.md)。

## 实验基本原则

- 固定权重、tokenizer、镜像身份、客户端和负载；运行期间冻结代码及配置。
- smoke 用于验证功能。正式测量保留 CUDA Graph 等正常优化，不能沿用仅为加快启动而禁用优化的设置。
- 保留 JIT 缓存，预热与测量分开保存；日志静默只表示没有识别到已知事件，不证明性能已经收敛。
- 禁用前缀复用的实验必须显式关闭 prefix/radix cache；检查实际成功数、失败数及输出 token 总量。
- 跨引擎先对齐语义，再检查实际日志。显存比例相同不等于 KV 容量相同。
- 按所选协议接纳结果；快速初筛中的事件样本须明确标注，不能冒充无JIT基线。保留 fallback 和 warning，不为通过验收而隐藏问题。
- GPU 被其他任务占用时停止，不终止别人的任务；只清理本次所属容器，保留结果和缓存。

执行器会保存解析配置、命令、镜像/模型信息、源码指纹、GPU/CPU/拓扑静态快照、完整服务日志和原始指标。设置 target 的 `binding` 后，还会检查并保存对应容器和线程的 CPU/内存允许集合，见[绑定配置](docs/configuration.md#cpu--numa-绑定)。显式多副本模式还保存 CPU/GPU/cgroup 资源时间序列；普通单服务模式的时间序列、实际内存页分布及 Git 本地差异需另行保存。

## 验证与打包

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 scripts/package.py
```

`tests/fixtures/` 存放自动化测试使用的固定配置，与实际项目配置分开维护。测试使用模拟进程和回环 HTTP，不启动真实推理服务。`.github/workflows/tests.yml` 在 push 或 PR 时自动用 Python 3.10、3.12 运行测试并检查打包。

源码包包含框架、项目配置、精选报告及模板；排除本地运行产物和缓存。打包不是原始结果备份。
