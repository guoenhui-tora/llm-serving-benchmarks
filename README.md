# LLM Serving Benchmarks

通过独立配置组合硬件、模型、推理引擎、镜像和负载，运行可追溯的 LLM serving 实验。
一个工程支持 vLLM 与 SGLang；客户端独立固定版本，实验按 case 顺序执行。

## 当前实现范围

- Linux 本机执行、NVIDIA GPU、Docker + NVIDIA Container Toolkit、host network / host IPC。
- vLLM 与 SGLang 的独立启动及参数校验。
- 独立的 `vllm bench serve` 客户端，通过 OpenAI `/v1/completions` 流式接口压测两个引擎。
- random token 负载、并发扫描、重复测量、禁用 prefix/radix cache。
- `/health`、`/v1/models`、可配置 chat probe、kernel 日志检查。
- 镜像/模型/GPU/CLI preflight，预热与测量期间已知编译事件检查，失败取证和容器清理。
- 可迁移的结果目录及跨运行汇总。

暂不实现 SSH 自动调度、多节点推理、ROCm、Nsight profiling、真实对话数据集、开启 prefix cache 的实验。
这些能力需要扩展明确的模块接口，不能靠填一个未知配置值自动启用。

**两套 smoke 已在 46 / 48 的真实镜像与模型上通过，最终代码在本机和两台服务器均通过 44 项测试。**
新增的缓存禁用开关、CLI 检查、统一 CPU 客户端、接口及固定输出长度均已验证，详见 [验证记录](docs/validation.md)。
性能 recipe 是未验证候选配置，不是最优配置或性能结论。

## 安装

需要 Python 3.10+。在工程目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./bench --help
```

`./bench` 从源码运行。也可以 `python -m pip install .` 安装 `bench` 命令；配置仍从工程目录提供。
宿主不安装 vLLM/SGLang。服务端和客户端各自在对应的 Docker 镜像中运行。
Docker 镜像必须事先准备好，执行器使用 `--pull never`，不会自动下载或升级。

48 上除了 SGLang 服务端镜像，也需要固定的 vLLM 客户端镜像。
客户端容器不申请 GPU。测试期间不允许其他计算进程占用所选 GPU。

## Git 与服务器使用

源码、测试、文档及 `configs/` 下的实验配置纳入版本管理。保留硬件、模型、
镜像版本和 recipe 配置，服务器 clone 后即可选择对应 campaign。

服务器首次使用：

```bash
git clone <仓库地址> llm-serving-benchmarks
cd llm-serving-benchmarks
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./bench validate configs/campaigns/46-glm52-vllm-smoke.yaml
```

按服务器选择 campaign，并事先准备配置要求的 Docker 镜像和 `/data/models/` 下的模型。
没有本地源码改动时，实验结束后用 `git pull --ff-only` 更新；实验运行期间不要更新源码。
多台服务器对比时使用同一个 commit，可用 `git rev-parse HEAD` 核对。

默认运行产物全部写入根目录 `results/`，包括原始结果、日志、命令和环境快照；
Git 只保留其中的 `.gitkeep`。手动导出的汇总放在 `reports/`，例如：

```bash
mkdir -p reports
./bench report results/CAMPAIGN/RUN_ID --json > reports/comparison.json
```

根目录的 `reports/`、`logs/`、`artifacts/`、`tmp/`、`cache/`、`.cache/`，
以及虚拟环境、Python 缓存和构建产物均忽略。若使用 `--run-root`，请放在
`results/` 内或仓库之外，任意自定义目录不会自动被忽略。
当前模型和 JIT 缓存位于仓库之外，不会随 Git 传输。

原始结果由各机器保留、另行归档，Git 不会备份被忽略的文件。
需要发布的精选报告可放入 `docs/` 并正常提交。不要全局忽略 `*.json`、`*.yaml`
或所有名为 `results` 的目录，以免隐藏配置、测试数据和结果处理源码。

## 配置组织

```text
configs/
├── targets/       # 本机地址、GPU、硬件要求、模型根目录、缓存根目录、端口
├── models/        # 模型身份、目录名、架构、文件要求
├── runtimes/      # engine: vllm/sglang、镜像、版本、镜像 ID
├── recipes/       # 原生参数、适用范围、kernel 检查、chat probe
├── clients/       # 独立压测工具与镜像
├── workloads/     # 数据、并发、请求量、采样、预热与重复次数
└── campaigns/     # 一组实验的显式组合
```

所有文档必须包含 `schema_version: 1`、`kind`、`id`。
**配置引用相对于 `configs/` 根目录**，不是当前 YAML 所在目录。
自定义配置根目录用 `--config-root`。引用不能逃出该目录，也不支持隐式继承、环境变量模板或任意覆盖。
未知字段、重复键、参数冲突、组合不匹配会直接报错。

campaign 公用一个 target、client 和 workload 列表；`cases` 中逐个选择 model、runtime 和 recipe。
一个 campaign 只在一台机器运行。两台机器分别运行各自的 campaign，再合并结果。
详细字段与扩展方法见 [配置说明](docs/configuration.md) 和 [架构说明](docs/architecture.md)。

## 先检查，再验证

以下是检查配置与执行实机验证的入口。

46 的 GLM52 + vLLM：

```bash
./bench validate configs/campaigns/46-glm52-vllm-smoke.yaml
./bench plan configs/campaigns/46-glm52-vllm-smoke.yaml
./bench preflight configs/campaigns/46-glm52-vllm-smoke.yaml
./bench run configs/campaigns/46-glm52-vllm-smoke.yaml
```

48 的 DeepSeek V4 + SGLang：

```bash
./bench validate configs/campaigns/48-dsv4-sglang-smoke.yaml
./bench plan configs/campaigns/48-dsv4-sglang-smoke.yaml
./bench preflight configs/campaigns/48-dsv4-sglang-smoke.yaml
./bench run configs/campaigns/48-dsv4-sglang-smoke.yaml
```

`validate` / `plan` 不访问 GPU、Docker 或网络。`plan` 显示完整配置、服务端和第一个 workload 的客户端命令。
`preflight` 检查本机地址、GPU 型号/架构/显存/计算进程、端口、模型分片、镜像 ID，
并在无网络的临时容器中检查 CLI 帮助。服务端帮助命令可见一张 GPU，以满足引擎的设备检测；不加载模型。
客户端帮助和实际压测均不申请 GPU。vLLM 0.29 顶层 CLI 会提前构造服务端配置，因此客户端直接调用同一官方 benchmark 模块的参数解析和 main 函数。
CLI 存在不等于硬件支持该参数，实际支持仍需 startup + probe + workload 验证。

`run` 自动再次 preflight，然后按 case 顺序执行。可用 `--case CASE_ID` 选择部分 case，参数可重复。
GPU 文件锁防止本执行器的任务相互重叠；它不是 Docker 层的 GPU 独占机制。

## 从 smoke 到性能测量

每个已提供的模型/引擎组合都有三个 campaign：

| 后缀 | 参数及负载 | 用途 |
|---|---|---|
| `-smoke` | 关闭部分优化、短上下文、C1、128/32、2 请求、1 次 | 功能验证，不比较性能 |
| `-calibration` | 性能候选 recipe、1024/256、C1/2/4、32 请求、1 次 | 检查候选设置及容量 |
| `-performance` | 性能候选 recipe、8192/1024、C1/2/4、32 请求、3 次 | 候选稳定后再正式测量 |

性能 recipe 恢复引擎默认编译/图优化行为，放宽上下文与批处理容量；这不保证所有优化都被启用或适合当前硬件。
同一个 smoke recipe 不允许用于 calibration/performance workload。
更大并发、更长上下文、DP/EP/PP、显存预算和通信后端需要按目标环境单独设计 recipe。

预热必须达到配置数量的**连续无已知编译日志事件**的轮次，才进入测量。
测量期间检测到事件会保存并舍弃该次数据，再预热和重试。日志静默不证明没有未被记录的编译，也不证明延迟已经收敛。
性能负载默认至少两轮安静预热。客户端的内部 warmup 次数设为 0；vLLM benchmark 仍会发送自己的初始单请求连通性测试，该请求不计入测量请求数。每个阶段的完整客户端日志都会保存。

`ignore_eos=true` 且固定输出长度时，结果必须包含预期的输出 token 总量；提前结束不能冒充更高吞吐。
采样种子、客户端镜像和 tokenizer 一致时复用相同随机生成流程。random token 不是业务质量评测。
chat probe 单独保存请求和回复，不进入性能统计；thinking 参数在 recipe 中明确配置。

JIT 缓存挂载在 `/root/.cache`，按服务端实际镜像 ID、GPU 架构、模型 ID、recipe 内容隔离。
删除服务容器不会删除该缓存；需要比较冷启动时应单独规划缓存状态。当前实验衡量预热后的 serving，不对冷启动速度做公平比较。

## 结果与停止

```text
results/<campaign>/<run-id>/
├── run.json                 # 状态、代码指纹、容器所属运行 ID、case 相对路径
├── plan.json                # 完整解析配置和配置指纹
├── run.log
└── cases/<case-id>/
    ├── case.json
    ├── resolved.json
    ├── environment.json     # GPU、模型文件哈希/分片大小、镜像 inspect、CLI 帮助
    ├── host.json            # 驱动、GPU 拓扑、CPU、内存、Docker 信息
    ├── argv.json / command.sh
    ├── server.log / server-inspect.json
    ├── kernel-checks.json
    ├── probes/
    └── trials/<workload>/cXXXX/rXX/
        ├── workload.json
        ├── measurement-checks.json
        ├── warmup-XX-XX/
        └── measurement-XX/  # raw.json、metrics.json、client.log、命令、编译证据
```

实际命令固定使用 preflight 确认过的镜像 ID，避免检查后 tag 变化。
模型身份包含配置、tokenizer 文件哈希及权重分片大小；没有对数百 GB 权重全文哈希，因此不能证明同大小权重内容完全相同。
结果索引使用相对路径，命令和环境快照中的原始宿主绝对路径仅用于取证。

```bash
./bench report results/CAMPAIGN/RUN_ID
./bench report /path/to/run-on-46 /path/to/run-on-48
./bench report /path/to/run-on-46 /path/to/run-on-48 --json > comparison.json
./bench stop results/CAMPAIGN/RUN_ID
```

`stop` 必须在原机器执行，只移除带该运行所有权标签的容器。正常结束、异常和 Ctrl-C 同样保留日志并清理容器。
强制杀掉宿主进程（SIGKILL）无法执行 finally；此时可用 `stop` 清理，该中断运行的残留状态不会被当作成功测量。
已存在的结果目录不允许覆盖，也不支持隐式断点续跑。

报表只汇总 `case.status=PASS` 的测量。按硬件、模型身份、服务端/客户端镜像、recipe、workload、并发和代码指纹等区分组，
不同并发和负载不会混在一起求平均；重复传入同一 run 或它的复制件不会重复计数。
缺失指标显示 `N/A`，单次测量没有标准差；各次 P95 的均值不等于合并请求后的 P95。
`kernel_checks=UNVERIFIED` 代表未配置必须出现的证据，不代表 kernel 验证通过。

GLM52/vLLM 与 DSV4/SGLang 是两个不同部署，不能据此得出哪个引擎更快。
引擎对比应固定模型、硬件、客户端和 workload，并分别验证两份 recipe。
目前只提供上述两个有手工启动依据的组合；另外两个模型/引擎交叉组合需要独立建立和验证 recipe。

## 测试与打包

```bash
python -m unittest discover -s tests -v
# 未安装项目包、使用 ./bench 的环境：
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/package.py
```

测试不需要 Docker、模型或 GPU；集成测试仅启动本机回环 HTTP 模拟服务。
默认打包到 `dist/llm-serving-benchmarks-source.tar.gz`，旁边生成 SHA-256。
归档包含源码、配置、文档和测试，排除结果、缓存、虚拟环境和历史工程。
