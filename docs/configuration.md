# 配置格式

每类配置独立变化，全部使用 `schema_version: 1` 和对应 `kind`。未知字段、重复键、非有限数值、越界引用直接失败。
实验工作区使用 `experiments/<项目>/configs/`，精选配置使用 `projects/<项目>/configs/`；两者都作为独立配置根。CLI 默认从 campaign 的父目录寻找名为 `configs` 的目录，也可用 `--config-root` 显式指定。路径引用统一相对于配置根目录。可以使用子目录整理文件；程序不从文件名推断引擎、模型或硬件。

## Target

必填：`executor: docker`、`address`（本机 IP）、`gpus`（显式索引列表）、`gpu`、`model_root`、`cache_root`、`port`。
`gpu` 必填 `vendor: nvidia`、`name_regex`、字符串 `compute_capability` 和 `min_memory_mib`。
`address` 用于防止在错误机器启动，执行器不会自动 SSH。

可选：`model_paths: {model-id: /absolute/path}`、`environment`、`ulimits`、`cap_add`、`binding`。
环境变量值必须是字符串。模型根目录下使用 `model.directory` 找权重；`model_paths` 可替换某个模型的路径。
目标固定使用 host network、host IPC，HTTP 绑定 `127.0.0.1`。未实现其他网络模式。
硬件参数只用于校验，不会自动猜测 NUMA、TP/EP 或 kernel 最优值。

### CPU / NUMA 绑定

`binding` 分别控制服务和压测客户端，编号均为宿主编号。每个指定的角色必须同时填写字符串 `cpus`、`mems`；支持 Linux 列表格式，如 `"0-3,8"`。未指定的角色保持原来的不绑定行为。

```yaml
binding:
  server:
    cpus: "32-47"
    mems: "2"
  client:
    cpus: "48-51"
    mems: "3"
```

执行器将它们转换为各容器的 `--cpuset-cpus`、`--cpuset-mems`。`validate/plan` 检查语法、角色和逻辑 CPU 重叠；本机 `preflight` 再检查 CPU 在线、内存节点存在，以及服务/客户端是否通过 SMT 共享物理核。它不会自动选择最优核数或保证后台任务不干扰；选取方法见[绑定规则](engine-comparison.md#cpugpu-绑定与同机多服务)。

服务就绪后及每轮客户端结束后，保存服务的 Docker inspect 和容器内线程亲和性；客户端在压测入口前后保存亲和性，结束后保存 inspect，再按所属运行清理容器。文件为 `server-binding-inspect.json`、`server-affinity.json`、`client-binding-inspect.json`、`client-affinity-start.json`、`client-affinity-end.json`。检查在客户端内部计时窗口外执行，不持续轮询。客户端已失败或被用户停止时，额外取证失败写入 `client-binding-error.json`，保留原始失败或中断状态。

Docker 配置必须与请求集合一致，进程/线程允许集合可以在其中进一步缩小。越界或无法核验会使运行失败，保留证据；这些快照不能证明整个测量期间的亲和性不变，也不代表所有内存页都在本地。绑定改变会进入 target 指纹，结果不会与其他绑定配置自动合并。

## Model

必填：`directory`（相对目录）、`served_name`、`architecture`（与 config.json 对应）、`required_files`（相对路径列表）。
可选：描述性的 `quantization`。
始终检查 config.json；存在 safetensors index 时会检查它列出的每个分片。
模型/Tokenizer 同一目录只读挂载到服务端和客户端的 `/model`。

## Runtime 与 Client

Runtime 必填：`engine: vllm|sglang`、`image`、字符串 `version`。
可选：`image_id`、`environment`、`ready_timeout_s`（默认 1800）。
`image_id` 是 `docker image inspect --format '{{.Id}}' IMAGE` 的值，不是 RepoDigest；名称明确，不混用两者。
版本是配置声明，实际工件以镜像 ID、RepoDigests、完整 inspect 和启动日志追溯。

当前精选runtime显式设置 `TRITON_CACHE_DIR`：vLLM为 `/root/.cache/triton`，SGLang为 `/root/.cache/sglang/triton`（保持该镜像原有重定向位置）。两者都在既有 `/root/.cache` 持久挂载内，不需要新挂载或修改recipe。宿主根目录仍取target的 `cache_root`；缓存按镜像ID、GPU架构、模型ID和recipe隔离，本次runtime变量调整不改变已有缓存目录指纹。变量优先级为target < runtime < recipe，覆盖时须确认新路径仍被持久挂载。

已有experiments工作区不会自动获得此设置，续接时同步对应runtime的环境变量。旧容器的 `/root/.triton/cache` 不会自动搬迁；已删除容器中未另行保存的产物无法靠新配置恢复。引擎可能在Inductor初始化时进一步重定向缓存，完整模型的实际路径仍需看启动证据。

Client 必填：`tool: vllm-bench`、`image`、字符串 `version`。
可选：`image_id`、`environment`、`trust_remote_code`（默认 false）。
客户端目前固定使用 OpenAI completions 流式传输和 `ttft,tpot,itl,e2el` 指标，50/90/95/99 分位数。
工具参数 `--backend vllm` 是这个客户端的协议适配器名称，不限制服务端引擎。
实际入口是 `python3` 调用 `vllm.benchmarks.serve` 的官方 `add_cli_args` / `main`，
绕过顶层 vLLM CLI 在无 GPU 环境构造服务端配置的设备检测错误。客户端容器不申请 GPU。

## Recipe

必填：`compatible`、`mode: smoke|performance`、`flags`、`options`。
`compatible` 声明 `engine`、模型 ID 列表 `models`、GPU 架构字符串列表 `compute_capabilities`。
可选 `runtime_versions` 是允许版本的精确列表；升级后应检查新镜像的参数/行为再扩展列表。

`flags` 是明确的**原生开关名**，不包含 `--`：

```yaml
flags: [trust-remote-code, no-enable-prefix-caching]
options:
  tensor-parallel-size: 8
  max-model-len: 16384
  kernel-config: {enable_flashinfer_autotune: false}
```

标量 options 生成 `--key value`；列表生成一个选项后跟多个参数；字典编码成一个 JSON 参数。
options 顶层不接受 bool/null。不要写 `enforce-eager: false`；不启用该开关就从 flags 中省略它。
需要否定开关时写出实际 CLI 接受的名字，适配器不会自动添加 `--no-`。
模型、served name、host、port 由执行器管理，不能放入 recipe 的 options/flags。

当前资源检查：vLLM 的 TP×PP×DP 等于选中 GPU 数；SGLang 的 TP×PP 等于 GPU 数。
SGLang 的 DP attention 必须显式启用且 DP 整除 TP，不按副本数额外乘 GPU 数。
这些是本执行器支持的单机并行约定；其他分布式模式需新增适配和测试。
每份 recipe 都必须明确配置引擎的上下文长度。

禁用缓存的负载要求：vLLM flags 中含 `no-enable-prefix-caching`；SGLang 含 `disable-radix-cache`。
两者实现和默认行为不同，不通过字符串替换推断其他参数的等价性。

可选 `environment` 覆盖优先级：target < runtime < recipe。客户端使用自己的环境配置。
可选 `provenance` 记录来源和验证范围。

可选 `checks`：

- `required`：必须出现的正则；缺少则失败。
- `forbidden`：不能出现的正则；出现则失败。
- `warnings`：保存并展示的警告；不会被自动忽略，也不直接判失败。
- `compilation`：补充引擎默认编译事件模式。

正则忽略大小写。未设置 required 时检查状态为 UNVERIFIED。
日志检查只验证选择/启动证据，不能等同于 GPU profiler、完整数值精度验证或所有优化均生效。

可选 `probe`：必填 `prompt`、`max_tokens`；可选 `request`（如 chat_template_kwargs）、`expect_regex`。
probe 总是验证非空消息；expect_regex 提供简单语义断言。请求/响应分别落盘。
不同模型的 thinking 开关分别填写，不硬编码到通用 HTTP 流程。

## Workload

必填：`purpose: smoke|calibration|performance`、`dataset`、`traffic`、`sampling`、`measurement`、`cache: disabled`。

```yaml
dataset: {name: random, input_tokens: 8192, output_tokens: 1024, range_ratio: 0}
traffic: {concurrency: [32], requests: 128, request_rate: inf}
sampling: {seed: 0, temperature: 0, ignore_eos: true}
measurement:
  protocol: quick
  budget_s: 3600
  warmup_rounds: 1
  repetitions: 3
  timeout_s: 1800
```

`requests`不小于最大并发；每轮使用此完整请求量，不自动乘4。输入+输出不得超过上下文。固定长度且ignore_eos时核对实际输入输出token总量。同case只启动一次服务，顺序执行负载、并发和轮次。

### 新协议字段（版本1）

| 字段 | 默认与约束 |
| --- | --- |
| `protocol` | 显式选择 `quick`、`jit_clean`、`stable`；新实验默认推荐quick；不省略此字段 |
| `budget_s` | 必填正整数；每workload/并发的总秒数，不含服务启动与最终清理 |
| `repetitions` | 性能/校准默认3且≥3；smoke的quick/jit_clean默认1且≥1；stable至少3。分别表示固定轮数、累计目标或窗口长度 |
| `timeout_s` | 默认1800；单客户端阶段超时，实际还受剩余budget_s限制 |
| `max_rounds` | jit_clean/stable默认12，必须≥repetitions；quick固定等于repetitions |
| `warmup_rounds` | 仅quick接受，默认1，只允许1或2；每轮请求数自动为2×当前并发 |
| `stability_threshold` | 仅stable接受，默认0.02；有限数，0<值≤1，0.01表示1% |

quick保留全部正式样本及事件标记；jit_clean累计最先无已知事件的样本；stable检查连续窗口内output_throughput、mean_ttft_ms、mean_tpot_ms的相对极差均≤阈值。旧预热/重试字段已移除，必须完整替换measurement块；缺少protocol也会报错。不允许把稳定性参数填到其他模式而静默忽略。

选择其他协议只需替换measurement块，例如：

```yaml
# 无已知编译事件验收
measurement:
  protocol: jit_clean
  budget_s: 3600
  max_rounds: 12
```

```yaml
# 稳定性确认
measurement:
  protocol: stable
  budget_s: 3600
  max_rounds: 12
  stability_threshold: 0.02
```

模板包含这三种C32 workload。预算是示例，运行前按模型和单轮耗时调整；不会自动重启、恢复或跨启动拼接。预算用尽标记PARTIAL；正常轮次边界可继续同服务的下一负载，预算中止了运行中的客户端时结束本case，避免残留请求干扰。故障按FAIL处理。完整资格与统计边界见[压测协议](benchmark-methodology.md)。

### 从旧配置切换

`warmup_requests`、`min_warmup_rounds`、`max_warmup_rounds`、`max_attempts`不再支持。整个measurement块替换为上面的三种模式之一；保留原负载、请求量和重复数，并明确总预算。当前仓库内配置已迁移，本地experiments不自动改写，使用前手动新增配置并validate。

功能验证可用 `purpose: smoke` 配合 `protocol: jit_clean`、`repetitions: 1`，不强制测三次。历史结果不重新判定；需原样复现时使用报告记录的源码commit与配置快照。只读的旧结果汇总能力继续保留，不等于保留旧实验执行分支。

## Campaign

必填：`target`、`client`、`workloads`（引用列表）、`cases`。
每个 case 必填 `id`、`model`、`runtime`、`recipe`；不支持隐式覆盖、继承或自动笛卡尔积。

要比较新镜像与旧镜像，保持 target/client/workloads 不变，在 cases 写两个 runtime 引用。
同一 recipe 可服务多个版本，前提是适用范围允许且 CLI preflight 通过。
要比较两个引擎，同一模型分别写对应的 runtime 和 recipe。
case 不必都使用相同模型，但报表不将不同模型数据聚合，也不会自动给出引擎胜负结论。

失败 case 保留证据后通常继续下一 case；如果清理/取证失败，停止 campaign，避免在不确定的资源状态下继续。
任何 case 失败，campaign 最终为 FAIL；仅有预算未完成则为 PARTIAL，两者CLI均返回非零。使用 `--case` 可单独重跑到新的 run 目录。

## 同步多副本部署

可选的case字段 `replica_targets` 指向一或两份target，使用同一model/runtime/recipe。campaign.target声明整机GPU与CPU/内存总预算；副本target明确划分资源，不能隐式继承或覆盖字段。

```yaml
cases:
- id: dual-tp4
  model: models/dsv4-flash-nvfp4.yaml
  runtime: runtimes/vllm-0.29.0.yaml
  recipe: recipes/node8-dual-tp4.yaml
  replica_targets:
  - targets/rtx6000d-48-node8-front.yaml
  - targets/rtx6000d-48-node8-rear.yaml
```

GPU必须不重叠且并集等于整机target；服务和客户端各自的CPU、内存集合并集也必须等于整机预算。副本间CPU及API端口不得重叠，preflight另查跨副本SMT物理核重叠。每副本TP×PP×DP与其GPU数匹配；model路径、硬件声明、缓存根及本机地址必须一致。

workload中的并发、正式请求数均为整机总量，须能被副本数整除；quick的预热量自动取整机并发的两倍。当前支持request_rate=inf和已验证的vLLM 0.29.0客户端；固定全局数据集在计时前按索引交错分片，其他客户端版本或未知官方计时结构拒绝运行。客户端CLI的num-prompts表示全局生成量，实际每侧请求分片见 `request-partition.json`。

每个case将全部副本启动一次，顺序执行所有workload及轮次。每轮同步执行，任一侧事件都会计入整机事件。jit_clean/stable据此拒绝整轮；quick保留并标记。stable依据整机聚合指标判断，单副本明细仍保存；没有额外的旧预热/重试执行路径。

各客户端在官方benchmark计时点同步起跑，保存同宿主monotonic时钟的 `benchmark-window.json`，同步就绪等待最多300秒（不超过客户端阶段超时），起跑偏差不得超过0.5秒。整机窗口为最早开始到最晚结束，不是客户端容器生命周期；吞吐按窗口总量计算。`requests.json`保存成功状态、输入输出tokens、TTFT、延迟和ITL，不保存生成文本。整机分位数由请求/事件合并计算；每次重复之间仍分别统计。

阶段目录包含 `replica-0/`、`replica-1/` 的官方raw/metrics、日志、亲和性及逐请求记录，根层保存整机metrics与窗口。服务证据在case的 `replicas/` 下。`bench report`按部署target、源码和协议指纹分组，不混入旧单服务协议。单副本也可显式使用此模式，作为同协议八卡参照。

当前采用固定等分负载，不包含HTTP负载均衡代理、跨节点编排或PD传输。新模式额外保存资源遥测；CPU高占用提示仅用于观察，资源背景是否可比需要验收。旧campaign不含此字段时保持原单服务路径。

## 自定义日志文件

recipe 中的日志路径是容器路径，声明环境变量不会自动复制宿主文件。当前 SGLang 项目通过 `python3 scripts/prepare_logging.py CAMPAIGN` 将项目 `configs/logging/` 的 JSON 放入对应缓存挂载目录；`--check` 只检查，`--case` 可选择 case。此步骤不改 recipe、启动容器或清空缓存。
