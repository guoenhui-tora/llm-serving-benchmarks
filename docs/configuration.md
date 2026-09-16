# 配置格式

每类配置独立变化，全部使用 `schema_version: 1` 和对应 `kind`。未知字段、重复键、非有限数值、越界引用直接失败。
实验工作区使用 `experiments/<项目>/configs/`，精选配置使用 `projects/<项目>/configs/`；两者都作为独立配置根。CLI 默认从 campaign 的父目录寻找名为 `configs` 的目录，也可用 `--config-root` 显式指定。路径引用统一相对于配置根目录。可以使用子目录整理文件；程序不从文件名推断引擎、模型或硬件。

## Target

必填：`executor: docker`、`address`（本机 IP）、`gpus`（显式索引列表）、`gpu`、`model_root`、`cache_root`、`port`。
`gpu` 必填 `vendor: nvidia`、`name_regex`、字符串 `compute_capability` 和 `min_memory_mib`。
`address` 用于防止在错误机器启动，执行器不会自动 SSH。

可选：`model_paths: {model-id: /absolute/path}`、`environment`、`ulimits`、`cap_add`。
环境变量值必须是字符串。模型根目录下使用 `model.directory` 找权重；`model_paths` 可替换某个模型的路径。
目标固定使用 host network、host IPC，HTTP 绑定 `127.0.0.1`。未实现其他网络模式。
硬件参数只用于校验，不会自动猜测 NUMA、TP/EP 或 kernel 最优值。

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

必填：

- `purpose: smoke|calibration|performance`。
- `dataset: {name: random, input_tokens: 128, output_tokens: 32, range_ratio: 0}`，range_ratio 可省略。
- `traffic: {concurrency: [1, 2, 4], requests: 32, request_rate: inf}`；requests 不小于最大并发。
- `sampling: {seed: 0, temperature: 0, ignore_eos: true}`。
- `measurement: {warmup_requests: 8, repetitions: 3}`。
- `cache: disabled`。

measurement 可选：`min_warmup_rounds`（连续安静轮数，默认 2）、`max_warmup_rounds`（默认 4）、
`max_attempts`（默认 2）、`timeout_s`（单次客户端阶段，默认 1800）。
实际每轮预热请求数取 max(warmup_requests, 当前并发)。输入+输出长度不得超过 recipe 上下文长度。
固定输出且 ignore_eos 时检查返回的 token 总量，防止协议差异悄悄改变测试负载。
相同 workload 列表在同一 case 的一次服务启动中顺序运行，不会为每个并发重启模型。

## Campaign

必填：`target`、`client`、`workloads`（引用列表）、`cases`。
每个 case 必填 `id`、`model`、`runtime`、`recipe`；不支持隐式覆盖、继承或自动笛卡尔积。

要比较新镜像与旧镜像，保持 target/client/workloads 不变，在 cases 写两个 runtime 引用。
同一 recipe 可服务多个版本，前提是适用范围允许且 CLI preflight 通过。
要比较两个引擎，同一模型分别写对应的 runtime 和 recipe。
case 不必都使用相同模型，但报表不将不同模型数据聚合，也不会自动给出引擎胜负结论。

失败 case 保留证据后通常继续下一 case；如果清理/取证失败，停止 campaign，避免在不确定的资源状态下继续。
任何 case 失败，campaign 最终为 FAIL。使用 `--case` 可单独重跑到新的 run 目录。

## 自定义日志文件

recipe 中的日志路径是容器路径，声明环境变量不会自动复制宿主文件。当前 SGLang 项目通过 `python3 scripts/prepare_logging.py CAMPAIGN` 将项目 `configs/logging/` 的 JSON 放入对应缓存挂载目录；`--check` 只检查，`--case` 可选择 case。此步骤不改 recipe、启动容器或清空缓存。
