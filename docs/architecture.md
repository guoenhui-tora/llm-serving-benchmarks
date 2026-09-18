# 实现与扩展边界

这是一个单机实验执行器，不包含模型内部代码或长期服务管理。

```text
CLI → config.resolve → resolved plan
                    → runner
                        ├── executor: Docker + NVIDIA preflight/lifecycle
                        ├── engine: native launch, resource/cache checks, log patterns
                        ├── HTTP probes
                        ├── client: fixed load generator, raw-result normalization
                        └── result manifests + report grouping
```

## 模块职责

| 模块 | 负责 |
|---|---|
| `config.py` | 严格配置校验、根目录引用解析、适用条件、负载上下文检查、完整快照 |
| `engines/` | 各引擎的命令入口、原生参数、并行资源和缓存语义、编译事件默认模式 |
| `executors/docker.py` | 镜像、设备、挂载、CLI 检查、取证、容器所属运行及清理 |
| `clients/vllm_bench.py` | 独立客户端命令和结果解析；输入输出协议不跟服务端镜像变化 |
| `clients/jsonl_dataset.py` | 标准库JSONL读取、固定选样、实际tokenizer长度检查及请求清单；官方发送和计时不变 |
| `checks/` | HTTP 协议检查及配置驱动的日志证据 |
| `protocols.py` | 三套协议的轮次接纳、稳定性窗口、总预算及逐轮记录；不持有服务生命周期 |
| `runner.py` | 顺序执行、预热、测量重试、运行状态、异常/中断恢复 |
| `deployment.py` / `clients/synchronized.py` | 本机一或两副本的同步阶段、官方客户端计时点检查、请求分片及整机指标；不实现负载均衡代理 |
| `telemetry.py` | 同步部署的只读CPU/GPU/cgroup时间序列；不自动处理外部任务 |
| `locks.py` | 当前宿主 GPU 索引的非阻塞建议锁 |
| `results/` | 相对路径解析、保守分组、汇总；不需要 GPU 或原始模型路径 |

runner 不判断模型名称，也不转换 vLLM/SGLang 参数。当前直接使用 Docker executor 和固定 vLLM 压测客户端。

## 新增配置通常不改代码

- 新 NVIDIA 硬件：新增 target，按实际拓扑与 kernel 支持建立 recipe。
- 新镜像：新增 runtime，更新 recipe 的适用范围并检查 CLI。
- 新模型：新增 model，建立对应 recipe；不要让模型差异进入 runner。
- 新负载长度/并发：新增 workload；确认上下文、容量和请求量。
- 同一实验不同机器：独立 campaign 或复制后只改 target，保持负载和客户端一致。

## 新能力的代码扩展

- 新引擎：新增 engines 模块，提供 `ENTRYPOINT`、`PREFIX`、`HELP_FLAG`、`server_args(case)`、
  `validate(case)`（返回上下文上限）和 `COMPILATION_PATTERNS`；注册并加入命令/缓存/资源语义测试。
- 新客户端：新增命令构造和结果标准化模块，再增加显式分派。保留原始 JSON，统一单位和缺失值约定。
  不混用两个客户端版本的测量来宣称服务端差异。
- 新 GPU 厂商/执行方式：拆出对应设备检测、透传和环境取证实现，增加 executor 分派；
  不把 NVIDIA 命令换成空操作后继续运行。
- 新数据集/协议：增加请求生成/模板/tokenizer 身份记录及客户端支持，明确长度和采样语义后开放 schema。
- Profiling：在已验证的普通 serving 流程之外新增独立命令；不把带 profiler 开销的结果混入正常 serving 基线。

## 已知测量边界

旧单服务模式的随机负载由固定客户端根据 seed 和 tokenizer 生成；保存完整参数，不保存每条流式 token 事件，因此不能恢复合并请求后的分位数。
显式replica_targets模式增加同步屏障、同一全局请求集的等分分片和逐请求/流式事件记录，可在一次整机测量内合并分位数；不合并不同重复。日志编译检查是启发式证据，不能代替 profiler 或稳定性统计。
JSONL模式在计时前核对文件SHA256、实际tokenizer长度并保存每侧请求清单；按顺序取前N条，同步副本交错分片。文件哈希进入workload分组，项目整体移动不改变此身份。
warmup 不清空 JIT 缓存；prefix/radix cache 则显式关闭。
kernel warnings 保留在结果中；不能把未触发 forbidden 自动解释成所有优化均有效。

长时运行使用 tmux。正常信号会清理容器，SIGKILL/宿主掉电无法保证 finally 执行。
没有自动重试失败请求；三套协议在同次启动内处理JIT轮次并保存全部尝试，不再保留旧预热/重试执行分支。协议日志按每轮落盘；没有跨进程自动恢复或跨启动拼接。
模型配置/tokenizer 哈希及分片大小用于追溯，不等价于完整权重校验和。

## 测试策略

本地测试覆盖配置错配、路径逃逸、重复键、原生开关、GPU 校验、模型缺失分片、所有权清理、GPU 锁、
两引擎顺序运行、真实回环 HTTP、请求校验、编译重试、失败/中断、相对路径迁移和汇总分组。
Docker 和负载生成进程在集成测试中模拟，不能替代真实镜像与模型验证。绑定测试覆盖 SMT 冲突、角色独立配置、容器亲和性证据及绑定不符时的失败清理。
测试配置位于 `tests/fixtures/configs/`，与不断演进的项目配置分开。项目 campaign 另做离线解析和命令检查。实机验证先功能探测，再按约定预算筛选性能候选。
