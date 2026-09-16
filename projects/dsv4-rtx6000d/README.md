# DeepSeek V4 Flash NVFP4 在 RTX6000D 上的推理优化

本项目使用 NVIDIA 发布的 [nvidia/DeepSeek-V4-Flash-0731-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4)，目标是在 RTX6000D 上优化推理吞吐与延迟。先完成 vLLM / SGLang 镜像选型和参数对齐，建立性能基线，再测试并行拓扑与调度配置；当前以单机 8 卡为主。

模型本地路径为 `/data/models/DeepSeek-V4-Flash-0731-NVFP4`。权重使用 NVFP4 routed experts，其余部分保留高精度；优化期间保持同一权重与 tokenizer。

## 当前基线

**选用 vLLM 0.29.0，关闭 FlashInfer autotune，作为后续调优起点。** 2026-09-15 的对齐实验中，输入 8192、输出 1024 tokens，vLLM 在 C16 / C32 的平均输出吞吐为 **532.57 / 651.35 tok/s**，均优于同轮 SGLang autotune off/on。具体延迟、波动与适用边界见选型报告。

保留三套已实测配置：vLLM autotune off、SGLang autotune off、SGLang autotune on。每套在 C16 / C32 各有三次有效重复，共 18 次测量、1728 成功请求、0 失败。

| 内容 | 入口 |
| --- | --- |
| 镜像、参数对齐与性能比较 | [镜像选型](reports/image-selection.md) |
| 18 次测量的逐次指标 | [baseline-samples.csv](data/baseline-samples.csv) |
| 环境准备与运行命令 | [基线复现](reports/reproduction.md) |
| 已遇到的问题和处理方法 | [排查记录](reports/lessons.md) |
| RTX6000D 通信拓扑与带宽 | [互联报告](reports/interconnect.html) |

CSV 中 `case` 区分引擎配置，`C` 为并发，`R` 为重复序号；延迟字段单位为 ms，输出吞吐单位为 tokens/s。

## 48 节点 TP4 对齐结果（2026-09-17）

已完成 gpu-6000d-48 上 SGLang GPU0–3 / vLLM GPU4–7 的三对同步测量。KV分配预算均约32.21 GiB/卡，每实例C32、8192/1024 tokens、128请求；vLLM **708.47 ± 1.30 tok/s**，SGLang **566.78 ± 1.68 tok/s**，vLLM高约25.0%，吞吐CV分别0.184%/0.296%。六次测量768请求全部成功，测量期无已知JIT事件。

结论适用于当前同机双服务及宿主后台CPU负载，较慢引擎存在另一侧空闲的测量尾段；不是独占TP4或TP8基线。预热曾发现无编译日志但吞吐未收敛，以及真实Triton/TileLang JIT，均在正式测量前排除。详见[完整报告、参数与复现入口](reports/tp4-48-aligned.md)、[逐次数据](data/tp4-48-samples.csv)。45 节点结果见下方校准进度，46、47 节点仍待执行。

## 配置与运行

这里的 `configs/` 保存精选 TP8 基线与 45、48 节点 TP4 对齐配置，serving 参数放在 `configs/recipes/`。继续实验时先复制到 `experiments/dsv4-rtx6000d/configs/`，步骤见 [基线复现](reports/reproduction.md)。下面只离线检查归档配置，从仓库根目录执行：

```bash
./bench validate projects/dsv4-rtx6000d/configs/campaigns/48-dsv4-aligned-final-c16-c32.yaml
./bench plan projects/dsv4-rtx6000d/configs/campaigns/48-dsv4-aligned-final-c16-c32.yaml
```

| Campaign | 用途 |
| --- | --- |
| `48-dsv4-aligned-final-c16-c32.yaml` | 三套配置 × C16/C32 × 三次重复，对应已完成的基线实验 |
| `48-dsv4-vllm-baseline-c32.yaml` | 单独选取 vLLM C32，作为优化对照入口；已离线校验 |
| `48-dsv4-functional.yaml` | 两引擎的短功能验证入口；已离线校验，尚未实机运行该 workload |
| `45-dsv4-vllm-tp4-aligned-c32.yaml` | 45 节点 vLLM GPU0–3；已完成三次同步测量，需项目内固定协议协调脚本 |
| `45-dsv4-sglang-tp4-aligned-c32.yaml` | 45 节点 SGLang GPU4–7；与上项成对运行 |
| `48-dsv4-vllm-tp4-aligned-c32.yaml` | 48 节点 vLLM GPU4–7；已完成三次同步测量，需报告中的协调脚本 |
| `48-dsv4-sglang-tp4-aligned-c32.yaml` | 48 节点 SGLang GPU0–3；与上项成对运行 |

实际运行按 [基线复现](reports/reproduction.md) 准备日志配置，再做 preflight/run。TP8 基线 target 对应 48 节点，TP4 campaign 分别引用 45、48 节点的独立 target；运行前需核对本机身份、GPU 分组和路径。

## TP4 节点校准计划（45、48 已完成，46、47 待执行）

四台机器都测试 **vLLM TP4 与 SGLang TP4，autotune 均关闭**。每台八卡分为两个四卡组，各部署一个模型实例，同时压测，比较节点差异和 TP4 下的引擎表现。完成校准后再决定基线及拓扑分工；首轮不重跑 TP8，不测试 PP/EP 或跨节点推理。

2026-09-17 已完成 **gpu-6000d-45**：同机两服务 TP4、每实例 C32，vLLM / SGLang 三次平均输出吞吐 **674.06 / 583.47 tok/s**，CV **0.39% / 0.59%**。KV 存储预算按实测池规模对齐至约 32.43 GiB/卡；六次正式样本共 768 成功请求、0 失败，均无已知测量编译事件。适用范围、预热异常、同步尾段及复现入口见 [45 节点 TP4 报告](reports/tp4-45.md)，逐次数据见 [tp4-45-samples.csv](data/tp4-45-samples.csv)。此结果不替代上面的 TP8 基线。

### 节点分工

| 节点 | GPU0–3 | GPU4–7 |
| --- | --- | --- |
| gpu-6000d-45 | vLLM TP4 | SGLang TP4 |
| gpu-6000d-46 | SGLang TP4 | vLLM TP4 |
| gpu-6000d-47 | vLLM TP4 | SGLang TP4 |
| gpu-6000d-48 | SGLang TP4 | vLLM TP4 |

交错分配避免一个引擎始终占用同一侧 GPU。逐机核对实际 GPU/NUMA 映射，为服务和客户端分配近端 CPU 资源，使用独立端口、容器名、日志和结果目录。

### 配置与测量

每个实例均为 TP4 / PP1 / DP1 / EP off，上下文 16384、活动容量 32、FP8 E4M3 KV、关闭前缀缓存与 autotune。对齐 prefill 预算 8192、decode Graph 模式和捕获尺寸；固定同一模型、镜像、客户端、采样参数，保留各引擎可用的 backend 和 JIT 日志。

**每实例输入 8192、输出 1024 tokens，C32、128 请求、三次有效重复**。因此整机合计 C64，不能与原 TP8 C32 直接比较整机吞吐。沿用 gate：每轮预热 64 请求，至少连续两轮无已知编译事件，最多五轮预热、两次测量尝试。

先检查四卡能否承载正式负载。根据日志估算显存比例，启动后核对实际 KV 分配、缓存策略、余量及抢占，不能直接照搬 TP8 的比例。以同一引擎的跨节点配置一致为目标，正式测量前冻结，跨节点比较前核对实际配置；遇到 OOM 或不支持时留证停止，不单方降低精度或缩短负载。

每台依次完成两个服务的加载、功能探测与预热，再成对开始正式测量。记录双方起止时间和重叠区间，避免一方测量时另一方仍在编译。若一方先结束，注明另一方测量尾段的资源状态。

- [x] **gpu-6000d-45**：两引擎各三次有效测量；[报告与原始结果位置](reports/tp4-45.md)，原始产物仅本机可用。
- [ ] **gpu-6000d-46**：两引擎各三次有效测量，检查日志并填写结果路径。
- [ ] **gpu-6000d-47**：两引擎各三次有效测量，检查日志并填写结果路径。
- [x] **gpu-6000d-48**：两引擎各三次有效测量，见[报告](reports/tp4-48-aligned.md)；原始结果 `experiments/dsv4-rtx6000d/results/tp4-paired-20260917-01/`（仅本机）。
- [ ] 汇总逐次结果、均值和标准差，确定引擎选择与后续拓扑任务。

共计划 **24 次正式测量**，每节点三对，不含预热及失败尝试；45、48 节点已完成其中 12 次。现有 GPU 锁和占用检查按选中 GPU 生效，可使用不重叠的卡组；但 runner 尚无双服务的测量同步机制，target 也未提供 CPU 绑定字段。执行前需准备对应配置与协调方式，不能直接并发启动原八卡 campaign。45 节点使用项目内 [固定协议协调脚本](scripts/paired_run.py)，48 节点使用其[报告](reports/tp4-48-aligned.md)记录的本地协调脚本，分别完成 CPU 绑定、双客户端计时屏障和证据保存。45 节点脚本针对该节点的既定资源分工，不能直接替代 48 节点协议；公共 runner 的能力边界未改变。

### 结果如何使用

目前 45、48 节点的结果分别完成了本机对齐，但尚不是完全相同协议的跨节点对照：45 节点按实测子池的张量布局计算约 32.43 GiB/卡，48 节点按引擎日志分配预算对齐约 32.21 GiB/卡；两台 CPU/NUMA 分配、预热吞吐收敛阈值也不同。应先核对这些差异与宿主后台负载，再判断节点差异，不能直接把吞吐差归因于硬件。

先按同一引擎比较四台，再比较各台两引擎的吞吐、TTFT、TPOT 和 P95。保留每次结果，核对请求/token 总量、JIT、抢占、CPU 负载和 GPU 功耗/频率；三次重复只能反映短期波动。

本轮结论适用于**两个引擎各占四卡、同机同时运行**。若四台排序一致且差距超过波动，可作为引擎选择依据；若排序随 GPU 组变化、差距接近波动或有资源争用，再小范围交换 GPU 组或独占复核。后续拓扑对照也需保持相同资源占用条件，不能直接把双服务结果当作独占基线。

TP4 优先测试，但不预设其比 TP8 快：它减少跨 NUMA 通信，也减少单实例计算、显存带宽和 KV 资源。两个同引擎 TP4 副本的整机性能需单独测量，不能从本轮混合引擎结果推算。
