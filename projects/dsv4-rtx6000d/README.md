# DeepSeek V4 Flash NVFP4 在 RTX6000D 上的推理优化

本项目使用 NVIDIA 发布的 [nvidia/DeepSeek-V4-Flash-0731-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4)，目标是在 RTX6000D 上优化推理吞吐与延迟，当前以单机为主。权重为NVFP4 routed experts与高精度其他部分的混合格式；实验保持同一权重与tokenizer。

**当前继续优先调优vLLM 0.29.0，关闭FlashInfer autotune。** TP8基线和四节点TP4对照均支持这一选择；TP4结果呈现与GPU分组一致的两组性能，下一步需要在同机控制绑定条件后比较拓扑，不能直接将分组差异解释为硬件或引擎的固有差异。

## TP4 四节点结果

2026-09-17，四台各部署vLLM TP4和SGLang TP4，同时压测。每实例输入8192、输出1024 tokens，C32、128请求、三次有效重复；整机合计C64。指标由四份逐次CSV重算，运行条件及异常依据各节点提交报告整理。

共同条件为TP4/PP1/DP1、EP off，上下文16384、活动容量32、FP8 E4M3 KV、关闭前缀缓存和autotune；prefill预算8192、chunked/mixed prefill，decode full Graph，捕获尺寸1/2/4/8/12/16/24/32，prefill Graph不启用。vLLM使用FLASHINFER_MLA_SPARSE_DSV4/FLASHINFER_CUTLASS；SGLang使用DeepseekV4AttnBackend/flashinfer_cutlass并关闭共享专家融合。block/page为256，FP4 indexer关闭，SGLang SWA/full ratio为0.1。未启用投机解码或CPU offload。

镜像固定为`vllm/vllm-openai:v0.29.0`、`lmsysorg/sglang:v0.5.19-cu130`，身份见[镜像记录](reports/image-selection.md#固定镜像与-recipe)。客户端统一vLLM 0.29.0、流式`/v1/completions`，seed=0、temperature=0、ignore_eos=true、range_ratio=0、request_rate=inf。

吞吐为三次均值 ± 样本标准差；延迟列为三次对应指标的均值，**P95 TTFT不是合并请求后的P95**。

| 节点 | 引擎 | GPU | 输出 tok/s | CV | Mean TTFT 秒 | P95 TTFT 秒 | Mean TPOT ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 45 | vLLM | 0–3 | 674.06 ± 2.63 | 0.39% | 7.32 | 22.67 | 40.30 |
| 45 | SGLang | 4–7 | 583.47 ± 3.45 | 0.59% | 7.49 | 25.83 | 47.50 |
| 46 | vLLM | 4–7 | 705.14 ± 4.32 | 0.61% | 6.93 | 20.85 | 38.59 |
| 46 | SGLang | 0–3 | 565.00 ± 1.01 | 0.18% | 7.90 | 27.32 | 48.89 |
| 47 | vLLM | 0–3 | 676.05 ± 3.22 | 0.48% | 7.17 | 22.80 | 40.31 |
| 47 | SGLang | 4–7 | 585.83 ± 2.61 | 0.45% | 7.45 | 25.72 | 47.32 |
| 48 | vLLM | 4–7 | 708.47 ± 1.30 | 0.18% | 6.93 | 20.87 | 38.37 |
| 48 | SGLang | 0–3 | 566.78 ± 1.68 | 0.30% | 7.92 | 27.39 | 48.70 |

24次有效测量共 **3072成功、0失败，输入25,165,824、输出3,145,728 tokens**。每个样本均完成128请求和131,072输出tokens。每轮预热64请求、最多五轮，至少连续两轮无已知编译事件；各节点额外检查与重试差异见下文。节点内vLLM吞吐领先约15.4%–25.0%，上表延迟指标也更低；吞吐CV均低于0.62%。有效样本未报告OOM或请求超时。46首对的第一次测量因JIT整对拒绝；47两轮诊断未计入正式数据。

| 节点 | 条件、逐次统计与异常 | 原始精度的逐次指标 |
| --- | --- | --- |
| 45 | [tp4-node45.md](reports/tp4-node45.md) | [tp4-node45.csv](data/tp4-node45.csv) |
| 46 | [tp4-node46.md](reports/tp4-node46.md) | [tp4-node46.csv](data/tp4-node46.csv) |
| 47 | [tp4-node47.md](reports/tp4-node47.md) | [tp4-node47.csv](data/tp4-node47.csv) |
| 48 | [tp4-node48.md](reports/tp4-node48.md) | [tp4-node48.csv](data/tp4-node48.csv) |

### GPU 分配与性能分组

**45/47是前四卡vLLM、后四卡SGLang；46/48恰好相反。** 按实际绑定分组后，同组两节点的各引擎吞吐均值差均不到0.5%，跨组差异则更明显：

| 分组 | GPU0–3 | GPU4–7 | vLLM节点均值的平均 tok/s | SGLang节点均值的平均 tok/s |
| --- | --- | --- | ---: | ---: |
| 45、47 | vLLM | SGLang | 675.06 | 584.65 |
| 46、48 | SGLang | vLLM | 706.80 | 565.89 |

46/48相对45/47，vLLM约高 **4.70%**，SGLang约低 **3.21%**；也就是两种引擎在GPU4–7上的结果都高于各自在GPU0–3上的结果。组均值仅用于描述这一模式，不合并为新的性能基线。TTFT/TPOT也呈同方向变化。

**这表明存在与绑定分组一致的性能差异，但不能据此确定GPU4–7本身更快。** 每个节点没有交换两种引擎的位置，且CPU布局、内存绑定、KV控制和观测协议也随分组变化：

| 节点 | 每侧服务/客户端物理核 | 内存绑定 | KV条件（vLLM / SGLang，每卡） |
| --- | --- | --- | --- |
| 45 | 14 / 2 | 服务和客户端均在GPU近端NUMA | 32.43 GiB显式预算 / 32.43 GiB张量重建值 |
| 46 | 16 / 8 | 未强绑定内存；客户端在同socket另一NUMA | 32.21 / 32.21 GiB日志分配预算 |
| 47 | 12 / 4 | 服务和客户端均在GPU近端NUMA | 32.73 GiB自动预算 / 32.43 GiB张量重建值 |
| 48 | 16 / 4 | 服务和客户端分别绑定同socket的两个NUMA | 32.21 / 32.21 GiB日志分配预算 |

SGLang张量重建值与日志预算是不同统计口径，不能当作可直接比较的物理显存占用。45/48额外要求末两轮预热吞吐差≤5%/3%；47增加编译产物检查，46增加编译器进程采样。各节点保留了这些协议差异，不能声称四台除GPU位置之外完全一致。

46/48明确记录了`doca_spcx_cc`后台CPU负载；47的CPU集合忙碌率也包含其他宿主活动，45缺少同口径量化。CPU cpuset只限制本次进程，不会排除后台任务。vLLM先结束后，SGLang仍有约29–47秒尾段；两侧完整窗口吞吐不能直接相加为整机吞吐，也不能用于证明独占TP4优于TP8。

有价值的后续验证是在同一节点保持CPU核数、客户端位置策略、内存约束和KV配置一致，交换引擎所在GPU组，或做四卡独占对照。当前结果足以支持优先调优vLLM；GPU位置效应与TP4相对TP8的纯拓扑收益仍待分离。

## TP8 基线与复现

2026-09-15的单机八卡、8192/1024对齐实验中，vLLM在C16/C32的输出吞吐为 **532.57 / 651.35 tok/s**。三套配置（vLLM off、SGLang off/on）各三次重复，合计18次测量。它是独立的TP8基线，不与上面的双TP4样本混合。

| 内容 | 入口 |
| --- | --- |
| TP8镜像、参数与性能表 | [镜像选型](reports/image-selection.md) |
| TP8逐次指标 | [baseline-samples.csv](data/baseline-samples.csv) |
| 环境与TP8运行命令 | [基线复现](reports/reproduction.md) |
| 已知问题 | [排查记录](reports/lessons.md) |
| RTX6000D通信测量 | [互联报告](reports/interconnect.html) |

`configs/` 保留三套TP8 recipe及其依赖：完整对比入口为`48-dsv4-aligned-final-c16-c32.yaml`，仅vLLM C32为`48-dsv4-vllm-baseline-c32.yaml`；`48-dsv4-functional.yaml`是已离线检查、尚未实机运行该负载的短验证入口。

TP4此次仅保留每节点CSV与报告，不提供双服务同步脚本或专用配置入口。继续研究时，按[工作区流程](../../README.md#先建立本地实验工作区)在`experiments/`新建候选，不能把原TP8配置直接当作TP4复现配置。
