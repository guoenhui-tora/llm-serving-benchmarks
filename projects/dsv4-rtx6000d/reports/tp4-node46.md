# 46 节点：vLLM / SGLang TP4 对齐结果

**2026-09-17，gpu-6000d-46 同机双服务测量中，vLLM TP4 平均输出吞吐为 705.14 ± 4.32 tok/s，SGLang TP4 为 565.00 ± 1.01 tok/s，vLLM 高 24.80%。** 三次吞吐 CV 分别为 0.61% / 0.18%，KV 分配预算均约 32.21 GiB/卡。

结论适用于本节点当前 CPU 背景负载下、每个引擎占四卡且同时运行、每实例 C32 的条件。CPU 背景负载未隔离且未完全归因，吞吐差不能全部归因于引擎。GPU 分组未交换复核，本轮也不直接代表独占 TP4、TP8 或两个同引擎副本的整机性能。

## 三次结果

“±”为三次测量的样本标准差，CV 为样本标准差除以均值。吞吐使用各客户端完整测量窗口；延迟为三次对应指标的均值。

| 引擎 | R1 tok/s | R2 tok/s | R3 tok/s | 平均输出吞吐 tok/s | 吞吐 CV | Mean TTFT 秒 | Mean TPOT ms | Mean E2E 秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM TP4 | 700.32 | 706.44 | 708.66 | 705.14 ± 4.32 | 0.61% | 6.927 ± 0.090 | 38.595 ± 0.349 | 46.410 ± 0.285 |
| SGLang TP4 | 566.13 | 564.20 | 564.67 | 565.00 ± 1.01 | 0.18% | 7.904 ± 0.007 | 48.889 ± 0.107 | 57.918 ± 0.103 |

六次有效测量共 **768 成功请求、0 失败，实际输入 6,291,456 / 输出 786,432 tokens**；逐请求长度均为 8192/1024。R1 为第二次尝试，R2/R3 为第一次尝试。全部逐次指标见 [CSV](../data/tp4-node46-samples.csv)：延迟字段单位为 ms，duration 为秒，吞吐为 tokens/s，时间戳为 UTC。

vLLM / SGLang 的吞吐最大值相对最小值分别高 1.19% / 0.34%；Mean TTFT CV 为 1.30% / 0.08%，Mean TPOT CV 为 0.90% / 0.22%。已报告延迟字段的最大 CV 为 vLLM P95 ITL 的 2.41%。三次重复支持本轮短期稳定性，不能据此排除系统干扰。

## 实测参数与 KV 对齐

| 控制项 | vLLM | SGLang |
| --- | --- | --- |
| 镜像版本 | 0.29.0 | 0.5.19-cu130 |
| GPU / 服务 CPU 集 | GPU4–7 / 32–47、96–111 | GPU0–3 / 0–15、64–79 |
| 客户端 CPU 集 | 48–55、112–119 | 16–23、80–87 |
| TP / PP / DP / EP | 4 / 1 / 1 / off | 同左 |
| 上下文 / 活动请求上限 | 16384 / 32 | 同左 |
| 权重 / 模型 dtype | NVFP4 混合精度 checkpoint / BF16 | 同左 |
| KV 精度 | FP8 E4M3 | 同左 |
| KV 分配预算 | kv-cache-memory-bytes=34585224151，约 32.21 GiB/卡 | mem-fraction-static=0.89，启动日志约 32.21 GiB/卡 |
| 前缀缓存 / autotune | 均关闭 | 均关闭 |
| Prefill | chunked，max-num-batched-tokens=8192 | mixed chunk，chunk/max-prefill 均 8192 |
| Decode Graph | FULL_DECODE_ONLY，保留镜像自动 breakable 路径 | full decode，prefill Graph disabled |
| Graph 捕获尺寸 | 1/2/4/8/12/16/24/32 | 同左 |
| 每实例负载 | 输入 8192 / 输出 1024 tokens，C32，128 请求 × 3 | 同左 |
| 客户端 | vLLM 0.29.0，流式 completions，seed=0、temperature=0、ignore_eos=true、rate=inf | 同左 |

模型与 tokenizer 均来自 `/data/models/DeepSeek-V4-Flash-0731-NVFP4`。模型身份为 `1e1b7e54442bcf7f15ef55192641b5c5da360bcaa608236aa0d06b273aedf8b4`，依据配置/tokenizer 哈希及权重分片大小，未逐字节哈希全部权重。固定镜像 ID：

- vLLM：`sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`
- SGLang：`sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9`

校准时 vLLM 的 gpu-memory-utilization=0.90 自动得到 32.73 GiB KV，因此正式测量改成显式字节预算；该比例参数仍保留，但不再决定 KV 大小。四个 worker 均确认预留 32.21 GiB。SGLang 四个 worker 的 available_bytes 同为约 32.21 GiB，包含约 0.34 GiB 的 c128 固定状态；日志虽写 GB，实际除数为 2³⁰。

对齐的是分配字节预算。两边压缩 KV、状态、padding 和逻辑 token 计数的实现不同，不能据此声称有效 token 容量完全相同。正式负载均达到 32 个活动请求；vLLM 抢占计数采样为 0，完整日志未发现抢占、SGLang retract 或 OOM。

## JIT 与测量窗口

每轮预热 64 请求，要求连续两轮无已知编译事件，最多五轮预热；每次重复最多两次测量尝试。双方串行完成加载与预热后，成对启动正式客户端，双方结束后才进入下一组。

首轮 vLLM 预热检测到 20 条编译事件，随后两轮为零，但第一次正式尝试仍出现 8 条编译事件，独立进程采样也捕获了编译器活动。该次两侧结果整对拒绝并保留，双方重新预热后第二次尝试通过；其余两对首次通过，没有按吞吐挑选样本。

最终共保存 vLLM 9 轮、SGLang 8 轮预热。六份有效窗口均无已知编译事件，1 秒间隔编译器采样也无匹配记录。SGLang 首轮预热虽无已知事件，吞吐仍从约 448 提升至后续约 567 tok/s，因此稳定性同时依据三次实测波动。校准及失败尝试的 JIT 缓存均保留，未放宽 gate。

按逐请求明细重建窗口，两侧实际起点差为 0.695 / 0.665 / 0.710 秒。vLLM 窗口 100% 与 SGLang 重叠；SGLang 窗口重叠比例为 80.84% / 79.86% / 79.68%，尾段为 43.67 / 46.11 / 46.45 秒。尾段 vLLM 服务仍驻留但无压测请求，也未开始下一轮预热；两个独立窗口的 tok/s 不能直接相加为统一窗口的整机吞吐。

## 环境限制与本机证据

有效窗口的 vLLM / SGLang 平均功耗为 259.20 / 260.00 W/卡，平均 SM 频率为 2418.18 / 2421.12 MHz，显存余量为 6546–6610 / 5230 MiB/卡。GPU 无其他计算任务，功耗上限保持 600 W/卡，未改变宿主锁频或功耗设置。

CPU 背景负载是本轮主要限制：本次容器全部退出后，宿主 CPU 忙碌率仍约 63%。事后发现 16 个实验前已存在的 doca_spcx_cc 进程，每个约占一个逻辑 CPU，允许 CPU 为 0–127，但它们不能完整解释总负载。有效窗口服务 CPU 集忙碌率为 71.10% / 69.70%，客户端 CPU 集为 65.31% / 65.45%；这些是集合内所有宿主进程的占用，不能当作客户端自身占用。cpuset 约束本次进程，但未隔离系统服务；未使用内存 NUMA 强绑定，也未修改这些后台服务。

两侧 kernel gate 均通过，warning 原样保留，包括 PCIe custom all-reduce / multicast 回退及 FP8 KV scale 提示。本轮没有进行完整模型质量评测。

本次仅归档报告与逐次数据；精确配置、执行代码和原始证据仅保存在 46 本机，不随本提交分发。以下路径相对于仓库根目录，复跑需使用本机留存的执行版本与完整配置：

- 正式结果：`experiments/dsv4-rtx6000d/results/46-tp4-aligned-c32-20260917-01/`
- 校准结果：`experiments/dsv4-rtx6000d/results/46-tp4-calibration-20260917-01/`
- 实测配置：`experiments/dsv4-rtx6000d/configs/`
- 正式执行源码：`experiments/dsv4-rtx6000d/reports/source-formal.tar.gz`；运行源码指纹为 `2364a3650b782926a5f700832769f1ca6a0f964b4685fa6b716e1371f35f1756`。
- 分析脚本、审计和 GPU/CPU/编译器采样：`experiments/dsv4-rtx6000d/reports/`
- JIT 缓存：`experiments/dsv4-rtx6000d/cache/`
