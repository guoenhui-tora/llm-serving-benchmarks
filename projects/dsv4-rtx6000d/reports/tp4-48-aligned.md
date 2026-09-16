# gpu-6000d-48：vLLM TP4 与 SGLang TP4 对齐结果

**在本轮同机双服务、每实例 C32 / 8192 输入 / 1024 输出的条件下，vLLM 平均输出吞吐比 SGLang 高约 25.0%，且三次重复稳定。** 两边的 TP、活动容量、KV 精度与预算、prefill、Graph 范围及客户端负载已对齐；KV 分配预算均约 **32.21 GiB/卡**。这支持在本机此负载下优先选择本轮 vLLM recipe，不代表官方最优配置或其他节点的结果。

适用范围：2026-09-17（Asia/Shanghai），gpu-6000d-48，SGLang GPU0–3 + vLLM GPU4–7 同时运行，整机合计 C64。每次同时发起等量请求，较快的 vLLM 完成后，SGLang 有约 45–47 秒尾段处于另一侧空闲状态。**CPU 不是独占资源**：宿主有 16 个早于实验启动的 `doca_spcx_cc` 进程，允许使用 CPU0–127；本次未改动它们。结果是在这个后台负载下得到的稳定对比，不能据此排除 CPU 争用，也不能直接当作独占 TP4 或旧 TP8 C32 基线。

## 三次正式测量

六次测量全部通过原 kernel/JIT/工作量检查，无测量重试、无按速度选样。每引擎 384 成功 / 0 失败；实际输入 3,145,728、输出 393,216 tokens。合计 **768 成功 / 0 失败，6,291,456 输入 / 786,432 输出 tokens**。客户端缺失 failed 字段时，以 completed=requested 推断失败数为零，逐次 CSV 保留该标记。

| 引擎 | R1 tok/s | R2 tok/s | R3 tok/s | 均值 ± 样本标准差 tok/s | 吞吐 CV |
|---|---:|---:|---:|---:|---:|
| vllm | 706.97 | 709.22 | 709.23 | 708.47 ± 1.30 | 0.184% |
| sglang | 567.61 | 564.85 | 567.88 | 566.78 ± 1.68 | 0.296% |

| 指标（三次均值 ± 样本标准差） | vLLM | SGLang |
|---|---:|---:|
| 平均 TTFT（秒） | 6.932 ± 0.096 | 7.917 ± 0.018 |
| P95 TTFT（秒） | 20.869 ± 0.027 | 27.390 ± 0.085 |
| 平均 TPOT（ms） | 38.374 ± 0.011 | 48.698 ± 0.178 |
| P95 TPOT（ms） | 42.462 ± 0.033 | 53.238 ± 0.081 |
| 平均 ITL（ms） | 38.374 ± 0.011 | 48.698 ± 0.178 |
| P95 ITL（ms） | 22.099 ± 0.143 | 25.498 ± 0.150 |
| 平均 E2E（秒） | 46.189 ± 0.085 | 57.736 ± 0.171 |
| P95 E2E（秒） | 62.898 ± 0.237 | 81.712 ± 0.251 |
| 请求吞吐（requests/s） | 0.692 ± 0.001 | 0.553 ± 0.002 |

各 P95 是三次独立测量 P95 的统计，不是合并请求后的 P95。vLLM/SGLang 平均 TTFT 的 CV 分别为 1.386% / 0.233%，平均 TPOT 为 0.028% / 0.366%。没有触发预先记录的波动分析标记（吞吐 CV>3%、平均 TPOT>5%、平均 TTFT>10%）。三次重复仅说明本轮短期稳定性，不是长期稳定性或质量评测。

逐次完整指标见 [正式样本 CSV](../data/tp4-48-samples.csv)，统计与资源汇总见 [analysis JSON](../data/tp4-48-analysis.json)。

## 公平性与 KV 设置

固定 `vllm/vllm-openai:v0.29.0` 与 `lmsysorg/sglang:v0.5.19-cu130`，镜像 ID 与既有[基线镜像](image-selection.md)一致。客户端统一使用同一 vLLM 0.29.0 镜像和 `/v1/completions` 流式协议。模型为同一 `/data/models/DeepSeek-V4-Flash-0731-NVFP4`；配置/tokenizer 哈希、48 个分片大小一致，未对完整权重逐字节哈希。

| 控制项 | vLLM | SGLang |
|---|---|---|
| GPU | 4–7 | 0–3 |
| TP / PP / DP / EP | 4 / 1 / 1 / off | 4 / 1 / 1 / ep-size=1 |
| 上下文 / 活动容量 | 16384 / 32 | 16384 / 32 |
| KV dtype | fp8_e4m3 | fp8_e4m3 |
| KV 分配控制 | kv-cache-memory-bytes=34585224151 | mem-fraction-static=0.89 |
| 日志确认的 KV 预算 | 32.21 GiB/卡 | 32.21 GiB/卡 |
| 前缀缓存 / autotune | 关闭 / 关闭 | 关闭 / 关闭 |
| Prefill | max-num-batched-tokens=8192；chunked prefill | chunked-prefill-size=max-prefill-tokens=8192；mixed chunk |
| Decode Graph | FULL_DECODE_ONLY | decode full、prefill disabled |
| Graph 尺寸 | 1,2,4,8,12,16,24,32 | 相同 |
| 模型、客户端 seed / temperature | 0 / 0 | 相同 |
| 输出与流量 | ignore_eos=true；range_ratio=0；request_rate=inf | 相同 |
| Attention / MoE | FLASHINFER_MLA_SPARSE_DSV4 / FLASHINFER_CUTLASS | DeepseekV4AttnBackend / flashinfer_cutlass |
| 模型布局适配 | block-size=256；FP4 indexer off | page-size自动256；swa-full-tokens-ratio=0.1；shared-experts fusion off |

先做启动校准，vLLM `gpu-memory-utilization=0.90` 得到 32.73 GiB/卡，而 SGLang 0.89 得到 32.21 GiB/卡，不能直接视为相同预算。正式 vLLM 新 recipe 固定为 **34,585,224,151 字节/卡**；此参数优先于仍保留的 0.90，日志确认跳过自动显存 profiling，不再由该比例决定 KV 大小。正式启动再次核对两边预算差不超过 1%，实际按日志精度一致。没有通过缩短负载、降低 KV 精度或改变容量取得 PASS。

这里对齐的是引擎日志确认的**分配预算**，不是逐个 tensor 的物理占用完全相等。SGLang 的预算包含不同压缩/状态池，日志 full_token=4,440,064、SWA=443,904、c4=1,110,016、c128=34,688、c128 固定状态约0.34 GiB；vLLM 日志逻辑 KV=84,081 tokens。两边布局和计数定义不同，不能相除宣称容量优势。实际 C32 下两边能保持32个活动请求；vLLM最终计数器抢占=0、abort=0、error=0，SGLang日志未发现请求回撤、OOM或异常。总显存占用不同也不能当作KV预算不同。

两引擎保留各自固定版本的调度与 backend，实现不可能逐项同构。例如 vLLM 日志选择 DSV4 runner，不能仅凭 V1 engine 字样推断 runner/异步语义；SGLang overlap schedule 保持默认开启。没有启用 speculative decoding、CPU offload 或跨节点并行。

## 预热、JIT 与同步

每轮预热64请求，输入/输出和并发与正式负载一致；至少连续2轮无已知编译日志，且最后2轮吞吐差≤3%，每次尝试最多5轮、每对最多2次测量尝试。若任一边测量出现JIT，整对拒绝并在预算内重试；本轮三对均第一次测量即通过。原 gate 未放宽。两边的缓存均从各自 TP4 校准缓存复制到正式 recipe 的独立缓存，原缓存保留。

- SGLang 第一组预热为443.49、571.50、563.99 tok/s，编译匹配均为0。前两轮相差28.9%，因未收敛继续第三轮；第二/三轮相差1.31%才通过。**日志没有编译事件并不等于性能稳定**；无法仅凭现有日志把首轮差异全部归因于JIT。
- vLLM 第一组为446.21、507.06、699.39、706.63 tok/s；前两轮分别36、24条编译相关日志，后两轮为0。Triton/TileLang事件保存在原始窗口，计数是日志条数，不是独立编译次数。
- 后两组双方各2轮预热即通过。共15轮、960个预热请求，与768个正式请求分开保存。完整[预热 CSV](../data/tp4-48-warmups.csv)保留未收敛轮次。vLLM JIT monitor verbose 与明确的 TileLang 开始/完成匹配同时启用；SGLang 使用既有自定义 INFO/编译日志设置。

两服务依次加载、探测与预热，之后客户端初始化完成，在原版 benchmark 计时前通过文件屏障共同释放；计时后仅保存墙钟边界，负载与指标公式不变。固定客户端实际日志为 `Skipping endpoint ready check`，其内部 warmup=0；功能探测由仓库完成，负载预热由外部独立阶段完成，不声称客户端额外执行了完整单请求预热。原/插桩函数哈希及源码逐阶段保存，插桩使用同一固定源码，并经过镜像内离线检查。

| 重复 | 起点偏差（微秒） | 同时测量（秒） | SGLang 尾段、vLLM 空闲（秒） |
|---|---:|---:|---:|
| 1 | 5.01 | 185.40 | 45.52 |
| 2 | 7.63 | 184.81 | 47.24 |
| 3 | 1.67 | 184.81 | 46.00 |

## 资源与观察边界

服务与客户端 CPU 采用对称且互不重叠的配置；运行中读取 worker 的 `/proc` 亲和性，确认实际生效：

| 引擎 | 服务 CPU / 内存 NUMA | 客户端 CPU / 内存 NUMA |
|---|---|---|
| SGLang | 0–15,64–79 / 0 | 16–19,80–83 / 1 |
| vLLM | 32–47,96–111 / 2 | 48–51,112–115 / 3 |

客户端位于对应 socket 的另一 NUMA，避免与服务共享同一组 CPU。Docker cpuset 只约束本次进程，不排斥宿主后台进程。5秒采样的服务 CPU 集合整体忙碌率约70%/71%，客户端集合约71%/69%（SGLang/vLLM）；这些包含全部宿主进程，不能解释成客户端自身CPU利用率或据此排除客户端瓶颈。16个root所有的`doca_spcx_cc`进程允许使用全部CPU，启动时间早于本次实验；其存在是本轮限制，不应归咎于引擎。

正式窗口未观察到外部 GPU 计算进程。GPU 利用率采样均值约99%；温度41–59℃，SM频率SGLang 2317–2422 MHz、vLLM 2415–2422 MHz，采样功耗最高约280 W/卡；没有修改600 W功耗上限或锁频。总显存占用SGLang 79,811 MiB/卡，vLLM 78,431–78,495 MiB/卡。5秒采样不能排除更短的干扰。

保留全部 warning：SGLang FP8 KV scale默认1.0、checkpoint实验性提示、tokenizer属性提示、PCIe multicast回退；vLLM平台通信/Graph等提示。功能短回答通过不代表量化质量已评测。双方kernel检查PASS，完整日志没有OOM或异常堆栈；日志gate是已知模式检查，不是所有kernel的profiler证明。

## 配置与本机复现入口

正式配置：[vLLM campaign](../configs/campaigns/48-dsv4-vllm-tp4-aligned-c32.yaml)、[SGLang campaign](../configs/campaigns/48-dsv4-sglang-tp4-aligned-c32.yaml)；全部配置依赖留在项目configs内部。原TP8 recipe保留。

```bash
./bench validate projects/dsv4-rtx6000d/configs/campaigns/48-dsv4-vllm-tp4-aligned-c32.yaml
./bench validate projects/dsv4-rtx6000d/configs/campaigns/48-dsv4-sglang-tp4-aligned-c32.yaml
./bench plan projects/dsv4-rtx6000d/configs/campaigns/48-dsv4-vllm-tp4-aligned-c32.yaml
./bench plan projects/dsv4-rtx6000d/configs/campaigns/48-dsv4-sglang-tp4-aligned-c32.yaml
```

继续运行先按根README建立或续用实验工作区，准备SGLang日志配置，再检查本机GPU/端口和固定工件。本轮**本地协调脚本不属于公共runner，也未随精选配置打包**；普通`bench run`不提供双服务计时同步/CPU绑定，不能把独立运行两个campaign称为复现本轮。以下命令依赖当前机器保留的协调脚本（其快照也保存在原始run目录），换机器需先迁移并检查这些本地工具：

```bash
python3 scripts/prepare_logging.py experiments/dsv4-rtx6000d/configs/campaigns/48-dsv4-sglang-tp4-aligned-c32.yaml --check
# 在 tmux 中执行；new-run目录必须尚不存在。
python3 experiments/dsv4-rtx6000d/tools/paired_tp4.py \
  experiments/dsv4-rtx6000d/configs/campaigns/48-dsv4-sglang-tp4-aligned-c32.yaml \
  experiments/dsv4-rtx6000d/configs/campaigns/48-dsv4-vllm-tp4-aligned-c32.yaml \
  --run-root experiments/dsv4-rtx6000d/results/new-run
```

以下链接为**仅本机保留、不随Git分发**的原始产物：

- [正式run](../../../experiments/dsv4-rtx6000d/results/tp4-paired-20260917-01/run.json)：两服务完整日志、逐阶段raw/metrics/compilation、镜像/模型身份、实际命令、源码指纹、协调脚本/客户端插桩快照、5秒遥测、亲和性、最终抢占计数。
- [启动校准](../../../experiments/dsv4-rtx6000d/results/tp4-calibration-20260917-01/run.json)：两引擎原始TP4显存比例的证据，未混入性能统计。
- [冻结前计划](../../../experiments/dsv4-rtx6000d/reports/tp4-plan.md)、[本地协调脚本](../../../experiments/dsv4-rtx6000d/tools/paired_tp4.py)、[客户端计时封装](../../../experiments/dsv4-rtx6000d/tools/client_sync.py)。

协调脚本4项离线检查覆盖三对测量、单边JIT整对重试、预热上限停止和CPU命令生成；固定镜像内验证计时封装；归档campaign再次validate/plan。公共src未修改，结束后本次容器全部清理，八卡空闲；模型、镜像、历史结果和JIT缓存均保留。
