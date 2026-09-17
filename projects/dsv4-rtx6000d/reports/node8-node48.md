# 48：八卡整机部署对照

## 结论与完成范围

**本节点完整通过的配置是 TP8 和 TP4×PP2；双 TP4 baseline 与 TP2×PP4 均在 C64 达到编译事件重试上限，未获得完整通过的 baseline。** 因而本轮不能完成四种部署的完整排名，也不能据此选择整机最优配置。

在两种完整通过的单套八卡配置中，TP4×PP2 相对 TP8 的整机吞吐在 C32/C64 分别提高 **11.45% / 32.47%**。TP4×PP2 的 C32 CV 为 **2.95%**，第三次明显高于前两次，结论应保留波动限制。TP2×PP4 的 C32 部分结果 CV 为 **7.65%**，不能视为稳定基线。

范围限定为本机 `gpu-6000d-48`、`10.90.1.48`，8×RTX 6000D、8192 输入／1024 输出，统一 CPU/NUMA 预算，保留既有 DOCA 后台负载。运行时间为北京时间 **2026-09-17 20:15 至 2026-09-18 00:14**，约四小时。未访问其他节点，未扩展参数矩阵、修改 gate 或追加重试。

| 顺序 / 配置 | C32 | C64 | 最终状态 / CSV |
| --- | --- | --- | --- |
| 1 / 双 TP4 | 三次通过测量 gate | 第一次重复的两次尝试均被 JIT gate 拒绝；第二、三次重复未运行 | FAIL；部分结果单列，不导出 |
| 2 / TP8 | 三次通过 | 三次通过 | PASS；导出六次 |
| 3 / TP4×PP2 | 三次通过 | 三次通过 | PASS；导出六次 |
| 4 / TP2×PP4 | 三次通过测量 gate，CV 7.65% | 第一次重复的两次尝试均被 JIT gate 拒绝；第二、三次重复未运行 | FAIL；部分结果单列，不导出 |

共完成 **22 次正式尝试：18 次通过测量 gate、4 次拒绝**。其中只有完整 PASS 配置的 **12 次**进入[逐次 CSV](../data/node8-node48.csv)；另外六次属于失败 case 的 C32 部分证据，不混入正式 CSV。四个配置均只启动一次，在同一套服务内顺序执行负载及重复。

## 运行条件与实际参数

配置入口为[本节点 campaign](../configs/campaigns/48-dsv4-node8.yaml)，完整统一协议见[项目 README](../README.md)。本轮原样复制公共 configs 到实验工作区，未修改公共代码或运行配置；以下是实测条件，不能与历史未绑定八卡结果合并。

| 项目 | 固定条件及核验 |
| --- | --- |
| 执行 commit | `aacdc39f0862b82527753ce3ac5cbf0de94c1c68`；开始时 Git 工作树干净 |
| runner 源码指纹 | 两个 run 均为 `edccf507e1246a4525a383ff4c4d3ce45a470378075bf3271922903c10020e34` |
| 服务 / 客户端 | `vllm/vllm-openai:v0.29.0`；image ID `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`，RepoDigest 同值 |
| 权重 / tokenizer | `/data/models/DeepSeek-V4-Flash-0731-NVFP4`；元数据、tokenizer 文件哈希及分片大小身份 `1e1b7e54442bcf7f15ef55192641b5c5da360bcaa608236aa0d06b273aedf8b4`，未全量计算权重内容哈希 |
| GPU | 0–7；每卡 85,651 MiB，驱动 580.159.04；开始和结束计算进程均为空；未改功耗或锁频 |
| 模型与调度 | 上下文 16384，FP8 E4M3 KV，显存比例 0.90 自动分配；每 DP rank prefill 8192；V2、async、mp executor、chunked prefill |
| 后端 | `FLASHINFER_MLA_SPARSE_DSV4`、日志实际 `FLASHINFER_CUTLASS` NVFP4 MoE、`fp8_ds_mla`；block size 256，FP4 indexer 关闭 |
| 关闭项 | FlashInfer autotune、prefix cache、EP、投机解码和 CPU offload；不启用 `--numa-bind` |
| Graph | `FULL_DECODE_ONLY`；双 TP4 每侧容量 32，尺寸 `[1,2,4,8,12,16,24,32]`；单套八卡容量 64，再含 `[48,64]`；实际启动保留 Graph 捕获 |
| 客户端 | 同一固定 vLLM 客户端和同步流程，流式 `/v1/completions`；seed 0、temperature 0、ignore_eos、range_ratio 0、request_rate=inf |
| 预算 | C32：128 正式请求、每轮预热 64；C64：256 正式请求、每轮预热 128；各三次重复，每次至少连续两轮安静、最多五轮预热、两次测量尝试；启动 1800 秒，客户端阶段 7200 秒 |

启动前四个 case 均通过 validate/plan、固定镜像与实际 CLI preflight；无自定义日志文件需要准备。各服务 health、models、关闭 thinking 的简短中文探测通过。每轮亲和性检查通过；测量期间源码、配置及缓存策略保持不变，未清空 JIT 缓存。

| 部署 | 服务 CPU / 内存 NUMA | 客户端 CPU / 内存 NUMA | API 端口 |
| --- | --- | --- | --- |
| 双 TP4 前侧 GPU0–3 | 0–15 / 0 | 16–19 / 1 | 31248 |
| 双 TP4 后侧 GPU4–7 | 32–47 / 2 | 48–51 / 3 | 31249 |
| 单套八卡 GPU0–7 | 0–15、32–47 / 0、2 | 16–19、48–51 / 1、3 | 31248 |

服务共 32 个物理核、客户端共 8 个物理核，每核一个线程，SMT 兄弟不分配给本次进程。Docker inspect 和线程允许集合均落盘。NUMA 页面快照也已保存，能看到共享页面驻留在允许集合以外的节点；cpuset 内存集合不等于全部页面本地驻留。

通过启动日志的 PID、global/local rank 和 `Worker_PP*_TP*` 名称交叉核对 GPU 映射：

| 配置 | 实际组与 rank 映射 | 日志 KV 预算 / 逻辑 token 容量 |
| --- | --- | --- |
| 双 TP4 | 两套独立 world size 4；各 local rank 0–3 对应本侧四卡，DP1/PP1，EP off | 每侧 rank0 均报 32.73 GiB / 每侧 85,450 tokens |
| TP8 | world size 8，rank 0–7 对应 GPU0–7；一个 TP8 组，DP1/PP1，EP off | rank0 报 51.72 GiB / 135,014 tokens |
| TP4×PP2 | PP0 的 TP0–3 对应 GPU0–3，PP1 的 TP0–3 对应 GPU4–7；PP 链为同一 TP rank 的前后卡 | rank0 报 52.05 GiB / 261,832 tokens |
| TP2×PP4 | PP0/1/2/3 的 TP0–1 分别对应 GPU0–1 / 2–3 / 4–5 / 6–7；PP 链为 0–2–4–6、1–3–5–7 | rank0 报 51.68 GiB / 475,128 tokens |

后两种均为 DP1、EP off。KV GiB 是日志报告的预算，单套部署只有 rank0 的对应日志，不据此证明每个 PP stage 的逐卡预算完全一致；逻辑容量也不是可跨拓扑等同的物理显存。双 TP4 的整机 prefill 上限 16384，单套八卡为 8192，属于协议中明确保留的调度预算差异。

NCCL 版本为 2.30.7。双 TP4、TP8、TP4×PP2 均明确记录超过两张 PCIe GPU 的 TP 组禁用 custom allreduce；TP2×PP4 未出现该禁用日志。所有配置保留 SM120 SymmMem 不支持、FlashInfer All Reduce 的 world-size 限制、TileLang fallback/串行化、FP8 scale 等 warning。未用 profiler 证明具体通信或 kernel 时间占比，也不把“没有禁用日志”当作所有通信路径的性能证明。四个 case 的完整 required/kernel 日志检查均 PASS，失败原因单列如下。

## 性能与真实工作量

下表只使用最终 PASS case 的整机行，吞吐为三次均值 ± 样本标准差；延迟单位均为 **ms**，各列是三次相应指标的均值。P95 均值不等于三次请求合并后的 P95，完整 mean/P95 TTFT、TPOT、ITL、E2EL 见 CSV。

| 配置 | 并发 | 输出 tok/s | CV | Mean TTFT | Mean TPOT | P95 ITL | P95 E2EL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP8 | 32 | 611.92 ± 0.91 | 0.15% | 9260.77 | 43.24 | 20.34 | 77065.72 |
| TP8 | 64 | 681.52 ± 0.69 | 0.10% | 14443.11 | 79.75 | 425.04 | 146175.19 |
| TP4×PP2 | 32 | 681.99 ± 20.09 | 2.95% | 4234.26 | 42.81 | 36.14 | 55254.16 |
| TP4×PP2 | 64 | 902.79 ± 3.83 | 0.42% | 5915.08 | 65.06 | 48.15 | 86808.44 |

TP4×PP2 的吞吐及表中 TTFT/E2EL 优于 TP8，但 C32 P95 ITL 更高，不能称为所有延迟指标都改善。

CSV 共 **24 行：12 个 node 样本及对应的 12 个 replica-0 样本**。本次完整 PASS 配置都是单套服务，因此两种 scope 的主要指标相同；统计只筛选 `scope=node`，不能重复相加。12 个 node 样本合计 **2304 成功、0 失败，输入 18,874,368、输出 2,359,296 tokens**。每次 C32 为 128 成功、输入 1,048,576／输出 131,072；每次 C64 为 256 成功、输入 2,097,152／输出 262,144。已独立检查每条请求 8192/1024、成功状态、请求索引覆盖和 Docker 绑定。

失败 case 的 C32 部分证据如下，**不在 CSV 中，也不作为完整部署通过结果**：

| 配置 | 三次输出 tok/s | 均值 ± 标准差 | CV |
| --- | --- | ---: | ---: |
| 双 TP4 | 1100.666348 / 1104.572394 / 1101.714944 | 1102.32 ± 2.02 | 0.18% |
| TP2×PP4 | 971.390398 / 1110.812934 / 1114.281170 | 1065.49 ± 81.52 | 7.65% |

这六次各 128 成功、0 失败，合计输入 6,291,456、输出 786,432 tokens。双 TP4 每侧 64 请求，索引交错分片无重叠；三次起跑偏差约 0.086 / 0.794 / 0.219 ms，结束尾段差 0.54 / 0.31 / 1.66 秒。整机吞吐按共同窗口计算，保留尾段，不相加两侧各自窗口吞吐。

## 异常、资源验收与限制

### C64 编译事件达到尝试上限

| 配置 / 第一次重复 | 尝试 1 事件条目 | 尝试 2 事件条目 | 处理 |
| --- | ---: | ---: | --- |
| 双 TP4 | 48 | 16 | 两次均拒绝，停止该 case，保留 C32 部分证据 |
| TP2×PP4 | 36 | 18 | 两次均拒绝，停止该 case，保留 C32 部分证据 |

每次尝试前都已连续两轮安静；第二次尝试前重新预热。事件包含测量窗口内 `jit_monitor` 明确报告的 TileLang `mhc_pre_big_fuse_with_norm_tilelang` 等编译，涉及正式请求调度出现的新形状。部分 TileLang 文本的内层时间戳早于 Docker 输出时间，但拒绝不只依赖这些延迟输出的文本，仍有测量期 JIT warning。未放宽规则、关闭 Graph、缩短负载或重启补测。四次拒绝结果均保留且不纳入性能统计；TP8 和 TP4×PP2 无被拒绝正式尝试。

没有观察到 OOM、模型不支持、请求失败或启动超时；所有配置清理成功后才进入下一项。C64 的第二、三次重复在上述两个失败 case 中未运行，因此计划的 24 个重复没有全部完成。

### 持续资源验收

启动前 8 卡空闲，无运行中的 Docker 容器；NUMA0/1/2/3 空闲内存约 227/230/168/75 GiB，未见内存压力阻塞。10 秒预采样的整机 CPU 忙碌率约 12.8%。已有 **16 个 `doca_spcx_cc` 忙进程**保留，与本节点历史记录一致；进程采样合计约占 16 个逻辑 CPU，其中 `roce-init.service` 的计数约为 6 个 CPU，不能只看该 service 就当作全部 DOCA 负载。

全程保存约 5 秒 CPU/GPU 遥测、约 30 秒 system service cgroup 计数，以及独立约 30 秒进程 CPU 增量；进程记录保留区间占用超过 0.02 个 CPU 的项。四个 case 共 2785 条 runner 遥测记录，无 observation error；全程最高整机 CPU 忙碌率约 35.9%，未触发 45% 提醒。逐次将 monotonic 官方计时窗口与资源采样对齐，核对了本次容器归属和后台服务，未发现矿工异常或新增高 CPU 外部任务。

完整 PASS 的 12 个样本，测量窗口平均整机 CPU 忙碌率为 **19.58%–20.00%**，单次采样最高 **20.67%**。包括六次部分结果在内的全部 18 个安静测量窗口，GPU 最高温度不超过 **62°C**，采样 SM 时钟不低于 **2415 MHz**；既有 DOCA 合计约 **16.00 个 CPU**，采到的其他非本次、非 DOCA 活跃进程合计峰值约 **0.131 个 CPU**。这些是低频采样证据，不能排除短暂争用。

CSV 在统一导出后逐次对照请求、绑定、窗口、遥测和进程证据，再标为 `accepted-background`：表示接受已约定的后台条件，**不表示 CPU 独占或性能已收敛**。DOCA 可以迁移到服务、客户端及其 SMT 兄弟上。保存的 `/metrics` 快照中累计抢占均为 0，完整服务日志无抢占记录；快照不是每个瞬间的独立证明。

### 波动诊断

TP4×PP2 C32 的第三次比前两次约快 5%，三次 CV 为 2.95%，接近协议观察阈值。三次无测量期 JIT，累计抢占为 0，GPU 平均 SM 约 2421 MHz，未见持续频率或背景负载突变；第三次 mean TPOT 从约 43.6 降至 41.2 ms。未确认其具体原因，不额外补测或舍弃慢样本。

TP2×PP4 C32 的 CV 为 7.65%，已按规则诊断。三次无测量期 JIT，累计抢占为 0，客户端采样峰值分别约 0.83/0.83/0.49 个 CPU，未显示客户端总 CPU 容量耗尽；GPU 平均 SM 频率接近。慢样本平均 GPU 利用率约 92.34%，后两次约 99.92%/100%；第一、二次服务 SMT 兄弟集合忙碌率约 3.55%/0.20%，客户端集合约 41.11%/34.66%。这些集合包含 DOCA 背景，分布确有变化，但不能据此唯一归因为 DOCA，也不能排除 PP 调度/通信等待。保留全部三次，不声称该组合在 C32 已收敛。

### 本机证据与复现

完整原始产物**仅在 48 本机**，不随 Git 分发：

- 工作区：`experiments/dsv4-node8-node48/`。
- baseline：`results/baseline-01/`；候选：`results/candidates-01/`。
- 各 case 的 `resolved.json`、`environment.json`、`replicas/*/server.log`、`kernel-checks.json`、启动命令及绑定证据。
- `trials/` 下全部预热、拒绝尝试、逐请求记录、请求分片、官方窗口与编译检查；各 case 的 `resource-telemetry.jsonl`。
- 工作区 `reports/` 下启动前后快照、CLI 预检、`process-telemetry.jsonl`、`sample-resource-audit.csv`、NUMA/metrics 快照、波动诊断与 `audit_samples.py`。

以下是本轮实际执行命令。复现时从仓库根目录复制公共 configs 到新的工作区，并替换工作区名称、run-root 和 CSV 输出路径，避免覆盖本轮产物；这些命令不是要求重跑失败组合：

```bash
./bench validate experiments/dsv4-node8-node48/configs/campaigns/48-dsv4-node8.yaml
./bench plan experiments/dsv4-node8-node48/configs/campaigns/48-dsv4-node8.yaml
./bench run experiments/dsv4-node8-node48/configs/campaigns/48-dsv4-node8.yaml \
  --case dual-tp4 --run-root experiments/dsv4-node8-node48/results/baseline-01
./bench run experiments/dsv4-node8-node48/configs/campaigns/48-dsv4-node8.yaml \
  --case tp8 --case tp4-pp2 --case tp2-pp4 \
  --run-root experiments/dsv4-node8-node48/results/candidates-01
python3 scripts/export_samples.py experiments/dsv4-node8-node48/results/baseline-01 \
  experiments/dsv4-node8-node48/results/candidates-01 \
  --output experiments/dsv4-node8-node48/reports/node8-node48.csv
```

导出默认 `review-required`，必须逐次资源复核后才能标注。收尾确认本次 Docker 容器已全部清理、GPU 计算进程为空，只读监控已停止；模型、镜像、缓存和历史结果保留。

归档自审已通过：公共 campaign 的 validate/plan、工作区与公共配置逐文件一致性、CSV 指标与统一导出的逐字段比对、真实 token 总量及吞吐重算、报告统计和相对链接检查，以及既有 `test_export_preserves_node_and_replica_scopes` 测试。未改公共源码，未重跑额外性能实验。
