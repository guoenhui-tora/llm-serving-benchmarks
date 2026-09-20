# 节点45：TP4 DSpark K1–K5 与独立复测

**2026-09-20，gpu-6000d-45 完成 TP4、EP off 的 off／K1–K5 六配置，全部 jit_clean PASS。核心矩阵优先保留 K3：807.66 ± 3.01 output tok/s，较本机 off 的 663.38 ± 0.20 提高 21.75%。** K4／K5 为 802.43／798.99 tok/s，继续增加 K 未提高本轮吞吐；三者差距较小，结合波动和较小 K 的优先规则，不能宣称 K3 是通用或官方最优。

范围限定单机后四卡、TP4/PP1/DP1、EP off、GovReport 固定前128条、总 C32／1024 输出，固定 vLLM 0.29.0＋本地原生 MXFP4 加载补丁。没有访问其他节点、测试 DP 拓扑或运行八卡双部署，不与历史随机输入结果混算收益。无已知事件不等于性能稳定；本轮未运行 stable，且 CPU 背景并非完全一致。

核心K3的平均TTFT／TPOT低于off，但各轮P95均值的TTFT／TPOT分别约高0.80%／4.23%；这项吞吐收益不代表所有尾延迟全面改善。

**预算内独立重启K3复测也PASS：809.15 ± 1.27 tok/s，CV 0.157%，较核心K3均值+0.184%。** 两次启动的三轮样本分别保存，没有合并成n=6；复测沿用已有缓存，不能用耗时差单独推断缓存的因果收益。

## 核心结果

以下各行 n=3，均为同一次启动中最先被协议接纳的三轮；±为样本标准差，CV=吞吐 SD／均值。延迟单位均为 ms。P95 汇总为各轮 P95 的均值和 SD，**不是合并请求后的 P95**。完整原始精度、时间戳和所有拒绝轮见[逐轮 CSV](../data/dspark-k-sweep-node45.csv)。

| DSpark | 输出 tok/s | CV | 相对同机 off | Mean TTFT ms | P95 TTFT ms | Mean TPOT ms | P95 TPOT ms | 接受率 % | 接受长度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OFF | 663.38 ± 0.20 | 0.030% | +0.00% | 5654.86 ± 2.92 | 19649.20 ± 12.45 | 42.69 ± 0.01 | 45.99 ± 0.01 | — | — |
| K1 | 733.17 ± 1.58 | 0.215% | +10.52% | 4711.46 ± 190.38 | 19835.12 ± 13.98 | 38.62 ± 0.20 | 48.89 ± 1.71 | 83.66 ± 0.18 | 1.8366 ± 0.0018 |
| K2 | 774.19 ± 4.16 | 0.537% | +16.70% | 4352.34 ± 101.72 | 19727.35 ± 105.14 | 36.37 ± 0.22 | 49.21 ± 1.54 | 74.33 ± 0.85 | 2.4866 ± 0.0170 |
| K3 | 807.66 ± 3.01 | 0.373% | +21.75% | 4333.36 ± 115.05 | 19807.33 ± 13.49 | 34.57 ± 0.31 | 47.94 ± 1.61 | 64.48 ± 1.19 | 2.9344 ± 0.0357 |
| K4 | 802.43 ± 8.63 | 1.075% | +20.96% | 4417.53 ± 27.32 | 19777.10 ± 18.82 | 34.32 ± 0.10 | 47.81 ± 1.07 | 55.68 ± 0.79 | 3.2271 ± 0.0317 |
| K5 | 798.99 ± 8.63 | 1.080% | +20.44% | 4289.84 ± 35.56 | 19728.65 ± 10.46 | 34.63 ± 0.17 | 48.78 ± 1.75 | 48.66 ± 0.92 | 3.4332 ± 0.0459 |

补充指标仍按每轮先统计、再计算均值 ± SD。DSpark 的 ITL 是流式事件间隔，并非逐 token 计算时间。

| DSpark | requests/s | Mean ITL ms | P95 ITL ms | Mean E2EL ms | P95 E2EL ms |
| --- | --- | --- | --- | --- | --- |
| OFF | 0.647832 ± 0.000197 | 42.69 ± 0.01 | 26.99 ± 0.08 | 49324.22 ± 14.82 | 64909.82 ± 30.72 |
| K1 | 0.715987 ± 0.001542 | 70.91 ± 0.30 | 127.53 ± 9.51 | 44220.26 ± 48.76 | 62660.90 ± 765.33 |
| K2 | 0.756044 ± 0.004060 | 90.37 ± 0.11 | 748.63 ± 2.64 | 41561.28 ± 134.75 | 60253.64 ± 309.92 |
| K3 | 0.788732 ± 0.002943 | 101.32 ± 0.42 | 761.06 ± 1.07 | 39694.84 ± 202.65 | 58638.08 ± 1090.98 |
| K4 | 0.783622 ± 0.008424 | 110.60 ± 0.76 | 760.67 ± 0.17 | 39526.56 ± 88.30 | 57814.23 ± 1269.87 |
| K5 | 0.780267 ± 0.008430 | 118.71 ± 1.06 | 758.81 ± 0.11 | 39719.93 ± 160.57 | 59136.28 ± 521.78 |

每轮均为128成功、0失败、输入1,042,149、输出131,072 tokens；这些四项在每组三个接纳样本中的 SD 均为0。核心共38轮、4,864成功请求、输入39,601,662／输出4,980,736 tokens；其中18轮接纳、20轮拒绝。CSV 的 accepted 只表示原协议接纳，不回溯改判延迟日志对应的轮次。

## 接受长度与开销

下面合并每组**三个接纳轮次的计数分子／分母**，与主表的轮次均值略有不同。位置数组为位置0起的接受次数，元素之和等于接受草稿 token 总数；不是百分比。

| DSpark | 提出草稿 tokens | 接受草稿 tokens | 验证步数 | 加权接受率 % | 加权接受长度（含 bonus） | 逐位置接受次数 |
| --- | --- | --- | --- | --- | --- | --- |
| K1 | 213965 | 179012 | 213965 | 83.6642 | 1.836642 | [179012] |
| K2 | 316222 | 235028 | 158111 | 74.3237 | 2.486475 | [132113, 102915] |
| K3 | 402075 | 259218 | 134025 | 64.4701 | 2.934102 | [110513, 86509, 62196] |
| K4 | 487612 | 271469 | 121903 | 55.6732 | 3.226926 | [99811, 77542, 56293, 37823] |
| K5 | 573110 | 278849 | 114622 | 48.6554 | 3.432770 | [94001, 73156, 53034, 35995, 22663] |

K1→K3 时，接受长度从约1.84增加至2.93，吞吐同步提高；K4/K5 进一步增加到约3.23/3.43，但吞吐没有继续提高。更长接受长度在本轮未抵消额外草稿／验证的综合开销。这里仅有端到端和计数证据，没有 profiler 分解草稿、验证、调度与通信时间，不能唯一归因某个 kernel，也不是模型质量评测。

指标定义来自固定镜像 `vllm/v1/spec_decode/metrics.py`：

- `vllm:spec_decode_num_draft_tokens_total` 为提出草稿 token 数 D；`vllm:spec_decode_num_accepted_tokens_total` 为接受草稿 token 数 A。
- `vllm:spec_decode_num_drafts_total` 为按请求累计的验证／draft 步数 V，并非合并批次的 GPU kernel 调用次数。
- 接受率为 `100 × ΔA / ΔD`；平均接受长度为 `1 + ΔA / ΔV`，按镜像约定包含目标模型 bonus token。它是引擎统计，不强制等于客户端最终输出长度的逐请求商。
- `vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0",...}` 保存逐位置接受次数；位置比例的分母为 ΔV。

本轮为 DP1，实际每个标量计数器只导出一组 `{engine="0",model_name="deepseek-v4-flash"}` 标签，位置计数再加 position；不按4个 TP worker 重复累加。每秒保存完整 `/metrics`，每轮选取协议 started_at 之后首个、finished_at 之前末个快照；两端均要求 running/waiting=0、成功请求增量128、标签集合一致、窗口内计数单调且未重置。快照涵盖完整128请求，排除了计时外中文探测。全部34个核心 on 轮通过这些核对，无缺失接受率；独立快照计算出的提出／接受／验证步数及接受率／长度与客户端 raw.json 逐轮交叉检查一致（容差1e-12）；逐轮边界时间、标签、前后值及原始快照保留在本机证据中。

## 逐轮与协议成本

完整负载直接进入 jit_clean，没有额外独立预热。每配置最多12轮／2700秒，单轮 timeout 900秒，启动上限1800秒；累计最先三轮通过工作量与事件 gate 的结果后停止。JIT 不触发重启，失败请求不自动重试。本轮遵守6小时后不再新增配置的排期规则。

| 配置 | run_id | 总轮数 | 接纳轮次 | 启动 s | 协议 s | benchmark 合计 s | run 合计 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OFF | off-core-01 | 4 | 2,3,4 | 107.39 | 948.68 | 843.81 | 1102.27 |
| K5 | k5-core-02 | 5 | 3,4,5 | 437.52 | 1355.29 | 1226.09 | 1843.85 |
| K1 | k1-core-02 | 8 | 5,7,8 | 376.98 | 2084.36 | 1872.66 | 2510.60 |
| K2 | k2-core-02 | 7 | 5,6,7 | 397.10 | 1780.66 | 1598.20 | 2228.71 |
| K3 | k3-core-02 | 7 | 4,5,7 | 419.63 | 1742.81 | 1559.48 | 2213.69 |
| K4 | k4-core-02 | 7 | 4,5,7 | 415.49 | 1742.75 | 1561.27 | 2209.26 |

启动为 runner 的 readiness 计时；协议包含客户端初始化、测量及检查；run 另含内部预检、启动、探测和最终取证／清理，不含外部准备。累计时间不重复写入每轮 CSV；CSV 的 duration_s 仅客户端 benchmark 计时，started_at／finished_at 为含初始化与检查的阶段窗口（UTC）。

| 配置 / run | 轮次 | 接纳 | 事件行 | 输出 tok/s | 接受率 % | 接受长度 |
| --- | --- | --- | --- | --- | --- | --- |
| off-core-01 | 1 | false | 52 | 522.0627 | — | — |
| off-core-01 | 2 | true | 0 | 663.5602 | — | — |
| off-core-01 | 3 | true | 0 | 663.1614 | — | — |
| off-core-01 | 4 | true | 0 | 663.4173 | — | — |
| k1-core-02 | 1 | false | 215 | 314.3026 | 83.9406 | 1.839406 |
| k1-core-02 | 2 | false | 84 | 487.9035 | 84.6967 | 1.846967 |
| k1-core-02 | 3 | false | 84 | 490.7775 | 84.3704 | 1.843704 |
| k1-core-02 | 4 | false | 24 | 640.6306 | 83.3053 | 1.833053 |
| k1-core-02 | 5 | true | 0 | 733.2166 | 83.6022 | 1.836022 |
| k1-core-02 | 6 | false | 12 | 732.2372 | 83.6598 | 1.836598 |
| k1-core-02 | 7 | true | 0 | 731.5697 | 83.5190 | 1.835190 |
| k1-core-02 | 8 | true | 0 | 734.7266 | 83.8716 | 1.838716 |
| k2-core-02 | 1 | false | 296 | 266.1629 | 74.3553 | 2.487105 |
| k2-core-02 | 2 | false | 48 | 597.4193 | 74.4115 | 2.488229 |
| k2-core-02 | 3 | false | 24 | 670.1188 | 74.0590 | 2.481181 |
| k2-core-02 | 4 | false | 24 | 716.8602 | 74.3416 | 2.486832 |
| k2-core-02 | 5 | true | 0 | 771.9599 | 74.1093 | 2.482185 |
| k2-core-02 | 6 | true | 0 | 771.6228 | 73.6073 | 2.472146 |
| k2-core-02 | 7 | true | 0 | 778.9860 | 75.2662 | 2.505324 |
| k3-core-02 | 1 | false | 312 | 265.0359 | 64.1213 | 2.923638 |
| k3-core-02 | 2 | false | 36 | 649.6299 | 63.7401 | 2.912203 |
| k3-core-02 | 3 | false | 24 | 693.9283 | 64.6868 | 2.940605 |
| k3-core-02 | 4 | true | 0 | 804.3234 | 63.2565 | 2.897694 |
| k3-core-02 | 5 | true | 0 | 810.1823 | 65.6315 | 2.968945 |
| k3-core-02 | 6 | false | 24 | 699.3218 | 64.9300 | 2.947900 |
| k3-core-02 | 7 | true | 0 | 808.4784 | 64.5512 | 2.936537 |
| k4-core-02 | 1 | false | 334 | 258.3918 | 55.6475 | 3.225899 |
| k4-core-02 | 2 | false | 16 | 693.7495 | 55.2485 | 3.209941 |
| k4-core-02 | 3 | false | 24 | 699.5370 | 56.3632 | 3.254530 |
| k4-core-02 | 4 | true | 0 | 793.9824 | 54.8937 | 3.195750 |
| k4-core-02 | 5 | true | 0 | 811.2251 | 56.4796 | 3.259184 |
| k4-core-02 | 6 | false | 24 | 698.5350 | 55.2980 | 3.211920 |
| k4-core-02 | 7 | true | 0 | 802.0792 | 55.6617 | 3.226468 |
| k5-core-02 | 1 | false | 338 | 253.1205 | 47.6801 | 3.384006 |
| k5-core-02 | 2 | false | 48 | 606.5624 | 48.0976 | 3.404881 |
| k5-core-02 | 3 | true | 0 | 791.8186 | 47.7371 | 3.386856 |
| k5-core-02 | 4 | true | 0 | 796.5883 | 48.6804 | 3.434021 |
| k5-core-02 | 5 | true | 0 | 808.5727 | 49.5732 | 3.478662 |

## 阻塞、日志及观察边界

off 首次正常结束后，五个 on 配置的 `kN-core-01` 均在内部 preflight 以 `Port 31249 is busy` 失败，未启动模型、未产生正式轮次；这些不是测量样本，所以不伪造 CSV 行。对应 case.json／run.log 全部保留。检查时容器与 GPU 均已释放，稍后端口可绑定；未及时捕获首次 socket 状态，因此仅判断为短暂退出残留，不能断言已实证 TIME_WAIT。

在没有服务运行时新增本地顺序续跑脚本，每个尚未加载模型的配置仅重试一次，使用新的 `kN-core-02`。相邻服务之间最多等待90秒确认端口可绑定；实测后续常需约59–60秒。未修改公共 runner、源码、gate 或任何 recipe。只清理本次拥有的容器，没有终止其他任务；初始工作区无本地改动，拉取快进至397e9d4，祖先检查成功，之后整个运行期间没有再次 pull。

所有核心配置的 kernel checks PASS。实际日志确认主模型 NVFP4／FLASHINFER_CUTLASS、草稿 DEEPGEMM_MXFP4、V2 runner、FP8 MLA KV、FULL_DECODE_ONLY、autotune关闭，以及目标和草稿捕获完成。on 的每个 worker 都出现 `LOCAL_DSPARK_FORMAT_FIX: validated 768 native MXFP4 draft experts`；未更换精度。保留 SM120 SymmMem 不可用、TP4 FlashInfer All Reduce 不支持、PCIe超过两卡 custom allreduce禁用、NVFP4实验性、FP4 indexer字段弃用、TileLang向量化退回串行循环、共享内存广播等待等 warning/fallback，不将 warning 数量等同于独立问题数。

K1／K3／K4 第6轮都发现延迟 TileLang 输出：例如 K1 Docker时间 `21:45:52Z` 的日志内部时间为 `21:39:02`；K3 `23:04:52Z` 对应 `22:55:47`；K4 `23:42:39Z` 对应 `23:33:04`。原始时间属于2026-09-19 UTC。这些反例说明日志窗口包含异步刷出，**事件行数不是独立编译次数，也不能把所有拒绝都解释为该轮实际重编译**。保守维持原 gate 与原拒绝轮，不隐藏 warning、不改判来取得 PASS。

## 容量、绑定与资源

GPU4–7；服务CPU32–47／NUMA2，客户端CPU48–51／NUMA3，各只使用一个SMT线程，兄弟为ID＋64。Docker inspect、worker/client线程亲和性均按runner检查保存。独立复测的宿主PID／GPU UUID交叉核验进一步确认 TP0/1/2/3→宿主GPU4/5/6/7，前四卡未参与。NUMA页面快照每60秒保存，确实存在其他NUMA节点的驻留页面；cpuset只约束允许集合，不能声称页面全部本地化。

| 配置 | KV等效容量 tokens | KV观测峰值 % | 抢占增量 | 温度 °C | 功耗 W/卡 | SM频率 MHz | warning匹配行 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OFF | 85450 | 24.105 | 0.0 | 35–60 | 78.23–297.09 | 2272–2430 | 48 |
| K1 | 76101 | 27.540 | 0.0 | 39–60 | 76.76–301.36 | 2295–2430 | 208 |
| K2 | 74629 | 27.933 | 0.0 | 39–60 | 76.27–299.05 | 2295–2430 | 196 |
| K3 | 74566 | 27.930 | 0.0 | 39–60 | 76.76–313.87 | 2302–2430 | 200 |
| K4 | 74427 | 27.911 | 0.0 | 39–60 | 76.76–318.79 | 2295–2430 | 200 |
| K5 | 74317 | 27.916 | 0.0 | 39–60 | 76.27–312.26 | 2295–2430 | 196 |

KV 容量取日志中唯一 DP rank 值，DSV4 压缩缓存等效 token 数不能当作普通全注意力容量。KV 每秒采样，峰值为所有轮次、全部可见rank的观测最大值，可能遗漏瞬时峰值；温度／功耗／频率为协议窗口内后四卡5秒采样范围，包含编译与客户端初始化空闲。未见持续热降频证据，低频采样不能排除瞬时干扰。未更改宿主锁频、功耗或NUMA设置。

下面仅统计三个**接纳阶段窗口**的 CPU ticks（含阶段初始化与检查）；数值为各CPU集合整体忙碌率，包含其他任务，并非本服务独占占用。

| 配置 | 整机 % | 服务核 % | 服务SMT % | 客户端核 % | 客户端SMT % |
| --- | --- | --- | --- | --- | --- |
| OFF | 10.156 | 29.777 | 0.156 | 4.072 | 0.132 |
| K1 | 10.094 | 29.073 | 0.139 | 20.770 | 8.434 |
| K2 | 10.114 | 29.288 | 0.296 | 14.861 | 6.332 |
| K3 | 10.056 | 29.044 | 0.148 | 3.976 | 0.290 |
| K4 | 10.098 | 29.266 | 0.155 | 3.700 | 0.206 |
| K5 | 10.084 | 29.256 | 0.185 | 3.669 | 0.290 |

所有配置的 roce-init.service CPU增量约8.000个逻辑CPU，8个既存DOCA进程持续存在；未观察到新增矿工等可识别异常。K1/K2客户端核及SMT负载明显较高，进程主线程的最后运行CPU快照不足以定位全部忙线程或中断，不能把差异全部归为DOCA或网卡IRQ。总体CPU均值接近并不证明资源无干扰；这限制小幅跨启动差异的解释，故保留K3作为本条件候选，不能把排序泛化为普遍最优。

## 身份与复现入口

本机原始工件均为**仅本机可用**：`experiments/dspark-k-sweep-node45/`，包含 configs、data、results、evidence、reports、scripts；原始日志／全部轮次JSON／每秒计数器／5秒资源／30秒进程／60秒NUMA页面／辅助脚本保留，不批量上传。每个实际run使用新的 `results/<run_id>/`，下表给出原始本地campaign入口；完整依赖来自本项目 configs 和 GovReport JSONL，不依赖归档目录指向实验区的软链接。

固定Git HEAD：`397e9d42dedee00c181c5fb10f5686ace069c111`；`git merge-base --is-ancestor 397e9d4 HEAD` 成功。所有实际run源码指纹为 `f6e873002b8cf61963bd221d63b0fe0a520e0bc6118f4ca63acfa54fc6726d52`。初始与运行前后文件哈希核对保留于 evidence；没有公共修复或实验配置变更需要归档。

服务和客户端 image ID：`sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`，tag `vllm/vllm-openai:v0.29.0`。权重目录 `/data/models/DeepSeek-V4-Flash-0731-NVFP4`，metadata/tokenizer＋48分片大小身份 `1e1b7e54442bcf7f15ef55192641b5c5da360bcaa608236aa0d06b273aedf8b4`；没有全量重算权重内容哈希。config SHA256 `bb0d2286d6761439e41d3cef31d16489411b816ed8688922f59730bbd5567cdb`，tokenizer SHA256 `8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf`。

数据 SHA256：`33bdbd36a8b335300974e1b07aa66b2c24c88cc4c4b55a21350aa3564a945d53`，见[GovReport说明](../data/govreport-near8k.md)及[JSONL](../data/govreport-near8k.jsonl)。补丁版本采用[manifest](../patches/dspark-native-mxfp4/manifest.json)：原模块 SHA256 `56f6b81f5712817689e24088b2c5302d5832fd4e6d5b2630d7e28be73c84b298`，覆盖 utils `e0c060632a5b2a9ce7016edb7e10ac666e3a58464c0216803b9318a083f1b902`，辅助函数 `33df649eec0abfe5350fafc18dc84bcbd492238e65b9652a13fcea9e3bdf4730`；用法见[补丁说明](dspark-compatibility.md)。

### 参数与启动命令

基于[off recipe](../configs/recipes/tp4.yaml)和[K5 recipe](../configs/recipes/tp4-dspark-k5.yaml)派生，不改变原recipe；全部使用[GovReport jit_clean workload](../configs/workloads/govreport-c32-jit-clean.yaml)。本地target将[rear target](../configs/targets/rear.yaml)地址改为10.90.1.45、id改为rtx6000d-45-k-sweep-rear；每个campaign只有一个case，无 replica_targets。

下列完整K3服务命令从实测 `cases/tp4-epoff-k3/command.sh` 原样导出。其余on仅修改K、Graph列表、对应recipe缓存目录以及runner生成的容器name/owner；off另去掉 speculative-config 和 PYTHONPATH，使用普通Graph列表。完整差异在后表，不手工维护第二份YAML标准答案。

```bash
docker run -d --pull never --name sb-4f1dda10ee054707-server --label io.serving-bench.run=4f1dda10ee054707 --network host --ipc host --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/vllm/9c9243b9cc6c3a46192f:/root/.cache:rw --cpuset-cpus 32-47 --cpuset-mems 2 --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --cap-add SYS_NICE -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 -e PYTHONPATH=/root/.cache/dspark-native-mxfp4 --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model --served-model-name deepseek-v4-flash --host 127.0.0.1 --port 31249 --trust-remote-code --enable-auto-tool-choice --no-enable-prefix-caching --no-enable-expert-parallel --enable-chunked-prefill --jit-monitor-verbose --async-scheduling --tensor-parallel-size 4 --pipeline-parallel-size 1 --data-parallel-size 1 --max-model-len 16384 --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8_e4m3 --block-size 256 --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' --moe-backend auto --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' --kernel-config '{"enable_flashinfer_autotune":false}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[3,4,6,8,12,16,24,32,36,48,64,72,96,128],"max_cudagraph_capture_size":128}' --seed 0 --distributed-executor-backend mp --speculative-config '{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":false}'
```

| 配置 | 本地campaign（工作区内） | Graph token列表 | 上限 | 缓存目录末段 | 初始文件数 | 初始字节 |
| --- | --- | --- | --- | --- | --- | --- |
| OFF | configs/campaigns/sweep-off.yaml | [1, 2, 4, 8, 12, 16, 24, 32] | 32 | d19496fcae0f32e092d6 | 334 | 12550676 |
| K1 | configs/campaigns/sweep-k1.yaml | [1, 2, 4, 8, 12, 16, 24, 32, 48, 64] | 64 | ef878f98e9a9e7944458 | 4 | 9079 |
| K2 | configs/campaigns/sweep-k2.yaml | [2, 3, 4, 6, 8, 12, 16, 24, 32, 36, 48, 64, 72, 96] | 96 | 57e57609b6d3b4f525f1 | 4 | 9079 |
| K3 | configs/campaigns/sweep-k3.yaml | [3, 4, 6, 8, 12, 16, 24, 32, 36, 48, 64, 72, 96, 128] | 128 | 9c9243b9cc6c3a46192f | 4 | 9079 |
| K4 | configs/campaigns/sweep-k4.yaml | [4, 5, 8, 10, 16, 20, 32, 40, 48, 60, 64, 80, 96, 120, 128, 160] | 160 | 64e714b37e4ce3809d20 | 4 | 9079 |
| K5 | configs/campaigns/sweep-k5.yaml | [5, 6, 10, 12, 20, 24, 40, 48, 60, 72, 80, 96, 120, 144, 160, 192] | 192 | b25bbebbe392cf487b86 | 4 | 9079 |

on 的捕获列表为基础批次 `[1,2,4,8,12,16,24,32]` 分别乘K与K+1后的排序去重并集，草稿覆盖32K、目标验证覆盖32(K+1)。镜像实际DSpark实现使用anchor作为首个预测位置，草稿每请求K个query token；实际CLI、初始化参数和捕获完成日志均已核对。仅K6/K7候选做了离线validate/plan/CLI preflight及补丁准备，未启动服务或测量，不能称为已支持完整负载。

缓存位于 `/home/enhui/.cache/serving-bench/vllm/<末段>`，原路径取实际image ID解析；`bench plan`使用image tag预览时缓存路径可能不同，因此以实测command.sh和cache-before.json为准。on初始缓存只有四个加载补丁文件，off已有334个缓存文件；各recipe独立，首轮成本不是严格同缓存起点的K值比较。全部JIT缓存保留，K3独立复测沿用其核心缓存，没有复制或清空。

客户端使用固定vLLM 0.29.0官方serve benchmark入口、仓库JSONL长度校验hook和亲和性记录hook，流式 `/v1/completions`；功能探测通过 `/health`、`/v1/models` 和关闭thinking的中文短生成后才进入完整负载。以下客户端参数从实测第4轮command.sh提取，完整容器命令与Python hook保存在对应原始目录：

```text
--backend vllm --base-url http://127.0.0.1:31249 --endpoint /v1/completions --model /model --tokenizer /model --served-model-name deepseek-v4-flash --num-prompts 128 --max-concurrency 32 --request-rate inf --num-warmups 0 --seed 0 --temperature 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 --save-result --result-dir /results --result-filename raw.json --dataset-name custom --dataset-path /dataset.jsonl --custom-output-len 1024 --skip-chat-template --disable-shuffle --no-oversample --ignore-eos --trust-remote-code
```

实际JSONL挂载为工作区 `data/govreport-near8k.jsonl:/dataset.jsonl:ro`，固定前128条，输入平均8141.7890625 tokens，输出固定1024；seed=0、temperature=0、ignore_eos、request_rate=inf、num_warmups=0。每轮实际tokenizer校验清单保存为 dataset-manifest.json。本地复现入口为 `./bench run experiments/dspark-k-sweep-node45/configs/campaigns/sweep-<off|k1|k2|k3|k4|k5>.yaml --run-root experiments/dspark-k-sweep-node45/results/<新的run_id>`；运行前沿用validate、plan、prepare.py（仅on）、prepare_logging和preflight步骤。

## 条件扩展：独立重启K3

六个核心配置全部完成后，同机K5较K4的吞吐变化为 **-0.428%**，没有超过 `max(2%, 2×max(CV_K4,CV_K5)) = 2.161%`，因此没有启动K6/K7。按约定选择核心均值最高且K较小的K3，仅追加一次独立重启，没有用足“最多两次”去反复挑选结果。这个阈值是预定扩展规则，不是显著性检验。

复测run为 `k3-restart-01`，仍用 `configs/campaigns/sweep-k3.yaml`，参数／绑定／Graph／负载／gate均未改变。缓存沿用核心K3，起点1,669个文件、46,713,849字节。共6轮，接纳第2、5、6轮；首轮128条事件，第3／4轮4／8条JIT monitor事件，全部保留原判定。

| 输出 tok/s | CV | 相对核心K3 | 相对本批off | Mean TTFT ms | P95 TTFT ms | Mean TPOT ms | P95 TPOT ms | 接受率 % | 接受长度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 809.15 ± 1.27 | 0.157% | +0.184% | +21.97% | 4356.23 ± 84.20 | 19808.20 ± 6.06 | 34.48 ± 0.15 | 46.42 ± 0.92 | 64.73 ± 0.09 | 2.9418 ± 0.0026 |

| requests/s | Mean ITL ms | P95 ITL ms | Mean E2EL ms | P95 E2EL ms |
| --- | --- | --- | --- | --- |
| 0.790185 ± 0.001239 | 101.34 ± 0.40 | 761.05 ± 0.50 | 39634.19 ± 73.45 | 58275.59 ± 522.78 |

三个接纳轮计数合计：D=401,046、A=259,581、V=133,682，计数加权接受率64.725992%，含bonus的加权接受长度2.941780，位置接受次数为 `[110867, 86655, 62059]`。公式、标签、窗口与核心相同。所有6轮仍各128成功、0失败、输入1,042,149／输出131,072 tokens；完整逐轮值如下：

| 轮次 | 接纳 | 事件行 | benchmark s | 输出 tok/s | 接受率 % | 接受长度 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | false | 128 | 168.5337 | 777.7198 | 63.8943 | 2.916828 |
| 2 | true | 0 | 162.0681 | 808.7464 | 64.8132 | 2.944395 |
| 3 | false | 4 | 163.5566 | 801.3864 | 64.0547 | 2.921640 |
| 4 | false | 8 | 162.7034 | 805.5884 | 64.2344 | 2.927032 |
| 5 | true | 0 | 161.7034 | 810.5706 | 64.7231 | 2.941694 |
| 6 | true | 0 | 162.1915 | 808.1310 | 64.6418 | 2.939255 |

启动103.34秒，协议1136.18秒（18.94分钟），所有轮次benchmark合计980.76秒，run总耗时1285.68秒。kernel checks为PASS，warning匹配188行，继续保留相同通信／精度／编译观察警告。

KV等效容量74,566 tokens，6轮观测KV峰值27.976%，抢占增量0。协议GPU温度35–60°C，功耗78.71–308.23 W/卡，SM频率2287–2430 MHz；roce-init.service约8.000个逻辑CPU。接纳阶段整机／服务核／服务SMT／客户端核／客户端SMT忙碌率分别为 10.075%／29.172%／0.170%／3.868%／0.164%。

两次K3启动结果接近，支持它作为本负载下的后续候选；一次重启不能证明长期无干扰稳定，也未重新做紧邻的off参照。所有收益仍仅除以本批、本机、同TP4/EP off的off，不跨节点或跨负载借用分母。

## 最终核对与归档

本节点7次实际服务启动均PASS（6个核心配置＋K3一次独立复测）；另有5次前述端口preflight失败，无测量轮。合计**44轮、21轮接纳、23轮拒绝**，5,632成功、0失败，输入45,854,556／输出5,767,168 tokens。CSV按一轮一行保留全部44轮，复测study_phase为restart_check，与core区分；没有中断轮或请求失败被遗漏。实际run从北京时间04:20:36到08:16:45，约3.94小时，包含端口等待／准备间隙，未触及6小时新增配置截止。

结束检查确认所有实验源码／配置及运行脚本的冻结哈希一致、镜像／模型身份一致，Docker无运行容器、GPU无计算进程；所有本次服务和客户端均由所属runner清理。归档仅本报告、完整CSV及README节点45完成标记，其他节点标记和总表保持原内容。数据与相对链接检查后提交，不push。
