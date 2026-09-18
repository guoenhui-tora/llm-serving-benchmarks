# 节点 48：四卡位置与 TP/PP 拓扑对照

## 1. 结论与完成范围

**在本轮 gpu-6000d-48、8192 输入 / 1024 输出、全局 C32、单服务四卡条件下，优先继续优化 GPU4–7 上的 TP2×PP2，prefill 暂保留 8192。** 其三次输出吞吐为 **762.92 ± 17.85 tokens/s，CV 2.34%**，相对同组 TP4 提高 **6.60%**；mean/P95 TTFT 分别降低 **35.01% / 44.69%**。这是本轮测得的组合表现，后台 CPU 活动与顺序测试限制了纯拓扑归因，不是跨节点或官方最优结论。

**前后四卡 TP4 未分出优劣。** 后四卡为715.72、前四卡为715.23 tokens/s，前四卡相对差异仅 **-0.07%**，小于重复波动。统一绑定后的本轮结果不支持沿用历史双服务数据中“后四卡更快”的推断。

prefill **4096** 的三次吞吐761.14 tokens/s与8192均值接近，CV **4.17%**超过3%诊断阈值；mean TTFT降低 **24.10%**，但 mean TPOT上升 **2.82%**、P95端到端延迟上升 **10.01%**。保留为首 token 延迟优化候选，不能声称吞吐更优或稳定。prefill **16384**只完成一次初筛，没有显示值得补测的整体收益。

| 范围 | 完成情况 |
| --- | --- |
| 主矩阵：后四卡 TP4 → 前四卡 TP4 → 后四卡 TP2×PP2 | 各三次有效重复，共9次；全部功能、工作量、绑定和kernel gate通过 |
| 可选：TP2×PP2 prefill4096 | 一次初筛＋两次补测，共3次；gate通过，吞吐不稳定 |
| 可选：TP2×PP2 prefill16384 | 一次有效初筛；未补测，不估计标准差或CV |
| 诊断 | 完成4096的日志、JIT、抢占、CPU/NUMA、温度/功耗/频率检查；未追加诊断测量 |
| 未执行 | 其他拓扑上的可选prefill、其他并发、八卡、跨节点及PD均未开展 |

本轮实验窗口为 **2026-09-17 05:58:15–08:37:37（UTC+8）**，约 **2.66 小时**，主矩阵和可选实验均在八小时检查点前收尾。原先的运行前CPU阻塞记录在本机保留；用户解除大面积占用后重新检查并开始本轮。结束时本次容器与GPU计算进程均已清理；保留模型、镜像、缓存和全部历史产物。

## 2. 运行条件与配置差异

基线来自 [公共 TP4 recipe](https://github.com/guoenhui-tora/llm-serving-benchmarks/blob/4642443b1274251b6a47f0d2742e0e59e2dccc95/projects/dsv4-rtx6000d/configs/recipes/vllm-tp4-baseline.yaml)、[公共 workload](https://github.com/guoenhui-tora/llm-serving-benchmarks/blob/4642443b1274251b6a47f0d2742e0e59e2dccc95/projects/dsv4-rtx6000d/configs/workloads/dsv4-8192-1024-c32-n128-repeat3.yaml)、[节点48 target](https://github.com/guoenhui-tora/llm-serving-benchmarks/blob/4642443b1274251b6a47f0d2742e0e59e2dccc95/projects/dsv4-rtx6000d/configs/targets/rtx6000d-48-tp4-baseline.yaml) 和 [节点48 campaign](https://github.com/guoenhui-tora/llm-serving-benchmarks/blob/4642443b1274251b6a47f0d2742e0e59e2dccc95/projects/dsv4-rtx6000d/configs/campaigns/48-dsv4-vllm-tp4-baseline.yaml)。先复制项目configs建立本机工作区，保留公共原文件；下表给出从公共配置重建候选所需的全部功能差异，记录ID、描述和provenance另行命名。

| 配置 | recipe options差异 | GPU / 服务CPU、内存NUMA / 客户端CPU、内存NUMA | workload repetitions |
| --- | --- | --- | --- |
| TP4 后四卡 / 8192 | 无 | GPU4–7 / CPU32–47、NUMA2 / CPU48–51、NUMA3 | 3 |
| TP4 前四卡 / 8192 | 无；只修改target GPU与binding | GPU0–3 / CPU0–15、NUMA0 / CPU16–19、NUMA1 | 3 |
| TP2×PP2 后四卡 / 8192 | TP=2、PP=2，DP仍为1 | 与后四卡TP4相同 | 1＋2 |
| TP2×PP2 后四卡 / 4096 | TP=2、PP=2、`max-num-batched-tokens=4096` | 与后四卡TP4相同 | 1＋2 |
| TP2×PP2 后四卡 / 16384 | TP=2、PP=2、`max-num-batched-tokens=16384` | 与后四卡TP4相同 | 1 |

环境变量、flags、其他options、模型、runtime与client均沿用公共配置：`VLLM_USE_V2_MODEL_RUNNER=1`，async scheduling、mp、EP off；上下文16384、活动容量32、FP8 E4M3 KV、显存比例0.90自动分配、block-size256、FP4 indexer关闭；关闭autotune和prefix cache，保留chunked prefill和FULL_DECODE_ONLY Graph，捕获尺寸1/2/4/8/12/16/24/32。没有引入投机解码、offload、`numa-bind`、额外线程设置或精度回退。

所有正式样本使用同一vLLM客户端、流式`/v1/completions`、seed0、temperature0、ignore_eos=true、range_ratio0、request_rate=inf，C32、128请求、8192/1024。每轮预热64请求，至少连续两轮无已知编译事件，每次尝试最多五轮预热、每次重复最多两次测量尝试；启动1800秒、单客户端阶段7200秒上限未变。初筛和补测只改变workload ID与重复数，已逐字段核对服务、target、client及其余workload一致；CSV重复编号合并为1/2/3，未修改公共指纹分组逻辑。

执行身份与证据：

- Git commit：`c54734568603d1ef52f6ccc24ce7673768777c37`；公共源码与配置没有改动。本地Git差异仅节点48报告、CSV和README任务行，探索配置及辅助脚本留在本机工作区。
- 所有run的runner源码指纹一致：`f5c33cf418ed84629e3786be4f4c7e651d66dccc3827e4226685a42e3361b583`。实验期间冻结源码及全部实验配置，运行包装逐文件检查配置SHA-256。
- [runtime](https://github.com/guoenhui-tora/llm-serving-benchmarks/blob/4642443b1274251b6a47f0d2742e0e59e2dccc95/projects/dsv4-rtx6000d/configs/runtimes/vllm-0.29.0.yaml) 与 [client](https://github.com/guoenhui-tora/llm-serving-benchmarks/blob/4642443b1274251b6a47f0d2742e0e59e2dccc95/projects/dsv4-rtx6000d/configs/clients/vllm-bench-0.29.0.yaml) 均固定`vllm/vllm-openai:v0.29.0`，实际完整ID：`sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`；未拉取或更换镜像。
- [模型配置](https://github.com/guoenhui-tora/llm-serving-benchmarks/blob/4642443b1274251b6a47f0d2742e0e59e2dccc95/projects/dsv4-rtx6000d/configs/models/dsv4-flash-nvfp4.yaml)为同一NVFP4权重与tokenizer。所有run的模型身份相同：`1e1b7e54442bcf7f15ef55192641b5c5da360bcaa608236aa0d06b273aedf8b4`；此身份依据配置/tokenizer哈希及分片大小，不等价于全部权重内容的校验和。
- 所有新配置均通过validate/plan和实际镜像CLI检查；vLLM无需额外自定义日志文件。每次服务启动均通过health、models与关闭thinking的中文语义探测，保留请求/响应。
- Docker inspect及服务/客户端线程快照验证实际CPU和内存允许集合符合上表；使用单个SMT线程，不修改宿主SMT、功耗或锁频。`numa_maps`快照可见非目标NUMA上的共享/文件映射页，不能将cpuset限制解释为所有页本地驻留。
- TP2×PP2实际GPU映射由worker名、宿主PID、GPU UUID和容器DeviceIDs共同核验：PP0_TP0/1落在GPU4/5，PP1_TP0/1落在GPU6/7。GPU4–5、6–7各为PXB近端组；仅凭启动参数推断的情况未作为映射证据。

实际启动日志确认DeepseekV4ForCausalLM、V2、FLASHINFER_MLA_SPARSE_DSV4、FLASHINFER_CUTLASS、expert_dtype=fp4、fp8_ds_mla、autotune关闭与Graph捕获。自动显存分配的实际日志如下；GiB为日志中rank0可用KV预算，tokens为引擎报告的KV容量，不能混同为跨拓扑等价的物理KV字节数。

| 配置 | 日志可用KV（GiB） | 日志GPU KV容量（tokens） |
| --- | ---: | ---: |
| TP4 后/前四卡 / 8192 | 32.73 | 85,450 |
| TP2×PP2 / 8192 | 32.21 | 167,509 |
| TP2×PP2 / 4096 | 33.00 | 227,735 |
| TP2×PP2 / 16384 | 30.61 | 160,704 |

recipe相同显存比例不保证相同KV容量，prefill预算也影响实际分配。本轮不人为固定KV字节数，这些差异属于测得配置的运行表现。上述日志与功能探测不能代替完整kernel数值精度或模型质量验证。

## 3. 性能汇总

[逐次CSV](../data/topology-node48.csv)保存13个有效样本的原始数值精度、唯一标识和连续重复编号。有效样本合计 **1664成功、0失败，输入13,631,488、输出1,703,936 tokens**；每个样本均为128成功、0失败、输入1,048,576、输出131,072 tokens。预热、一次gate拒绝尝试和资源诊断不进入CSV。

吞吐列为均值±样本标准差，CV=样本标准差/均值；所有延迟单位为 **ms**，是各次对应指标的均值。多个P95的均值**不是合并请求的P95**；n=1的标准差和CV不可用，不填零。

| 配置 / prefill | n | 输出 tokens/s | CV | requests/s | Mean TTFT ms | P95 TTFT ms | Mean TPOT ms | P95 TPOT ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP4 后四卡 / 8192 | 3 | 715.72 ± 1.40 | 0.20% | 0.69894 | 6381.40 | 20340.25 | 38.45 | 42.20 |
| TP4 前四卡 / 8192 | 3 | 715.23 ± 2.31 | 0.32% | 0.69847 | 6321.89 | 20346.69 | 38.54 | 42.31 |
| TP2×PP2 后四卡 / 8192 | 3 | 762.92 ± 17.85 | 2.34% | 0.74504 | 4147.24 | 11249.98 | 37.91 | 41.89 |
| TP2×PP2 后四卡 / 4096 | 3 | 761.14 ± 31.73 | 4.17% | 0.74331 | 3147.88 | 10779.10 | 38.98 | 41.35 |
| TP2×PP2 后四卡 / 16384 | 1 | 753.08（单次） | — | 0.73543 | 5782.95 | 11659.09 | 36.86 | 41.75 |

| 配置 / prefill | Mean ITL ms | P95 ITL ms | Mean端到端 ms | P95端到端 ms |
| --- | ---: | ---: | ---: | ---: |
| TP4 后四卡 / 8192 | 38.45 | 23.46 | 45720.24 | 62128.86 |
| TP4 前四卡 / 8192 | 38.54 | 22.95 | 45752.91 | 62591.03 |
| TP2×PP2 后四卡 / 8192 | 37.91 | 32.36 | 42928.99 | 47248.30 |
| TP2×PP2 后四卡 / 4096 | 38.98 | 32.17 | 43023.57 | 51977.78 |
| TP2×PP2 后四卡 / 16384 | 36.86 | 32.60 | 43491.74 | 46062.05 |

相对同GPU4–7的TP4，TP2×PP2/8192的吞吐提高6.60%、mean TPOT降低1.42%、P95端到端降低23.95%；但P95 ITL从23.46增至32.36 ms，不能宣称所有延迟指标都更优。ITL按固定客户端流式事件口径保留，不从均值重建逐token分布。本节点没有八卡结果，不计算四卡到八卡扩展效率。

TP2×PP2/8192的mean TTFT CV为4.13%、mean TPOT CV为2.45%，比TP4更易波动。prefill4096的吞吐相对8192为-0.23%，结合4.17% CV应表述为未分出吞吐优劣；它的首token改善伴随更高的端到端尾延迟与波动。16384初筛吞吐相对8192为-1.29%，mean TTFT上升39.44%，未继续补测。

## 4. 异常、诊断与限制

**一次真实JIT引起的测量拒绝。** 前四卡TP4的第1次重复预热用满五轮，事件数为20/0/16/0/0；第3次重复首次测量检测到24条编译日志，吞吐630.24 tokens/s，被gate拒绝。窗口中部分TileLang日志延迟刷出，但同时有当时新形状902 tokens的JIT告警和真实编译开始/完成证据，拒绝不是仅由旧日志造成。保持原gate，重新两轮安静预热后，第2次尝试通过。没有隐藏warning、修改识别规则或增加尝试上限。其他正式样本均第一次尝试被接受；预热最多五轮、其余配置首个重复主要需要三轮，4096补测首个重复需要四轮。

**4096稳定性诊断。** 三次吞吐CV4.17%、mean/P95 TPOT CV4.82%/5.11%；全部样本保留。有效测量窗口无已知编译事件，所有完整服务日志未识别到抢占、OOM或ERROR；因此不能用已检测JIT或请求不足解释慢样本。按测量阶段（包含客户端初始化及阶段取证、不是严格GPU benchmark计时边界）汇总10秒GPU/CPU采样：

| 4096重复 | 客户端CPU48–51平均忙碌率 | 客户端SMT112–115忙碌率 | GPU温度范围°C | SM频率范围MHz | 换入/换出页增量 |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 28.09% | 19.70% | 46–61 | 2407–2422 | 0 / 0 |
| 2（慢样本） | 73.62% | 0.11% | 47–61 | 2407–2430 | 0 / 0 |
| 3 | 23.57% | 24.52% | 46–61 | 2407–2430 | 0 / 0 |

4096三次显存频率均为12481 MHz、采样clock-event mask均为0；服务CPU平均忙碌率约31.69%/34.47%/34.19%，GPU功耗观测范围约79.69–361.00 W。NUMA与换页快照未发现对应内存压力。慢样本伴随客户端集合负载升高，提示CPU竞争可能影响表现，但集合忙碌率包含本次客户端及后台活动，缺少该历史窗口的完整线程级CPU分解，不能据此确定因果。

资源恢复后仍有16个`doca_spcx_cc`后台进程。收尾额外20秒只读线程采样发现各有一个约100%逻辑CPU占用的活动线程，允许范围CPU0–127，结束瞬间有线程落在CPU48以及客户端SMT兄弟CPU113等位置。这能证明仍存在共享资源竞争条件，**不能反推慢样本当时的具体线程归属**。前四卡客户端SMT与部分8192样本也有背景负载；因此主矩阵收益限于本轮实测运行条件，不声称已隔离所有CPU干扰。

没有进行额外诊断测量：尚不能在保持本轮绑定和不干预他人任务的前提下构造可控的CPU隔离对照，继续相同条件重跑也不能确证因果。保留4096不稳定结论，不用新样本替代原三次结果。未出现OOM、不支持、启动/客户端超时或未完成的主矩阵配置；没有更改精度、V2、Graph或既定gate。

重要warning与实现限制均保留：SM120上SymmMemCommunicator不可用；FlashInfer All Reduce对当前world_size2/4不可用；TP4的PCIe四卡custom allreduce禁用；TileLang部分循环降为串行；PP有NCCL非批量P2P通信器和PyTorch非可写buffer提示；ModelOpt NVFP4格式实验性、FP4 indexer旧字段弃用、EXT4自动prefetch关闭。模型generation_config默认采样参数覆盖提示也保留，正式客户端明确传temperature0等固定请求参数。TP2未出现同一TP4 custom-allreduce禁用提示，不据此推断所有通信kernel均优化或验证完毕。

本轮为顺序运行、每配置有限重复，未做交换顺序实验；随机负载及客户端ITL定义限制了对实际业务的外推。所有有效样本的PASS只代表既定功能、工作量、绑定与日志检查通过，4096的性能不稳定仍成立。

原始JSON、完整日志、预热、拒绝尝试、CPU/GPU/NUMA遥测、解析配置、实际CLI、命令、Git差异和源文件哈希**仅保存在节点48本机工作区**，不随Git分发；精确目录、运行对应关系与重算入口见本机交接记录。仓库只归档本报告、逐次CSV和节点48任务行，探索recipe/campaign未批量复制到projects；公共配置加上述差异足以重建本轮功能设置。
