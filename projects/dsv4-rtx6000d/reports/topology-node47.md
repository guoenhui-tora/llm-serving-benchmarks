# 节点47：vLLM并行拓扑与prefill预算对照

## 1. 结论与完成范围

2026-09-17，`gpu-6000d-47` 在同一NVFP4权重、vLLM 0.29.0、8192输入/1024输出、全局C32条件下，完成TP4、TP8、TP4＋EP4、TP8＋EP8各三次有效测量。**本轮四种拓扑中，GPU4–7的TP4、EP off吞吐最高：718.48 ± 2.62 output tokens/s，CV 0.37%。** 这是本节点、规定绑定及用户允许的残余后台负载下的结果，不推广为模型或平台的官方最优配置。

- 同四卡、同绑定：EP4相对TP4吞吐变化 **-1.68%**，没有吞吐收益；mean TPOT略高，mean TTFT差距较小。
- 同八卡、均不绑定：EP8相对TP8吞吐变化 **+3.71%**，mean TTFT由9836.84降至8591.70 ms，mean TPOT由42.72降至42.07 ms。收益超过本轮重复吞吐波动，但仍低于四卡TP4。
- 四卡到八卡整体扩展：TP8/TP4吞吐比 **0.8509**（-14.91%）；八卡EP8/四卡TP4为 **0.8824**。GPU数量和CPU绑定策略同时变化，不能将差异全部归因于纯拓扑。

主矩阵于北京时间05:57开始、08:45前完成，约2小时48分，早于13:57的8小时检查点。四组吞吐CV均低于0.4%，未触发>3%的额外诊断；TTFT、TPOT也保留逐次数据。可选prefill对照也已收尾：4096完成一次初筛，16384完成初筛及两次补测，全部实验于北京时间09:58结束，总计约4小时1分。没有未完成或阻塞的运行；4096未补测，其余拓扑上的prefill配置仅离线准备，未执行。

**本轮最高吞吐为TP4、EP off、prefill 16384：727.97 ± 4.44 output tokens/s，N=3、CV 0.61%，较8192提高1.32%。** 三次有效吞吐均高于8192的三次样本，但增益较小；mean TTFT由6.33升至8.15秒（+28.80%），mean TPOT由38.33降至36.00 ms。吞吐优先时可将16384作为继续优化候选，重视平均首token延迟时保留8192；不是全面占优。16384初筛和补测各有一次测量因JIT事件被拒绝，补测的一次预热用满五轮后才达到连续两轮安静，详见异常记录。各配置顺序执行，没有通过交错实验排除时间漂移。

4096初筛为710.98 output tokens/s，相对8192为-1.04%，未见吞吐收益，按预算不补测。其mean TTFT为5.28秒，但P95 ITL为192.42 ms，明显高于8192的22.81 ms；仅有一次初筛，不对延迟收益或稳定性作定论。

## 2. 运行条件与配置差异

从[公共TP4 recipe](https://github.com/guoenhui-tora/llm-serving-benchmarks/blob/4642443b1274251b6a47f0d2742e0e59e2dccc95/projects/dsv4-rtx6000d/configs/recipes/vllm-tp4-baseline.yaml)、[节点47 target](https://github.com/guoenhui-tora/llm-serving-benchmarks/blob/4642443b1274251b6a47f0d2742e0e59e2dccc95/projects/dsv4-rtx6000d/configs/targets/rtx6000d-47-tp4-baseline.yaml)、[C32 workload](https://github.com/guoenhui-tora/llm-serving-benchmarks/blob/4642443b1274251b6a47f0d2742e0e59e2dccc95/projects/dsv4-rtx6000d/configs/workloads/dsv4-8192-1024-c32-n128-repeat3.yaml)复制到本机独立工作区后派生，公共原文件未改。下表足以重建候选；探索配置和临时观测脚本继续仅保留本机，未批量归档。

| 配置 | 相对公共recipe的变化 | 环境变量变化 | GPU | 服务CPU / NUMA | 客户端CPU / NUMA |
| --- | --- | --- | --- | --- | --- |
| TP4、EP off | 无 | 无 | 4–7 | 32–47 / 2 | 48–51 / 3 |
| TP8、EP off | `tensor-parallel-size: 8` | 无 | 0–7 | 删除整个target `binding` | 同左 |
| TP4＋EP4 | flags中`no-enable-expert-parallel`替换为`enable-expert-parallel` | 无 | 4–7 | 32–47 / 2 | 48–51 / 3 |
| TP8＋EP8 | TP=8，并替换EP开关 | 无 | 0–7 | 删除整个target `binding` | 同左 |
| TP4，prefill 4096 | 仅`max-num-batched-tokens: 4096` | 无 | 4–7 | 32–47 / 2 | 48–51 / 3 |
| TP4，prefill 16384 | 仅`max-num-batched-tokens: 16384` | 无 | 4–7 | 32–47 / 2 | 48–51 / 3 |

recipe的其他差异仅ID、描述及来源信息，campaign更新ID和引用。PP=1、DP=1；EP off对应专家组大小1。EP候选初筛使用公共workload副本，仅修改ID和重复数为1；补测仅改ID和重复数为2。初筛及补测的服务、target、模型、客户端、单次负载和源码一致性已按解析配置核验，CSV重复编号连续为1/2/3。可选prefill也采用同一初筛/补测口径；未测数据不补造。

固定上下文16384、活动序列容量32、FP8 E4M3 KV、显存比例0.90自动分配、block size 256、chunked prefill；`FULL_DECODE_ONLY`捕获尺寸1/2/4/8/12/16/24/32，保留正常Graph。`VLLM_USE_V2_MODEL_RUNNER=1`、async scheduling、mp executor，FlashInfer autotune及prefix cache关闭，未启用投机解码、CPU offload或八卡NUMA绑定。原有`TILELANG_CACHE_DIR`及JIT日志设置保留，未更改gate或添加兼容性修复。

统一流式`/v1/completions`、seed=0、temperature=0、ignore_eos=true、range_ratio=0、request_rate=inf。每次128请求、C32，实际输入1,048,576、输出131,072 tokens；每轮预热64请求、同样输入输出长度，至少连续两轮无已知编译事件，最多五轮。每次测量最多两次尝试；启动超时1800秒，单个客户端阶段7200秒。每个新服务均通过health、models及关闭thinking的简短中文探测。

### 工件与实际实现证据

- Git commit：`c54734568603d1ef52f6ccc24ce7673768777c37`；runner源码指纹：`f5c33cf418ed84629e3786be4f4c7e651d66dccc3827e4226685a42e3361b583`。通用源码与公共配置未改；启动前本地tracked差异仅为前次节点47阻塞报告的README状态，完整diff和配置哈希保存在本机。实验期间执行源码和配置保持冻结。
- 服务及客户端均为[固定runtime](https://github.com/guoenhui-tora/llm-serving-benchmarks/blob/4642443b1274251b6a47f0d2742e0e59e2dccc95/projects/dsv4-rtx6000d/configs/runtimes/vllm-0.29.0.yaml)/[固定客户端](https://github.com/guoenhui-tora/llm-serving-benchmarks/blob/4642443b1274251b6a47f0d2742e0e59e2dccc95/projects/dsv4-rtx6000d/configs/clients/vllm-bench-0.29.0.yaml)，image ID：`sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`；没有拉取或换镜像。
- [模型](https://github.com/guoenhui-tora/llm-serving-benchmarks/blob/4642443b1274251b6a47f0d2742e0e59e2dccc95/projects/dsv4-rtx6000d/configs/models/dsv4-flash-nvfp4.yaml)的48个权重分片及所需文件核对通过；配置/tokenizer哈希与分片大小身份：`1e1b7e54442bcf7f15ef55192641b5c5da360bcaa608236aa0d06b273aedf8b4`。分片合计175,550,788,904字节，没有计算完整权重内容哈希。
- 实际日志确认V2 Model Runner、`expert_dtype='fp4'`、`FLASHINFER_CUTLASS`、`FLASHINFER_MLA_SPARSE_DSV4`、`fp8_ds_mla`、关闭autotune及Graph捕获。EP4日志明确组大小4、每rank 64/256专家；EP8明确组大小8、每rank 32/256专家，均linear分配。NCCL版本2.30.7；这些是启动/配置路径证据，不是全部kernel或模型数值精度验证。
- 四卡Docker inspect、worker/客户端线程允许集合检查通过；八卡Docker cpuset字段为空，额外保存服务、客户端和线程快照，观察到CPU0–127、NUMA0–3。实际worker NUMA映射并非全部本地：四卡末次测量快照约79%的worker映射页计数在NUMA2、约20%在NUMA3；八卡分布更分散。该计数可含共享映射，不能相加当作独占物理内存用量。

## 3. 性能汇总

[逐次CSV](../data/topology-node47.csv)保留原始数值精度。当前有效数据共**16次、2,048成功、0失败，输入16,777,216、输出2,097,152 tokens**。主矩阵为12次；可选项按实际有效重复数列出。吞吐为均值±样本标准差；只有一个样本时SD/CV不可用，记为“—”。延迟均为毫秒，P95列是各次P95的算术平均，**不是合并请求后的P95**。

| 配置 | N | Output tokens/s | CV | Requests/s | Mean TTFT ms | P95 TTFT ms | Mean TPOT ms | P95 TPOT ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP4，EP off | 3 | 718.48 ± 2.62 | 0.37% | 0.701641 | 6330.44 | 20364.94 | 38.33 | 41.99 |
| TP8，EP off | 3 | 611.35 ± 1.16 | 0.19% | 0.597020 | 9836.84 | 29382.83 | 42.72 | 48.75 |
| TP4＋EP4 | 3 | 706.39 ± 2.70 | 0.38% | 0.689837 | 6286.98 | 20055.58 | 39.14 | 42.78 |
| TP8＋EP8 | 3 | 634.00 ± 1.74 | 0.28% | 0.619143 | 8591.70 | 27206.27 | 42.07 | 47.21 |
| TP4，prefill 4096 | 1 | 710.98 ± — | — | 0.694312 | 5275.60 | 20659.19 | 39.78 | 42.67 |
| TP4，prefill 16384 | 3 | 727.97 ± 4.44 | 0.61% | 0.710910 | 8153.65 | 20018.93 | 36.00 | 41.50 |

| 配置 | Mean ITL ms | P95 ITL ms | Mean E2EL ms | P95 E2EL ms |
| --- | ---: | ---: | ---: | ---: |
| TP4，EP off | 38.33 | 22.81 | 45545.97 | 62213.13 |
| TP8，EP off | 42.72 | 20.38 | 53543.75 | 76558.64 |
| TP4＋EP4 | 39.14 | 24.13 | 46322.73 | 62646.94 |
| TP8＋EP8 | 42.07 | 21.05 | 51625.96 | 73386.81 |
| TP4，prefill 4096 | 39.78 | 192.42 | 45967.75 | 63808.60 |
| TP4，prefill 16384 | 36.00 | 22.49 | 44981.76 | 58094.12 |

GPU每2秒采样，下面覆盖有效测量所属客户端阶段（含客户端准备/收尾，非精确压测计时窗口），各次按采样数汇总。温度为范围，功耗和SM频率为均值；未调整宿主功耗或锁频。

| 配置 | 日志KV预算 GiB/卡 | 功耗 W/卡 | SM MHz | 温度 ℃ |
| --- | ---: | ---: | ---: | ---: |
| TP4，EP off | 32.73 | 240.76 | 2419.7 | 47–60 |
| TP8，EP off | 51.75 | 183.57 | 2422.1 | 46–55 |
| TP4＋EP4 | 32.77 | 233.57 | 2420.5 | 47–59 |
| TP8＋EP8 | 51.81 | 180.10 | 2422.1 | 45–56 |
| TP4，prefill 4096 | 33.28 | 240.03 | 2418.9 | 47–59 |
| TP4，prefill 16384 | 31.09 | 242.76 | 2421.0 | 47–61 |

## 4. 异常与限制

启动前曾因明显CPU后台冲突阻塞，未加载模型。用户处理后要求重新实验；复核仍见CPU32、49及客户端SMT兄弟112、115接近满载，用户明确授权“允许保留，记录后继续实验”。没有终止、移动或更改后台任务。CPU cpuset不隔离后台负载，因此本报告结论适用于这次被接受的资源条件。CPU逐核负载、GPU功耗/温度/频率和约30秒一次的NUMA快照均保存，不能据低频观测排除短时干扰。

以下测量尝试因编译事件被拒绝，**全部排除CSV和性能统计**；它们的请求及token工作量仍完整保存。部分TileLang日志存在缓冲延迟，gate保守保留原行为；未为获得PASS放宽识别。JIT监控事件不等于每条都发生完整CPU重编译，缓存读取和编译需要结合原始时间戳理解。

| 本机运行标识 | 拒绝的重复/尝试 | 编译事件条数 | 成功 / 失败请求 |
| --- | --- | ---: | ---: |
| `tp4-baseline-01` | `r01/measurement-01` | 48 | 128 / 0 |
| `tp4-baseline-prefill16384-screen-01` | `r01/measurement-01` | 4 | 128 / 0 |
| `tp4-baseline-prefill16384-supplement-01` | `r01/measurement-01` | 12 | 128 / 0 |
| `tp4-ep4-supplement-01` | `r02/measurement-01` | 16 | 128 / 0 |
| `tp8-ep8-supplement-01` | `r01/measurement-01` | 32 | 128 / 0 |

当前共保存54个完成的预热阶段，未混入正式数据。未出现模型OOM、不支持、启动超时、预热上限耗尽或最终测量gate阻塞。 已完成配置的完整服务日志未识别抢占记录；这属于日志观察范围，不等于证明任何时刻都不存在抢占。三次重复不构成长期负载或质量验证。

保留的重要warning包括：SM120不支持SymmMemCommunicator、FlashInfer All Reduce不支持当前world size、PCIe-only超过两卡时custom allreduce禁用、TileLang部分向量化循环降为串行、NVFP4格式experimental、旧indexer字段deprecated。客户端无GPU容器中的CUDA/vllm._C导入warning、服务generation_config默认值提示也保留；实际压测显式发送temperature=0等采样设置，且请求/token验收通过。没有通过隐藏warning或关闭正常优化消除这些记录。

原始运行目录、完整JSON/日志、预热、拒绝尝试、解析配置、命令、镜像inspect和遥测仅本机可用，具体位置见本地续接记录；仓库报告不将本机绝对路径作为复现入口。按[项目执行步骤](../README.md#准备本机配置)从公共配置重建表中差异、validate/plan/preflight后，在新的工作区run-root运行即可。源码无修复，临时脚本仅做顺序调度、观测和离线汇总，不改变runner语义。本次所属容器按每次运行自动清理，模型、镜像、历史结果和JIT缓存保留。
