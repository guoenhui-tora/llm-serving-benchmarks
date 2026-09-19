# 节点45：独立 quick／jit_clean 对照

**2026-09-20，gpu-6000d-45 的两次独立启动均完成并 PASS。** 单实例 TP4、DSpark off、C32、GovReport 前128条／1024输出，完整128请求预热后的 quick 为 **664.56 ± 0.28 output tok/s**；独立 jit_clean 为 **663.21 ± 0.51 tok/s**。quick 的三轮正式测量均无已知事件；clean 首轮因52条事件被拒绝，随后最先三轮无事件被接纳。

**本次完整预热后未出现明显慢首轮，quick 与 clean 的性能和实际时间成本接近。** quick 相对 clean 的吞吐、Mean TTFT、Mean TPOT均值分别差 **+0.204%／-0.052%／-0.223%**，均在预先约定的3%内。两组都实际执行4×128请求，协议耗时 **15.79／15.81分钟**，整个 run 均约18.35分钟。没有观察到本次 quick 明显节约时间，也不能据一次配对宣称两协议普遍等价或跨启动耗时稳定。

范围仅限本机本轮固定镜像、权重、缓存起点及后台负载。两段 CPU／SMT 忙碌率存在差异；0.20%的吞吐差虽然大于各组三次相对极差，也不足以排除启动、顺序或资源影响。未执行 stable，不将低波动改称 stable PASS。未测试 DSpark on、其他负载、拓扑、并发或其他节点。

## 同机对照与逐轮证据

以下统计来自[完整逐轮CSV](../data/dspark-protocol-node45.csv)重新读取计算。正式接纳各 n=3，±为样本标准差，CV=样本标准差/均值；P95行是各轮P95的均值与标准差，不是合并请求后的P95。延迟统一使用 ms。

| 指标 | quick 均值 ± SD | CV | jit_clean 均值 ± SD | CV |
| --- | ---: | ---: | ---: | ---: |
| 输出 tok/s | 664.5642 ± 0.2798 | 0.042% | 663.2128 ± 0.5122 | 0.077% |
| 请求 requests/s | 0.648988 ± 0.000273 | 0.042% | 0.647669 ± 0.000500 | 0.077% |
| 客户端 benchmark 秒 | 197.2300 ± 0.0830 | 0.042% | 197.6320 ± 0.1526 | 0.077% |
| Mean TTFT ms | 5649.9929 ± 2.6592 | 0.047% | 5652.9483 ± 2.7604 | 0.049% |
| P95 TTFT ms | 19631.9901 ± 21.7907 | 0.111% | 19636.8171 ± 13.6395 | 0.069% |
| Mean TPOT ms | 42.6065 ± 0.0194 | 0.046% | 42.7020 ± 0.0372 | 0.087% |
| P95 TPOT ms | 45.9016 ± 0.0355 | 0.077% | 45.9884 ± 0.0372 | 0.081% |
| Mean ITL ms | 42.6065 ± 0.0194 | 0.046% | 42.7020 ± 0.0372 | 0.087% |
| P95 ITL ms | 26.6053 ± 0.0475 | 0.178% | 26.9172 ± 0.1605 | 0.596% |
| Mean E2EL ms | 49236.4811 ± 21.2018 | 0.043% | 49337.0661 ± 38.8511 | 0.079% |
| P95 E2EL ms | 64809.7260 ± 23.5556 | 0.036% | 64891.1501 ± 58.5109 | 0.090% |

吞吐／Mean TTFT／Mean TPOT的三次相对极差分别为 quick **0.077%／0.083%／0.080%**，clean **0.149%／0.088%／0.171%**；各组均满足本次描述“三次接近”的≤3%观察标准。

| run／阶段 | 原轮次 | 接纳 | 事件行数 | benchmark 秒 | 输出 tok/s | Mean TTFT ms | Mean TPOT ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| quick／warmup | 1 | false | 52 | 250.7914 | 522.6335 | 9284.4174 | 52.1424 |
| quick／measurement | 1 | true | 0 | 197.2691 | 664.4326 | 5648.3718 | 42.6182 |
| quick／measurement | 2 | true | 0 | 197.1347 | 664.8856 | 5648.5449 | 42.5841 |
| quick／measurement | 3 | true | 0 | 197.2863 | 664.3745 | 5653.0619 | 42.6173 |
| jit_clean／measurement | 1 | false | 52 | 251.0114 | 522.1755 | 9259.9924 | 52.2205 |
| jit_clean／measurement | 2 | true | 0 | 197.4611 | 663.7865 | 5651.1444 | 42.6614 |
| jit_clean／measurement | 3 | true | 0 | 197.6802 | 663.0506 | 5656.1261 | 42.7100 |
| jit_clean／measurement | 4 | true | 0 | 197.7546 | 662.8014 | 5651.5745 | 42.7345 |

每轮均128成功、0失败、实际输入1,042,149／输出131,072 tokens；这四项每组三个接纳样本的SD均为0；成功数和token总量的CV为0，失败均值为0、CV不定义。每组含预热或拒绝轮共512请求、输入4,168,596／输出524,288 tokens；两组合计1024成功、0失败、输入8,337,192／输出1,048,576 tokens。每组正式接纳384请求、输入3,126,447／输出393,216 tokens。

CSV含 quick 预热和全部正式轮，以及 clean 的拒绝轮；时间戳为UTC、含时区，`started_at/finished_at`是含客户端初始化与检查的阶段窗口，`duration_s`仅官方 benchmark 计时。`failed=0`由128请求全部成功推定（固定客户端原始结果未提供failed，原始normalized保留推定标记）。DSpark接受率和接受长度不适用，留空。quick的accepted按协议接纳规则填写，不预设等同无JIT。

## 独立启动的时间成本

每种协议只启动一次；quick先运行，clean后运行，缓存副本事先准备。全部时间使用秒：

| 阶段／口径 | quick | jit_clean |
| --- | ---: | ---: |
| run开始至服务launch（含内置preflight、取证） | 45.5905 | 44.3244 |
| 服务launch日志至首次health 200 | 105.7880 | 105.8505 |
| health至功能响应落盘（含绑定、models与中文probe） | 0.3456 | 0.3466 |
| 协议墙钟：全部轮次、初始化和检查 | 947.3978 | 948.8670 |
| 所有轮次客户端benchmark计时之和 | 842.4815 | 843.9073 |
| 三个接纳轮次benchmark计时之和 | 591.6901 | 592.8959 |
| 协议结束至run结束：取证＋清理＋落盘 | 1.4615 | 1.4475 |
| 其中服务清理：Docker kill至destroy | 1.2775 | 1.2644 |
| run.json总墙钟（不含外部准备） | 1100.5886 | 1100.8381 |
| 外部bench进程墙钟（含Python入口退出） | 1100.8375 | 1101.0912 |

quick预热客户端计时为 250.7914 秒；clean被拒绝首轮为 251.0114 秒，均计入各自协议总成本。相较quick，clean协议多 1.4693 秒（0.155%），run总墙钟多 0.2495 秒。每组服务日志均为512次 `/v1/completions` POST及1次计时外中文chat探测；探测不计入CSV或上述负载token汇总。

run时间窗为北京时间 quick 00:29:29–00:47:50、clean 00:50:24–01:08:45。服务计时从run的Starting日志取起点，以带Docker时间戳的首个health成功日志为终点；功能完成取chat-response.json落盘时间。清理使用本次容器的历史Docker事件，正常runner所属清理产生signal 9／exitCode 137，不是推理OOM。两组结束前inspect的OOMKilled均为false。外部validate、plan、preflight、缓存准备、监控线程退出和本报告取证不混入run.json总成本。

## 日志、资源与解释边界

两组kernel checks均PASS、缺失与forbidden均为空。实际日志确认 DeepseekV4ForCausalLM、V2 runner、FLASHINFER_MLA_SPARSE_DSV4、FLASHINFER_CUTLASS、expert_dtype=fp4、fp8_ds_mla、混合tokens=16预热、autotune关闭以及decode Graph捕获完成。每卡KV预算32.73 GiB、等效容量85,450 tokens；两组采样活动请求峰值均32，KV峰值分别24.105%／24.065%，抢占计数均为0。完整日志未发现ERROR、Traceback、CUDA error或OOM。

quick预热与clean第1轮各52条已知事件，涵盖Triton JIT monitor和TileLang编译开始／完成及JIT monitor；其余六轮事件均为0。匹配行数不等于独立编译次数，日志窗口包含初始化与刷新，不能严格归属到客户端计时内。两组均有48条warning匹配，完整保留：SM120不支持SymmMem，TP4不支持FlashInfer All Reduce，超过两张PCIe GPU时custom allreduce禁用，NVFP4格式实验性、弃用的FP4 indexer字段、generation_config默认采样覆盖等；客户端显式temperature=0、seed=0。没有放宽gate或隐藏fallback。

CPU每5秒读取/proc/stat，GPU同步低频采样；每15秒保存进程列表和KV指标。下表为整个协议阶段／三个接纳阶段内CPU平均忙碌率（按CPU ticks加权，包含服务、客户端、后台进程和采样工具，不等于本次进程独占使用率）：

| CPU集合 | quick 协议／接纳 | clean 协议／接纳 |
| --- | ---: | ---: |
| 服务32–47 | 34.75%／35.23% | 29.06%／29.57% |
| 服务SMT兄弟96–111 | 0.16%／0.12% | 0.14%／0.16% |
| 客户端48–51 | 3.90%／3.97% | 14.37%／18.78% |
| 客户端SMT兄弟112–115 | 0.21%／0.21% | 7.21%／10.06% |

已知8个doca_spcx_cc持续存在，roce-init.service的CPU增量两段均约8.00个逻辑CPU；进程位置快照中的2个DOCA位于服务CPU32/33，其余在其他CPU。未观察到新增矿工等可识别异常进程；clean首轮额外捕获了ptxas/cicc编译进程。两段客户端及其SMT忙碌率明显不同，采样的进程最后运行CPU与生命周期平均%CPU不足以完整归因这些差异；不将两段资源背景称为完全一致，也不据此认定协议造成性能变化。

接纳阶段（含每轮初始化／检查）GPU4–7采样SM频率均在2415–2430 MHz，显存频率均12481 MHz，温度48–60°C；平均功耗quick约250.80 W、clean约251.11 W，峰值295.67／297.73 W。阶段包含客户端初始化的空闲段，平均功耗不能作为纯benchmark功耗；低频采样也不能排除短暂竞争。没有锁频、改功耗上限或宿主NUMA设置。绑定取证通过，只清理本次容器，结束GPU计算进程为空。

## 固定身份与缓存起点

- Git：运行前 `git pull --ff-only` 从b32e0c1快进到 `fa18d8079c68fc2976fa32690a267ca82834581d`，工作树干净；两次运行期间源码、配置和数据冻结。
- 两组runner源码指纹：`f6e873002b8cf61963bd221d63b0fe0a520e0bc6118f4ca63acfa54fc6726d52`。
- 服务和客户端镜像：`vllm/vllm-openai:v0.29.0`；image ID及RepoDigest SHA256：`c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`。执行始终pull never。
- 原始NVFP4模型：`/data/models/DeepSeek-V4-Flash-0731-NVFP4`；模型身份指纹 `1e1b7e54442bcf7f15ef55192641b5c5da360bcaa608236aa0d06b273aedf8b4`。config/tokenizer/index哈希及48个分片大小已保存，不声称完整权重内容校验；没有补丁或权重转换。
- [GovReport数据](../data/govreport-near8k.jsonl) SHA256：`33bdbd36a8b335300974e1b07aa66b2c24c88cc4c4b55a21350aa3564a945d53`，固定前128条且顺序不变，实际tokenizer逐轮验证。
- 本机：`gpu-6000d-45`／`10.90.1.45`，RTX6000D GPU4–7，SM120、每卡85,651 MiB、驱动580.159.04；服务CPU32–47/NUMA2，客户端CPU48–51/NUMA3，SMT兄弟与GPU近端映射已核实。

工作区从公共configs新复制到 `experiments/dspark-protocol-node45/`，本地rear target仅改节点ID/IP；另外创建rear-quick.yaml与rear-clean.yaml两份薄target，仅改变cache_root，两campaign各引用对应target。服务recipe逐字节保留。

缓存源由当前镜像ID、SM、模型ID和完整off recipe实际计算得到：`/home/enhui/.cache/serving-bench/vllm/d19496fcae0f32e092d6`，不是猜测历史路径。2026-09-20 00:27:33+08:00先冻结快照，再在任何服务启动前复制两份：

```text
/home/enhui/.cache/serving-bench/dspark-protocol-node45/20260920-01/
├── snapshot/
├── quick/vllm/d19496fcae0f32e092d6/
└── clean/vllm/d19496fcae0f32e092d6/
```

各副本334个普通文件、463个文件/目录条目，相对路径、SHA256和权限清单逐项一致；没有符号链接或硬链接，共享快照不用于服务写入。源目录有root所有的不可读文件，普通用户读取报PermissionError后未跳过：使用同一固定镜像的无网络、无GPU辅助容器将源目录只读挂载，完整复制并验证复制前后源清单未变；仅将新运行副本改为UID/GID 2002，权限模式不变且检查可读写。原缓存和快照保留。clean启动前再次逐项确认它未被quick写入。

缓存准备脚本内部耗时单列：冻结快照及首次核验0.1976秒，quick副本复制/核验0.1143秒，clean副本0.1138秒，总0.4305秒；不含辅助容器启动和权限排查，均不计入服务或协议耗时。宿主页缓存、目录路径、先后顺序和独立进程状态仍有残余差异；内容一致不保证逐worker运行状态完全相同。

## 复现入口和实际命令

公共入口：[off quick campaign](../configs/campaigns/dspark-off-quick.yaml)、[off clean campaign](../configs/campaigns/dspark-off-clean.yaml)、[TP4 recipe](../configs/recipes/tp4.yaml)、[完整预热workload](../configs/workloads/govreport-c32-quick-full.yaml)、[jit_clean workload](../configs/workloads/govreport-c32-jit-clean.yaml)。按[项目任务与缓存规则](../README.md#2026-09-20四节点预热与负载对照)在本地复制、修改薄target并克隆缓存后执行，不能直接使用归档target的48地址。

服务条件为TP4/PP1/DP1、EP off、上下文16384、活动容量32、prefill8192、显存比例0.90、FP8 E4M3 KV、V2＋async、autotune/prefix cache off，Graph尺寸1/2/4/8/12/16/24/32、上限32。客户端为固定vLLM 0.29.0、流式 `/v1/completions`、C32、128请求、temperature=0、seed=0、ignore_eos=true、request_rate=inf、num_warmups=0。quick单轮full预热＋3轮；clean无独立预热、最先3轮无事件、最多12轮；两组budget_s=2700、timeout_s=900，实际没有重试或扩展预算。

每组启动前分别完成validate、plan、prepare_logging和preflight；该off recipe不需要自定义日志文件，使用jit-monitor-verbose，未使用K5补丁脚本。以下是本次实际bench命令（通过tmux中的只读监控wrapper调用）：

```bash
./bench run /home/enhui/llm-serving-benchmarks/experiments/dspark-protocol-node45/configs/campaigns/dspark-off-quick.yaml \
  --run-root /home/enhui/llm-serving-benchmarks/experiments/dspark-protocol-node45/results/off-quick-01
./bench run /home/enhui/llm-serving-benchmarks/experiments/dspark-protocol-node45/configs/campaigns/dspark-off-clean.yaml \
  --run-root /home/enhui/llm-serving-benchmarks/experiments/dspark-protocol-node45/results/off-clean-01
```

以下完整Docker服务命令直接取自各run的 `cases/tp4-off/command.sh`，包含当次容器名、绑定、环境、只读模型挂载和缓存副本，不手工另维护一套参数。复现时需新容器名及新的run-root；缓存应从保留快照再生成新的两个副本，不能把本次跑过的副本当成原始起点。

<details>
<summary>quick 实际服务命令</summary>

```bash
docker run -d --pull never --name sb-a27439e5efe842ef-server --label io.serving-bench.run=a27439e5efe842ef --network host --ipc host --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/dspark-protocol-node45/20260920-01/quick/vllm/d19496fcae0f32e092d6:/root/.cache:rw --cpuset-cpus 32-47 --cpuset-mems 2 --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --cap-add SYS_NICE -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model --served-model-name deepseek-v4-flash --host 127.0.0.1 --port 31249 --trust-remote-code --enable-auto-tool-choice --no-enable-prefix-caching --no-enable-expert-parallel --enable-chunked-prefill --jit-monitor-verbose --async-scheduling --tensor-parallel-size 4 --pipeline-parallel-size 1 --data-parallel-size 1 --max-model-len 16384 --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8_e4m3 --block-size 256 --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' --moe-backend auto --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' --kernel-config '{"enable_flashinfer_autotune":false}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,12,16,24,32],"max_cudagraph_capture_size":32}' --seed 0 --distributed-executor-backend mp
```

</details>

<details>
<summary>clean 实际服务命令</summary>

```bash
docker run -d --pull never --name sb-f7080118afad4348-server --label io.serving-bench.run=f7080118afad4348 --network host --ipc host --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/dspark-protocol-node45/20260920-01/clean/vllm/d19496fcae0f32e092d6:/root/.cache:rw --cpuset-cpus 32-47 --cpuset-mems 2 --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --cap-add SYS_NICE -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model --served-model-name deepseek-v4-flash --host 127.0.0.1 --port 31249 --trust-remote-code --enable-auto-tool-choice --no-enable-prefix-caching --no-enable-expert-parallel --enable-chunked-prefill --jit-monitor-verbose --async-scheduling --tensor-parallel-size 4 --pipeline-parallel-size 1 --data-parallel-size 1 --max-model-len 16384 --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8_e4m3 --block-size 256 --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' --moe-backend auto --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' --kernel-config '{"enable_flashinfer_autotune":false}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,12,16,24,32],"max_cudagraph_capture_size":32}' --seed 0 --distributed-executor-backend mp
```

</details>

全部原始结果**仅本机可用**，不随Git分发：`experiments/dspark-protocol-node45/results/{off-quick-01,off-clean-01}/`保存plan/resolved、环境与绑定、HTTP probes、完整服务日志、kernel检查、逐轮raw/metrics/dataset-manifest/compilation和protocol；`reports/`保存独立validate/plan/preflight输出、缓存SHA256/权限清单、冻结输入哈希、两个observation目录内CPU/GPU/KV时间线及进程快照、Docker事件、成本汇总和本地重算脚本。归档CSV与本报告可独立用于核算性能，不依赖本机日志才能理解结论。

交付前检查统一CSV字段、8轮顺序与原始精度、接纳语义、每轮请求/token总量、数字重算、文档链接及 `git diff --check`。仅新增本节点报告与CSV，并更新README中45的完成行；不提交探索配置或大日志，不push。
