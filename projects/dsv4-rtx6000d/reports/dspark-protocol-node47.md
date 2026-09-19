# 47节点：随机等长与GovReport完整预热对照

**已完成本节点两次独立启动、共2轮预热和6轮正式测量，两组case及quick协议均PASS。** 2026-09-20北京时间，GPU4–7、单TP4、DSpark off、C32、1024输出下，随机8192输入为 **718.42 ± 2.31 tok/s（CV 0.32%）**，GovReport为 **664.69 ± 0.37 tok/s（CV 0.06%）**，后者低 **7.48%**。本机同批重现了真实文本负载较慢的现象，不能混用两种负载作为性能基线。

GovReport的Mean TPOT高11.12%、Mean端到端延迟高8.07%，但Mean TTFT低10.81%。其输入总量还少0.61%，差异不能简单解释为输入token更多；本实验没有隔离文本内容、输入长度分布、生成路径与调度shape，也保留独立启动、顺序、宿主页缓存和后台负载差异，未确定唯一原因。

随机组预热／正式事件数为20／8、0、0，GovReport为52／0、0、0。**一轮完整128请求预热在随机组仍未消除全部已知事件；GovReport本次三轮无事件，未见慢首轮。** 这不证明所有负载只需一轮预热，也不证明跨启动稳定性。quick接纳全部正式轮，包括随机首轮事件样本；未追加clean、补测、重启或其他并发／拓扑。

原始精度、含预热的全部8轮见[统一CSV](../data/dspark-protocol-node47.csv)。仅比较本机，未访问45、46或48。

## 同机逐次结果

每轮均128成功、0失败、输出131,072 tokens；random每轮输入1,048,576，GovReport每轮输入1,042,149。正式6轮合计768成功、输入6,272,175、输出786,432 tokens；含预热8轮合计1024成功、输入8,362,900、输出1,048,576 tokens。CSV的时间戳表示客户端阶段边界（含初始化和检查），`duration_s`仅为客户端benchmark计时。

| 负载 | 阶段／轮次 | 接纳 | 事件日志行 | benchmark秒 | 输出tok/s | requests/s | Mean TTFT ms | Mean TPOT ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | warmup/1 | false | 20 | 182.916582 | 716.567075 | 0.699773 | 6426.446383 | 38.359058 |
| random | measurement/1 | true | 8 | 181.776823 | 721.060023 | 0.704160 | 6227.636233 | 38.274159 |
| random | measurement/2 | true | 0 | 182.693055 | 717.443801 | 0.700629 | 6391.697591 | 38.338026 |
| random | measurement/3 | true | 0 | 182.865360 | 716.767791 | 0.699969 | 6382.230128 | 38.388968 |
| govreport | warmup/1 | false | 52 | 249.796912 | 524.714254 | 0.512416 | 9261.962937 | 51.922349 |
| govreport | measurement/1 | true | 0 | 197.225760 | 664.578503 | 0.649002 | 5645.486703 | 42.609651 |
| govreport | measurement/2 | true | 0 | 197.280192 | 664.395138 | 0.648823 | 5655.248428 | 42.613701 |
| govreport | measurement/3 | true | 0 | 197.068776 | 665.107900 | 0.649519 | 5646.613829 | 42.570308 |

下表同样按原轮次排列。P95为各轮自身分位数，不合并请求或不同配置。

| 负载 | 阶段／轮次 | P95 TTFT ms | P95 TPOT ms | Mean ITL ms | P95 ITL ms | Mean E2EL ms | P95 E2EL ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | warmup/1 | 20550.020756 | 42.168926 | 38.359059 | 22.992744 | 45667.763091 | 62587.460141 |
| random | measurement/1 | 20375.568884 | 41.848419 | 38.274159 | 22.131322 | 45382.100929 | 61392.076700 |
| random | measurement/2 | 20385.918381 | 42.006874 | 38.338027 | 22.520466 | 45611.498214 | 61966.717924 |
| random | measurement/3 | 20376.529193 | 42.056381 | 38.388969 | 23.037034 | 45654.144543 | 61780.219764 |
| govreport | warmup/1 | 32732.884210 | 68.764092 | 51.922350 | 26.769749 | 62378.526031 | 77853.428404 |
| govreport | measurement/1 | 19609.175189 | 45.924910 | 42.609652 | 26.891996 | 49235.160144 | 64839.547751 |
| govreport | measurement/2 | 19649.035497 | 45.955069 | 42.613702 | 26.822277 | 49249.064438 | 64795.856168 |
| govreport | measurement/3 | 19609.430424 | 45.822597 | 42.570309 | 26.814204 | 49196.039224 | 64809.580334 |

## 三轮正式统计

均值±样本标准差，括号为CV；P95行表示“三个重复P95的均值±标准差”，不是全部请求合并后的P95。预热不进入统计，未测DSpark接受率／接受长度，CSV相应字段留空。

| 指标 | random，n=3 | GovReport，n=3 | GovReport相对random |
| --- | ---: | ---: | ---: |
| benchmark时间 s | 182.445 ± 0.585（0.321%） | 197.192 ± 0.110（0.056%） | +8.08% |
| 输出 tok/s | 718.424 ± 2.308（0.321%） | 664.694 ± 0.370（0.056%） | -7.48% |
| 请求/s | 0.701586 ± 0.002254（0.321%） | 0.649115 ± 0.000361（0.056%） | -7.48% |
| Mean TTFT ms | 6333.855 ± 92.110（1.454%） | 5649.116 ± 5.340（0.095%） | -10.81% |
| P95 TTFT ms | 20379.339 ± 5.718（0.028%） | 19622.547 ± 22.940（0.117%） | -3.71% |
| Mean TPOT ms | 38.334 ± 0.058（0.150%） | 42.598 ± 0.024（0.056%） | +11.12% |
| P95 TPOT ms | 41.971 ± 0.109（0.259%） | 45.901 ± 0.069（0.151%） | +9.36% |
| Mean ITL ms | 38.334 ± 0.058（0.150%） | 42.598 ± 0.024（0.056%） | +11.12% |
| P95 ITL ms | 22.563 ± 0.454（2.014%） | 26.843 ± 0.043（0.159%） | +18.97% |
| Mean E2EL ms | 45549.248 ± 146.316（0.321%） | 49226.755 ± 27.494（0.056%） | +8.07% |
| P95 E2EL ms | 61713.005 ± 293.158（0.475%） | 64814.995 ± 22.343（0.034%） | +5.03% |

按任务预先约定的三个核心指标判断：random的吞吐／Mean TTFT／Mean TPOT相对极差为0.60%／2.59%／0.30%，GovReport为0.11%／0.17%／0.10%，均≤3%，可描述为各组三次接近。两组均值差异均超过3%，不能描述为两个负载性能接近。random的P95 ITL相对极差为4.01%，不能扩展为“所有指标都接近”，也不能把本轮quick改称stable。

## 时间与预算

random的run时间为北京时间00:29:39.805–00:46:21.593，GovReport为00:48:28.038–01:06:44.555。每组一次启动，固定128请求预热1轮＋128请求正式3轮；协议预算2700秒、单客户端阶段900秒，无追加或重试。两组协议耗时均在上限内。

| 时间项，秒 | random | GovReport |
| --- | ---: | ---: |
| 容器实际启动至首次health 200 | 107.757 | 105.815 |
| health至功能响应保存 | 0.394 | 0.357 |
| 完整协议墙钟（预热＋正式，含初始化／检查） | 847.837 | 944.500 |
| 预热benchmark计时 | 182.917 | 249.797 |
| 三轮正式benchmark计时之和 | 547.335 | 591.575 |
| 四轮benchmark计时之和 | 730.252 | 841.372 |
| 协议结束后取证及清理 | 1.378 | 1.480 |
| run总墙钟（含内置preflight） | 1001.788 | 1096.517 |

启动至health由Docker `State.StartedAt`与完整服务日志首个health成功时间戳相减；runner在docker启动命令返回后开始的readiness计时另为107.369／105.436秒，口径略有不同。功能结束取`chat-response.json`保存时间，两次probes均PASS。清理列由protocol结束至case结束计算，包含最终日志／绑定取证，未单独测纯容器删除时间。run总耗时不包含独立执行的validate／plan／preflight、缓存准备及人工复核；外层观察器有最多5秒退出观察延迟，不冒充run时间。

GovReport协议比random多96.663秒，四轮benchmark累计多111.120秒，其中预热多66.880秒、正式三轮多44.239秒。客户端初始化与检查也有差异，不能用benchmark计时之和替代协议成本。两组独立启动各仅一次，不据此证明启动耗时稳定。

## 日志与资源复核

两组完整kernel检查均PASS，missing／forbidden为空；无OOM、ERROR、Traceback或抢占记录。日志确认`DeepseekV4ForCausalLM`、`FLASHINFER_MLA_SPARSE_DSV4`、`FLASHINFER_CUTLASS`、expert_dtype fp4、`fp8_ds_mla`、V2 runner、async、关闭prefix cache／FlashInfer autotune、Graph FULL_DECODE_ONLY及捕获列表1/2/4/8/12/16/24/32。每卡KV预算均32.73 GiB，日志逻辑容量85,450 tokens。未加载DSpark补丁或草稿模型。

random预热有20条事件、正式首轮8条MHC TileLang事件，后两轮无事件；GovReport预热52条事件，包含MHC TileLang编译起止与JIT监测、top-k首次路径，正式三轮无事件。匹配行数不等于独立编译次数；监测可能包含首次加载和延迟日志，窗口还含客户端初始化及日志刷新。原规则未修改，未删除事件样本。

两组保留相同非事件warning：SM120 SymmMem不支持、world_size=4时FlashInfer All Reduce不支持、超过两张PCIe GPU禁用custom allreduce、NVFP4格式实验性、FP4 indexer字段弃用、EXT4不自动prefetch、模型generation_config覆盖默认采样值。实际客户端显式传temperature=0，未采用warning中的默认temperature=1；关闭autotune的日志也按原检查保留。kernel报告warning总行数分别48／48，和协议事件计数不是同一个口径。

服务CPU32–47／NUMA2，客户端CPU48–51／NUMA3；SMT兄弟分别96–111、112–115。GPU4/5与6/7分别为近端PXB组，均靠近NUMA2。所有服务／客户端Docker cpuset及线程允许集合检查通过。就绪后`numa_maps`快照显示worker多数映射页在NUMA2，仍有NUMA3及少量NUMA0/1页面；允许集合不证明全部页面本地，映射计数也不能直接相加当作独占物理内存。

按正式客户端阶段窗口汇总`/proc/stat`差分（包含初始化／检查，约5秒采样），下列忙碌率为对应CPU集合的平均，包含后台进程：

| CPU集合 | random忙碌率 | GovReport忙碌率 |
| --- | ---: | ---: |
| 整机 | 11.68% | 11.71% |
| 服务32–47 | 30.27% | 30.16% |
| 服务SMT兄弟96–111 | 0.13% | 0.13% |
| 客户端48–51 | 13.09% | 18.94% |
| 客户端SMT兄弟112–115 | 10.98% | 3.23% |

约30秒一次的进程快照持续观察到10个`doca_spcx_cc`，未见新增矿工或类似异常。可读system.slice cgroup中`roce-init.service`在两组均约消耗6个逻辑CPU；这不是全部10个DOCA线程的总量。整机与服务集合平均忙碌率接近，客户端及其SMT兄弟分布并不相同；cpuset不独占物理核，不能断言资源完全无干扰，也不按CPU负载比例修正性能。

GPU采样约5秒，正式阶段GPU4–7且利用率≥90%的采样中，random／GovReport平均每卡功耗265.61／278.92 W，温度范围52–61／53–61°C，SM频率均2415–2422 MHz；未见持续热降频迹象，但低频采样不能排除瞬时干扰。KV峰值24.15%／24.10%，累计抢占均0。后台资源、KV和进程观察均保留原始时间戳；没有修改宿主功耗、锁频、SMT或全局NUMA设置。

## 缓存起点与身份

先执行`git pull --ff-only`，HEAD为`fa18d80`（完整提交见下表），确认包含该提交；从项目复制独立工作区。两组均使用同一完整recipe，按`cache_directory`计算的原缓存为`/home/enhui/.cache/serving-bench/vllm/d19496fcae0f32e092d6`，未凭历史路径猜测。

北京时间2026-09-20 00:27:31.647–00:27:32.196，先验证源缓存，再普通复制冻结快照及random／govreport两份独立副本。快照共320文件、124子目录，文件内容11,595,210 bytes；SHA256／相对路径／权限清单散列为`6bff46fd93b8ee103929636121ab44267a390c65701e5ae5c602a029cde7a9be`。前后源清单、快照和两副本清单完全一致，无软链接或硬链接共享写入。

源中`vllm/modelinfos/...json`为root所有，宿主用户首次读取被拒绝且未复制任何文件。随后以固定镜像、无GPU、无网络的临时准备容器只读挂载源缓存，完整读取和复制；仅将本次快照／副本所有权改为UID/GID 2002，模式保留。宿主用户再次核对两副本可读写且哈希一致；GovReport启动前重新核验其副本未被random写入。原缓存、冻结快照和更新后的两副本全部保留，未清空缓存改成冷启动。

缓存准备脚本总计0.549秒（含清单校验、复制和本次副本所有权调整），其中源与快照检查／复制0.214秒，random副本复制／校验0.107秒，GovReport为0.090秒。这个计时不含之前权限失败、准备容器启动或人工检查，单独列出，不计入服务／协议耗时。缓存目录克隆未清宿主页缓存，路径不同仍是剩余差异。源中已有flashinfer、vllm、tilelang目录，没有triton目录；本轮runtime显式设置`TRITON_CACHE_DIR=/root/.cache/triton`，两边起点相同。

| 身份 | 实测值 |
| --- | --- |
| 节点 | gpu-6000d-47，10.90.1.47，8×RTX6000D，仅用GPU4–7 |
| 驱动 | 580.159.04 |
| 执行Git commit | fa18d8079c68fc2976fa32690a267ca82834581d |
| runner源码指纹 | f6e873002b8cf61963bd221d63b0fe0a520e0bc6118f4ca63acfa54fc6726d52 |
| 服务及客户端镜像 | vllm/vllm-openai:v0.29.0 |
| image ID | sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 |
| 模型路径 | /data/models/DeepSeek-V4-Flash-0731-NVFP4 |
| 模型元数据／分片大小身份 | 1e1b7e54442bcf7f15ef55192641b5c5da360bcaa608236aa0d06b273aedf8b4 |
| config.json SHA256 | bb0d2286d6761439e41d3cef31d16489411b816ed8688922f59730bbd5567cdb |
| tokenizer.json SHA256 | 8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf |
| GovReport SHA256 | 33bdbd36a8b335300974e1b07aa66b2c24c88cc4c4b55a21350aa3564a945d53 |

两次模型元数据和48分片大小身份一致，未重算全量权重内容哈希。运行前工作树无本地差异；运行后逐文件核验源码及工作区配置未变。root-owned准备产物不进入提交。

## 实际命令与复现入口

公共入口：[random campaign](../configs/campaigns/tp4-random-quick.yaml)、[GovReport off campaign](../configs/campaigns/dspark-off-quick.yaml)、[TP4 recipe](../configs/recipes/tp4.yaml)、[rear target](../configs/targets/rear.yaml)、[random workload](../configs/workloads/random-c32-quick-full.yaml)、[GovReport workload](../configs/workloads/govreport-c32-quick-full.yaml)。预算和缓存规则见[节点任务](../README.md#2026-09-20四节点预热与负载对照)。本轮只改本地rear的IP／节点id，并新增两份薄target分别设置cache_root；两campaign分别引用rear-random／rear-govreport，公共recipe与workload保持原样。

本机工作区为`experiments/dspark-protocol-node47/`。两campaign分别完成validate、plan、日志准备和实际CLI preflight后执行以下命令，tmux中的外层观察器只运行CLI并读取资源，不修改runner或测量逻辑。复现需重新从保留的冻结快照制作两个新的可写副本，并使用新的run-root，不能直接复用已写入的缓存来声称相同起点。

```bash
./bench run experiments/dspark-protocol-node47/configs/campaigns/tp4-random-quick.yaml \
  --run-root experiments/dspark-protocol-node47/results/random-quick-01
./bench run experiments/dspark-protocol-node47/configs/campaigns/dspark-off-quick.yaml \
  --run-root experiments/dspark-protocol-node47/results/govreport-quick-01
```

以下完整服务命令从各次`cases/tp4-off/command.sh`直接导出，仅插入换行用于阅读。它们是本次实测快照，不是另一份维护的配置标准答案。

### random服务

```bash
docker run -d \
  --pull never \
  --name sb-437c282f502f4815-server \
  --label io.serving-bench.run=437c282f502f4815 \
  --network host \
  --ipc host \
  --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/dspark-protocol-node47/20260920/random/vllm/d19496fcae0f32e092d6:/root/.cache:rw \
  --cpuset-cpus 32-47 \
  --cpuset-mems 2 \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --cap-add SYS_NICE -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 \
  --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model \
  --served-model-name deepseek-v4-flash \
  --host 127.0.0.1 \
  --port 31249 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --no-enable-prefix-caching \
  --no-enable-expert-parallel \
  --enable-chunked-prefill \
  --jit-monitor-verbose \
  --async-scheduling \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 256 \
  --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' \
  --moe-backend auto \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,12,16,24,32],"max_cudagraph_capture_size":32}' \
  --seed 0 \
  --distributed-executor-backend mp
```

### govreport服务

```bash
docker run -d \
  --pull never \
  --name sb-de905cb80edc4656-server \
  --label io.serving-bench.run=de905cb80edc4656 \
  --network host \
  --ipc host \
  --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/dspark-protocol-node47/20260920/govreport/vllm/d19496fcae0f32e092d6:/root/.cache:rw \
  --cpuset-cpus 32-47 \
  --cpuset-mems 2 \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --cap-add SYS_NICE -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 \
  --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model \
  --served-model-name deepseek-v4-flash \
  --host 127.0.0.1 \
  --port 31249 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --no-enable-prefix-caching \
  --no-enable-expert-parallel \
  --enable-chunked-prefill \
  --jit-monitor-verbose \
  --async-scheduling \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 256 \
  --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' \
  --moe-backend auto \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,12,16,24,32],"max_cudagraph_capture_size":32}' \
  --seed 0 \
  --distributed-executor-backend mp
```

### 客户端负载

客户端固定镜像同上，CPU48–51／NUMA3，无GPU；入口使用`vllm.benchmarks.serve`官方add_cli_args／main，流式`/v1/completions`。以下参数从各组正式第一轮`command.sh`的`--backend`起截取；完整Docker命令、亲和性包装及GovReport固定选样hook保存在原始产物。GovReport四轮实际tokenizer请求清单完全一致，顺序为同一前128条；未对prompt追加chat模板、未shuffle／oversample。两组均显式temperature=0、seed=0、ignore_eos、request_rate=inf、num_warmups=0（预热由外层协议单独执行）。

random：

```text
--backend vllm
--base-url http://127.0.0.1:31249
--endpoint /v1/completions
--model /model
--tokenizer /model
--served-model-name deepseek-v4-flash
--num-prompts 128
--max-concurrency 32
--request-rate inf
--num-warmups 0
--seed 0
--temperature 0
--percentile-metrics ttft,tpot,itl,e2el
--metric-percentiles 50,90,95,99
--save-result
--result-dir /results
--result-filename raw.json
--dataset-name random
--random-input-len 8192
--random-output-len 1024
--random-range-ratio 0
--ignore-eos
--trust-remote-code
```

govreport：

```text
--backend vllm
--base-url http://127.0.0.1:31249
--endpoint /v1/completions
--model /model
--tokenizer /model
--served-model-name deepseek-v4-flash
--num-prompts 128
--max-concurrency 32
--request-rate inf
--num-warmups 0
--seed 0
--temperature 0
--percentile-metrics ttft,tpot,itl,e2el
--metric-percentiles 50,90,95,99
--save-result
--result-dir /results
--result-filename raw.json
--dataset-name custom
--dataset-path /dataset.jsonl
--custom-output-len 1024
--skip-chat-template
--disable-shuffle
--no-oversample
--ignore-eos
--trust-remote-code
```

## 本机原始证据与交付

原始产物仅在47本机可用，绝对根目录为`/home/enhui/llm-serving-benchmarks/experiments/dspark-protocol-node47/`：

- `results/random-quick-01/`、`results/govreport-quick-01/`：run／plan、解析配置、环境、实际命令、probes、完整server.log、kernel检查、所有轮次raw／metrics／compilation／binding、protocol记录。
- `reports/random-observation/`、`reports/govreport-observation/`：CPU/GPU/cgroup遥测、KV与进程观察、NUMA快照、启动日志、运行前后源码／配置哈希及观察器计时。
- `reports/cache-*-manifest.json`、`cache-preparation.json`：缓存内容／权限、起点与复制时间；快照及副本留在`/home/enhui/.cache/serving-bench/dspark-protocol-node47/20260920/`。
- `reports/*-validate.json`、`*-plan.json`、`*-preflight.json`、`*-bench-report.md`、`analysis.json`及本地分析脚本：校验、统计与复现证据。`.json`后缀的CLI输出按工具原格式保存，preflight为文本。

两次服务及客户端均由runner按owner清理，缓存准备容器自动移除；最终docker ps为空、全部GPU显存0 MiB。只提交本报告、统一CSV及README中47自己的完成行，不提交探索配置、缓存或大日志，不push。提交前从CSV重算本报告全部统计，核对统一字段、原始轮次、时间戳／单位／相对链接，并通过validate、plan和`git diff --check`；本轮未修改公共代码，未追加性能实验。
