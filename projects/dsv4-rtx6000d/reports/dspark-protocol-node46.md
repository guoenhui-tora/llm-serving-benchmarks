# 节点46：DSpark off／K5 的完整负载预热验证

**2026-09-20（北京时间），本机单实例TP4、C32、GovReport近8K输入／1024输出，按固定quick完成off→K5两次独立启动。off正式三轮为663.40 ± 0.51，K5为741.52 ± 116.34 output tok/s；以本机off为分母，平均提高11.78%，但K5吞吐CV达15.69%，不能称为稳定复现此前约21%的收益。** 两个case和quick协议均PASS；K5第二轮有40条已知事件，按quick完整接纳，没有剔除慢轮或补测clean。

一轮128请求预热后，off三轮接近；K5第一轮无事件，第二轮再次编译、吞吐降至607.21 tok/s，第三轮恢复。**一次完整预热并不保证后续无JIT，而且慢轮不限于第一轮。** 本轮只验证既定预热方案，不证明增加一轮就能覆盖所有形状。结论限定于本机这两次启动、当前缓存历程和后台负载；没有扩展K值、并发或拓扑。

## 同机正式结果

每组n=3，均值±样本标准差，CV=样本标准差/均值。所有正式轮均纳入；逐轮完整原始精度见[CSV](../data/dspark-protocol-node46.csv)。P95列是三次各自P95的均值，**不是合并请求后的P95**。预热不进入这些统计。

| 指标 | off，均值±标准差 | CV | K5，均值±标准差 | CV |
| --- | ---: | ---: | ---: | ---: |
| 输出 tok/s | 663.40 ± 0.51 | 0.08% | 741.52 ± 116.34 | 15.69% |
| 请求 requests/s | 0.647853 ± 0.000495 | 0.08% | 0.724142 ± 0.113614 | 15.69% |
| 客户端benchmark计时 s | 197.576 ± 0.151 | 0.08% | 180.009 ± 31.052 | 17.25% |
| Mean TTFT s | 5.6477 ± 0.0036 | 0.06% | 4.5737 ± 0.4754 | 10.39% |
| P95 TTFT s | 19.6194 ± 0.0158 | 0.08% | 20.3999 ± 1.2093 | 5.93% |
| Mean TPOT ms | 42.6933 ± 0.0380 | 0.09% | 38.3987 ± 6.8572 | 17.86% |
| P95 TPOT ms | 45.9915 ± 0.0595 | 0.13% | 56.2730 ± 11.7743 | 20.92% |
| Mean ITL ms | 42.6933 ± 0.0380 | 0.09% | 131.3289 ± 21.0939 | 16.06% |
| P95 ITL ms | 26.9499 ± 0.0935 | 0.35% | 757.7596 ± 0.5396 | 0.07% |
| Mean端到端 s | 49.3229 ± 0.0373 | 0.08% | 43.8556 ± 7.4785 | 17.05% |
| P95端到端 s | 64.8368 ± 0.0368 | 0.06% | 68.0269 ± 14.5850 | 21.44% |
| 草稿接受率 % | — | — | 48.6426 ± 1.1411 | 2.35% |
| 平均接受长度（含目标补充token） | — | — | 3.4321 ± 0.0571 | 1.66% |

K5平均TTFT降低19.02%、平均TPOT降低10.06%；P95 TTFT增加3.98%、P95 TPOT增加22.36%，不能描述为全部延迟改善。DSpark成组返回token，ITL衡量流式事件间隔，不能等同逐模型token耗时；其较长的尾间隔仍是本轮观测结果。接受率不替代模型质量评测。

off的吞吐、Mean TTFT、Mean TPOT相对极差分别为0.153%、0.126%、0.177%，均低于任务约定的3%，可描述为本组三次接近；K5分别为27.47%、19.82%、31.17%，均不接近。这是quick实验，没有运行或宣称通过stable。K5均值收益落在自身较大波动范围内，不能据此确认稳定提升11.78%。无事件两轮虽均约806–811 tok/s，也不能删去第二轮后另报为正式三轮收益。

## 全部轮次与实际工作量

| 配置 | 阶段/轮次 | 接纳 | 已知事件条数 | benchmark s | 输出 tok/s | Mean TTFT s | Mean TPOT ms | 草稿接受率 % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tp4-off | warmup/1 | false | 52 | 251.404 | 521.3597 | 9.3628 | 52.2166 | — |
| tp4-off | measurement/1 | true | 0 | 197.728 | 662.8903 | 5.6445 | 42.7335 | — |
| tp4-off | measurement/2 | true | 0 | 197.573 | 663.4110 | 5.6516 | 42.6884 | — |
| tp4-off | measurement/3 | true | 0 | 197.426 | 663.9041 | 5.6471 | 42.6580 | — |
| tp4-k5 | warmup/1 | false | 344 | 520.682 | 251.7313 | 12.3364 | 112.9826 | 48.2837 |
| tp4-k5 | measurement/1 | true | 0 | 162.532 | 806.4359 | 4.2031 | 34.5343 | 49.2804 |
| tp4-k5 | measurement/2 | true | 40 | 215.861 | 607.2067 | 5.1097 | 46.3160 | 47.3252 |
| tp4-k5 | measurement/3 | true | 0 | 161.634 | 810.9202 | 4.4084 | 34.3457 | 49.3222 |

每轮均128成功、0失败，实际输入1,042,149、输出131,072 tokens。每组成功数及输入/输出总量的逐轮标准差和CV均为0；失败数恒为0，CV因分母为0不定义。八轮合计**1,024成功、0失败，输入8,337,192、输出1,048,576 tokens**；其中六个正式轮为768成功、输入6,252,894、输出786,432 tokens。短功能探测不计入上述benchmark工作量。

CSV严格使用任务规定的30列：含两轮预热和全部六个正式轮，`accepted=true`仅用于case和protocol均PASS的正式轮；quick接纳允许已知事件。`duration_s`只取客户端benchmark计时，`started_at/finished_at`为含客户端初始化及检查的阶段窗口，ISO8601时区为UTC `+00:00`。接受率单位为百分数；off的草稿字段为空。没有使用仅导出接纳样本的旧导出脚本。

## 独立启动与协议成本

北京时间off运行于00:29:16–00:47:37，K5运行于00:49:06–01:16:40。每个campaign只执行一次`bench run`，同配置全部轮次使用同一个服务；两组之间审阅日志并确认GPU释放。没有失败重试、额外预热或额外正式轮。

| 时间边界（秒） | off | K5 |
| --- | ---: | ---: |
| 独立preflight准备（不计入run） | 44.634 | 44.159 |
| run开始至服务启动命令，含run内preflight | 44.511 | 44.643 |
| 服务启动命令至首次health成功 | 107.863 | 440.092 |
| 首次health成功至功能检查响应落盘 | 0.356 | 4.920 |
| 协议墙钟，含客户端初始化/测量/检查 | 946.675 | 1163.012 |
| 预热benchmark计时 | 251.404 | 520.682 |
| 三轮正式benchmark计时之和 | 592.727 | 540.027 |
| 协议完成至case结束：收尾取证及容器清理 | 1.440 | 1.607 |
| run总墙钟 | 1100.854 | 1654.280 |

启动边界来自`run.log`的Starting时间与Docker完整日志的首次`GET /health`成功；功能结束取`chat-response.json`写入时间，包含响应持久化，非引擎内部计时。Docker容器StartedAt至health分别107.772/439.998秒，runner就绪等待分别107.406/439.617秒，口径差来自启动调用边界。功能完成到协议开始不足0.006秒。

协议完成后收尾取证与清理分别1.440/1.607秒；**纯Docker删除耗时没有独立计时，不将这段全部当成删除耗时**。run总计包括内部preflight，不含此前validate/plan/preflight、补丁准备和人工取证间隔。两次run合计45.92分钟；协议阶段分别15.78/19.38分钟，均低于各自45分钟上限。K5正式benchmark计时之和较短，但首次编译/预热和启动更贵，因此本轮run总成本更高。

两边均固定1轮完整128请求预热＋3轮128请求测量，每组512请求、C32；每客户端阶段超时900秒、协议总预算2700秒、服务就绪上限1800秒。K5预热benchmark为520.682秒，包含客户端初始化/检查的整阶段也未触及900秒。46无需缓存克隆，本次未执行缓存复制；K5只准备文档规定的加载覆盖。缓存起点不同，不能据此量化DSpark本身的等缓存启动开销。

## 事件、后端与资源限制

- 两组内核检查均PASS，required匹配齐全，未见ERROR、Traceback、OOM、CUDA错误或草稿scale不匹配警告；功能检查的模型列表和短中文回答均PASS。保留所有warning，不修改gate。
- 日志确认V2 runner、async scheduling、主模型`FLASHINFER_MLA_SPARSE_DSV4`/`FLASHINFER_CUTLASS`、NVFP4专家及实际`fp8_ds_mla` KV，FlashInfer autotune和prefix cache关闭。off的Graph上限32，K5保留尺寸`[5,6,10,12,20,24,40,48,60,72,80,96,120,144,160,192]`、上限192；目标验证192/草稿160 tokens覆盖配置与Graph捕获日志均确认，未测逐步replay命中率。
- K5四个worker均输出`LOCAL_DSPARK_FORMAT_FIX: validated 768 native MXFP4 draft experts`；草稿使用`DEEPGEMM_MXFP4`，主模型NVFP4与草稿FP8 linear配置保留。K5为vLLM 0.29.0＋本地加载补丁，不能写成未修改的原生镜像功能已解决。
- off预热52条事件、正式0/0/0；K5预热344条、正式0/40/0。K5第二轮的40条含16条JIT监测消息及12条TileLang编译开始/12条完成，涉及`mhc_pre_big_fuse_broadcast_with_norm_tilelang`、`mhc_pre_big_fuse_with_norm_tilelang`等形状。该轮内部编译时间位于该轮内；部分日志输出晚于内部时间，仍保守保留原计数。条数不是独立编译次数，不跨worker去重或修改归属。
- 保留SM120 SymmMem不支持、world_size=4 FlashInfer All Reduce禁用、超过两张PCIe GPU的custom allreduce禁用、NVFP4实验格式、indexer参数弃用、EXT4不自动prefetch、TileLang串行化fallback和模型generation_config warning。实际客户端仍显式temperature=0。kernel-check warning匹配计数off/K5为48/196，包含重复和预期disabled消息，不能当作独立故障数量。

正式三轮阶段窗口内，CPU `/proc/stat` 与GPU约5秒采样，进程列表和`/metrics`约15秒采样，systemd服务cgroup计数约30秒采样。CPU值是集合内逻辑CPU平均忙碌率，包含准备/检查和后台活动，不是本次服务的专属利用率。

| CPU集合 | off平均忙碌率 | K5平均忙碌率 |
| --- | ---: | ---: |
| 整机128逻辑CPU | 13.31% | 13.15% |
| 服务32–47 | 40.57% | 32.96% |
| 服务SMT兄弟96–111 | 0.88% | 2.98% |
| 客户端48–51 | 3.70% | 23.15% |
| 客户端SMT兄弟112–115 | 50.06% | 29.43% |

DOCA进程采样始终12个，`roce-init.service`计数增量约占6个逻辑CPU；该cgroup不包含全部后台负载，不能把6当作全部DOCA开销。未观察到新增矿工等高占用进程，但进程低频快照不能排除瞬时任务或解释所有线程。

**CPU集合负载分布并不完全一致。** K5三轮客户端平均忙碌率分别23.85%/22.28%/23.56%，对应SMT兄弟30.18%/32.24%/25.13%；off分别约3.49%–4.11%与50.0%。K5慢轮服务SMT兄弟平均4.59%，第一/三轮为0.16%/3.67%。不能把集合变化全部归因于客户端或DOCA，也不能仅凭DOCA进程数量恒定声称资源等价；没有逐线程连续归因证据。已知编译与慢轮同现，但不是排除其他影响的因果隔离实验。

| 正式阶段资源采样 | off | K5 |
| --- | ---: | ---: |
| GPU4–7功耗范围 | 79.87–298.75 W | 79.86–325.87 W |
| GPU4–7温度范围 | 48–60 °C | 47–60 °C |
| GPU4–7 SM频率范围 | 2415–2430 MHz | 2415–2430 MHz |
| GPU4–7显存占用范围 | 78975–79039 MiB | 78697–78761 MiB |
| 每卡KV预算 / 逻辑token容量 | 32.73 GiB / 85,450 | 28.47 GiB / 74,317 |
| KV使用采样峰值 | 24.10% | 27.87% |
| 在途请求采样峰值 / 抢占计数 | 32 / 0 | 32 / 0 |

GPU温度最高60°C、SM频率2415–2430MHz，未见明显持续热降频；采样含轮间空闲/编译等待，功耗范围不能视为满载功耗比较。低频采样不能排除短时降频或CPU干扰。绑定证据检查通过；cpuset只限制本次进程，不独占CPU，也不证明所有内存页均本地驻留。本次未修改宿主功耗、频率、SMT或NUMA设置。

## 固定身份、缓存与复现

节点`gpu-6000d-46`、地址`10.90.1.46`，只在本机操作，无SSH或跨节点操作。先`git pull --ff-only`，执行代码commit为`fa18d8079c68fc2976fa32690a267ca82834581d`，初始tracked工作区干净；两次runner源码指纹均为`f6e873002b8cf61963bd221d63b0fe0a520e0bc6118f4ca63acfa54fc6726d52`。冻结的49份源码/配置/观测脚本哈希在两次运行后复核未变，运行期间没有pull或修改它们。

服务与客户端镜像固定`vllm/vllm-openai:v0.29.0`，image ID及RepoDigest SHA256均为`c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`，`--pull never`。权重目录`/data/models/DeepSeek-V4-Flash-0731-NVFP4`，两组模型身份`1e1b7e54442bcf7f15ef55192641b5c5da360bcaa608236aa0d06b273aedf8b4`一致；元数据/tokenizer哈希和48个分片大小已保存，未全量重算权重内容SHA256。

| 模型文件 | SHA256 |
| --- | --- |
| `config.json` | `bb0d2286d6761439e41d3cef31d16489411b816ed8688922f59730bbd5567cdb` |
| `model.safetensors.index.json` | `5d2ad3076e04081d6c0728cb4b004dc832850ec5ae732f3adb06cf87c4b437a5` |
| `tokenizer_config.json` | `6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547` |
| `tokenizer.json` | `8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf` |

硬件为RTX6000D/SM120，每卡85,651 MiB、驱动580.159.04，GPU4–7；服务CPU32–47/NUMA2，客户端CPU48–51/NUMA3，SMT兄弟为各CPU编号+64。配置固定TP4/PP1/DP1、EP off、上下文16384、max-num-seqs=32、prefill8192、显存比例0.90、FP8 E4M3 KV、block256、V2＋async、autotune/prefix cache off。K5/greedy/standard rejection/adaptive verification=false。

负载为[固定GovReport JSONL](../data/govreport-near8k.jsonl)，SHA256 `33bdbd36a8b335300974e1b07aa66b2c24c88cc4c4b55a21350aa3564a945d53`；每轮实际tokenizer核验前128条且顺序固定，不使用全部256条。流式`/v1/completions`，C32、128请求、1024输出、seed0、temperature0、ignore_eos=true、request_rate=inf，客户端内部num-warmups=0，预热由外层quick完成。

从公共configs复制到本机工作区后，只有`targets/rear.yaml`的address与描述身份ID由48改为46，recipe/workload不变，cache_root仍为`~/.cache/serving-bench`。根据完整recipe与固定镜像解析出的缓存：

| 配置 | 本机持久缓存 | 启动前状态 |
| --- | --- | --- |
| off | `/home/enhui/.cache/serving-bench/vllm/d19496fcae0f32e092d6` | 318文件，11,533,873字节；vllm169、tilelang80、flashinfer69文件，没有triton子目录 |
| K5 | `/home/enhui/.cache/serving-bench/vllm/2b1375ee0b032aecf642` | 原目录不存在；仅先用prepare.py放置四个加载覆盖文件，运行中生成其余缓存 |

缓存起点清单时间为2026-09-20 00:26:26 +08:00，包含文件相对路径、大小、权限及mtime，源目录和新增缓存均保留。46不要求相同缓存副本，未清空缓存，也未把K5新缓存伪装成与off一致。两组显式设置`TRITON_CACHE_DIR=/root/.cache/triton`并持久挂载；运行后两目录均有Triton产物，不据此证明全部kernel可跨启动复用。补丁准备仅传入K5 campaign，manifest校验通过。

复用入口：[off quick](../configs/campaigns/dspark-off-quick.yaml)、[K5 quick](../configs/campaigns/dspark-k5-quick.yaml)、[完整负载workload](../configs/workloads/govreport-c32-quick-full.yaml)、[补丁准备说明](dspark-compatibility.md#后续实验如何使用)。公共target仍保留原地址；在新的本地工作区改为实际本机地址。两组均完成独立validate/plan/preflight，run再次preflight并执行health/models/短中文探测。

本次实际campaign调用（容器及完整参数由实测快照导出如下）：

```bash
./bench run experiments/dspark-protocol-node46/configs/campaigns/dspark-off-quick.yaml \
  --run-root experiments/dspark-protocol-node46/results/off-quick-01
./bench run experiments/dspark-protocol-node46/configs/campaigns/dspark-k5-quick.yaml \
  --run-root experiments/dspark-protocol-node46/results/k5-quick-01
```

精确原始调用使用相同工作区的绝对路径；上述相对形式从仓库根执行等价。复跑须新建run-root，不能覆盖本批。以下服务命令逐参数来自各case的`argv.json`，保留本次容器名/缓存路径，非手工维护的第二份recipe。

<details><summary>tp4-off 完整实际服务启动命令</summary>

```bash
docker run \
  -d \
  --pull never \
  --name sb-4d943eabb3c84c99-server \
  --label io.serving-bench.run=4d943eabb3c84c99 \
  --network host \
  --ipc host \
  --gpus '"device=4,5,6,7"' \
  -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro \
  -v /home/enhui/.cache/serving-bench/vllm/d19496fcae0f32e092d6:/root/.cache:rw \
  --cpuset-cpus 32-47 \
  --cpuset-mems 2 \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --cap-add SYS_NICE \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_DEBUG=WARN \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e TILELANG_CACHE_DIR=/root/.cache/tilelang \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
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

</details>

<details><summary>tp4-k5 完整实际服务启动命令</summary>

```bash
docker run \
  -d \
  --pull never \
  --name sb-b03455cdec8e41a2-server \
  --label io.serving-bench.run=b03455cdec8e41a2 \
  --network host \
  --ipc host \
  --gpus '"device=4,5,6,7"' \
  -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro \
  -v /home/enhui/.cache/serving-bench/vllm/2b1375ee0b032aecf642:/root/.cache:rw \
  --cpuset-cpus 32-47 \
  --cpuset-mems 2 \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --cap-add SYS_NICE \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_DEBUG=WARN \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e TILELANG_CACHE_DIR=/root/.cache/tilelang \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e PYTHONPATH=/root/.cache/dspark-native-mxfp4 \
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
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[5,6,10,12,20,24,40,48,60,72,80,96,120,144,160,192],"max_cudagraph_capture_size":192}' \
  --seed 0 \
  --distributed-executor-backend mp \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":false}'
```

</details>

全部原始产物**仅本机可用，不随Git分发**：`experiments/dspark-protocol-node46/results/{off-quick-01,k5-quick-01}/`保存run/case/解析配置、实际服务/客户端命令、镜像模型信息、绑定、探测、完整服务日志和每轮raw/metrics/compilation/dataset-manifest；`reports/`保存独立validate/plan/preflight、准备环境、缓存起点、冻结哈希、两次资源时间线、执行墙钟及统计脚本/summary。八轮数据集manifest均经过实际tokenizer与总量检查，未混用历史随机负载。

结束确认两组本次所属容器均已清理，GPU计算进程为空，原模型、镜像、缓存和历史结果保留。归档只新增本报告与CSV，并更新README中46自己的完成行；不改公共源码/配置，不push。全部性能数字从本次CSV重算，公共入口重新validate/plan，Markdown相对链接、30列CSV、接受标志、时间戳/单位及`git diff --check`均在提交前核验。
