# 48节点：完整负载预热验证（2026-09-20）

**已完成计划中的一次 off quick，campaign、case、协议与启动日志检查均 PASS。** 单实例 TP4、C32、GovReport 前128条、输出1024 tokens，完整负载预热一轮后，三轮正式吞吐为 **664.41 ± 0.32 output tok/s（样本标准差，CV 0.048%）**，没有复现明显慢首轮。正式三轮的吞吐、Mean TTFT、Mean TPOT相对极差分别为 **0.096%、0.117%、0.115%**，均低于预先约定的3%观察阈值，可描述为本组三次接近；本轮执行的是quick，不能改称通过stable，也不能证明任意缓存起点下只需一轮预热。

范围严格限定gpu-6000d-48（10.90.1.48）、GPU4–7和此次持久缓存／DOCA后台负载。仅一次服务启动、1轮128请求预热＋3轮128请求正式测量，没有追加clean、重启补测或访问其他节点。历史clean仅作旁证，不是同批配对。

## 逐轮结果与同机历史参照

完整精度及全部字段见[逐轮CSV](../data/dspark-protocol-node48.csv)。预热不进入正式统计；三轮正式数据全部接纳。时间戳保存UTC时区，本机运行时间为北京时间 **2026-09-20 00:28:16–00:45:38**。

| 阶段 | output tok/s | Mean TTFT ms | Mean TPOT ms | 客户端计时 s | 阶段墙钟 s | 已知事件行数 | 接纳 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 预热1 | 661.19 | 5753.45 | 42.752 | 198.236 | 223.929 | 28 | 否 |
| 正式1 | 664.46 | 5661.09 | 42.604 | 197.262 | 225.172 | 0 | 是 |
| 正式2 | 664.08 | 5654.48 | 42.637 | 197.375 | 224.511 | 0 | 是 |
| 正式3 | 664.71 | 5657.48 | 42.588 | 197.186 | 225.166 | 0 | 是 |

每轮128成功、0失败，输入 **1,042,149**、输出 **131,072 tokens**；4轮合计 **512成功、0失败，输入4,168,596、输出524,288 tokens**。正式测量合计384请求、输入3,126,447、输出393,216 tokens。完整服务日志恰有512条completions HTTP 200；另有计时外中文功能探测，回答“2”，输入17、输出2 tokens，不混入压测总量。

下表仅统计正式三轮，n=3，标准差为样本标准差；P95各列的均值是三次分位数的均值，**不是合并全部请求后的P95**。延迟统一为ms。

| 指标 | 第1轮 | 第2轮 | 第3轮 | 均值 ± 标准差 | CV |
| --- | ---: | ---: | ---: | ---: | ---: |
| 输出 tok/s | 664.455 | 664.077 | 664.712 | 664.415 ± 0.320 | 0.048% |
| 请求/s | 0.648882 | 0.648512 | 0.649133 | 0.648842 ± 0.000312 | 0.048% |
| 客户端计时 s | 197.262 | 197.375 | 197.186 | 197.274 ± 0.095 | 0.048% |
| Mean TTFT ms | 5661.095 | 5654.479 | 5657.479 | 5657.684 ± 3.313 | 0.059% |
| P95 TTFT ms | 19665.018 | 19644.928 | 19663.090 | 19657.679 ± 11.084 | 0.056% |
| Mean TPOT ms | 42.604 | 42.637 | 42.588 | 42.610 ± 0.025 | 0.059% |
| P95 TPOT ms | 45.926 | 45.956 | 45.930 | 45.937 ± 0.016 | 0.035% |
| Mean ITL ms | 42.604 | 42.637 | 42.588 | 42.610 ± 0.025 | 0.059% |
| P95 ITL ms | 26.874 | 26.996 | 26.966 | 26.945 ± 0.063 | 0.235% |
| Mean E2EL ms | 49244.805 | 49272.526 | 49225.239 | 49247.523 ± 23.760 | 0.048% |
| P95 E2EL ms | 64836.679 | 64836.518 | 64847.703 | 64840.300 ± 6.412 | 0.010% |

三轮成功数、输入／输出总量分别恒为128、1,042,149／131,072，标准差与CV均为0；失败数恒为0（标准差0、CV不定义）。DSpark关闭，因此接受率和接受长度字段留空。

| 同机参照 | 输出 tok/s | Mean TTFT ms | Mean TPOT ms |
| --- | ---: | ---: | ---: |
| 本轮quick，正式n=3 | 664.41 | 5657.68 | 42.610 |
| 2026-09-19历史clean，n=3 | 663.92 | 5583.45 | 42.718 |
| 本轮均值相对历史 | +0.074% | +1.330% | −0.253% |

历史数据来自[上轮报告](dspark-tp4-20260919.md)和[原始精度CSV](../data/dspark-tp4-20260919.csv)的off-paired／jit_clean三轮。三个均值差异均在3%内，但历史clean在同一服务先quick后取得，缓存历程也不同；本次不能独立归因于“完整负载预热优于两轮64请求”，更不能据此证明跨启动成本稳定。

## 时间成本与预算

| 阶段 | 秒 | 口径 |
| --- | ---: | --- |
| run内预检至启动命令 | 45.643 | run开始到Starting日志 |
| 服务启动命令至health | 95.713 | Starting日志到服务health 200日志 |
| health至功能探测完成 | 0.359 | health日志到chat-response.json落盘，近似值 |
| quick协议墙钟 | 898.784 | 含客户端初始化、测量、绑定及日志检查 |
| 其中预热benchmark | 198.236 | 客户端计时 |
| 其中3轮正式benchmark合计 | 591.823 | 客户端计时 |
| 取证与清理尾段 | 1.464 | 协议结束至case结束，含日志取证和容器清理 |
| run总耗时 | 1041.969 | 17.366分钟，包含run内预检 |

四轮benchmark合计 **790.059秒**，协议墙钟 **14.980分钟**，其余约 **108.725秒**为客户端初始化与检查等。Docker StartedAt到health为95.624秒，runner的wait_ready计时为95.261秒；起点不同，不混用。功能完成用产物mtime近似，清理尾段没有单独的docker删除计时。

执行前单独的validate／plan／preflight和缓存清单准备不计入run总耗时；本节点未克隆缓存，无缓存复制成本。协议预算2700秒、每客户端阶段900秒；固定1＋3轮、不重试，均在预算内完成。服务ready超时配置1800秒，实际约96秒。

## 日志、资源与验收边界

启动检查确认DeepseekV4ForCausalLM、FLASHINFER_MLA_SPARSE_DSV4、FLASHINFER_CUTLASS、expert_dtype=fp4、fp8_ds_mla、V2 Model Runner、async、关闭FlashInfer autotune及mixed tokens=16预热。Graph为FULL_DECODE_ONLY，尺寸[1,2,4,8,12,16,24,32]；prefix cache关闭，DSpark／EP关闭。实际KV预算每卡32.73 GiB，日志逻辑容量85,450 tokens。Docker inspect及每轮服务／客户端线程允许集合检查通过。

预热匹配28条已知事件（20条TileLang、8条Triton），正式轮为0／0／0。这里统计的是日志匹配行数，不能当作28次独立CPU重编译，也不能排除首次进程内缓存加载。未改变gate或事件正则。完整日志检查PASS、无缺失必需项和禁止项，无OOM、Traceback、请求错误或抢占。共保留48条warning匹配，包含JIT观察、SM120 SymmMem不可用、world_size=4的FlashInfer All Reduce禁用、超过两张PCIe GPU的custom allreduce禁用、NVFP4格式实验性和FP4 indexer参数弃用；FP8 KV日志仍提示scale不足可能损害精度。模型generation_config默认temperature=1的warning保留，实际客户端显式temperature=0。没有进行数值质量评测。

资源每约10秒采样一次，KV每约2秒采样；下表按各阶段内首末资源样本的CPU计数增量计算，含客户端准备与检查，不等于纯benchmark区间。每阶段22个样本、覆盖约216秒；集合忙碌率包括本次进程和背景，不能解释为本次进程独占率。

| 阶段 | 整机忙碌率 | 服务32–47 | SMT96–111 | 客户端48–51 | SMT112–115 | DOCA逻辑CPU当量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 预热1 | 13.26% | 30.27% | 4.32% | 20.45% | 10.56% | 12.00 |
| 正式1 | 13.28% | 30.40% | 0.73% | 31.13% | 0.20% | 12.00 |
| 正式2 | 13.31% | 30.60% | 0.19% | 21.88% | 0.14% | 12.00 |
| 正式3 | 13.34% | 30.67% | 0.20% | 19.14% | 3.54% | 12.00 |

逐进程CPU增量显示12个doca_spcx_cc持续各占约1个逻辑CPU；收尾cgroup映射为roce-init.service内6个、两个user session内共6个，允许CPU0–127。服务cgroup采样的roce-init约6 CPU与此吻合，不能只看system.slice误记成全部只有6 CPU。未见新增矿工式大额背景负载，但客户端与SMT集合忙碌率确有变化，cpuset不能排除竞争，不称为无干扰运行。进程首末快照可能遗漏在窗口内新建又退出的进程。

正式阶段GPU4–7的采样温度48–61°C、SM频率2415–2430 MHz、单卡功耗81.06–298.28 W，包含轮次准备／空闲区间；显存78,975–79,039 MiB。未见持续热降频证据，低频采样不能排除瞬时影响。正式轮KV峰值约24.10%，运行请求最高32、排队最高29，累计preemptions始终0；未改宿主功耗、锁频或NUMA设置。结束后本次容器全部清理，GPU恢复空闲，模型、镜像、缓存和原始结果保留。

## 固定身份与复现入口

- Git：`fa18d8079c68fc2976fa32690a267ca82834581d`，先执行git pull --ff-only并确认包含fa18d80；启动前Git无本地差异，结束复核源码／配置哈希全部一致。
- runner源码指纹：`f6e873002b8cf61963bd221d63b0fe0a520e0bc6118f4ca63acfa54fc6726d52`。
- 服务／客户端：`vllm/vllm-openai:v0.29.0`，固定image ID `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`；不拉取镜像。
- 模型：`/data/models/DeepSeek-V4-Flash-0731-NVFP4`；元数据／tokenizer哈希与分片大小身份 `1e1b7e54442bcf7f15ef55192641b5c5da360bcaa608236aa0d06b273aedf8b4`，未全量哈希权重内容。
- tokenizer.json SHA256：`8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf`。
- 数据SHA256：`33bdbd36a8b335300974e1b07aa66b2c24c88cc4c4b55a21350aa3564a945d53`；[GovReport数据](../data/govreport-near8k.jsonl)前128条，顺序固定，不用全部256条。每轮客户端实际tokenizer计数通过。
- 缓存沿用 `/home/enhui/.cache/serving-bench/vllm/d19496fcae0f32e092d6`，由当前recipe与固定image ID计算；启动前1,167文件、25,734,657字节，含目录共1,407项，已保存相对路径／SHA256／权限／所有者／mtime清单。用无GPU、无网络容器只读挂载核对root所有文件，未漏读、未清空或改权限。没有将缓存快照冒称冷启动。
- [公共campaign](../configs/campaigns/dspark-off-quick.yaml)、[recipe](../configs/recipes/tp4.yaml)、[workload](../configs/workloads/govreport-c32-quick-full.yaml)、[target](../configs/targets/rear.yaml)复制到新工作区；48地址、target、cache_root和recipe均未修改。本轮是该完整预热配置在48的实际测量，不代表其余节点已验证。

服务TP4/PP1/DP1，GPU4–7，服务CPU32–47／NUMA2；客户端CPU48–51／NUMA3，SMT兄弟分别96–111／112–115。上下文16384、活动容量32、prefill8192、显存比例0.90、FP8 E4M3 KV。客户端流式 `/v1/completions`，C32、128请求、seed0、temperature0、ignore_eos=true、request_rate=inf；GovReport变长输入、固定输出1024。

实际执行入口（在仓库根目录；重跑使用新的run-root）：

```bash
./bench validate experiments/dspark-protocol-node48/configs/campaigns/dspark-off-quick.yaml
./bench plan experiments/dspark-protocol-node48/configs/campaigns/dspark-off-quick.yaml
python3 scripts/prepare_logging.py experiments/dspark-protocol-node48/configs/campaigns/dspark-off-quick.yaml
./bench preflight experiments/dspark-protocol-node48/configs/campaigns/dspark-off-quick.yaml
./bench run experiments/dspark-protocol-node48/configs/campaigns/dspark-off-quick.yaml \
  --run-root experiments/dspark-protocol-node48/results/off-quick-01
```

上述run由本地tmux监控包装器启动，包装器只采集资源和调用未修改的bench。以下完整服务命令直接来自本次 `cases/tp4-off/command.sh`；它是实测快照，容器名用于追溯，复现应由plan重新生成。

<details><summary>完整实际服务命令</summary>

```bash
docker run -d --pull never --name sb-5adb64485a0d42fa-server --label io.serving-bench.run=5adb64485a0d42fa --network host --ipc host --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/vllm/d19496fcae0f32e092d6:/root/.cache:rw --cpuset-cpus 32-47 --cpuset-mems 2 --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --cap-add SYS_NICE -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model --served-model-name deepseek-v4-flash --host 127.0.0.1 --port 31249 --trust-remote-code --enable-auto-tool-choice --no-enable-prefix-caching --no-enable-expert-parallel --enable-chunked-prefill --jit-monitor-verbose --async-scheduling --tensor-parallel-size 4 --pipeline-parallel-size 1 --data-parallel-size 1 --max-model-len 16384 --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8_e4m3 --block-size 256 --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' --moe-backend auto --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' --kernel-config '{"enable_flashinfer_autotune":false}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,12,16,24,32],"max_cudagraph_capture_size":32}' --seed 0 --distributed-executor-backend mp
```

</details>

原始证据**仅本机可用**：`experiments/dspark-protocol-node48/`。`results/off-quick-01/`保存run／case身份、实际argv、解析配置、模型与镜像身份、启动与每轮完整日志、probe、亲和性、dataset manifest、raw／metrics／compilation／protocol JSON；`results/dspark-off-quick-launch/`保存资源与KV时间线、源码配置哈希和runner日志；`evidence/`保存离线plan、预检、缓存清单、拓扑及结束状态；`reports/analysis.json`保存从本CSV重算的统计和资源汇总。原始日志、探索配置、工具、缓存均未提交。

归档核验：CSV包含预热及全部正式轮4行、30字段，时间戳带时区，原始精度未截断；逐轮指标与原始metrics一致。仅新增本节点报告与CSV，并更新README中48自己的完成行，不修改其他节点状态。
