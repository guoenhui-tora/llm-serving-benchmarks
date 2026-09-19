# 节点47：DSpark K 值实验与 DP 草稿启动阻塞

**本节点部分完成：TP2×DP2、EP off、DSpark off 已取得三轮 clean，输出吞吐 710.39 ± 20.66 tok/s，CV 2.91%。K5 在启动内存 profile 的共同 DP／草稿路径触发 `AssertionError: 8192 80`，未进入功能探测或正式测量；按约定停止相关 K1–K4。** 没有获得 DSpark 的 K 值曲线、相对收益或接受统计，不能据此评价 DSpark 性能。off 的无事件结果仍有2.91%吞吐CV，不是稳定性验证。

范围为2026-09-20本机 `gpu-6000d-47`／`10.90.1.47`，仅GPU4–7、四卡一套服务。执行依据为[项目下一轮方案](../README.md#下一轮四节点-dspark-拓扑与-k-值实验)。全部正式轮次见[44字段 CSV](../data/dspark-k-sweep-node47.csv)，包括六轮拒绝和三轮接纳；没有把启动失败伪造成一轮测量。

## 完成情况与主要结果

| 配置 | 状态／接纳 | 输出 tok/s，均值±样本标准差 | 较本机同拓扑 off | Mean／P95 TTFT s | Mean／P95 TPOT ms | 草稿接受率／平均接受长度 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| tp2-dp2-epoff-off | PASS，3/9轮 | 710.39 ± 20.66 | 0%（自身参照） | 6.057／14.777 | 36.082／39.441 | 不适用 |
| tp2-dp2-epoff-k5 | FAIL，启动profile断言 | — | — | — | — | 未测得 |
| tp2-dp2-epoff-k1 | 未运行，共同路径阻塞 | — | — | — | — | 未测得 |
| tp2-dp2-epoff-k2 | 未运行，共同路径阻塞 | — | — | — | — | 未测得 |
| tp2-dp2-epoff-k3 | 未运行，共同路径阻塞 | — | — | — | — | 未测得 |
| tp2-dp2-epoff-k4 | 未运行，共同路径阻塞 | — | — | — | — | 未测得 |

核心配置均已完成工作区准备、validate、plan；K1–K4的“未运行”不是GPU实测失败。K6/K7只做配置与补丁准备。由于核心六配置未全部完成，最多两次的K6/K7或独立重启扩展均未触发；未追加其他拓扑或EP状态，也未重复off以挑选更低CV。

以下为off三轮接纳样本7、8、9的补充统计。所有P95汇总均为**三轮各自P95的平均和样本标准差**，不是合并请求的P95。

| 指标 | 三轮均值 ± 样本标准差 |
| --- | ---: |
| requests/s | 0.69374 ± 0.02018 |
| Mean TTFT s | 6.057 ± 0.422 |
| P95 TTFT s | 14.777 ± 0.636 |
| Mean TPOT ms | 36.082 ± 0.583 |
| P95 TPOT ms | 39.441 ± 0.759 |
| Mean ITL ms | 36.082 ± 0.583 |
| P95 ITL ms | 27.017 ± 0.060 |
| Mean端到端延迟 s | 42.969 ± 1.000 |
| P95端到端延迟 s | 51.535 ± 1.113 |

## 全部轮次与协议成本

统一 `jit_clean` 版本1：总C32、每轮128请求，累计最先三轮通过工作量和已知事件检查的结果；同配置全程一次服务启动，无独立预热、无JIT触发重启。最多12轮／2700秒，单轮900秒，启动1800秒。六小时排期上限从本地调度04:16:06起计；本次未触及排期上限，停止原因是共同路径故障。

| 轮次 | 北京时间，阶段起止 | 客户端计时 s | 输出 tok/s | 已知事件行数 | 接纳 | KV观测峰值 % |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| 1 | 04:23:12–04:30:02 | 384.713 | 340.701 | 98 | 否 | 23.394 |
| 2 | 04:30:02–04:35:06 | 277.321 | 472.637 | 48 | 否 | 23.443 |
| 3 | 04:35:06–04:38:58 | 205.776 | 636.965 | 26 | 否 | 23.366 |
| 4 | 04:38:58–04:42:53 | 207.750 | 630.914 | 16 | 否 | 23.381 |
| 5 | 04:42:53–04:46:45 | 205.193 | 638.773 | 24 | 否 | 23.384 |
| 6 | 04:46:45–04:51:20 | 248.567 | 527.311 | 30 | 否 | 23.428 |
| 7 | 04:51:20–04:54:51 | 185.150 | 707.922 | 0 | 是 | 23.422 |
| 8 | 04:54:51–04:58:27 | 189.663 | 691.079 | 0 | 是 | 23.391 |
| 9 | 04:58:27–05:01:52 | 179.016 | 732.182 | 0 | 是 | 23.378 |

九轮均128成功、0失败，每轮输入1,042,149／输出131,072 tokens；全部合计 **1,152成功、0失败，输入9,379,341／输出1,179,648 tokens**。接纳三轮合计384成功、输入3,126,447／输出393,216 tokens。前六轮共242条已知事件，事件行数不是独立编译次数；未放宽gate。阶段起止包含客户端初始化和日志检查，`duration_s`为客户端原始benchmark计时，二者不混用。

| run_id | 结果 | 北京时间run起止 | 就绪启动耗时 s | 协议总耗时 s | run总耗时 s |
| --- | --- | --- | ---: | ---: | ---: |
| core-off-01 | PASS，接纳7／8／9轮 | 04:16:49–05:01:53 | 336.521 | 2319.752 | 2704.097 |
| core-k5-02 | FAIL，未就绪 | 05:05:55–05:09:48 | — | 未开始 | 233.510 |

run总耗时含内部preflight和清理，不含外部配置、补丁准备与外部preflight。K5从05:06:39开始启动到05:09:48检测退出约189秒。此前 `core-k5-01` 外部preflight报告 `Port 31249 is busy`，未启动模型、未创建run；之后只读检查确认无监听、无容器且端口释放，以新编号重试一次。原失败文件保留；暂时不可绑定的确切原因未证实，不归咎于其他任务。

## K5共同路径阻塞证据

四个worker均输出补丁校验标记，backend日志确认以下草稿分派，但执行尚未通过：

```text
LOCAL_DSPARK_FORMAT_FIX: validated 768 native MXFP4 draft experts;
draft=Mxfp4MoEMethod group32 E8M0; target NVFP4 unchanged;
FP8 linear configuration retained.
Using 'DEEPGEMM_MXFP4' Mxfp4 MoE backend.
Using MoEPrepareAndFinalizeNaiveDPEPModular
Using DeepGemmFP4Experts
```

随后 `gpu_worker.determine_available_memory → model_runner.profile_run → _dummy_run → DFlashSpeculator.propose → DSparkSpeculator._generate_draft → _run_model → set_forward_context → DPMetadata.make` 触发：

```text
vllm/forward_context.py:96
assert num_tokens_across_dp_cpu[dp_rank] == batchsize
AssertionError: 8192 80
```

四个worker日志均记录该断言。固定镜像实际源码显示：profile使用 `max_num_batched_tokens=8192` 做目标dummy run；`num_reqs=min(num_tokens,max_num_reqs)=16`。DSpark在本权重的anchor模式下每请求query数为K，K5的草稿forward因此是16×5=80；DFlash继承路径在跳过attention的profile分支直接传入目标的 `dp_sync.num_tokens_across_dp`，未按草稿query数重建DP元数据。于是80与8192不一致，断言发生在草稿model forward和Graph捕获之前。

这解释了为什么不能把它当作K5显存不足或Graph上限太小。相同路径下K1–K4分别需要16、32、48、64，都不等于8192；这是源码推导，**没有将其写成四次实测失败**。按“共性DP／草稿路径故障停止其余相关K”的规则结束相关候选。未改8192预算、未禁用Graph、未去掉断言、未换精度或修改公共runner；已有MXFP4补丁只修加载，不修本次DP元数据问题。

K5未到 `/health` 就绪，models／中文探测与正式轮均未执行；未完成KV分配和Graph运行验证。最终kernel gate为FAIL，缺少禁用autotune和mixed tokens=16预热的必需日志，均保留。K1–K4后续应在单独修复并验证共同路径后重新测量，本报告不把配置准备称为通过。

## 绑定、资源与KV

实际Docker设备为 `[4,5,6,7]`。off日志有 `Worker_DP0_TP0/TP1` 与 `Worker_DP1_TP0/TP1`，四进程运行在本套后四卡；TP2近端分组按本机GPU拓扑为4–5／6–7，未涉及前四卡。单API端口31249，vLLM默认两个API worker，内部DP路由；并非runner启动两个独立服务。EP显式 `--no-enable-expert-parallel`，不把DP2描述为两份无通信的独立TP2。

服务CPU32–47／NUMA2，客户端CPU48–51／NUMA3，各自只使用每物理核一个线程；SMT兄弟分别96–111、112–115。off每轮Docker inspect及线程允许集合检查通过，GPU0–3全程无本次计算任务。NUMA页面快照显示worker有约79万–81万页在N2、约19.7万页在N3，另有少量N0/N1页面；允许集合不代表所有文件共享页或已有页都迁移到本地，不能宣称页面全部本地驻留。

| off接纳轮 | 整机CPU忙碌 % | 服务核 % | 服务SMT兄弟 % | 客户端核 % | 客户端SMT兄弟 % | GPU平均功耗 W/卡 | GPU温度范围 °C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 7 | 12.45 | 36.19 | 0.24 | 7.20 | 2.91 | 265.92 | 48–61 |
| 8 | 12.49 | 36.12 | 0.17 | 6.69 | 0.17 | 269.23 | 49–61 |
| 9 | 12.42 | 35.73 | 0.11 | 6.23 | 0.19 | 267.77 | 48–62 |

资源按每轮阶段窗口内完整5秒采样区间汇总，含客户端初始化和收尾空闲段，不等同GPU纯计算段。clean三轮GPU SM时钟观测2415–2430 MHz；未观察到明显持续热降频，但低频采样不能排除瞬时干扰。10个已知 `doca_spcx_cc` 进程合计约10.000个逻辑CPU持续忙碌；30秒进程CPU增量及5秒整机/逐核时间线未发现新增矿工类大幅背景负载。第7轮客户端SMT兄弟均值2.91%、局部采样最高约25.35%，高于8/9轮，不能把后台活动当作已完全隔离。未改变宿主锁频、功耗或NUMA设置。

off每个DP rank的等效KV容量均为78,506 tokens，每worker日志KV分配为30.07 GiB，DSV4压缩缓存不等同普通全注意力token容量。`vllm:kv_cache_usage_perc`在每轮按rank取观测最大值，再取两个rank的最大；全部九轮最高23.443%，三个clean轮分别23.422%、23.391%、23.378%。`vllm:num_preemptions_total`按rank检查无重置后取窗口前后增量并求和，九轮均0；整段计数器保持0。两rank各自运行请求峰值均为16，保存了总C32实际容量证据。K5没有成功分配KV，容量、峰值和抢占均记未测得。

warning原样保留：off最终457条、K5 24条，包括SM120 SymmMem／FlashInfer All Reduce限制、TileLang回退及编译、DP coordinator统计乱序、FP8 KV scale和弃用提示等；warning计数并非独立问题数量。clean仅排除规则识别到的编译事件，不替代资源审阅或质量评测。

## 接受统计口径与缺失项

本地观察脚本每2秒读取一次 `/metrics`，保存请求起止时间及完整原始文本；按轮次 `protocol.json` 的阶段起止关联。off各轮窗口内103–205份快照，首尾采样间隔均不足2秒；counter检查保留各rank和原始标签，KV峰值仅为采样观测值，不保证抓到瞬时峰值。辅助脚本不发送生成请求，不修改runner和gate。off功能探测先于正式阶段，数据集检查在客户端计时前完成。

实际镜像 `v1/spec_decode/metrics.py` 定义如下，标签为 `model_name="deepseek-v4-flash", engine="0"/"1"`，位置计数再带 `position`。后续成功的DSpark轮应按rank求增量后累加分子／分母，不累加TP worker，也不累加重复导出的相同标签序列：

- 提出草稿token：`vllm:spec_decode_num_draft_tokens_total`。
- 接受草稿token：`vllm:spec_decode_num_accepted_tokens_total`。
- 验证／draft次数：`vllm:spec_decode_num_drafts_total`。
- 草稿接受率：`100 × ΣΔaccepted_tokens / ΣΔdraft_tokens`。
- 平均接受长度：`1 + ΣΔaccepted_tokens / ΣΔnum_drafts`，按验证次数加权，**包含目标补充token**的约定。
- 逐位置：`vllm:spec_decode_num_accepted_tokens_per_pos_total`的增量次数，按位置排序；若转比例，分母为总验证次数。

counter缺失、重置或窗口无法可靠关联时不计算。**本次off无草稿指标，K5未就绪无成功metrics窗口，因此CSV所有草稿字段为空**，不以0、K或日志百分比代替测量。不能判断接受长度是否抵消草稿与验证开销。ITL始终是流式事件间隔；若DSpark成功，也不能解释为逐token计算时间。

## 固定身份与复现入口

初始工作区干净；`git pull --ff-only`从 `f25251e` 快进到 `397e9d42dedee00c181c5fb10f5686ace069c111`，`git merge-base --is-ancestor 397e9d4 HEAD`成功。运行期间未pull、未修改源码或配置。

| 身份 | 值 |
| --- | --- |
| 执行Git commit | `397e9d42dedee00c181c5fb10f5686ace069c111` |
| runner源码指纹，两次run相同 | `f6e873002b8cf61963bd221d63b0fe0a520e0bc6118f4ca63acfa54fc6726d52` |
| 服务和客户端镜像 | `vllm/vllm-openai:v0.29.0` |
| Image ID／RepoDigest SHA256 | `c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1` |
| 模型路径 | `/data/models/DeepSeek-V4-Flash-0731-NVFP4` |
| 模型元数据/tokenizer/分片大小身份 | `1e1b7e54442bcf7f15ef55192641b5c5da360bcaa608236aa0d06b273aedf8b4` |
| JSONL SHA256 | `33bdbd36a8b335300974e1b07aa66b2c24c88cc4c4b55a21350aa3564a945d53` |
| 补丁版本 | 本commit的[manifest.json](../patches/dspark-native-mxfp4/manifest.json)；patched utils SHA256 `e0c060632a5b2a9ce7016edb7e10ac666e3a58464c0216803b9318a083f1b902` |

模型身份未全量重算权重内容SHA256；补丁另校验768个草稿专家header。主模型NVFP4、草稿原生MXFP4、FP8线性层保持既定格式，使用方式见[补丁说明](dspark-compatibility.md)。

共同负载为[GovReport固定JSONL](../data/govreport-near8k.jsonl)前128条，不逐rank重复取128条；平均输入8141.789 tokens、输出固定1024。流式 `/v1/completions`，总C32、seed0、temperature0、ignore_eos=true、request_rate=inf，tokenizer_mode=deepseek_v4。短中文探测使用 `chat_template_kwargs.thinking=false`；off通过，K5未到该阶段。客户端、服务均固定同镜像。服务PP1、上下文16384、每rank max-num-seqs16、每rank prefill8192（聚合最多16384，和TP4调度语义不同）、显存比例0.90、FP8 E4M3 KV、V2＋async、FULL_DECODE_ONLY；关闭prefix cache、autotune和EP。

配置从本项目完整configs及JSONL复制到 `experiments/dspark-k-sweep-node47/`。由[四卡DP服务参数](../configs/recipes/dual-tp2-dp2-epon.yaml)派生单rear target，EP flag替换成 `no-enable-expert-parallel`；不设置replica_targets。所有case引用[GovReport jit_clean](../configs/workloads/govreport-c32-jit-clean.yaml)，target改为节点47。on合并[DSpark环境与检查](../configs/recipes/tp4-dspark-k5.yaml)，固定greedy／standard／adaptive=false，仅改变K和下表Graph。每份on recipe变化后均独立执行prepare并校验补丁，旧recipe原样保留。

| 配置 | Graph捕获列表 | 目标上限／草稿上限 | 验证范围 |
| --- | --- | --- | --- |
| off | `[1,2,4,8,12,16]` | 16／— | Graph完成，六档普通decode |
| K1 | `[1,2,4,8,12,16,24,32]` | 32／16 | 仅离线准备 |
| K2 | `[2,3,4,6,8,12,16,24,32,36,48]` | 48／32 | 仅离线准备 |
| K3 | `[3,4,6,8,12,16,24,32,36,48,64]` | 64／48 | 仅离线准备 |
| K4 | `[4,5,8,10,16,20,32,40,48,60,64,80]` | 80／64 | 仅离线准备 |
| K5 | `[5,6,10,12,20,24,40,48,60,72,80,96]` | 96／80 | 实际CLI已传入，profile失败，未完成捕获 |
| K6 | `[6,7,12,14,24,28,48,56,72,84,96,112]` | 112／96 | 仅离线准备 |
| K7 | `[7,8,14,16,28,32,56,64,84,96,112,128]` | 128／112 | 仅离线准备 |

捕获列表按 `[1,2,4,8,12,16]` 分别乘K和K+1后合并去重排序；off使用普通列表。off各worker实际六档Graph捕获完成。K5的目标96／草稿80已体现在实际CLI，但尚无运行Graph覆盖通过结论。

下列off完整启动命令直接从实测 `argv.json` 导出，只重新排版；owner、容器名、缓存路径均为当次快照：

```bash
docker run -d \
  --pull never \
  --name sb-7af67685bcae485b-server \
  --label io.serving-bench.run=7af67685bcae485b \
  --network host \
  --ipc host \
  --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/vllm/638faf7b431b04afccaa:/root/.cache:rw \
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
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 2 \
  --max-model-len 16384 \
  --max-num-seqs 16 \
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
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,12,16],"max_cudagraph_capture_size":16}' \
  --seed 0 \
  --distributed-executor-backend mp \
  --data-parallel-size-local 2
```

K5实际命令保持上述公共参数，差异从其 `argv.json` 导出如下；不将失败启动命令称为可用性能配置：

```text
--name sb-011f4fc9237b4aee-server
--label io.serving-bench.run=011f4fc9237b4aee
-v /home/enhui/.cache/serving-bench/vllm/f59831a76839e0f0452c:/root/.cache:rw
-e PYTHONPATH=/root/.cache/dspark-native-mxfp4
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[5,6,10,12,20,24,40,48,60,72,80,96],"max_cudagraph_capture_size":96}'
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":false}'
```

复现其他候选时按上表替换K及compilation-config，并保留DSpark完整采样参数、补丁环境、必需检查；不能沿用K5缓存路径。实际每份缓存目录如下，根均为 `~/.cache/serving-bench/vllm/`：

| 配置 | cache目录名 | 起点 |
| --- | --- | --- |
| tp2-dp2-epoff-off | `638faf7b431b04afccaa` | 准备前不存在；保留本次产生的JIT缓存 |
| tp2-dp2-epoff-k5 | `f59831a76839e0f0452c` | 准备前不存在；保留本次产生的JIT缓存 |
| tp2-dp2-epoff-k1 | `01ef75bae5856e5ae33e` | 准备前不存在；只准备补丁，未启动模型 |
| tp2-dp2-epoff-k2 | `001f0925bfe652912135` | 准备前不存在；只准备补丁，未启动模型 |
| tp2-dp2-epoff-k3 | `cbd821289e4f6196c450` | 准备前不存在；只准备补丁，未启动模型 |
| tp2-dp2-epoff-k4 | `2090a0732583ad097930` | 准备前不存在；只准备补丁，未启动模型 |
| tp2-dp2-epoff-k6 | `4598c63d3a70d1011b80` | 准备前不存在；只准备补丁，未启动模型 |
| tp2-dp2-epoff-k7 | `8f79d4f58422681b9db0` | 准备前不存在；只准备补丁，未启动模型 |

缓存未删除，也未从off复制给on。各recipe从各自起点独立编译；off的9轮共享同次启动与其持久缓存。今后热缓存复测的启动和协议成本可能不同。

实际run入口：

```bash
./bench run experiments/dspark-k-sweep-node47/configs/campaigns/tp2-dp2-epoff-off.yaml \
  --run-root experiments/dspark-k-sweep-node47/results/core-off-01
./bench run experiments/dspark-k-sweep-node47/configs/campaigns/tp2-dp2-epoff-k5.yaml \
  --run-root experiments/dspark-k-sweep-node47/results/core-k5-02
```

以上为当次历史入口，重跑必须使用新run-root，不覆盖现有结果。每次run之前均完成validate、plan、日志准备和preflight；on先执行项目 `patches/dspark-native-mxfp4/prepare.py`，不传off。长时测量由工作区脚本在tmux中启动，观测与测量同步保存。

原始产物**仅节点47本机可用**，不随Git分发，目录根为 `/home/enhui/llm-serving-benchmarks/experiments/dspark-k-sweep-node47/`：

- `configs/`、`data/`：完整派生配置、JSONL及依赖；`evidence/*-plan.json`保存全部候选计划。
- `results/core-off-01/`、`results/core-k5-02/`：完整启动命令、解析配置、镜像与模型身份、服务日志、探测、逐轮JSON、绑定与gate。
- `evidence/*-observations/`：压缩原始metrics快照、资源时间线、进程CPU增量和相关system-service cgroup快照。
- `evidence/dp-draft-failure-source.json`：固定镜像相关源码原文及SHA256；`image-metrics-dspark-source.txt`保存指标定义。
- `evidence/core-k5-01-preflight.txt`：端口预检失败；`core-off-01-numa-pages.json`保存实际页面分布。
- `scripts/observe.py`、`run_one.py`、`summarize.py`、`write_report.py`及 `reports/summary.json`：本地辅助脚本、统计与报告生成入口。

结束已确认本次服务/客户端及预检临时容器清理，GPU空闲；未清理他人tmux任务、模型、镜像、历史结果或缓存。最终仅提交本报告、完整CSV及README本节点完成标记，不推送。离线复核包括CSV逐行与原始JSON对照、统计重算、配置validate/plan、相对链接和 `git diff --check`；未修改公共源码，不重跑性能实验。
