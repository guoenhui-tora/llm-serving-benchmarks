# 节点47：DP＋DSpark 补丁后 K5／K3／K4 续测

**本节点核心任务全部完成：EP off 的 K5、K3、K4 均取得三轮 clean、协议 PASS。三组输出吞吐为 862.60、860.77、874.25 tok/s，均值最大差约 1.57%，而 K3／K4 的吞吐 CV 为 1.79%／2.56%，尚未分出可靠胜负。** K5 本次重复最接近，但没有独立重启验证，不能称为稳定最优。历史 off 的客户端核及 SMT 背景与本轮明显不同，本报告将其作为历史值并列，不计算 DSpark 相对收益。

范围：2026-09-20，本机 `gpu-6000d-47`／`10.90.1.47`，GPU4–7 单套 TP2×DP2／PP1、EP off，总 C32，GovReport 近8K／1024。按[当前节点任务](../README.md#下一轮dpdspark-补丁后续测)顺序 K5 → K3 → K4，各配置只启动一次。原 K 值报告和失败证据保留于[上轮报告](dspark-k-sweep-node47.md)；本轮 [44 列 CSV](../data/dspark-dp-resume-node47.csv) 包含全部 30 轮，旧 off 不复制、不合并。

## 核心结果与停止决定

以下每项均为三轮均值 ± 样本标准差，吞吐 CV 为标准差／均值；P95 是逐轮 P95 的重复统计，不是合并请求 P95。接受长度只含接受的草稿 token，不含目标补充 token。

| 配置／批次 | 接纳轮／总轮 | 输出 tok/s | CV | 较历史 off | Mean／P95 TTFT s | Mean／P95 TPOT ms | 草稿接受率 %／长度 |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| off，历史独立启动 | 7、8、9／9 | 710.39 ± 20.66 | 2.91% | 历史参照 | 6.057 ± 0.422／14.777 ± 0.636 | 36.082 ± 0.583／39.441 ± 0.759 | 不适用 |
| K5，本轮 PASS | 7、8、10／10 | 862.60 ± 2.49 | 0.29% | 参照失效，不计算 | 3.906 ± 0.060／13.690 ± 0.001 | 31.907 ± 0.187／45.361 ± 0.903 | 49.15 ± 0.62／2.457 ± 0.031 |
| K3，本轮 PASS | 5、7、12／12 | 860.77 ± 15.39 | 1.79% | 参照失效，不计算 | 3.672 ± 0.176／13.699 ± 0.005 | 32.409 ± 0.328／45.976 ± 1.068 | 64.57 ± 0.65／1.937 ± 0.020 |
| K4，本轮 PASS | 6、7、8／8 | 874.25 ± 22.34 | 2.56% | 参照失效，不计算 | 3.691 ± 0.304／13.694 ± 0.004 | 32.032 ± 0.213／44.001 ± 1.378 | 57.10 ± 0.45／2.284 ± 0.018 |

K3 相对 K4 均值变化为 **-1.54%**，未超过追加筛选阈值 **5.11%**（`max(2%, 2×两组较大CV)`）。
K3 相对 K5 均值变化为 **-0.21%**，未超过追加筛选阈值 **3.58%**（`max(2%, 2×两组较大CV)`）。
因此不触发 K2，也不运行 K1；两者只提前完成配置、validate／plan 与补丁准备。没有追加 K6/K7、其他并发、双部署或独立重启。资源变化亦不支持将小差距解释为可靠收益；这些阈值不是统计显著性检验。

| 补充指标，均值 ± 样本标准差 | 历史 off | K5 | K3 | K4 |
| --- | ---: | ---: | ---: | ---: |
| requests/s | 0.69374 ± 0.02018 | 0.84239 ± 0.00243 | 0.84060 ± 0.01503 | 0.85376 ± 0.02182 |
| Mean ITL ms | 36.082 ± 0.583 | 110.149 ± 0.739 | 95.084 ± 0.371 | 105.033 ± 0.140 |
| P95 ITL ms | 27.017 ± 0.060 | 779.053 ± 0.578 | 758.686 ± 1.011 | 778.302 ± 0.984 |
| Mean 端到端延迟 s | 42.969 ± 1.000 | 36.547 ± 0.187 | 36.826 ± 0.466 | 36.460 ± 0.403 |
| P95 端到端延迟 s | 51.535 ± 1.113 | 51.617 ± 0.701 | 51.237 ± 0.684 | 50.131 ± 1.179 |

DSpark 的 ITL 是流式事件间隔，不等于逐 token 计算时间；不能因为吞吐较高就宣称所有延迟改善。历史 off 的原始精度及资源限制见[旧 CSV](../data/dspark-k-sweep-node47.csv)与旧报告。

## 全部轮次、实际工作量与时间成本

统一 `jit_clean` 版本1，同配置一次启动内直接逐轮执行完整负载，无独立预热；累计最先三轮无已知编译事件且工作量正确的样本。每配置最多12轮／2700秒、单轮900秒、服务启动1800秒，没有重启补满或修改 gate。K3 在第12轮取得第三轮 clean；所有拒绝轮均计入成本。

| K | 轮 | 阶段起止，北京时间 | 客户端计时 s | 输出 tok/s | 事件行数 | 累计 clean | 接纳 | KV峰值 % |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |
| 5 | 1 | 12:11:19–12:19:57 | 491.571 | 266.639 | 208 | 0 | 否 | 28.086 |
| 5 | 2 | 12:19:57–12:23:44 | 199.633 | 656.564 | 30 | 0 | 否 | 28.135 |
| 5 | 3 | 12:23:44–12:26:41 | 152.268 | 860.796 | 10 | 0 | 否 | 28.086 |
| 5 | 4 | 12:26:41–12:29:55 | 166.613 | 786.687 | 4 | 0 | 否 | 28.105 |
| 5 | 5 | 12:29:55–12:32:54 | 153.451 | 854.160 | 10 | 0 | 否 | 28.082 |
| 5 | 6 | 12:32:54–12:36:18 | 177.055 | 740.290 | 16 | 0 | 否 | 28.090 |
| 5 | 7 | 12:36:18–12:39:18 | 152.430 | 859.886 | 0 | 1 | 是 | 28.112 |
| 5 | 8 | 12:39:18–12:42:15 | 151.851 | 863.161 | 0 | 2 | 是 | 28.179 |
| 5 | 9 | 12:42:15–12:45:16 | 155.093 | 845.118 | 10 | 2 | 否 | 28.120 |
| 5 | 10 | 12:45:16–12:48:13 | 151.569 | 864.768 | 0 | 3 | 是 | 28.116 |
| 3 | 1 | 13:01:00–13:09:41 | 494.999 | 264.792 | 219 | 0 | 否 | 28.092 |
| 3 | 2 | 13:09:41–13:13:08 | 180.869 | 724.678 | 24 | 0 | 否 | 27.910 |
| 3 | 3 | 13:13:08–13:16:57 | 203.387 | 644.446 | 36 | 0 | 否 | 28.059 |
| 3 | 4 | 13:16:57–13:19:57 | 152.939 | 857.019 | 4 | 0 | 否 | 28.096 |
| 3 | 5 | 13:19:57–13:22:58 | 154.720 | 847.155 | 0 | 1 | 是 | 28.040 |
| 3 | 6 | 13:22:58–13:25:56 | 151.570 | 864.761 | 4 | 1 | 否 | 28.107 |
| 3 | 7 | 13:25:56–13:28:52 | 149.375 | 877.467 | 0 | 2 | 是 | 28.107 |
| 3 | 8 | 13:28:52–13:31:53 | 153.594 | 853.365 | 4 | 2 | 否 | 28.081 |
| 3 | 9 | 13:31:53–13:35:06 | 167.373 | 783.115 | 2 | 2 | 否 | 28.059 |
| 3 | 10 | 13:35:06–13:38:02 | 149.841 | 874.738 | 6 | 2 | 否 | 28.111 |
| 3 | 11 | 13:38:02–13:41:00 | 152.614 | 858.847 | 2 | 2 | 否 | 27.844 |
| 3 | 12 | 13:41:00–13:43:59 | 152.818 | 857.698 | 0 | 3 | 是 | 28.029 |
| 4 | 1 | 14:53:55–15:03:41 | 560.240 | 233.957 | 233 | 0 | 否 | 28.126 |
| 4 | 2 | 15:03:41–15:06:52 | 164.461 | 796.980 | 28 | 0 | 否 | 28.134 |
| 4 | 3 | 15:06:52–15:09:51 | 152.913 | 857.167 | 2 | 0 | 否 | 28.074 |
| 4 | 4 | 15:09:51–15:12:59 | 161.211 | 813.047 | 16 | 0 | 否 | 28.104 |
| 4 | 5 | 15:12:59–15:15:58 | 153.573 | 853.481 | 4 | 0 | 否 | 28.156 |
| 4 | 6 | 15:15:58–15:18:59 | 154.484 | 848.452 | 0 | 1 | 是 | 28.074 |
| 4 | 7 | 15:18:59–15:21:52 | 147.793 | 886.859 | 0 | 2 | 是 | 28.100 |
| 4 | 8 | 15:21:52–15:24:46 | 147.697 | 887.440 | 0 | 3 | 是 | 28.115 |

每轮均 **128 成功、0 失败，输入 1,042,149／输出 131,072 tokens**。30轮合计 **3,840 成功、0 失败，输入31,264,470／输出3,932,160 tokens**；接纳9轮合计1,152成功、输入9,379,341／输出1,179,648 tokens。21轮被事件检查拒绝，共872条匹配日志；匹配行数不是独立编译次数。包含真实 TileLang 编译和 Triton JIT 事件，不重新解释或放宽判定。

| run_id | 北京时间 run 起止 | 启动就绪 s | 协议 s | run 总计 s |
| --- | --- | ---: | ---: | ---: |
| dpfix-tp2-dp2-epoff-k5-01 | 12:03:05–12:48:15 | 443.954 | 2214.058 | 2709.798 |
| dpfix-tp2-dp2-epoff-k3-01 | 12:52:49–13:44:01 | 444.035 | 2579.080 | 3072.219 |
| dpfix-tp2-dp2-epoff-k4-01 | 14:45:53–15:24:47 | 433.879 | 1850.566 | 2334.644 |

三个 run 合计 135.28 分钟；从第一组 run 开始至最后一组结束跨越 3.36 小时，包含组间空档，不等于连续占用 GPU 的时间。run 含内部 preflight、启动、协议与清理，不含外部准备／预检；协议阶段含客户端初始化与检查，`duration_s` 仅为客户端内部 benchmark 计时。

## 接受统计、KV 与日志证据

沿用固定镜像 `v1/spec_decode/metrics.py` 的实际定义，每约1秒读取一次本机 `/metrics`，保存请求起止时间和原文；按 `engine="0"/"1"` 区分 DP rank，不累加 TP worker。每轮使用阶段开始前最后一份快照与结束前最后一份完整快照，检查计数器在整个窗口不减小。30轮边界间隙均不足1秒；这仍是采样证据，不能排除采样间的瞬时峰值。

接受率为 `100×ΣΔaccepted_tokens／ΣΔdraft_tokens`；平均接受长度为 `ΣΔaccepted_tokens／ΣΔdrafts`，**不加目标补充 token**，CSV 的 `acceptance_length_includes_bonus=false`。另保存按 position 排序的接受计数。逐轮同时检查 `draft_tokens=K×drafts`、逐位置接受计数和等于总接受数、成功计数增量为128。表中的均值／标准差是三次逐轮比率统计，不冒充合并加权比率。

每组首轮采样窗口的 `request_success_total` 增量为129，混入此前单个中文探测的延迟计数，因此首轮全部草稿统计字段留空；正式客户端实际仍是128成功。其余27轮通过计数窗口核验，包括全部9个接纳轮；未把功能探测、日志百分比或K当作接受统计。没有计数器重置，所有原始快照和按rank增量留在本机。

| 配置／DP rank | 三个接纳轮 Σdrafts | Σ提出草稿 tokens | Σ接受草稿 tokens | 加权接受率 % | 加权接受长度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| K5／0 | 57140 | 285700 | 140582 | 49.206 | 2.4603 |
| K5／1 | 56656 | 283280 | 139025 | 49.077 | 2.4538 |
| K3／0 | 66761 | 200283 | 129875 | 64.846 | 1.9454 |
| K3／1 | 67132 | 201396 | 129495 | 64.299 | 1.9290 |
| K4／0 | 60197 | 240788 | 137547 | 57.124 | 2.2849 |
| K4／1 | 59605 | 238420 | 136050 | 57.063 | 2.2825 |

| 配置 | DP0／DP1 KV容量 tokens | 日志KV预算 GiB/worker | 全轮DP0／DP1使用率峰值 % | 各rank运行请求峰值 | 全轮抢占增量 |
| --- | --- | ---: | --- | --- | ---: |
| K5 | 65,416／65,416 | 25.06 | 28.131／28.179 | 16／16 | 0 |
| K3 | 65,618／65,618 | 25.14 | 28.107／28.111 | 16／16 | 0 |
| K4 | 65,453／65,453 | 25.07 | 28.156／28.134 | 16／16 | 0 |

容量来自每rank `cache_config_info`／启动日志，CSV取两个rank的最小值；使用率取各rank采样峰值再取最大，抢占按无重置计数器的逐rank增量求和。DSV4压缩缓存的等效token容量不等同普通全注意力KV容量。显存比例同为0.90，K与Graph变化后KV容量仍略有差异；本负载没有OOM或抢占证据。

三组日志均包含两个DP rank、各四份量化分派与DP补丁标记，确认主模型NVFP4／`FLASHINFER_CUTLASS`、草稿原生MXFP4／`DEEPGEMM_MXFP4`、attention `FLASHINFER_MLA_SPARSE_DSV4` 与 `fp8_ds_mla`、V2 runner、关闭autotune，以及目标／草稿Graph捕获完成。EP依据实际 `--no-enable-expert-parallel` 及运行配置核对，服务日志未单独输出显式的EP关闭标记。未出现旧 `8192 80` 断言、OOM或异常退出；日志选择标记与完整请求通过不代表验证全部kernel或模型质量。

警告原样保留：SM120 SymmMem不可用、TP2 FlashInfer All Reduce不支持、ModelOpt NVFP4实验格式、TileLang向量化回退、编译期间共享内存等待、DP统计乱序、多个API worker关闭不完整日志统计、弃用及generation_config默认值提示。正式客户端显式temperature0，不采用默认temperature1。最终warning匹配数K5／K3／K4为249／280／235，不代表独立问题数。

## 绑定与资源可比性

GPU4–7，TP2近端组4–5／6–7；服务CPU32–47／NUMA2，客户端CPU48–51／NUMA3，SMT兄弟分别96–111、112–115。每轮服务和客户端Docker inspect及线程允许集合检查通过；GPU0–3未被本次使用。没有收集本轮NUMA实际页面分布，允许集合不能证明所有共享页本地驻留。

| 配置／接纳轮 | 整机忙碌 % | 服务核 % | 服务SMT % | 客户端核 % | 客户端SMT % | GPU平均功耗 W/卡 | GPU温度范围 °C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| K5／7 | 12.23 | 34.70 | 0.15 | 37.24 | 11.80 | 244.27 | 48–62 |
| K5／8 | 12.23 | 34.70 | 0.14 | 25.88 | 24.85 | 239.80 | 48–60 |
| K5／10 | 12.27 | 35.00 | 0.13 | 25.51 | 25.12 | 250.86 | 48–61 |
| K3／5 | 12.28 | 40.40 | 0.13 | 18.05 | 10.92 | 245.24 | 48–61 |
| K3／7 | 12.23 | 40.32 | 0.16 | 21.63 | 7.41 | 245.74 | 48–61 |
| K3／12 | 12.25 | 40.32 | 0.13 | 16.33 | 12.62 | 243.94 | 48–60 |
| K4／6 | 12.17 | 34.44 | 0.13 | 25.67 | 25.14 | 243.22 | 48–61 |
| K4／7 | 12.25 | 34.94 | 0.18 | 8.37 | 25.16 | 243.72 | 48–60 |
| K4／8 | 12.18 | 34.39 | 0.19 | 22.60 | 10.24 | 243.46 | 48–60 |

资源以轮次阶段内完整约5秒区间统计，包含客户端初始化和收尾；进程及system.service cgroup约30秒记录一次。三组DOCA进程合计平均均约10.000逻辑核，未观察到新增矿工类大幅整机负载。clean轮GPU SM时钟为2415–2430 MHz，未见明显持续热降频；采样不能排除瞬时干扰，未改宿主功耗、锁频或NUMA设置。

**旧off作为相对收益分母的资源可比性不足。** 旧off接纳轮客户端核平均6.23%–7.20%、SMT兄弟0.17%–2.91%；本轮见上表，客户端及SMT占用和各组间分布明显变化。例如旧off第8轮CPU50／112为2.39%／0.15%，本轮K5第8轮为95.74%／99.01%。整机总忙碌率相近、DOCA总用量及主线程最后CPU位置相近，仍不能证明逐核竞争相同，也不能唯一归因于DOCA。

模型、镜像、客户端、输入集、协议、绑定声明、off与on共有的服务参数均一致，runner指纹也相同；on增加规定的DSpark／DP补丁与Graph，资源分布却发生实质变化。故仅并列历史off，收益列留空说明；未擅自重跑off或按CPU比例修正吞吐。K值之间同样保留资源差异，不能用K4高出约1%–2%的均值建立确定排名。

## 固定身份、实际命令与复现入口

先执行 `git pull`，快进至 `e8c1da1ce0aafe3331a7ff1dc68cad4980642796`，并核验HEAD包含 `e8c1da1`。启动前工作区干净，运行期间未pull或修改源码／配置。旧off基于 `397e9d42dedee00c181c5fb10f5686ace069c111`，旧失败结论不改判。

| 身份 | 本轮值 |
| --- | --- |
| Git | `e8c1da1ce0aafe3331a7ff1dc68cad4980642796` |
| runner源码指纹 | `f6e873002b8cf61963bd221d63b0fe0a520e0bc6118f4ca63acfa54fc6726d52` |
| 镜像tag | `vllm/vllm-openai:v0.29.0` |
| 服务／客户端Image ID、RepoDigest SHA256 | `c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1` |
| 模型路径 | `/data/models/DeepSeek-V4-Flash-0731-NVFP4` |
| 模型元数据／tokenizer／分片大小身份 | `1e1b7e54442bcf7f15ef55192641b5c5da360bcaa608236aa0d06b273aedf8b4` |
| JSONL SHA256 | `33bdbd36a8b335300974e1b07aa66b2c24c88cc4c4b55a21350aa3564a945d53` |
| 补丁manifest SHA256 | `2e30906bd35789d8e7cd5940f8e28d5b03eb10aff9a014814a23f780b9a5f368` |

三组模型身份和旧off完全一致，未全量重算权重内容哈希。补丁每个campaign均重新prepare并 `--check`，包括量化分派与上游PR #54856 profiling回移，见[兼容性说明](dspark-compatibility.md)。缓存按recipe分别隔离，未复制或删除旧JIT缓存，未覆盖旧补丁。

固定服务参数为TP2／DP2／PP1、local DP2，每rank max-num-seqs16、prefill8192，上下文16384、显存比例0.90、FP8 E4M3 KV、V2＋async、FULL_DECODE_ONLY；关闭EP、prefix cache、FlashInfer autotune，无CPU offload。DSpark为greedy、standard rejection、adaptive verification关闭。

负载为[GovReport JSONL](../data/govreport-near8k.jsonl)固定前128条，平均输入8141.789 tokens、固定输出1024，流式 `/v1/completions`；总C32而非每rank C32，seed0、temperature0、ignore_eos、request_rate=inf。中文功能探测设置 `chat_template_kwargs.thinking=false`，每组均通过。

下列K5完整启动命令从本轮 `argv.json` 导出，仅重新排版；owner和缓存路径是当次快照。K3／K4除owner、容器名、缓存路径以及下表K／Graph之外，服务参数相同。

```bash
docker run -d \
  --pull never \
  --name sb-68ab33cc0a4245e6-server \
  --label io.serving-bench.run=68ab33cc0a4245e6 \
  --network host \
  --ipc host \
  --gpus '"device=4,5,6,7"' \
  -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro \
  -v /home/enhui/.cache/serving-bench/vllm/59ec3b51ba5f816040f9:/root/.cache:rw \
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
  -e SERVING_BENCH_DSPARK_DP_PROFILE_FIX=1 \
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
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[5,6,10,12,20,24,40,48,60,72,80,96],"max_cudagraph_capture_size":96}' \
  --seed 0 \
  --distributed-executor-backend mp \
  --data-parallel-size-local 2 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":false}'
```

| 配置 | K | Graph捕获尺寸 | 目标／草稿上限 | cache目录名 | owner |
| --- | ---: | --- | --- | --- | --- |
| tp2-dp2-epoff-k5 | 5 | `[5, 6, 10, 12, 20, 24, 40, 48, 60, 72, 80, 96]` | 96／80 | `59ec3b51ba5f816040f9` | `68ab33cc0a4245e6` |
| tp2-dp2-epoff-k3 | 3 | `[3, 4, 6, 8, 12, 16, 24, 32, 36, 48, 64]` | 64／48 | `f1fccc4eba1e89c29bb8` | `0c9ddffb4e794f64` |
| tp2-dp2-epoff-k4 | 4 | `[4, 5, 8, 10, 16, 20, 32, 40, 48, 60, 64, 80]` | 80／64 | `5033496a42fcef25b8a0` | `ce5750009dfd486c` |

Graph列表覆盖基本批次 `[1,2,4,8,12,16]` 的K倍与K+1倍；启动捕获完成和完整C32轮次已实测。配置来自[公共DP recipe](../configs/recipes/tp2-dp2-dspark-k5.yaml)及[单case campaign](../configs/campaigns/dspark-dp-clean.yaml)，在本机工作区派生EP off与各K；工作区中target为47，runtime保持1800秒启动上限和持久化Triton缓存。探索配置不另复制进projects。

以下参数直接提取自首轮客户端 `argv.json` 中传给固定镜像官方 `vllm.benchmarks.serve` 的部分；runner另注入JSONL装载与线程绑定取证，不是可单独运行的完整Docker命令：

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

本机执行入口如下；每次必须使用新的run目录。实际三次run_id已见上表，不覆盖它们。公共依赖、完整解析配置、客户端封装及最终命令以各run快照为准。

```bash
study=experiments/dspark-k-sweep-node47
case_id=tp2-dp2-epoff-k5  # 本轮随后依次为k3、k4
campaign="$study/configs/campaigns/$case_id.yaml"
./bench validate "$campaign"
./bench plan "$campaign"
python3 projects/dsv4-rtx6000d/patches/dspark-native-mxfp4/prepare.py "$campaign"
python3 projects/dsv4-rtx6000d/patches/dspark-native-mxfp4/prepare.py "$campaign" --check
python3 scripts/prepare_logging.py "$campaign"
./bench preflight "$campaign"
./bench run "$campaign" --run-root "$study/results/dpfix-$case_id-<新编号>"
```

本轮通过tmux持久会话顺序运行，本地只读观察器为 `scripts/observe_dpfix.py`，每秒metrics、每约5秒CPU/GPU、每约30秒进程/cgroup；没有改公共runner或gate。三个run均正常PASS并由所属runner清理容器，结束核对GPU全部空闲；没有清理其他任务、模型、镜像或缓存。

## 本机原始产物与归档检查

以下均仅节点47本机可用，不随Git提交：

```text
experiments/dspark-k-sweep-node47/
  results/dpfix-tp2-dp2-epoff-k5-01/
  results/dpfix-tp2-dp2-epoff-k3-01/
  results/dpfix-tp2-dp2-epoff-k4-01/
  evidence/dpfix-*-observations/  # 原始metrics gzip、资源及进程JSONL
  evidence/dpfix-*-preflight.txt、*-console.log、*-exit.json
  reports/dpfix-preparation/     # 初始Git、计划、补丁manifest与准备记录
  reports/dpfix-summary.json     # 每轮按rank增量、窗口核验与资源摘要
  reports/dpfix-final-audit.json # 收尾一致性、工作量、亲和性核验
  scripts/summarize_dpfix.py、write_dpfix_report.py
```

提交前从44列CSV重算三组均值、样本标准差及CV，并与协议接纳顺序和原始metrics逐项交叉核对；核对模型／镜像、源码和缓存补丁身份，90份线程亲和性快照及对应Docker cpuset核验通过；复查工作量、单位、相对链接，执行validate／plan和 `git diff --check`。仅归档本报告、CSV和README中47节点进度，不修改其他节点与总表，不push。
