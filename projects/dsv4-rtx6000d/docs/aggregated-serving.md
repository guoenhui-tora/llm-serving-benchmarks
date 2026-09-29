# 聚合式推理：单节点四卡与八卡部署参照

**结论。** 早期 vLLM 0.29.0、**无 DSpark、随机 8192 输入/1024 输出**的实验显示：单四卡前后位置性能相近；两个独立四卡 TP4 服务在每卡并发对齐时接近单四卡的两倍吞吐，TTFT/TPOT 基本不变，四个节点的双服务基线也相近。单八卡是否更好取决于具体拓扑。

这里的聚合式推理指每个服务实例同时完成 prefill 和 decode；双四卡是两个独立实例，不是 P/D 分离。文末另有**真实语义输入**的四卡 DSpark off/K1–K5 对照，不与上述随机负载结果交叉排名。

## 无 DSpark：随机 8K/1K 的部署拓扑

### 硬件绑定与启动配置

以下部署实验统一使用 DSV4 NVFP4 权重、tokenizer，以及服务端和客户端镜像 `vllm/vllm-openai:v0.29.0`。各服务的硬件绑定如下，后四卡 TP4 的启动命令见表后。

| 服务位置 | GPU | 服务 CPU / 内存 NUMA | 对应客户端 CPU / 内存 NUMA | 服务 HTTP 端口 |
| --- | --- | --- | --- | --- |
| 前四卡 | 0–3 | 0–15 / 0 | 16–19 / 1 | 31248 |
| 后四卡 | 4–7 | 32–47 / 2 | 48–51 / 3 | 31249 |

单四卡按所在行启动一套；双四卡按两行分别启动两套独立服务；单八卡使用 GPU0–7、两行 CPU/NUMA 集合的并集和端口 31248。前/后四卡分别对应 NUMA0/2 的 GPU 区域；客户端绑在对应半区的独立 CPU 集合，**不与服务绑定同一个内存 NUMA**。Docker 绑定限制容器的 CPU 与内存允许集合，不保证宿主资源独占。

以**后四卡 TP4** 为例，换拓扑时按表替换硬件变量，并设置该行实际的 `TP`、`PP`、`DP` 和 `EP`。客户端并发 `C` 是全局同时发出且尚未完成的请求数上限；`--max-num-seqs=S` 是**每个 DP 引擎**能同时处理的请求数上限。单四卡 TP4 只有一个引擎，S32、容量 32；双四卡 TP4 各有一个引擎、各 S32，两套合计 64；单八卡按其 DP 引擎数分配 S，合计 64。因此单四卡测 C32，八卡部署测全局 C64，而不是每套四卡服务各测 C64。客户端 C 可以低于服务容量，不必为每档 C 更改 S。`--max-num-batched-tokens=8192` 是**每引擎**单轮 token 预算；Graph 按每引擎的 S 配置，不按整机合计容量配置。

<details>
<summary><strong>后四卡 TP4 · 服务启动命令</strong></summary>

```bash
IMAGE_ID=sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1
MODEL_DIR=/data/models/DeepSeek-V4-Flash-0731-NVFP4
NAME=aggregated-rear-tp4
CACHE_DIR="$HOME/.cache/serving-bench/$NAME"
GPUS=4,5,6,7; GPU_COUNT=4; CPUS=32-47; NUMA=2; PORT=31249
TP=4; PP=1; DP=1; EP=off

if (( TP * PP * DP != GPU_COUNT )); then echo 'TP×PP×DP must equal GPU_COUNT' >&2; exit 1; fi
SEQS=$((8 * GPU_COUNT / DP))
case "$SEQS" in
  8)  SIZES='[1,2,4,8]' ;;
  16) SIZES='[1,2,4,8,12,16]' ;;
  32) SIZES='[1,2,4,8,12,16,24,32]' ;;
  64) SIZES='[1,2,4,8,12,16,24,32,48,64]' ;;
  *) echo "Unknown historical Graph capacity: $SEQS" >&2; exit 1 ;;
esac
GRAPH="{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":$SIZES,\"max_cudagraph_capture_size\":$SEQS}"
DP_ARGS=()
if (( DP > 1 )); then DP_ARGS=(--data-parallel-size-local "$DP"); fi
if [[ "$EP" == on ]]; then EP_FLAG=--enable-expert-parallel
else EP_FLAG=--no-enable-expert-parallel; fi

docker run -d --pull never --name "$NAME" \
  --network host --ipc host --gpus "\"device=$GPUS\"" \
  --cpuset-cpus "$CPUS" --cpuset-mems "$NUMA" \
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  --cap-add SYS_NICE \
  -v "$MODEL_DIR:/model:ro" -v "$CACHE_DIR:/root/.cache:rw" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e VLLM_USE_V2_MODEL_RUNNER=1 -e TILELANG_CACHE_DIR=/root/.cache/tilelang \
  --entrypoint vllm "$IMAGE_ID" serve /model \
  --served-model-name deepseek-v4-flash --host 127.0.0.1 --port "$PORT" \
  --trust-remote-code --enable-auto-tool-choice --jit-monitor-verbose \
  --no-enable-prefix-caching "$EP_FLAG" --enable-chunked-prefill \
  --async-scheduling --tensor-parallel-size "$TP" \
  --pipeline-parallel-size "$PP" --data-parallel-size "$DP" \
  "${DP_ARGS[@]}" --distributed-executor-backend mp \
  --max-model-len 16384 --max-num-seqs "$SEQS" \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 \
  --kv-cache-dtype fp8_e4m3 --block-size 256 \
  --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' \
  --moe-backend auto --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --compilation-config "$GRAPH" --seed 0
```

</details>

**客户端负载。** 使用上述镜像，经流式 `/v1/completions` 发送 seed 0 的随机输入：每请求严格 8192 输入/1024 输出 tokens，`random_range_ratio=0`、`temperature=0`、`ignore_eos`、无限到达率。单四卡为 C32、每轮 128 请求；双四卡与单八卡为全局 C64、每轮 256 请求；均预热后正式测三轮。

### 前后四卡的部署位置

先在节点 48 固定单四卡 TP4、全局 C32，仅比较前/后四卡位置：

| 四卡部署 | Output tok/s，均值±样本 SD | Mean / P95 TTFT，s | Mean TPOT，ms |
| --- | ---: | ---: | ---: |
| 后四卡 TP4 | 715.72±1.40 | 6.381 / 20.340 | 38.45 |
| 前四卡 TP4 | 715.23±2.31 | 6.322 / 20.347 | 38.54 |

前/后四卡 TP4 吞吐仅差 0.07%，TTFT 和 TPOT 也接近；**前后四卡位置性能基本一致**。

### 节点间性能与双四卡扩容

八卡上运行两个**独立**的四卡 TP4，各服务处理完整输入与输出；两个客户端各以 C32 同步压测，总计 C64。四台节点的整机吞吐为：

| 节点 | Output tok/s，均值±样本 SD | Mean / P95 TTFT，s | Mean TPOT，ms |
| --- | ---: | ---: | ---: |
| 45 | 1428.22±4.14 | 6.518 / 20.714 | 38.34 |
| 46 | 1428.31±7.21 | 6.303 / 20.630 | 38.51 |
| 47 | 1429.63±6.19 | 6.396 / 20.654 | 38.41 |
| 48 | 1433.82±0.93 | 6.389 / 20.652 | 38.29 |

- **节点间性能接近**：四台节点的双四卡吞吐相差最多约 0.4%，TTFT、TPOT 也相近。
- **双四卡吞吐近乎翻倍，延迟基本不变**：节点 48 的单四卡 C32 与双四卡全局 C64 分别为 715.72/1433.82 tok/s（两倍基准的 100.17%）；Mean TTFT 为 6.381/6.389 s，Mean TPOT 为 38.45/38.29 ms。

测量时由**两个独立客户端同步发请求**，按同一随机请求集的索引奇偶均分，每侧 C32/128 请求，总计 C64/256 请求；每个客户端绑在上表对应侧的 CPU。整机吞吐使用双方输出总 tokens 除以计时窗口的**并集时长**，包含一侧先结束后的尾段，不能直接相加各自窗口的 tok/s。它是静态均分实验，不具有统一入口的动态路由能力。

### 单节点并行拓扑选型

固定单节点八张卡、全局 C64，每轮 256 请求，正式三轮。为了在同一时间考察双四卡和单八卡的多种 TP/PP/DP/EP 组合，四台节点分担不同候选；每台节点都测自己的双四卡 TP4 基线。用各节点自己的基线做相对比较，不直接拿不同节点上的候选互相排名。

| 节点 | 八卡部署 | 整机 output tok/s，均值±样本 SD | Mean / P95 TTFT，s | Mean TPOT，ms | 相对该节点双 TP4 吞吐 |
| --- | --- | ---: | ---: | ---: | ---: |
| 48 | 双四卡 TP4 | 1433.82±0.93 | 6.389 / 20.652 | 38.29 | 参照 |
| 48 | 单八卡 TP2×PP4 | 1472.15±6.18 | 5.105 / 12.451 | 38.48 | +2.67% |
| 48 | 单八卡 TP4×PP2 | 902.79±3.83 | 5.915 / 23.170 | 65.06 | -37.04% |
| 48 | 单八卡 TP8 | 681.52±0.69 | 14.443 / 57.591 | 79.75 | -52.47% |
| 45 | 双四卡 TP4 | 1428.22±4.14 | 6.518 / 20.714 | 38.34 | 参照 |
| 45 | 双四卡 TP1×PP4 | 1326.79±119.80 | 4.967 / 11.869 | 42.31 | -7.10% |
| 45 | 双四卡 TP2×PP2 | 1663.93±74.85 | 4.162 / 11.396 | 33.80 | **+16.50%** |
| 46 | 双四卡 TP4 | 1428.31±7.21 | 6.303 / 20.630 | 38.51 | 参照 |
| 46 | 双四卡 TP1×DP4、EP off | 1500.34±59.23 | 7.646 / 29.876 | 30.19 | +5.04% |
| 46 | 双四卡 TP1×DP4、EP on | 1487.73±56.02 | 7.604 / 21.528 | 30.83 | +4.16% |
| 46 | 双四卡 TP2×DP2、EP off | 1491.49±43.22 | 6.288 / 15.681 | 33.00 | +4.42% |
| 46 | 双四卡 TP2×DP2、EP on | 1620.37±17.99 | 5.017 / 12.306 | 31.16 | **+13.45%** |
| 47 | 双四卡 TP4 | 1429.63±6.19 | 6.396 / 20.654 | 38.41 | 参照 |
| 47 | 单八卡 TP2×DP4、EP off | 1029.12±9.54 | 13.296 / 34.181 | 46.01 | -28.01% |
| 47 | 单八卡 TP2×DP4、EP on | 1242.03±23.34 | 9.248 / 23.115 | 38.67 | -13.12% |
| 47 | 单八卡 TP4×DP2、EP off | 878.47±26.48 | 12.265 / 41.016 | 59.12 | -38.55% |
| 47 | 单八卡 TP4×DP2、EP on | 1151.59±48.74 | 7.877 / 26.041 | 45.77 | -19.45% |

相对各节点自己的双 TP4 基线：48 的单八卡 TP2×PP4 高 2.67%（TPOT 略高），45 的双四卡 TP2×PP2 高 16.50%（波动较大），46 的双四卡 TP2×DP2、EP on 高 13.45%；47 的单八卡 DP 候选均落后。**双四卡和单八卡各有占优的候选**，但不同节点测试的候选不同，本轮无法选出跨节点通用最优。

**后续双四卡估算。** 前面的 TP4 配对表明：对于两套**同配置、独立**的四卡服务，若各侧负载均分、每卡并发对齐，可先用单四卡吞吐乘二、TTFT/TPOT 近似单四卡来估计双四卡的性能。这是候选初筛的估算；其他四卡拓扑是否同样扩容仍需实测，单八卡实例也不能套用此估算。

**后续统一入口对照。** 这批数据是两个客户端分别直连两套服务，没有统一 HTTP 入口；静态均分的吞吐不等于代理动态分流的实测吞吐。若要检验统一入口，应保持两套完整四卡 `vllm serve` 实例不变，在它们前面加 HTTP 负载均衡器，让**一个客户端**通过代理向两个服务发起全局 C64 负载；单八卡对照也走相同代理路径，并检查两侧分配、代理开销和整机窗口。vLLM 官方的[多服务 + Nginx 示例](https://docs.vllm.ai/en/v0.29.0/deployment/nginx/)展示了这种部署方式。这只是未来对照方案，历史数据没有测过代理。

## DSpark off/K1–K5：真实 GovReport 的投机长度对照

实验固定 45 节点后四卡 TP4、EP off、v0.29、全局 C32、每引擎 S32/B8192；沿用上面的硬件绑定和服务基本参数。off 不启用投机。下面只列出历史 K3 相对上文启动命令的配置差异：替换 `GRAPH`，并在 `vllm serve` 参数末尾加入 `--speculative-config "$SPEC"`。

```bash
SPEC='{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":false}'
GRAPH='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[3,4,6,8,12,16,24,32,36,48,64,72,96,128],"max_cudagraph_capture_size":128}'
```

投机验证时，目标侧需覆盖每引擎最多 `S × (K+1)` 个 token，草稿侧需覆盖 `S × K`：S32 的 K3 对应 128/96，K5 对应 192/160。所需形状列在 `--compilation-config` 的 `cudagraph_capture_sizes` 中，`max_cudagraph_capture_size` 是捕获上限（K3 为 128，K5 为 192）。历史 v0.29 投机服务还需在 `docker run` 的镜像参数前增加 `-e PYTHONPATH=/root/.cache/dspark-native-mxfp4`，从挂载的缓存目录加载当时的原生 MXFP4 草稿修复；仅追加投机参数不足以复现结果。

客户端改用仓库的 JSONL 读取与 tokenizer 校验适配，逐行提交**真实语义输入**：GovReport 报告正文加摘要指令，取固定的前 128 条；每条输入 **7168–9213 tokens、平均 8141.79 tokens**，每条实际输出**严格 1024 tokens**，`temperature=0`、`ignore_eos`。off 与 K1–K5 各取当时协议接纳的三轮；输入不是严格等长的 8192 tokens，不与上面的拓扑结果交叉排名。

| DSpark | Output tok/s，均值±样本 SD | Mean TTFT，s | Mean TPOT，ms | 草稿接受率 | 平均接受长度，含 bonus |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | 663.38±0.20 | 5.655 | 42.69 | - | - |
| K1 | 733.17±1.58 | 4.711 | 38.62 | 83.66% | 1.837 |
| K2 | 774.19±4.16 | 4.352 | 36.37 | 74.32% | 2.486 |
| K3 | 807.66±3.01 | 4.333 | 34.57 | 64.47% | 2.934 |
| K4 | 802.43±8.63 | 4.418 | 34.32 | 55.67% | 3.227 |
| K5 | 798.99±8.63 | 4.290 | 34.63 | 48.66% | 3.433 |

接受率是接受草稿 tokens / 提出草稿 tokens；表中的平均接受长度按 `1 + 接受草稿 tokens / 验证步数` 汇总，**包含目标模型 bonus token**，不是每步纯草稿接受数。K 增大时接受率下降、接受长度增加；本轮 K1→K3 吞吐上升，K4/K5 没再提高吞吐。这组观察说明选 K 时要同时考虑接受长度与额外草稿、验证开销。
