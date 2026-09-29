# DSV4 / RTX6000D：16卡PD的24K/2K选型与优化

研究对象为DeepSeek-V4-Flash-0731-NVFP4、16张RTX6000D、真实GovReport精确24576-token输入和2048-token输出。本文的吞吐均为经过代理、P、远端KV交接和D的完整客户端窗口 output tok/s。

## 结论先读

**当前优先工作点是3P1D、P budget8200、仅P侧`NCCL_P2P_LEVEL=PHB`、全局C112：3409.3±26.5 output tok/s，Mean TTFT 9.342秒。** C120的吞吐更高，但Mean TTFT只剩88毫秒余量；C128已越线。C112是留有较多余量的已测优先点，不是全局最优或95%逐请求SLO达标点。各档的TPOT、ITL和完整结果见[最优配置并发扫描](#最优配置并发扫描)。

## 1. 四卡拓扑与原始16卡结果

早期四卡DSpark实验中，TP2×DP2、EP on、K5是已经验证可运行且有竞争力的单元；后续v0.30的16卡3P1D/24K实验进一步比较全TP2×DP2/NIXL与全TP2×PP2/Mooncake：C32～C80的DP2整机吞吐高约25%～28%。两组connector、NIC和每服务容量不同，这只是选型依据，不是单独证明DP优于PP的控制实验。因此后续以三个四卡P、一个四卡D的全TP2×DP2为基线，不重复拓扑矩阵。

原始16卡部署：46后四卡P0、47后四卡P1、46前四卡P2、48后四卡D；45的CPU承担代理和客户端，不计GPU。P预算24832，每引擎S32；D预算16384，**最初整机D容量80（两个引擎各S40）**。当时服务端v0.30、客户端仍为v0.29。真实24K/2K、全局C32/C48/C64/C80的正式结果：

| 全局并发 | Mean TTFT (ms) | Output (tok/s，均值±SD) | Mean TPOT (ms) | Mean ITL (ms) | P99 ITL (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 5167 | 1607.4±14.6 | 15.98 | 55.84 | 209.36 |
| 48 | 6315 | 2056.5±21.7 | 18.55 | 64.90 | 211.76 |
| 64 | 7436 | 2430.0±12.1 | 20.87 | 72.89 | 222.11 |
| 80 | 9107 | 2759.7±5.3 | 22.41 | 78.69 | 239.33 |

这四档按每档2C条完整请求预热、三轮各4C条正式测量。**不可把上表与后文D容量128、客户端v0.30优化扫描当作只改budget/PHB的严格配对**；后面的单项对照均使用新批次的相同D容量和客户端。

## 2. P budget：先固定容量再调度

新批次固定D为两个引擎各S64（整机名义容量128），服务端、客户端均用v0.30。在相同24K/2K和C80下，以P预算24832作新基线：

| P budget／P通信 | Output (tok/s，均值±SD) | Mean TTFT (s) | 判断 |
| --- | ---: | ---: | --- |
| 24832／默认 | 2758.8±5.1 | 9.053 | 新容量下的初始基线 |
| 8192／默认 | 2806.0±15.5 | 8.600 | 正式有18次已知JIT日志匹配；确认批次仍有4次 |
| 8200／默认 | 2807.3±13.1 | 7.734 | 吞吐与8192相近，TTFT更低，正式零已知JIT匹配 |

四卡24K/1 prefill-only曾用于筛选约8K的预算，但不等价于完整PD。后续同配置独立启动、固定C80的邻近预算对照为：**8200：2825.8±18.3 tok/s、TTFT7.794秒；8208：2806.5±3.1、7.880秒；8216：2839.0±17.9、7.855秒**。8216的吞吐微增小于轮间波动，8208没有改善，因此选8200作为后续比较基线，**不是证明8200为数学上的唯一最优**。预算影响24K输入的分块和排队节奏，不能仅凭这些端到端数字断言具体kernel机制。

## 3. 仅P侧开启PHB

固定P budget8200、D预算16384、全局C80，仅在三个P服务设置`NCCL_P2P_LEVEL=PHB`。各配置分别启动一次，客户端与数据保持一致：

| P通信 | Output (tok/s，均值±SD) | Mean TTFT (s) | Mean TPOT (ms) |
| --- | ---: | ---: | ---: |
| 默认 | 2825.8±18.3 | 7.794 | 22.53 |
| P侧PHB | 2842.2±36.6 | 7.453 | 22.52 |

Mean TTFT下降0.341秒（4.38%）；吞吐仅增加0.58%，低于轮次波动，不视作明确吞吐收益。P内部跨卡对的NCCL选路观测从SHM/direct变成P2P/CUMEM；**这描述的是P内部GPU通信，不是P→D远端KV经NIXL/UCX的传输协议**。单次独立启动的比较不能排除背景及启动差异，故后续扫描只把P侧PHB作为已测试过的工作配置。

## 4. D侧PHB：不保留

P继续使用PHB、P预算8200、全局C80，再在D加`NCCL_P2P_LEVEL=PHB`：

| P／D通信 | Output (tok/s，均值±SD) | Mean TTFT (s) | Mean TPOT (ms) |
| --- | ---: | ---: | ---: |
| PHB／默认 | 2842.2±36.6 | 7.453 | 22.524 |
| PHB／PHB | 2839.7±22.4 | 7.601 | 22.459 |

没有明确收益，因此最终扫描保持D默认。新增D侧PHB组的NCCL日志验证了P2P/CUMEM，但前一组D未记录相应初始化详情，不能断言原D的准确选路；新组还增加了D侧启动INFO诊断，比较并非纯粹单变量。这里的0.09%吞吐差异不具备机制归因强度。

## 5. 扫描协议与复现要点

最终扫描保持P8200＋仅P侧PHB配置，**只改变单个客户端入口的全局并发C**。每档预热2C条完整请求，正式固定三轮、每轮4C条完整请求；输入取[精确24K GovReport JSONL](../datasets/README.md)的互异请求，每条24576输入tokens、2048输出tokens，`temperature=0`、`ignore_eos`、无限到达率，P/D本地prefix cache关闭。C32～C112同一服务启动；C128与随后补测的C120是另一服务启动。每轮吞吐均含客户端起跑与排空，不选最快轮。

| 固定项 | 本次值 |
| --- | --- |
| 服务／客户端镜像 | `vllm/vllm-openai:v0.30.0`，image ID `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90` |
| 模型、传输、调度 | `DeepSeek-V4-Flash-0731-NVFP4`；FP8 KV、NIXL/UCX、V2 async、DSpark K5、EP on、TP2×DP2 |
| P（每引擎） | `--max-num-seqs 32 --max-num-batched-tokens 8200`；三个P容器均设`NCCL_P2P_LEVEL=PHB`；K5 target/draft Graph上限192/160 |
| D（每引擎） | `--max-num-seqs 64 --max-num-batched-tokens 16384`；D保持默认NCCL选路；K5 target/draft Graph上限384/320 |

四个实例须各自启动完整模型；同一宿主上的HTTP、实例内DP RPC和NIXL side-channel端口不能冲突，不同节点可以复用端口号。单个代理负责P→D路由和KV交接，单个客户端连接代理。端口关系和准备步骤见[PD启动说明](pd-startup.md)；压测时以经tokenizer校验的24K JSONL逐行适配v0.30客户端，不能用原生随机数据命令代替这份负载。

最新扫描与补测所有正式窗口均为4C/4C成功、每请求2048输出tokens；三条P→D路由均覆盖、远端输入KV命中、D整段输入重算和抢占为0，P/D target及K5 draft Graph捕获完成。C32及C128预热分别有60次已知JIT日志匹配，**正式轮次为零已知匹配**；不能由此声称任何运行时形状必定走Graph。C112采样D每引擎running/waiting峰值56/2（配置S64），C128为63/2；CPU上存在其他背景进程，采样不能证明完全隔离。均值TTFT线只是选档依据，尾延迟和逐请求达标仍需单独评估。

### 最优配置并发扫描

三P一D各为独立四卡TP2×DP2服务、EP on、DSpark K5；P每引擎S32/budget8200且仅P启用PHB，D每引擎S64/budget16384、默认通信。C32～C112属于同一次服务启动；C128先测且越线后，在另一次启动中补测C120。两次启动配置相同，但跨批次差值不能全部归因于并发。

| 全局并发 | Mean TTFT (ms) | Output (tok/s，均值±SD) | Mean TPOT (ms) | Mean ITL (ms) | P99 ITL (ms) | 均值SLO |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 32 | 4790 | 1653.1±22.8 | 15.90 | 56.57 | 210.23 | 通过 |
| 48 | 5772 | 2122.1±11.1 | 18.35 | 64.78 | 211.34 | 通过 |
| 64 | 6559 | 2465.9±11.7 | 20.96 | 73.38 | 227.23 | 通过 |
| 80 | 7575 | 2821.2±13.7 | 22.63 | 79.54 | 237.09 | 通过 |
| 96 | 8632 | 3161.9±20.9 | 24.30 | 85.25 | 235.68 | 通过 |
| **112** | **9342** | **3409.3±26.5** | **25.97** | **90.87** | **238.41** | **通过，优先** |
| 120 | 9912 | 3542.6±20.3 | 26.71 | 93.87 | 240.92 | 通过，余量小 |
| 128 | 10477 | 3674.2±37.6 | 27.21 | 95.67 | 238.52 | TTFT越线 |

各档三轮正式测量的吞吐、延迟分位数和达标情况见[逐轮 CSV](../data/pd-16gpu-24k-2k-optimized-rounds.csv)。`num_prompts`、`completed`、`failed`分别是计划、成功、失败请求数；`slo_joint_pass_rate`是同一请求同时满足CSV中TTFT与TPOT限值的占比（0～1），`slo_goodput_req_s`是达标请求数除以完整客户端窗口秒数。总token吞吐包含输入和输出；生成吞吐看`output_throughput_tok_s`。

“±”为三轮 output tok/s 的样本标准差；其余指标为三轮正式值的算术均值，P99 ITL是**各轮P99的均值**，不是合并所有token重算的P99。Mean ITL统计流式token事件间隔，与每请求均值TPOT口径不同。均值SLO要求三轮Mean TTFT均值≤10000ms且Mean TPOT均值≤50ms；不能用它代替逐请求联合达标率。C112的P95 TTFT为37.836秒、联合达标率79.02%；C120为40.362秒、78.75%。C120 TTFT三轮为9.971／9.886／9.880秒，其中一轮距线仅29毫秒；C128三轮均越线。

<details>
<summary>展开最优配置的启动与压测命令</summary>

以下由C112实测服务、代理、客户端argv参数化整理；省去实验编排和观测脚本，**不是这些模板又完成了一次性能复测**。先核对每个节点GPU、CPU/NUMA、NIC、端口、固定镜像和权重。服务缓存中还必须按[PD启动说明](pd-startup.md#2-启动前准备)准备DSpark MXFP4草稿加载修复与`pd-logging.json`；只有原版镜像和CLI不能复现本次实验。实测代理与客户端适配使用冻结源码，下面改从仓库`src/`挂载，复测前需核对两者版本和行为。每个服务独立持久缓存，保留JIT缓存。

| 节点 | 实例 | GPU | CPU／NUMA | HTTP／NIXL基数／DP RPC |
| --- | --- | --- | --- | --- |
| 46 | P0 | 4,5,6,7 | 32-47／2 | 31449／29300／29600 |
| 47 | P1 | 4,5,6,7 | 32-47／2 | 31449／29300／29600 |
| 46 | P2 | 0,1,2,3 | 0-15／0 | 31450／29320／29620 |
| 48 | D0 | 4,5,6,7 | 32-47／2 | 31449／29300／29600 |

在各实例所在节点运行下列命令，先按表设置第一段变量。示例为46的P0；D0设置`ROLE=D`，其余按表换地址、绑核和端口。`NIXL_PORT`是TP2×DP2两个worker的端口基数，占用本节点`base`和`base+1`。

```bash
ROLE=P; SERVICE=p0; HOST_IP=10.90.1.46
GPUS=4,5,6,7; CPUS=32-47; NUMA=2
HTTP_PORT=31449; NIXL_PORT=29300; RPC_PORT=29600
REPO="$HOME/llm-serving-benchmarks"
CACHE_DIR="$REPO/experiments/dsv4-pd-deploy/cache/$SERVICE"
MODEL_DIR=/data/models/DeepSeek-V4-Flash-0731-NVFP4
IMAGE_ID=sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90

P_GRAPH='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[5,6,10,12,20,24,40,48,60,72,80,96,120,144,160,192],"max_cudagraph_capture_size":192}'
D_GRAPH='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[5,6,10,12,20,24,40,48,60,72,80,96,120,144,160,192,200,240,280,288,320,336,384],"max_cudagraph_capture_size":384}'
if [[ "$ROLE" == P ]]; then
  SEQS=32; BUDGET=8200; KV_ROLE=kv_producer; GRAPH="$P_GRAPH"
  ROLE_ENV=(-e NCCL_P2P_LEVEL=PHB -e NCCL_DEBUG=INFO
    -e NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,P2P,SHM,NET)
else
  SEQS=64; BUDGET=16384; KV_ROLE=kv_consumer; GRAPH="$D_GRAPH"
  ROLE_ENV=(-e NCCL_DEBUG=WARN)
fi
test -f "$CACHE_DIR/dsv4-v030-mxfp4/sitecustomize.py"
test -f "$CACHE_DIR/pd-logging.json"

docker run -d --pull never --name "sb-pd16-$SERVICE" \
  --label io.serving-bench.run=pd16-24k-guide --network host --ipc host \
  --gpus "\"device=$GPUS\"" --cpuset-cpus "$CPUS" --cpuset-mems "$NUMA" \
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  --cap-add SYS_NICE --device /dev/infiniband \
  -v "$MODEL_DIR:/model:ro" -v "$CACHE_DIR:/root/.cache:rw" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e PYTHONPATH=/root/.cache/dsv4-v030-mxfp4 \
  -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_LOGGING_CONFIG_PATH=/root/.cache/pd-logging.json \
  -e VLLM_NIXL_SIDE_CHANNEL_HOST="$HOST_IP" \
  -e VLLM_NIXL_SIDE_CHANNEL_PORT="$NIXL_PORT" \
  -e UCX_NET_DEVICES=mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1 \
  -e UCX_TLS=rc,cuda_copy,cuda_ipc,sm,self -e UCX_IB_GID_INDEX=3 \
  -e UCX_LOG_LEVEL=info -e NIXL_LOG_LEVEL=INFO "${ROLE_ENV[@]}" \
  --entrypoint vllm "$IMAGE_ID" serve /model \
  --served-model-name deepseek-v4-flash --host 0.0.0.0 --port "$HTTP_PORT" \
  --trust-remote-code --enable-auto-tool-choice \
  --no-enable-prefix-caching --enable-expert-parallel --enable-chunked-prefill \
  --jit-monitor-verbose --async-scheduling \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --data-parallel-size 2 --data-parallel-size-local 2 \
  --data-parallel-rpc-port "$RPC_PORT" --distributed-executor-backend mp \
  --max-model-len 32768 --max-num-seqs "$SEQS" \
  --max-num-batched-tokens "$BUDGET" --gpu-memory-utilization 0.9 \
  --kv-cache-dtype fp8_e4m3 --block-size 256 \
  --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV4","indexer_kv_dtype":"auto"}' \
  --moe-backend auto --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --compilation-config "$GRAPH" --seed 0 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":false}' \
  --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"$KV_ROLE\",\"kv_buffer_device\":\"cuda\",\"kv_load_failure_policy\":\"fail\",\"kv_connector_extra_config\":{\"backends\":[\"UCX\"],\"enforce_handshake_compat\":true}}"
```

四个服务全部READY后，在45启动仓库代理（CPU16-19／NUMA1）。代理的`--prefill`是三个P的**服务HTTP地址**，不是NIXL地址；客户端连接代理`127.0.0.1:31580`。模型服务本身不做P/D选路，代理启动后才会从所列P、D实例中选路。

```bash
REPO="$HOME/llm-serving-benchmarks"
IMAGE_ID=sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
docker run -d --pull never --name sb-pd16-proxy \
  --label io.serving-bench.run=pd16-24k-guide --network host \
  --cpuset-cpus 16-19 --cpuset-mems 1 \
  -v "$REPO/src:/src:ro" -e PYTHONPATH=/src \
  --entrypoint python3 "$IMAGE_ID" -m serving_bench.pd_proxy \
  --host 127.0.0.1 --port 31580 --timeout 1200 \
  --prefill http://10.90.1.46:31449 http://10.90.1.47:31449 \
            http://10.90.1.46:31450 \
  --decode http://10.90.1.48:31449
```

正式压测须加载仓库的[JSONL适配](../../../src/serving_bench/clients/jsonl_dataset.py)，否则原生`custom`不会逐行使用上述精确输入。下面以C112的一轮正式请求为例，完整预热将`N`设为`2*C`、正式每轮设为`4*C`，共三轮且每轮使用新的`RESULT_DIR`；服务不重启。此命令省去了实测的客户端CPU亲和性记录hook，其模型请求与数据适配参数来自实测argv；须另行核验亲和性、逐请求token和完整窗口。

```bash
REPO="$HOME/llm-serving-benchmarks"
MODEL_DIR=/data/models/DeepSeek-V4-Flash-0731-NVFP4
IMAGE_ID=sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
C=112; N=$((4*C))
RESULT_DIR="$REPO/experiments/dsv4-rtx6000d/results/pd16-24k-repro/c112/round-1"
mkdir -p "$RESULT_DIR"
docker run --rm --pull never --network host \
  --cpuset-cpus 20-23 --cpuset-mems 1 \
  -v "$MODEL_DIR:/model:ro" \
  -v "$REPO/projects/dsv4-rtx6000d/datasets/govreport-isl24576-exact-n512.jsonl:/dataset.jsonl:ro" \
  -v "$REPO/src/serving_bench/clients/jsonl_dataset.py:/jsonl_dataset.py:ro" \
  -v "$RESULT_DIR:/results:rw" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint python3 "$IMAGE_ID" -c '
import runpy
from pathlib import Path
import vllm.benchmarks.serve as serve
runpy.run_path("/jsonl_dataset.py")["install"](serve, {
    "max_input_tokens": 24576, "output_tokens": 2048,
    "sha256": "615edb29e5a171e8f4dc7679739437202cddf8d93887f7b9a7260bc9e6d68170"
}, Path("/results"), 0, 1)
from vllm.benchmarks.serve import add_cli_args, main
from vllm.utils.argparse_utils import FlexibleArgumentParser
p = FlexibleArgumentParser(); add_cli_args(p); main(p.parse_args())' \
  --backend vllm --base-url http://127.0.0.1:31580 \
  --endpoint /v1/completions --model /model --tokenizer /model \
  --served-model-name deepseek-v4-flash \
  --dataset-name custom --dataset-path /dataset.jsonl \
  --custom-output-len 2048 --skip-chat-template \
  --disable-shuffle --no-oversample --ignore-eos --trust-remote-code \
  --num-prompts "$N" --max-concurrency "$C" --request-rate inf \
  --num-warmups 0 --seed 0 --temperature 0 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 \
  --goodput ttft:10000 tpot:50 --save-result --save-detailed \
  --result-dir /results --result-filename raw.json
```

原扫描C112每轮正式448条请求、11,010,048输入tokens、917,504输出tokens；完整预热224条请求单独保存。跨机迁移需重新核对IP、GPU/NUMA、NIC及可达性；只清理本次实际启动的容器，不删除镜像、权重、结果和JIT缓存。

</details>
