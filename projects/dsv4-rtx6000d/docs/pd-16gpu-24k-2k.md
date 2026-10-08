# 分离式推理：16 卡 24K/2K

本研究使用DeepSeek-V4-Flash-0731-NVFP4和16张RTX6000D；负载为真实GovReport文本，每请求输入精确24576 tokens、输出2048 tokens。吞吐按经过代理、P、远端KV交接和D的完整客户端窗口计算；早期[聚合式推理拓扑](aggregated-serving.md)只提供候选线索，不是同条件的性能基线。

## 结论先读

**本次已测方案的优先工作点为3P1D、P每引擎budget8200、仅P侧PHB、全局C112：输出吞吐3409.3±26.5 tok/s，Mean TTFT 9.342秒。** C120吞吐更高，但TTFT均值只剩88毫秒余量；C128已越线。C112不是全局最优，也未达到95%的逐请求联合达标率。完整取舍见[最优配置并发扫描](#4-最优配置并发扫描)。

## 1. 拓扑来源与原始表现

最初借用[聚合式推理四卡／八卡对照](aggregated-serving.md)选取PD实例候选，但聚合式服务同时承担prefill和decode，不能据此认定其四卡拓扑分别作为P、D也最优。更合适的研究方法是分别用prefill-only和decode-only缩小单侧候选，再由真实P→D端到端实验验收；本研究的24K prefill-only预算初筛属于前者，D拓扑则没有完成同等条件的单侧筛选。

早期四卡DSpark实验中，TP2×DP2、EP on、K5是已验证且有竞争力的候选。本负载据此从三个四卡P、一个四卡D的3P1D部署起步，各实例均采用TP2×DP2、EP on、K5。四卡实验只提供起点，不能证明P、D各自的拓扑或3P1D配比已是最优；下面先记录原始表现，再讨论参数优化。

原始16卡部署：46后四卡P0、47后四卡P1、46前四卡P2、48后四卡D；45的CPU承担代理和客户端，不计GPU。P预算24832，每引擎S32；D预算16384，**最初整机D容量80（两个引擎各S40）**。当时服务端v0.30、客户端仍为v0.29。真实24K/2K、全局C32/C48/C64/C80的正式结果：

| 全局并发 | Mean TTFT (ms) | Output (tok/s，均值±SD) | Mean TPOT (ms) | Mean ITL (ms) | P99 ITL (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 5167 | 1607.4±14.6 | 15.98 | 55.84 | 209.36 |
| 48 | 6315 | 2056.5±21.7 | 18.55 | 64.90 | 211.76 |
| 64 | 7436 | 2430.0±12.1 | 20.87 | 72.89 | 222.11 |
| 80 | 9107 | 2759.7±5.3 | 22.41 | 78.69 | 239.33 |

这四档按每档2C条完整请求预热、三轮各4C条正式测量。后续D容量改为128、客户端改为v0.30；**不能把原始表与最终扫描的差值全部算作budget或PHB收益**。下面的单项对照则固定新容量和客户端。

## 2. P预算：固定D容量后的对照

新批次固定D为两个引擎各S64（整机名义容量128），服务端、客户端均用v0.30。在相同24K/2K和C80下，以P预算24832作新基线：

| P budget／P通信 | Output (tok/s，均值±SD) | Mean TTFT (s) | 判断 |
| --- | ---: | ---: | --- |
| 24832／默认 | 2758.8±5.1 | 9.053 | 新容量下的初始基线 |
| 8192／默认 | 2806.0±15.5 | 8.600 | 正式有18次已知JIT日志匹配；确认批次仍有4次 |
| 8200／默认 | 2807.3±13.1 | 7.734 | 吞吐与8192相近，TTFT更低，正式零已知JIT匹配 |

四卡24K输入、1 token输出的prefill-only实验筛出约8K的预算候选，但8192与8200在单侧接近，最终仍由完整PD选取工作配置。同容量、固定C80的邻近预算补测中，8208没有改善，8216的微小吞吐差异落在轮间波动内，因此采用8200，**不证明它是唯一最优值**。预算可能改变输入分块和排队，不能仅凭端到端数字确定具体kernel机制。

## 3. P侧与D侧PHB对照

固定P budget8200、D预算16384、全局C80，仅在三个P服务设置`NCCL_P2P_LEVEL=PHB`。各配置分别启动一次，客户端与数据保持一致：

| P通信 | Output (tok/s，均值±SD) | Mean TTFT (s) | Mean TPOT (ms) |
| --- | ---: | ---: | ---: |
| 默认 | 2825.8±18.3 | 7.794 | 22.53 |
| P侧PHB | 2842.2±36.6 | 7.453 | 22.52 |

Mean TTFT下降0.341秒（4.38%）；吞吐仅增加0.58%，低于轮次波动，不视作明确吞吐收益。P内部跨卡对的NCCL选路观测从SHM/direct变成P2P/CUMEM；**这描述的是P内部GPU通信，不是P→D远端KV经NIXL/UCX的传输协议**。单次独立启动的比较不能排除背景及启动差异，故后续扫描只把P侧PHB作为已测试过的工作配置。

### D侧PHB：不保留

P继续使用PHB、P预算8200、全局C80，再在D加`NCCL_P2P_LEVEL=PHB`：

| P／D通信 | Output (tok/s，均值±SD) | Mean TTFT (s) | Mean TPOT (ms) |
| --- | ---: | ---: | ---: |
| PHB／默认 | 2842.2±36.6 | 7.453 | 22.524 |
| PHB／PHB | 2839.7±22.4 | 7.601 | 22.459 |

没有明确收益，因此最终扫描保持D默认。新增D侧PHB组的NCCL日志验证了P2P/CUMEM，但前一组D未记录相应初始化详情，不能断言原D的准确选路；新组还增加了D侧启动INFO诊断，比较并非纯粹单变量。这里的0.09%吞吐差异不具备机制归因强度。

## 4. 最优配置并发扫描

最终扫描保持P8200＋仅P侧PHB配置，**只改变单个客户端入口的全局并发C**。输入取[精确24K GovReport JSONL](../datasets/README.md)的互异请求，每条24576输入tokens、2048输出tokens；`temperature=0`、`ignore_eos`、无限到达率，P/D本地prefix cache关闭。每档预热2C条完整请求，正式固定三轮、每轮4C条；吞吐包含客户端起跑与排空。C32～C112属于同一次服务启动；C128及随后补测的C120在另一次启动中完成，服务配置不变。所有预定轮次均保留。

| 固定项 | 本次值 |
| --- | --- |
| 服务／客户端镜像 | `vllm/vllm-openai:v0.30.0`，image ID `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90` |
| 模型、传输、调度 | `DeepSeek-V4-Flash-0731-NVFP4`；FP8 KV、NIXL/UCX、V2 async、DSpark K5、EP on、TP2×DP2 |
| P（每引擎） | `--max-num-seqs 32 --max-num-batched-tokens 8200`；三个P容器均设`NCCL_P2P_LEVEL=PHB`；K5 target/draft Graph上限192/160 |
| D（每引擎） | `--max-num-seqs 64 --max-num-batched-tokens 16384`；D保持默认NCCL选路；K5 target/draft Graph上限384/320 |

四个实例分别加载模型，代理负责P→D路由与KV交接，单个客户端连接代理；压测客户端逐行读取经tokenizer校验的24K JSONL。端口及启动命令见下文的[启动与复现](#5-启动与复现)。

所有正式窗口均为4C/4C成功、每请求2048输出tokens；三条P→D路由均覆盖，远端KV命中，D整段输入重算和抢占为0，P/D target及K5 draft Graph捕获完成。C32及C128预热分别有60次已知JIT日志匹配，正式轮次没有已知匹配；这不证明所有运行形状都走Graph。C112采样D每引擎running/waiting峰值56/2（配置S64），C128为63/2；CPU仍有背景进程。

三P一D各为独立四卡TP2×DP2服务、EP on、DSpark K5；C128先测且越线后，补测C120。下面的吞吐为三轮均值±样本标准差，其他指标为三轮均值；P99 ITL是各轮P99的均值，不是合并请求后的P99。

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

各档三轮正式测量的吞吐、延迟分位数及达标指标见[最终配置逐轮 CSV](results/pd-16gpu-24k-2k.csv)。`slo_joint_pass_rate`是同一请求同时满足TTFT和TPOT限值的比例，`slo_goodput_req_s`是达标请求数除以完整窗口时长；与表中的均值SLO不是同一口径。总token吞吐包含输入和输出，生成吞吐使用`output_throughput_tok_s`。

Mean ITL统计流式token事件间隔，与每请求均值TPOT口径不同。均值SLO要求三轮Mean TTFT均值≤10000ms且Mean TPOT均值≤50ms，不能用它代替逐请求联合达标率。C112的P95 TTFT为37.836秒、联合达标率79.02%；C120为40.362秒、78.75%。C120 TTFT三轮为9.971／9.886／9.880秒，其中一轮距线仅29毫秒；C128三轮均越线。故优先选择TTFT余量更大的C112，而非宣称C120的均值SLO在更长运行中必然稳定通过。

## 5. 启动与复现

<details>
<summary><strong>固定环境、服务启动与压测命令</strong></summary>

以下按C112实测服务、代理、客户端参数整理，省去实验编排和观测脚本。先核对各节点GPU、CPU/NUMA、NIC、端口、固定镜像与权重；并按[PD启动说明](pd-startup.md#2-启动前准备)准备DSpark MXFP4草稿加载修复和`pd-logging.json`，只有原版镜像与CLI不能复现本次实验。代理与客户端适配从仓库`src/`挂载，复测时须核对版本与行为。各服务使用独立的持久缓存，保留JIT缓存。

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
CACHE_DIR="$HOME/.cache/dsv4-pd/$SERVICE"
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

正式压测须加载仓库的[JSONL适配](../../../src/serving_bench/clients/jsonl_dataset.py)，否则原生`custom`不会逐行使用上述精确输入。下面以C112的一轮正式请求为例，运行前为每轮设置不同的绝对路径`RESULT_DIR`；完整预热将`N`设为`2*C`、正式每轮设为`4*C`，共三轮，服务不重启。此命令省去了实测的客户端CPU亲和性记录hook，其模型请求与数据适配参数来自实测argv；须另行核验亲和性、逐请求token和完整窗口。

```bash
REPO="$HOME/llm-serving-benchmarks"
MODEL_DIR=/data/models/DeepSeek-V4-Flash-0731-NVFP4
IMAGE_ID=sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
C=112; N=$((4*C))
RESULT_DIR="${RESULT_DIR:?请先指定本轮独立的结果目录}"
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
