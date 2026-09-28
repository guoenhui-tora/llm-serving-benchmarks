# DSV4 在 RTX6000D 上如何启动 PD 分离

本页以已验证的 24 卡 4P2D 部署为例，说明如何启动模型服务、配置 P/D 通信、启动代理并验通请求。**只启动六个模型服务还不够：客户端通过代理访问 P 和 D；代理传递 KV 元数据，实际 KV 由 D 通过 NIXL/UCX 从 P 读取，不经过代理。**

## 1. 进程、网络和请求路径

每个 P 或 D 都是一个独立的四卡 vLLM 服务，单独加载同一套模型权重，TP2×DP2、EP on、DSpark K5。一个服务内部有两个 DP 引擎，**不需要为每个 DP 引擎再启动一个容器**。`gpu-6000d-46`、`gpu-6000d-47` 各起两个 P，`gpu-6000d-48` 起两个 D；代理和客户端在 `gpu-6000d-45`，其 GPU 不算入 24 张服务 GPU。控制端放在 45 只是本次部署方式。P 不绑定固定 D：代理按在途请求数从两个池中选一对，四个 P 都必须能与两个 D 通信。

一次请求依次经过：

1. 客户端向代理发送完整请求，代理选一个 P 和一个 D。
2. 代理通过 HTTP 将 prompt 发给 P；P 计算输入并返回 `kv_transfer_params`（远端地址、请求标识、KV block 等元数据）。
3. 代理通过 HTTP 将原始请求和这组元数据发给 D；D 通过 NIXL/UCX 从刚才的 P 读取 KV。
4. D 生成输出，代理将 D 的响应转发给客户端。代理只传元数据和 HTTP 响应，不承载实际 KV 数据。

部署涉及三类模型服务端口：HTTP 供代理发请求，NIXL side channel 供 P/D 建立连接，DP RPC 供实例内部协调；KV 数据由连接器通过 UCX 传输。客户端使用的是代理自己的 HTTP 端口 `31580`。下表列出六个模型服务的端口和设备分配。

| 节点 | 实例 | GPU | CPU / NUMA | 服务 HTTP | NIXL 端口基数 | DP RPC |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 46 (`10.90.1.46`) | P0 | 4-7 | 32-47 / 2 | 31449 | 29300 | 29600 |
| 47 (`10.90.1.47`) | P1 | 4-7 | 32-47 / 2 | 31449 | 29300 | 29600 |
| 46 (`10.90.1.46`) | P2 | 0-3 | 0-15 / 0 | 31450 | 29320 | 29620 |
| 47 (`10.90.1.47`) | P3 | 0-3 | 0-15 / 0 | 31450 | 29320 | 29620 |
| 48 (`10.90.1.48`) | D0 | 4-7 | 32-47 / 2 | 31449 | 29300 | 29600 |
| 48 (`10.90.1.48`) | D1 | 0-3 | 0-15 / 0 | 31450 | 29320 | 29620 |

以 46 节点的 P0 为例，下面只摘出端口和并行参数；完整的 Docker 启动命令在第 3 节：

```bash
VLLM_NIXL_SIDE_CHANNEL_HOST=10.90.1.46 \
VLLM_NIXL_SIDE_CHANNEL_PORT=29300 \
vllm serve /model --host 0.0.0.0 --port 31449 \
  --tensor-parallel-size 2 --data-parallel-size 2 \
  --data-parallel-size-local 2 --data-parallel-rpc-port 29600
```

代理访问 P0 的 HTTP 端口 `31449`；两个 DP worker 的 NIXL 端口为 `29300`、`29301`；`29600` 用于该实例的 DP RPC。完整命令使用 Docker `--network host`，因此三类端口都占用 **46 节点的主机网络端口**，不需 `-p` 映射；同一节点不能冲突，不同节点可以复用。DP RPC 虽占主机端口，仍是实例内部通信，不是客户端或 KV 传输入口。代理转发 P 返回的连接信息，不硬编码远端 NIXL 端口；节点间仍需保证 NIXL/UCX 通信可达。

## 2. 启动前准备

在 46/47/48 分别确认八卡空闲、GPU↔NUMA/CPU 对应、NIC (`mlx5_4`～`mlx5_7`)、端口未被占用，以及同一权重、tokenizer 和固定镜像：

```bash
IMAGE_ID=sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
docker image inspect vllm/vllm-openai:v0.30.0 --format '{{.Id}}'
docker image inspect "$IMAGE_ID" --format '{{.Id}}'
MODEL_DIR=/data/models/DeepSeek-V4-Flash-0731-NVFP4
test -f "$MODEL_DIR/config.json"
```

两次 image ID 必须相同。45 上的代理和客户端也要有这一镜像；代理需能读取本仓库 `src/`，客户端需能读取模型目录中的 tokenizer，并检查代理端口 `31580` 空闲。每个模型服务使用独立、持久的 `CACHE_DIR`，不删除旧 JIT 缓存。固定镜像**必须额外加载**本项目的 [MXFP4 草稿加载修复](../patches/v030-dspark-mxfp4/README.md)；只设置原版镜像及 CLI 无法等价复现。

将修复中的 `sitecustomize.py`、`dspark_native_mxfp4.py`、`manifest.json` 放入每个服务缓存的 `dsv4-v030-mxfp4/`；启动命令用 `PYTHONPATH` 加载它们。更换镜像或权重时必须重新核对修复的哈希断言。已测启动还从各自缓存加载 `pd-logging.json`，用于采集 NIXL 诊断日志，因此每个服务都要预先准备该文件。

下面假设各节点的仓库位于 `~/llm-serving-benchmarks`。Shell 命令使用 `$HOME/llm-serving-benchmarks`，避免引号中的 `~` 无法展开。对**本节点每个实例的独立缓存**执行：

```bash
set -e
REPO="$HOME/llm-serving-benchmarks"
CACHE_DIR="$REPO/experiments/dsv4-pd-deploy/cache/p0" # 依实例改成 p1/p2/p3/d0/d1
install -d "$CACHE_DIR/dsv4-v030-mxfp4"
PATCH_DIR="$REPO/projects/dsv4-rtx6000d/patches/v030-dspark-mxfp4"
for name in sitecustomize.py dspark_native_mxfp4.py manifest.json; do
  source="$PATCH_DIR/$name"
  target="$CACHE_DIR/dsv4-v030-mxfp4/$name"
  if [[ -e "$target" ]]; then
    cmp -s "$source" "$target" || { echo "缓存中 $name 与归档版本不同，请核对" >&2; exit 1; }
  else
    install -m 0644 "$source" "$target"
  fi
done
```

实测服务从缓存读取下面的 vLLM logging JSON。在每个服务的独立 `CACHE_DIR` 下创建它（已有文件先比对，不覆盖）；旧工作区缓存不是 Git 交付物。

<details>
<summary>展开每个服务启动前需要准备的 pd-logging.json</summary>

```bash
if [[ -e "$CACHE_DIR/pd-logging.json" ]]; then
  echo '日志配置已存在，先逐字段比对，勿覆盖'
else
  cat > "$CACHE_DIR/pd-logging.json" <<'JSON'
{
  "version": 1,
  "disable_existing_loggers": false,
  "formatters": {"vllm": {
    "class": "vllm.logging_utils.NewLineFormatter",
    "datefmt": "%Y-%m-%d %H:%M:%S",
    "format": "%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s"
  }},
  "handlers": {"vllm": {
    "class": "logging.StreamHandler", "formatter": "vllm",
    "level": "DEBUG", "stream": "ext://sys.stdout"
  }},
  "loggers": {
    "vllm": {"handlers": ["vllm"], "level": "INFO", "propagate": false},
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl": {
      "handlers": ["vllm"], "level": "DEBUG", "propagate": false
    }
  }
}
JSON
fi
python3 -m json.tool "$CACHE_DIR/pd-logging.json" >/dev/null
```

以上是从实测日志配置按 JSON 等价排版的内容；若已有文件，保留并逐字段核对后再继续。日志等级只是观察配置，仍须检查实际输出。

</details>

启动前确认文件在容器挂载的缓存中。准备好后冻结服务端源码、镜像与启动参数，直到该批实验结束。

## 3. 在 46/47/48 启动 P 和 D

以下模板按保存的 v0.30 实际 `docker run` argv 整理；**在相应节点本地执行**，不是由普通 `./bench run` 跨机编排。先按上表给出本节点每个实例的环境变量，再执行相同的启动段。实例间只变表中的地址/绑核/端口、P 或 D 的容量和角色；不要把 P、D 的 Graph 上限互换。

```bash
# 例：在 46 启动 P0；其他实例按上表修改。
set -e
ROLE=P; NAME=sb-pd24-p0; HOST_IP=10.90.1.46
GPUS=4,5,6,7; CPUS=32-47; NUMA=2
HTTP_PORT=31449; NIXL_PORT=29300; RPC_PORT=29600
CACHE_DIR="$HOME/llm-serving-benchmarks/experiments/dsv4-pd-deploy/cache/p0"
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
python3 -m json.tool "$CACHE_DIR/pd-logging.json" >/dev/null

docker run -d --pull never --name "$NAME" \
  --label io.serving-bench.run=pd24-guide --network host --ipc host \
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

P 的每 DP 引擎 S32、budget8200，K5 target/draft Graph 上限 192/160；D 的每引擎 S64、budget16384，Graph 上限 384/320。两个 D 合计四个 DP 引擎，名义请求容量 256，不等于实际 decode batch。`NCCL_P2P_LEVEL=PHB` **只加在四个 P 上**，调整 P 实例内部 GPU 间的 NCCL P2P 路径；它不是跨节点 P→D 的协议设置。`FULL_DECODE_ONLY` 也不表示 P 的 prefill 被 Graph 捕获。以上地址和 NIC 绑定只适用于实测节点，迁移环境时必须重做拓扑与可达性检查。

## 4. 启动代理

六个服务各自 READY 后，运行一个独立的代理进程。本次代理在 45，只占 CPU、不占服务 GPU；它使用**同一固定 v0.30 镜像的 Python** 和仓库的 `serving_bench.pd_proxy`（需有 `aiohttp`）。模型加载修复也是本项目的文件依赖，因此整个部署并非只依赖代理。

`--prefill` 和 `--decode` 填的是**服务 HTTP 地址池**，不是 NIXL 地址。列表顺序会影响负载相同时的选路，下面保留已测顺序。

```bash
REPO="$HOME/llm-serving-benchmarks"
IMAGE_ID=sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
docker run -d --pull never --name sb-pd24-proxy \
  --label io.serving-bench.run=pd24-guide --network host \
  --cpuset-cpus 16-19 --cpuset-mems 1 \
  -v "$REPO/src:/src:ro" -e PYTHONPATH=/src \
  --entrypoint python3 "$IMAGE_ID" -m serving_bench.pd_proxy \
  --host 127.0.0.1 --port 31580 --timeout 1200 \
  --prefill http://10.90.1.46:31449 http://10.90.1.47:31449 \
            http://10.90.1.46:31450 http://10.90.1.47:31450 \
  --decode http://10.90.1.48:31449 http://10.90.1.48:31450
```

`127.0.0.1` 只接受 45 本机客户端；远端客户端需要显式调整 bind 地址并限制访问。代理向 P 发送原始 prompt，设置 `stream=false`、`max_tokens=1` 和 `do_remote_decode=true`。P 返回 `remote_engine_id`、`remote_request_id`、`remote_host`、`remote_port`、`remote_block_ids` 等元数据；代理校验后将整组 `kv_transfer_params` 附到原请求，交给 D 生成并转发其响应。

这些远端 ID 和端口由 P 在运行时生成，代理不生成或复制 KV 数据。P 名额占用至元数据返回，D 名额占用至输出结束。代理不自动重试，也不回退到 D 本地整段 prefill；D 的 `kv_load_failure_policy=fail` 会暴露 KV 加载故障。

**能否换代理？** 可以另行实现相同的 HTTP 编排与 KV 元数据转发，但 `vllm serve` 不会自动提供这里的 PD 选路。本次仅验收了仓库的 least-inflight 代理；固定 v0.30 镜像是否自带可运行的上游代理并未核实。替换时需重新验证多 P/D 选路、流式响应、传输失败处理和客户端计时，不能视为同一配置的复测。

代理、客户端都可以放在 45，也可以放在 46/47/48，只要能访问所有 P/D 的服务 HTTP 地址；代理和客户端甚至不必同机。同机时客户端使用 `127.0.0.1:31580`；异机时让代理绑定可达的宿主地址，将客户端 URL 改为 `http://<代理所在节点IP>:31580`，并限制访问。若放在 46/47/48，需重新选择空闲的 CPU/NUMA、避免与模型服务/客户端的绑核冲突，并核对 31580 端口；模型服务的 GPU 仍是原来的 24 张，但共享 CPU/网卡可能影响性能，不能把这样的测量当作与 45 上控制端的原实验同环境。

## 5. 验通服务和客户端

先确认六个容器存活，并从代理所在节点检查所有模型服务的 HTTP `/health` 和 `/v1/models`，最后检查代理 `/health`：

```bash
set -e
for host in 10.90.1.46 10.90.1.47 10.90.1.48; do
  for port in 31449 31450; do
    curl -fsS --max-time 10 "http://$host:$port/health" >/dev/null
    curl -fsS --max-time 10 "http://$host:$port/v1/models" >/dev/null
  done
done
curl -fsS http://127.0.0.1:31580/health
```

代理 `/health` **只检查代理自身**，不能代替六个服务的 READY 检查。再从代理所在节点发一个短请求检查 P→D→客户端的完整链路：

```bash
curl -fsS http://127.0.0.1:31580/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","prompt":"Summarize this report.","max_tokens":8,"temperature":0,"ignore_eos":true,"stream":false}'
```

单条请求只能验证一条 P→D 路径；结合代理日志检查 P 返回的 KV 元数据、D 的远端 KV 命中和真实输出。启动日志还应确认 DSpark MXFP4 修复、K5 及预期 Graph 已加载，没有 NIXL 传输失败或 D 整段输入重算。单次短请求不能替代真实输入长度下的验收。

还可在代理所在节点运行下面的 **vLLM bench 短负载**：8 个随机请求、最多同时 2 个，用于验通 PD 链路，**不是 16K 正式性能压测**。命令沿用已测客户端的入口；异机客户端须将 `--base-url` 改为代理宿主地址：

```bash
IMAGE_ID=sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
MODEL_DIR=/data/models/DeepSeek-V4-Flash-0731-NVFP4
docker run --rm --pull never --network host -v "$MODEL_DIR:/model:ro" \
  --entrypoint python3 "$IMAGE_ID" \
  -c 'from vllm.benchmarks.serve import add_cli_args, main; from vllm.utils.argparse_utils import FlexibleArgumentParser; p=FlexibleArgumentParser(); add_cli_args(p); main(p.parse_args())' \
  --backend vllm --base-url http://127.0.0.1:31580 \
  --endpoint /v1/completions --model /model --tokenizer /model \
  --served-model-name deepseek-v4-flash --dataset-name random \
  --random-input-len 128 --random-output-len 16 --random-range-ratio 0 \
  --num-prompts 8 --max-concurrency 2 --request-rate inf \
  --temperature 0 --ignore-eos --trust-remote-code
```

`--max-concurrency 2` 限制的是**整个客户端经过代理**的同时在途请求数，不是每个 P/D 各 2 个。`--model /model` 标识客户端使用的本地模型目录，`--tokenizer /model` 从中读取 tokenizer 来生成和统计请求；`--served-model-name deepseek-v4-flash` 是发给服务的 API 模型名。客户端只读挂载该目录，**不加载模型权重做推理，也不占服务 GPU**；真正运行模型的是六个 P/D 服务。短测后检查代理选路日志和各服务请求记录；8 条请求不保证覆盖全部八种 P→D 组合，更不代表正式负载已验收。

清理时只停止、删除本次 `io.serving-bench.run=pd24-guide` 标签下**确属自己启动**的六个服务和代理；不删除权重、镜像、历史实验、缓存，也不清理其他人的容器。
