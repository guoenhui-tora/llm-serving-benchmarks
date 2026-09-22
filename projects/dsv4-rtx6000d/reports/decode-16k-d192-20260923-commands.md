# 16K D192扫描实测命令

以下由实测command.json直接导出，不是另行手写的配置。实际服务为45后四卡、S96/DP引擎、总容量192，target Graph576、draft最高480。完整客户端CLI包含JSONL适配与逐请求记录hook，见[命令JSON](../data/decode-16k-d192-20260923-commands.json)。

## 实际服务启动

```bash
docker run -d --pull never --name sb-n23-decode-d192 --label io.serving-bench.run=n23-decode --network host --ipc host --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/vllm/b9d81f448a5dba070c35:/root/.cache:rw --cpuset-cpus 32-47 --cpuset-mems 2 --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --cap-add SYS_NICE --device /dev/infiniband -v /home/enhui/llm-serving-benchmarks/projects/dsv4-rtx6000d/patches/input-kernel-warmup:/root/.cache/dsv4-input-warmup:ro -v /home/enhui/llm-serving-benchmarks/projects/dsv4-rtx6000d/patches/dspark-native-mxfp4:/root/.cache/dspark-native-mxfp4:ro -v /home/enhui/llm-serving-benchmarks/projects/dsv4-rtx6000d/patches/mhc-startup-warmup-extended/dspark:/root/.cache/dsv4-mhc-warmup:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-pd-screen/reports/pd-logging.json:/root/pd-logging.json:ro -e VLLM_ENGINE_READY_TIMEOUT_S=1800 -e VLLM_SERVER_DEV_MODE=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONPATH=/root/.cache/dsv4-input-warmup:/root/.cache/dspark-native-mxfp4:/root/.cache/dsv4-mhc-warmup -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_LOGGING_CONFIG_PATH=/root/pd-logging.json -e SERVING_BENCH_DSPARK_DP_PROFILE_FIX=1 --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model --served-model-name deepseek-v4-flash --host 0.0.0.0 --port 31350 --trust-remote-code --enable-auto-tool-choice --enable-prefix-caching --enable-expert-parallel --enable-chunked-prefill --jit-monitor-verbose --async-scheduling --tensor-parallel-size 2 --pipeline-parallel-size 1 --data-parallel-size 2 --max-model-len 32768 --max-num-seqs 96 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8_e4m3 --block-size 256 --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' --moe-backend auto --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' --kernel-config '{"enable_flashinfer_autotune":false}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[5,6,10,12,20,24,40,48,60,72,80,96,120,144,160,192,200,240,288,320,360,384,432,480,512,576],"max_cudagraph_capture_size":576}' --seed 0 --distributed-executor-backend mp --worker-cls dsv4_input_worker.Worker --data-parallel-size-local 2 --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":false}'
```

## 客户端与协议

固定镜像内官方vLLM benchmark serve，端口31350，CPU48–55/NUMA3；每档先reset+prime（窗口外），1轮2C预热，再3轮4C正式，request-rate=inf并持续补位。JSON保留每档第一正式轮及C32预热的完整Docker命令，其他重复除产物目录/容器名外相同。

| C | 预热请求数 | 每正式轮请求数 | 正式轮数 |
|---:|---:|---:|---:|
| 32 | 64 | 128 | 3 |
| 64 | 128 | 256 | 3 |
| 80 | 160 | 320 | 3 |
| 96 | 192 | 384 | 3 |
| 128 | 256 | 512 | 3 |
| 144 | 288 | 576 | 3 |
| 192 | 384 | 768 | 3 |

实际依赖为固定镜像、模型、仓库input-kernel-warmup/mhc-startup-warmup-extended/dspark/native-mxfp4补丁、原生JSONL适配器与request_records.py hook。模型启动前已扩展draft预热安全界限到576并验证GPU scratch路径；服务启动4个worker均通过target/input/draft两遍覆盖验收。日志、prime响应和全部28轮实际命令仅在本机`experiments/dsv4-night-20260923/decode/`保存，不能只复制本页CLI而省略补丁和数据。

[结果与完整负载说明](decode-16k-d192-20260923.md)。
