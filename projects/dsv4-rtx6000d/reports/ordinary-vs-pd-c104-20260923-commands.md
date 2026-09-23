# 普通16卡C104：实测命令

从产物导出，需先准备模型、缓存内补丁和真实JSONL。

## r0 — gpu-6000d-46

```bash
docker run -d --pull never --name sb-o104-20260923-01-r0 --label io.serving-bench.run=o104-20260923-01 --network host --ipc host --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/vllm/bfd3829bfea9c9999415:/root/.cache:rw --cpuset-cpus 32-47 --cpuset-mems 2 --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --cap-add SYS_NICE --device /dev/infiniband -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONPATH=/root/.cache/dsv4-input-warmup:/root/.cache/dspark-native-mxfp4:/root/.cache/dsv4-mhc-warmup -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_NIXL_SIDE_CHANNEL_HOST=10.90.1.46 -e VLLM_NIXL_SIDE_CHANNEL_PORT=26300 -e UCX_NET_DEVICES=mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1 -e UCX_TLS=rc,cuda_copy,cuda_ipc,sm,self -e UCX_IB_GID_INDEX=3 -e UCX_LOG_LEVEL=info -e NIXL_LOG_LEVEL=INFO -e VLLM_LOGGING_CONFIG_PATH=/root/.cache/pd-logging.json -e SERVING_BENCH_DSPARK_DP_PROFILE_FIX=1 --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model --served-model-name deepseek-v4-flash --host 0.0.0.0 --port 31249 --trust-remote-code --enable-auto-tool-choice --no-enable-prefix-caching --enable-expert-parallel --enable-chunked-prefill --jit-monitor-verbose --async-scheduling --tensor-parallel-size 2 --pipeline-parallel-size 1 --data-parallel-size 2 --max-model-len 32768 --max-num-seqs 32 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8_e4m3 --block-size 256 --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' --moe-backend auto --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' --kernel-config '{"enable_flashinfer_autotune":false}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[5,6,10,12,20,24,40,48,60,72,80,96,120,144,160,192],"max_cudagraph_capture_size":192}' --seed 0 --distributed-executor-backend mp --worker-cls dsv4_input_worker.Worker --data-parallel-size-local 2 --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":false}'
```

## r1 — gpu-6000d-47

```bash
docker run -d --pull never --name sb-o104-20260923-01-r1 --label io.serving-bench.run=o104-20260923-01 --network host --ipc host --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/vllm/ee8d157e54f4b168fc23:/root/.cache:rw --cpuset-cpus 32-47 --cpuset-mems 2 --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --cap-add SYS_NICE --device /dev/infiniband -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONPATH=/root/.cache/dsv4-input-warmup:/root/.cache/dspark-native-mxfp4:/root/.cache/dsv4-mhc-warmup -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_NIXL_SIDE_CHANNEL_HOST=10.90.1.47 -e VLLM_NIXL_SIDE_CHANNEL_PORT=26300 -e UCX_NET_DEVICES=mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1 -e UCX_TLS=rc,cuda_copy,cuda_ipc,sm,self -e UCX_IB_GID_INDEX=3 -e UCX_LOG_LEVEL=info -e NIXL_LOG_LEVEL=INFO -e VLLM_LOGGING_CONFIG_PATH=/root/.cache/pd-logging.json -e SERVING_BENCH_DSPARK_DP_PROFILE_FIX=1 --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model --served-model-name deepseek-v4-flash --host 0.0.0.0 --port 31249 --trust-remote-code --enable-auto-tool-choice --no-enable-prefix-caching --enable-expert-parallel --enable-chunked-prefill --jit-monitor-verbose --async-scheduling --tensor-parallel-size 2 --pipeline-parallel-size 1 --data-parallel-size 2 --max-model-len 32768 --max-num-seqs 32 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8_e4m3 --block-size 256 --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' --moe-backend auto --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' --kernel-config '{"enable_flashinfer_autotune":false}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[5,6,10,12,20,24,40,48,60,72,80,96,120,144,160,192],"max_cudagraph_capture_size":192}' --seed 0 --distributed-executor-backend mp --worker-cls dsv4_input_worker.Worker --data-parallel-size-local 2 --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":false}'
```

## r2 — gpu-6000d-48

```bash
docker run -d --pull never --name sb-o104-20260923-01-r2 --label io.serving-bench.run=o104-20260923-01 --network host --ipc host --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/vllm/810cd4ce21052b2fde1b:/root/.cache:rw --cpuset-cpus 32-47 --cpuset-mems 2 --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --cap-add SYS_NICE --device /dev/infiniband -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONPATH=/root/.cache/dsv4-input-warmup:/root/.cache/dspark-native-mxfp4:/root/.cache/dsv4-mhc-warmup -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_NIXL_SIDE_CHANNEL_HOST=10.90.1.48 -e VLLM_NIXL_SIDE_CHANNEL_PORT=26300 -e UCX_NET_DEVICES=mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1 -e UCX_TLS=rc,cuda_copy,cuda_ipc,sm,self -e UCX_IB_GID_INDEX=3 -e UCX_LOG_LEVEL=info -e NIXL_LOG_LEVEL=INFO -e VLLM_LOGGING_CONFIG_PATH=/root/.cache/pd-logging.json -e SERVING_BENCH_DSPARK_DP_PROFILE_FIX=1 --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model --served-model-name deepseek-v4-flash --host 0.0.0.0 --port 31249 --trust-remote-code --enable-auto-tool-choice --no-enable-prefix-caching --enable-expert-parallel --enable-chunked-prefill --jit-monitor-verbose --async-scheduling --tensor-parallel-size 2 --pipeline-parallel-size 1 --data-parallel-size 2 --max-model-len 32768 --max-num-seqs 32 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8_e4m3 --block-size 256 --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' --moe-backend auto --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' --kernel-config '{"enable_flashinfer_autotune":false}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[5,6,10,12,20,24,40,48,60,72,80,96,120,144,160,192],"max_cudagraph_capture_size":192}' --seed 0 --distributed-executor-backend mp --worker-cls dsv4_input_worker.Worker --data-parallel-size-local 2 --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":false}'
```

## r3 — gpu-6000d-46

```bash
docker run -d --pull never --name sb-o104-20260923-01-r3 --label io.serving-bench.run=o104-20260923-01 --network host --ipc host --gpus '"device=0,1,2,3"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/vllm/249d7e89c3ccee9dd0f8:/root/.cache:rw --cpuset-cpus 0-15 --cpuset-mems 0 --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --cap-add SYS_NICE --device /dev/infiniband -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONPATH=/root/.cache/dsv4-input-warmup:/root/.cache/dspark-native-mxfp4:/root/.cache/dsv4-mhc-warmup -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_NIXL_SIDE_CHANNEL_HOST=10.90.1.46 -e VLLM_NIXL_SIDE_CHANNEL_PORT=26301 -e UCX_NET_DEVICES=mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1 -e UCX_TLS=rc,cuda_copy,cuda_ipc,sm,self -e UCX_IB_GID_INDEX=3 -e UCX_LOG_LEVEL=info -e NIXL_LOG_LEVEL=INFO -e VLLM_LOGGING_CONFIG_PATH=/root/.cache/pd-logging.json -e SERVING_BENCH_DSPARK_DP_PROFILE_FIX=1 --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model --served-model-name deepseek-v4-flash --host 0.0.0.0 --port 31250 --trust-remote-code --enable-auto-tool-choice --no-enable-prefix-caching --enable-expert-parallel --enable-chunked-prefill --jit-monitor-verbose --async-scheduling --tensor-parallel-size 2 --pipeline-parallel-size 1 --data-parallel-size 2 --max-model-len 32768 --max-num-seqs 32 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8_e4m3 --block-size 256 --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' --moe-backend auto --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' --kernel-config '{"enable_flashinfer_autotune":false}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[5,6,10,12,20,24,40,48,60,72,80,96,120,144,160,192],"max_cudagraph_capture_size":192}' --seed 0 --distributed-executor-backend mp --worker-cls dsv4_input_worker.Worker --data-parallel-size-local 2 --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":false}'
```

## 代理

```bash
docker run -d --pull never --name o104-20260923-01-proxy --label io.serving-bench.run=o104-20260923-01 --network host --cpuset-cpus 16-19 --cpuset-mems 1 -v /home/enhui/llm-serving-benchmarks/src:/src:ro -e PYTHONPATH=/src --entrypoint python3 sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 -m serving_bench.pd_proxy --port 31250 --timeout 600 --ordinary http://10.90.1.46:31249 http://10.90.1.47:31249 http://10.90.1.48:31249 http://10.90.1.46:31250 --ordinary-policy least-inflight
```

## 208条预热

```bash
docker run -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-ordinary-c104-20260923/hooks/request_records.py:/request_records.py:ro --pull never --name o104-20260923-01-client-c104-warmup-01 --label io.serving-bench.run=o104-20260923-01 --network host -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-ordinary-c104-20260923/results/O104-01/benchmark/c104/warmup-01:/results:rw -v /home/enhui/llm-serving-benchmarks/src/serving_bench/clients/jsonl_dataset.py:/jsonl_dataset.py:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-ordinary-c104-20260923/data/govreport-16k-prefix-1024-v2.jsonl:/dataset.jsonl:ro --cpuset-cpus 20-23 --cpuset-mems 1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 --entrypoint python3 sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 -c 'import runpy; runpy.run_path('"'"'/request_records.py'"'"')
import re, json
from pathlib import Path
def id_set(value: str) -> set[int]:
    """Parse the Linux CPU/node list syntax used by Docker and /proc."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", value):
        raise ValueError("Expected a quoted CPU/node list, e.g. '"'"'0-3,8'"'"'")
    result = set()
    for part in value.split(","):
        ends = part.split("-")
        start, end = int(ends[0]), int(ends[-1])
        if end < start or end > 1048575:
            raise ValueError("Invalid CPU/node range")
        items = set(range(start, end + 1))
        if result & items:
            raise ValueError("Duplicate CPU/node IDs")
        result.update(items)
    return result

def snapshot() -> dict:
    """Snapshot visible threads, including workers; vanished threads are normal."""
    result = {"threads": [], "unreadable": []}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            tasks = list((proc / "task").iterdir())
        except FileNotFoundError:
            continue
        except PermissionError:
            result["unreadable"].append(str(proc))
            continue
        for task in tasks:
            try:
                status = dict(line.split(":", 1) for line in (task / "status").read_text().splitlines() if ":" in line)
                result["threads"].append({"pid": int(proc.name), "tid": int(task.name),
                    "name": status["Name"].strip(), "cpus": status["Cpus_allowed_list"].strip(),
                    "mems": status["Mems_allowed_list"].strip()})
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError:
                result["unreadable"].append(str(task))
    return result

def verify_snapshot(binding: dict, observed: dict) -> None:
    if observed["unreadable"] or not observed["threads"]:
        raise ValueError("Cannot verify container thread affinity")
    expected = {key: id_set(binding[key]) for key in ("cpus", "mems")}
    for thread in observed["threads"]:
        for key in expected:
            # Engines may narrow worker affinity within the container allocation.
            actual = id_set(thread[key])
            if not actual <= expected[key]:
                raise ValueError(f"Thread {thread['"'"'tid'"'"']} {key} escapes requested binding: {thread[key]}")

_binding = {'"'"'cpus'"'"': '"'"'20-23'"'"', '"'"'mems'"'"': '"'"'1'"'"'}
def record_affinity(stage):
    observed = snapshot()
    Path('"'"'/results/client-affinity-'"'"' + stage + '"'"'.json'"'"').write_text(json.dumps(observed, indent=2))
    verify_snapshot(_binding, observed)
record_affinity('"'"'start'"'"')
try:
    exec("import runpy\nfrom pathlib import Path\nimport vllm.benchmarks.serve as serve\nrunpy.run_path('"'"'/jsonl_dataset.py'"'"')['"'"'install'"'"'](serve, {'"'"'name'"'"': '"'"'jsonl'"'"', '"'"'path'"'"': '"'"'data/govreport-16k-prefix-1024-v2.jsonl'"'"', '"'"'max_input_tokens'"'"': 16384, '"'"'output_tokens'"'"': 1024, '"'"'sha256'"'"': '"'"'025246b18d8ba7e1bc4d98c1a20b32b570860bb06a0baf0cea4ae97403d982e7'"'"'}, Path('"'"'/results'"'"'), 0, 1)\nfrom vllm.benchmarks.serve import add_cli_args, main; from vllm.utils.argparse_utils import FlexibleArgumentParser; p=FlexibleArgumentParser(); add_cli_args(p); main(p.parse_args())")
finally:
    record_affinity('"'"'end'"'"')
' --backend vllm --base-url http://127.0.0.1:31250 --endpoint /v1/completions --model /model --tokenizer /model --served-model-name deepseek-v4-flash --num-prompts 208 --max-concurrency 104 --request-rate inf --num-warmups 0 --seed 0 --temperature 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 --save-result --result-dir /results --result-filename raw.json --dataset-name custom --dataset-path /dataset.jsonl --custom-output-len 1024 --skip-chat-template --disable-shuffle --no-oversample --ignore-eos --trust-remote-code --goodput ttft:10000 tpot:50 --save-detailed
```

## 三轮各416条正式：第一轮命令

```bash
docker run -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-ordinary-c104-20260923/hooks/request_records.py:/request_records.py:ro --pull never --name o104-20260923-01-client-c104-measurement-01 --label io.serving-bench.run=o104-20260923-01 --network host -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-ordinary-c104-20260923/results/O104-01/benchmark/c104/measurement-01:/results:rw -v /home/enhui/llm-serving-benchmarks/src/serving_bench/clients/jsonl_dataset.py:/jsonl_dataset.py:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-ordinary-c104-20260923/data/govreport-16k-prefix-1024-v2.jsonl:/dataset.jsonl:ro --cpuset-cpus 20-23 --cpuset-mems 1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 --entrypoint python3 sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 -c 'import runpy; runpy.run_path('"'"'/request_records.py'"'"')
import re, json
from pathlib import Path
def id_set(value: str) -> set[int]:
    """Parse the Linux CPU/node list syntax used by Docker and /proc."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", value):
        raise ValueError("Expected a quoted CPU/node list, e.g. '"'"'0-3,8'"'"'")
    result = set()
    for part in value.split(","):
        ends = part.split("-")
        start, end = int(ends[0]), int(ends[-1])
        if end < start or end > 1048575:
            raise ValueError("Invalid CPU/node range")
        items = set(range(start, end + 1))
        if result & items:
            raise ValueError("Duplicate CPU/node IDs")
        result.update(items)
    return result

def snapshot() -> dict:
    """Snapshot visible threads, including workers; vanished threads are normal."""
    result = {"threads": [], "unreadable": []}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            tasks = list((proc / "task").iterdir())
        except FileNotFoundError:
            continue
        except PermissionError:
            result["unreadable"].append(str(proc))
            continue
        for task in tasks:
            try:
                status = dict(line.split(":", 1) for line in (task / "status").read_text().splitlines() if ":" in line)
                result["threads"].append({"pid": int(proc.name), "tid": int(task.name),
                    "name": status["Name"].strip(), "cpus": status["Cpus_allowed_list"].strip(),
                    "mems": status["Mems_allowed_list"].strip()})
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError:
                result["unreadable"].append(str(task))
    return result

def verify_snapshot(binding: dict, observed: dict) -> None:
    if observed["unreadable"] or not observed["threads"]:
        raise ValueError("Cannot verify container thread affinity")
    expected = {key: id_set(binding[key]) for key in ("cpus", "mems")}
    for thread in observed["threads"]:
        for key in expected:
            # Engines may narrow worker affinity within the container allocation.
            actual = id_set(thread[key])
            if not actual <= expected[key]:
                raise ValueError(f"Thread {thread['"'"'tid'"'"']} {key} escapes requested binding: {thread[key]}")

_binding = {'"'"'cpus'"'"': '"'"'20-23'"'"', '"'"'mems'"'"': '"'"'1'"'"'}
def record_affinity(stage):
    observed = snapshot()
    Path('"'"'/results/client-affinity-'"'"' + stage + '"'"'.json'"'"').write_text(json.dumps(observed, indent=2))
    verify_snapshot(_binding, observed)
record_affinity('"'"'start'"'"')
try:
    exec("import runpy\nfrom pathlib import Path\nimport vllm.benchmarks.serve as serve\nrunpy.run_path('"'"'/jsonl_dataset.py'"'"')['"'"'install'"'"'](serve, {'"'"'name'"'"': '"'"'jsonl'"'"', '"'"'path'"'"': '"'"'data/govreport-16k-prefix-1024-v2.jsonl'"'"', '"'"'max_input_tokens'"'"': 16384, '"'"'output_tokens'"'"': 1024, '"'"'sha256'"'"': '"'"'025246b18d8ba7e1bc4d98c1a20b32b570860bb06a0baf0cea4ae97403d982e7'"'"'}, Path('"'"'/results'"'"'), 0, 1)\nfrom vllm.benchmarks.serve import add_cli_args, main; from vllm.utils.argparse_utils import FlexibleArgumentParser; p=FlexibleArgumentParser(); add_cli_args(p); main(p.parse_args())")
finally:
    record_affinity('"'"'end'"'"')
' --backend vllm --base-url http://127.0.0.1:31250 --endpoint /v1/completions --model /model --tokenizer /model --served-model-name deepseek-v4-flash --num-prompts 416 --max-concurrency 104 --request-rate inf --num-warmups 0 --seed 0 --temperature 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 --save-result --result-dir /results --result-filename raw.json --dataset-name custom --dataset-path /dataset.jsonl --custom-output-len 1024 --skip-chat-template --disable-shuffle --no-oversample --ignore-eos --trust-remote-code --goodput ttft:10000 tpot:50 --save-detailed
```
