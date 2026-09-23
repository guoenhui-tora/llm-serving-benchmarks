# 16K/1 EP on/off实际命令

以下从46实测`argv.json`机械导出；原始绝对路径仅46本机可用。复现使用[工作区准备入口](../reproduction/prefill-ep-20260923/README.md)创建新owner、端口检查和独立缓存。

## EP on：服务

```bash
docker run -d --pull never --name sb-prefill-ep-on-scan-01-server --label io.serving-bench.run=prefill-ep-on-scan-01 --network host --ipc host --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-prefill-ep-20260923/cache/on/vllm/4bc1d0883cb10d18149e:/root/.cache:rw --cpuset-cpus 32-47 --cpuset-mems 2 --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --cap-add SYS_NICE -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 -e PYTHONPATH=/root/.cache/dsv4-input-warmup:/root/.cache/dspark-native-mxfp4:/root/.cache/dsv4-mhc-warmup -e SERVING_BENCH_DSPARK_DP_PROFILE_FIX=1 -e VLLM_LOGGING_CONFIG_PATH=/root/.cache/pd-logging.json --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model --served-model-name deepseek-v4-flash --host 127.0.0.1 --port 31349 --trust-remote-code --enable-auto-tool-choice --no-enable-prefix-caching --enable-expert-parallel --enable-chunked-prefill --jit-monitor-verbose --async-scheduling --tensor-parallel-size 2 --pipeline-parallel-size 1 --data-parallel-size 2 --max-model-len 32768 --max-num-seqs 32 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8_e4m3 --block-size 256 --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' --moe-backend auto --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' --kernel-config '{"enable_flashinfer_autotune":false}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[5,6,10,12,20,24,40,48,60,72,80,96,120,144,160,192],"max_cudagraph_capture_size":192}' --seed 0 --distributed-executor-backend mp --worker-cls dsv4_input_worker.Worker --data-parallel-size-local 2 --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":false}' --data-parallel-rpc-port 31400
```

## EP on：C16预热64请求

```bash
docker run -v /home/enhui/llm-serving-benchmarks/src:/benchsrc:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-prefill-ep-20260923/scripts:/hooks:ro --pull never --name sb-prefill-ep-on-scan-01-client --label io.serving-bench.run=prefill-ep-on-scan-01 --network host -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-prefill-ep-20260923/results/scan-01/on/trials/c16/warmup-01:/results:rw -v /home/enhui/llm-serving-benchmarks/src/serving_bench/clients/jsonl_dataset.py:/jsonl_dataset.py:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-prefill-ep-20260923/data/govreport-16k-prefix-1-v1.jsonl:/dataset.jsonl:ro --cpuset-cpus 48-51 --cpuset-mems 3 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 --entrypoint python3 sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 -c 'import sys; sys.path.insert(0,'"'"'/benchsrc'"'"'); sys.path.insert(0,'"'"'/hooks'"'"'); import vllm.benchmarks.serve as serve; from client_observe import install; install(serve,31349)
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

_binding = {'"'"'cpus'"'"': '"'"'48-51'"'"', '"'"'mems'"'"': '"'"'3'"'"'}
def record_affinity(stage):
    observed = snapshot()
    Path('"'"'/results/client-affinity-'"'"' + stage + '"'"'.json'"'"').write_text(json.dumps(observed, indent=2))
    verify_snapshot(_binding, observed)
record_affinity('"'"'start'"'"')
try:
    exec("import runpy\nfrom pathlib import Path\nimport vllm.benchmarks.serve as serve\nrunpy.run_path('"'"'/jsonl_dataset.py'"'"')['"'"'install'"'"'](serve, {'"'"'name'"'"': '"'"'jsonl'"'"', '"'"'path'"'"': '"'"'data/govreport-16k-prefix-1-v1.jsonl'"'"', '"'"'max_input_tokens'"'"': 16384, '"'"'output_tokens'"'"': 1, '"'"'sha256'"'"': '"'"'4effcb83c8d3203e53f3c81b61537e5d24da8c814263539f2aa29cc100266101'"'"'}, Path('"'"'/results'"'"'), 0, 1)\nfrom vllm.benchmarks.serve import add_cli_args, main; from vllm.utils.argparse_utils import FlexibleArgumentParser; p=FlexibleArgumentParser(); add_cli_args(p); main(p.parse_args())")
finally:
    record_affinity('"'"'end'"'"')
' --backend vllm --base-url http://127.0.0.1:31349 --endpoint /v1/completions --model /model --tokenizer /model --served-model-name deepseek-v4-flash --num-prompts 64 --max-concurrency 16 --request-rate inf --num-warmups 0 --seed 0 --temperature 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 --save-result --result-dir /results --result-filename raw.json --dataset-name custom --dataset-path /dataset.jsonl --custom-output-len 1 --skip-chat-template --disable-shuffle --no-oversample --ignore-eos --trust-remote-code --save-detailed
```

## EP on：C16正式256请求

```bash
docker run -v /home/enhui/llm-serving-benchmarks/src:/benchsrc:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-prefill-ep-20260923/scripts:/hooks:ro --pull never --name sb-prefill-ep-on-scan-01-client --label io.serving-bench.run=prefill-ep-on-scan-01 --network host -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-prefill-ep-20260923/results/scan-01/on/trials/c16/measurement-01:/results:rw -v /home/enhui/llm-serving-benchmarks/src/serving_bench/clients/jsonl_dataset.py:/jsonl_dataset.py:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-prefill-ep-20260923/data/govreport-16k-prefix-1-v1.jsonl:/dataset.jsonl:ro --cpuset-cpus 48-51 --cpuset-mems 3 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 --entrypoint python3 sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 -c 'import sys; sys.path.insert(0,'"'"'/benchsrc'"'"'); sys.path.insert(0,'"'"'/hooks'"'"'); import vllm.benchmarks.serve as serve; from client_observe import install; install(serve,31349)
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

_binding = {'"'"'cpus'"'"': '"'"'48-51'"'"', '"'"'mems'"'"': '"'"'3'"'"'}
def record_affinity(stage):
    observed = snapshot()
    Path('"'"'/results/client-affinity-'"'"' + stage + '"'"'.json'"'"').write_text(json.dumps(observed, indent=2))
    verify_snapshot(_binding, observed)
record_affinity('"'"'start'"'"')
try:
    exec("import runpy\nfrom pathlib import Path\nimport vllm.benchmarks.serve as serve\nrunpy.run_path('"'"'/jsonl_dataset.py'"'"')['"'"'install'"'"'](serve, {'"'"'name'"'"': '"'"'jsonl'"'"', '"'"'path'"'"': '"'"'data/govreport-16k-prefix-1-v1.jsonl'"'"', '"'"'max_input_tokens'"'"': 16384, '"'"'output_tokens'"'"': 1, '"'"'sha256'"'"': '"'"'4effcb83c8d3203e53f3c81b61537e5d24da8c814263539f2aa29cc100266101'"'"'}, Path('"'"'/results'"'"'), 0, 1)\nfrom vllm.benchmarks.serve import add_cli_args, main; from vllm.utils.argparse_utils import FlexibleArgumentParser; p=FlexibleArgumentParser(); add_cli_args(p); main(p.parse_args())")
finally:
    record_affinity('"'"'end'"'"')
' --backend vllm --base-url http://127.0.0.1:31349 --endpoint /v1/completions --model /model --tokenizer /model --served-model-name deepseek-v4-flash --num-prompts 256 --max-concurrency 16 --request-rate inf --num-warmups 0 --seed 0 --temperature 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 --save-result --result-dir /results --result-filename raw.json --dataset-name custom --dataset-path /dataset.jsonl --custom-output-len 1 --skip-chat-template --disable-shuffle --no-oversample --ignore-eos --trust-remote-code --save-detailed
```

## EP off：服务

```bash
docker run -d --pull never --name sb-prefill-ep-off-scan-01-server --label io.serving-bench.run=prefill-ep-off-scan-01 --network host --ipc host --gpus '"device=0,1,2,3"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-prefill-ep-20260923/cache/off/vllm/27bdb3834f168d5c8d8b:/root/.cache:rw --cpuset-cpus 0-15 --cpuset-mems 0 --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --cap-add SYS_NICE -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 -e PYTHONPATH=/root/.cache/dsv4-input-warmup:/root/.cache/dspark-native-mxfp4:/root/.cache/dsv4-mhc-warmup -e SERVING_BENCH_DSPARK_DP_PROFILE_FIX=1 -e VLLM_LOGGING_CONFIG_PATH=/root/.cache/pd-logging.json --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model --served-model-name deepseek-v4-flash --host 127.0.0.1 --port 31350 --trust-remote-code --enable-auto-tool-choice --no-enable-prefix-caching --enable-chunked-prefill --jit-monitor-verbose --async-scheduling --tensor-parallel-size 2 --pipeline-parallel-size 1 --data-parallel-size 2 --max-model-len 32768 --max-num-seqs 32 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8_e4m3 --block-size 256 --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' --moe-backend auto --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' --kernel-config '{"enable_flashinfer_autotune":false}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[5,6,10,12,20,24,40,48,60,72,80,96,120,144,160,192],"max_cudagraph_capture_size":192}' --seed 0 --distributed-executor-backend mp --worker-cls dsv4_input_worker.Worker --data-parallel-size-local 2 --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":false}' --data-parallel-rpc-port 31500
```

## EP off：C16预热64请求

```bash
docker run -v /home/enhui/llm-serving-benchmarks/src:/benchsrc:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-prefill-ep-20260923/scripts:/hooks:ro --pull never --name sb-prefill-ep-off-scan-01-client --label io.serving-bench.run=prefill-ep-off-scan-01 --network host -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-prefill-ep-20260923/results/scan-01/off/trials/c16/warmup-01:/results:rw -v /home/enhui/llm-serving-benchmarks/src/serving_bench/clients/jsonl_dataset.py:/jsonl_dataset.py:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-prefill-ep-20260923/data/govreport-16k-prefix-1-v1.jsonl:/dataset.jsonl:ro --cpuset-cpus 16-19 --cpuset-mems 1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 --entrypoint python3 sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 -c 'import sys; sys.path.insert(0,'"'"'/benchsrc'"'"'); sys.path.insert(0,'"'"'/hooks'"'"'); import vllm.benchmarks.serve as serve; from client_observe import install; install(serve,31350)
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

_binding = {'"'"'cpus'"'"': '"'"'16-19'"'"', '"'"'mems'"'"': '"'"'1'"'"'}
def record_affinity(stage):
    observed = snapshot()
    Path('"'"'/results/client-affinity-'"'"' + stage + '"'"'.json'"'"').write_text(json.dumps(observed, indent=2))
    verify_snapshot(_binding, observed)
record_affinity('"'"'start'"'"')
try:
    exec("import runpy\nfrom pathlib import Path\nimport vllm.benchmarks.serve as serve\nrunpy.run_path('"'"'/jsonl_dataset.py'"'"')['"'"'install'"'"'](serve, {'"'"'name'"'"': '"'"'jsonl'"'"', '"'"'path'"'"': '"'"'data/govreport-16k-prefix-1-v1.jsonl'"'"', '"'"'max_input_tokens'"'"': 16384, '"'"'output_tokens'"'"': 1, '"'"'sha256'"'"': '"'"'4effcb83c8d3203e53f3c81b61537e5d24da8c814263539f2aa29cc100266101'"'"'}, Path('"'"'/results'"'"'), 0, 1)\nfrom vllm.benchmarks.serve import add_cli_args, main; from vllm.utils.argparse_utils import FlexibleArgumentParser; p=FlexibleArgumentParser(); add_cli_args(p); main(p.parse_args())")
finally:
    record_affinity('"'"'end'"'"')
' --backend vllm --base-url http://127.0.0.1:31350 --endpoint /v1/completions --model /model --tokenizer /model --served-model-name deepseek-v4-flash --num-prompts 64 --max-concurrency 16 --request-rate inf --num-warmups 0 --seed 0 --temperature 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 --save-result --result-dir /results --result-filename raw.json --dataset-name custom --dataset-path /dataset.jsonl --custom-output-len 1 --skip-chat-template --disable-shuffle --no-oversample --ignore-eos --trust-remote-code --save-detailed
```

## EP off：C16正式256请求

```bash
docker run -v /home/enhui/llm-serving-benchmarks/src:/benchsrc:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-prefill-ep-20260923/scripts:/hooks:ro --pull never --name sb-prefill-ep-off-scan-01-client --label io.serving-bench.run=prefill-ep-off-scan-01 --network host -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-prefill-ep-20260923/results/scan-01/off/trials/c16/measurement-01:/results:rw -v /home/enhui/llm-serving-benchmarks/src/serving_bench/clients/jsonl_dataset.py:/jsonl_dataset.py:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-prefill-ep-20260923/data/govreport-16k-prefix-1-v1.jsonl:/dataset.jsonl:ro --cpuset-cpus 16-19 --cpuset-mems 1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 --entrypoint python3 sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 -c 'import sys; sys.path.insert(0,'"'"'/benchsrc'"'"'); sys.path.insert(0,'"'"'/hooks'"'"'); import vllm.benchmarks.serve as serve; from client_observe import install; install(serve,31350)
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

_binding = {'"'"'cpus'"'"': '"'"'16-19'"'"', '"'"'mems'"'"': '"'"'1'"'"'}
def record_affinity(stage):
    observed = snapshot()
    Path('"'"'/results/client-affinity-'"'"' + stage + '"'"'.json'"'"').write_text(json.dumps(observed, indent=2))
    verify_snapshot(_binding, observed)
record_affinity('"'"'start'"'"')
try:
    exec("import runpy\nfrom pathlib import Path\nimport vllm.benchmarks.serve as serve\nrunpy.run_path('"'"'/jsonl_dataset.py'"'"')['"'"'install'"'"'](serve, {'"'"'name'"'"': '"'"'jsonl'"'"', '"'"'path'"'"': '"'"'data/govreport-16k-prefix-1-v1.jsonl'"'"', '"'"'max_input_tokens'"'"': 16384, '"'"'output_tokens'"'"': 1, '"'"'sha256'"'"': '"'"'4effcb83c8d3203e53f3c81b61537e5d24da8c814263539f2aa29cc100266101'"'"'}, Path('"'"'/results'"'"'), 0, 1)\nfrom vllm.benchmarks.serve import add_cli_args, main; from vllm.utils.argparse_utils import FlexibleArgumentParser; p=FlexibleArgumentParser(); add_cli_args(p); main(p.parse_args())")
finally:
    record_affinity('"'"'end'"'"')
' --backend vllm --base-url http://127.0.0.1:31350 --endpoint /v1/completions --model /model --tokenizer /model --served-model-name deepseek-v4-flash --num-prompts 256 --max-concurrency 16 --request-rate inf --num-warmups 0 --seed 0 --temperature 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 --save-result --result-dir /results --result-filename raw.json --dataset-name custom --dataset-path /dataset.jsonl --custom-output-len 1 --skip-chat-template --disable-shuffle --no-oversample --ignore-eos --trust-remote-code --save-detailed
```

其他C的客户端命令仅改变`--max-concurrency`及产物目录/容器名，N始终64预热／256正式，模型不重启。全部C实际argv保存在原始run-root，各配置见复现快照。

相对归档P0：移除NIXL connector、PD专用NIXL/UCX环境和设备挂载；监听127.0.0.1；独立HTTP/RPC端口、owner、GPU/CPU绑定和缓存路径；off另移除`--enable-expert-parallel`。TP2/DP2/PP1、S32、budget16384、maxlen32768、Graph192、K5完整配置及其余模型参数保留。
