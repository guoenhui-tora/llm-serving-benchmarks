# 普通双TP4恢复千级吞吐，1P1D收益仍未确认（2026-09-21）

**修正普通代理分流、复用已有缓存后，GovReport总C32的双TP4多轮达到1027–1033 output tok/s；第7轮1033.08无已知编译事件。旧682.09不能代表普通推理稳态能力。** 本批只完成普通对照诊断，尚未取得三轮clean，不重新宣称稳定基线；PD复测按前置条件跳过，不能宣称PD已加速。

[计划](pd-calibration-plan.md)于3e721e6提交后执行，公共路由修复为ac6b8f1。原[quick报告](pd-screen-20260920.md)和历史random结果保留。本次同时改变普通路由、缓存起点及测量协议，不能把全部改善单独归因于代理，也不能用新普通结果与旧PD结果计算可靠加速比。

## 具体修复及效果

旧代理按请求轮询，遇到某实例停顿时仍继续给它分配请求。根据旧start/done日志重建正式第1/2/3轮在途请求，最大失衡分别为31/1、3/29、25/7（总32）。这是代理在途HTTP请求，包含完整响应时间，不等于GPU活动序列数。历史同机双TP4则为两个同步客户端各固定C16。

增加显式`--ordinary-policy least-inflight`，以完整响应期间的在途数选副本；成功、上游错误或取消均释放计数。默认round-robin保留，PD流程不变。固定镜像内7项PD相关回归全部通过，新增真实回环HTTP测试覆盖慢流式响应期间连续分流与上游错误后释放；已有缺失元数据拒绝、完整输出预算转发测试仍通过。aiohttp的既有NotAppKeyWarning保留。

本次896条请求全部成功。`route`日志显示每实例最多16、全系统最多32；634个分派后总32的记录均为16/16。完成量46为454、48为442，是动态分流而非固定各半；并发均衡不等于token工作量均衡。每轮全局请求集仍固定前128条。

## 全部七轮

两节点各一个TP4普通服务、每服务四卡、总八卡，DSpark off。GovReport前128条、全系统C32、1024输出；每轮输入1,042,149、输出131,072 tokens。jit_clean记录每个完整轮次；事件轮次不进入clean统计。

| 轮次 | 输出tok/s | Mean TTFT秒 | Mean TPOT毫秒 | 事件日志行 | 实际开始编译日志行 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 871.32 | 4.214 | 29.942 | 76 | 4 |
| 2 | 1027.24 | 3.986 | 27.148 | 32 | 4 |
| 3 | 1026.61 | 3.997 | 27.199 | 24 | 0 |
| 4 | 876.62 | 4.237 | 29.735 | 40 | 8 |
| 5 | 1027.91 | 3.961 | 27.135 | 16 | 0 |
| 6 | 1030.34 | 4.041 | 27.057 | 8 | 0 |
| 7 | 1033.08 | 4.017 | 27.009 | 0 | 0 |

七轮整体诊断均值984.73±75.71 tok/s，CV7.69%，包含全部事件轮次，**不是接纳后的性能均值**。只有第7轮eligible；n=1不能提供三轮均值或稳定性结论。事件计数为日志行，包含各worker重复报告，不是独立kernel数量。

原计划各模式最多8轮或25分钟、取最先三轮clean，普通达标后才测PD。第6轮结束时clean为0，剩余两轮已无法完成三轮目标；第7轮已开始，故让其完整结束后用工作区STOP停止，不再执行第8轮。`reports/early-stop.json`在第7轮完成前记录决定，因而不是看到1033后挑选停止点。

原始collector将STOP记为`FAIL / User STOP`，控制器为STOPPED，外层为STOPPED_WITH_EVIDENCE；这是代理主动按剩余预算判断停止，**不是用户发送了停止指令，也不是模型请求失败**。这些原状态保持原样，不改判PASS或PARTIAL。外层mode状态停在PERFORMANCE是finally保留的未完成阶段，清理状态以控制器及最终实机核查为准。

## 为什么继续出现JIT事件

固定镜像源码及实测日志将主要剩余事件定位到DSV4 mHC的`mhc_pre_big_fuse_broadcast_with_norm_tilelang`和`mhc_pre_big_fuse_with_norm_tilelang`。`model_executor/kernels/mhc/tilelang.py`根据当前token批次计算`n_splits`；`tilelang_kernels.py:19`的`compute_num_split`先取SM数除以grid大小，再按K维度限制。`num_tokens`在kernel中是动态维度，但`n_splits`进入编译key。因此相同GovReport请求仍可因chunked-prefill调度出现未覆盖的参数组合。

例如普通46侧第6轮805-token运行形状触发`n_splits=12`，两种mHC函数各四个worker报告一次，共8行。前六轮仍逐步出现新的key，不能认为重复输入自然等于完整预热。第1/2/4轮另外分别有4/4/8行TileLang实际开始编译记录；后续仅有监控warning时，不将其一概等同于真实CPU重编译。

`utils/jit_monitor.py`对TileLang的监控先检查进程内`_kernel_cache`，未命中会在调用原函数前发warning，因此也可能覆盖磁盘缓存加载。当前保守gate保持原规则，没有隐藏warning或放宽验收。源码只读检查使用固定镜像无GPU容器，快照和哈希保存在本机工作区reports，未修改镜像、kernel或精度。

## 资源与验收

46/48各GPU4–7、服务CPU32–47/NUMA2；45代理CPU16–19/NUMA1、客户端CPU20–23/NUMA1。两服务TP4/PP1/DP1、EP off，FP8 KV、block256、上下文16384、prefill8192、maxseq32、显存比例0.9、V2/async/FULL_DECODE_ONLY；prefix cache和autotune关闭，与前批普通recipe逐字一致。此处prefill/decode目录名只是沿用目标名称，两端均做完整普通推理。

固定镜像`vllm/vllm-openai:v0.29.0`，实际ID `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`；模型`/data/models/DeepSeek-V4-Flash-0731-NVFP4`。启动前image ID和两端模型元数据/tokenizer哈希与前批一致；没有全量重算权重哈希。HTTP端口31249，45代理31250。普通服务不带connector；保留前批RDMA环境和设备参数不代表发生KV传输。

普通46/48分别复用缓存`c122390d01ef192d39f3`和`67b1eb1b36bade5f682b`，位于`/home/enhui/.cache/serving-bench/vllm/`；未改recipe身份、清缓存或换精度。解析命令与旧服务一致，启动日志和真实后端保留。

七轮含轮间间隔的遥测中，两节点整机CPU忙碌率约12.96%/12.93%，服务CPU集合约34.20%/33.54%，GPU利用率约81.18%/78.19%，平均SM频率约2416/2418 MHz。既有DOCA背景仍在；cpuset不代表独占，低频采样不能排除瞬时竞争。这些背景限制保留，不据此作吞吐补偿。

总896成功、0失败，输入7,295,043、输出917,504 tokens；服务成功计数与代理一致，两端computed-prefill增量合计7,295,043，抢占计数增量均0。没有额外初始生成请求。客户端完整计时包含45代理、跨节点HTTP、排队和推理。

生命周期为北京时间09月20日23:54:57至09月21日00:15:28，约20.52分钟；两服务启动至联合READY约101.32秒。没有重启或补测。源码/config与工作区脚本哈希均未变化，清理无错误；最终SSH核查46/48无GPU计算进程、45/46/48无运行容器，缓存保留。

## 接下来如何处理1P1D

当前值得采用的普通对照应保留均衡分流，不能再用旧682作为PD参照。下一步先用固定版本的有限`n_splits`组合设计定向预热，覆盖进程内加载和磁盘编译，独立保存预热耗时；这是候选适配，尚未实现或实测，不保证覆盖所有kernel。

随后只复测相同八卡、GovReport总C32的普通与1P1D，不重复整套功能筛查，也不扩DSpark或配比矩阵。若PD仍低于普通，再按P耗时、D利用率与NIXL传输等待选择一个具体调整；不能先假定PD一定优于普通或将历史random的1095作为硬性通过阈值。

[逐轮CSV](../data/pd-calibration-20260921.csv)、[证据JSON](../data/pd-calibration-20260921-evidence.json)。原始工作区`experiments/dsv4-pd-calibration/`、全部日志/缓存及源码只读快照仅本机可用。运行入口和预算见[计划](pd-calibration-plan.md)；下列命令直接从本次command.json导出。

## 实际命令

<details>
<summary>普通46服务</summary>

```bash
docker run -d --pull never --name sb-pd-calibration-20260920-ordinary-prefill --label io.serving-bench.run=pd-calibration-20260920-ordinary --network host --ipc host --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/vllm/c122390d01ef192d39f3:/root/.cache:rw --cpuset-cpus 32-47 --cpuset-mems 2 --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --cap-add SYS_NICE --device /dev/infiniband -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_NIXL_SIDE_CHANNEL_HOST=10.90.1.46 -e VLLM_NIXL_SIDE_CHANNEL_PORT=56300 -e UCX_NET_DEVICES=mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1 -e UCX_TLS=rc,cuda_copy,cuda_ipc,sm,self -e UCX_IB_GID_INDEX=3 -e UCX_LOG_LEVEL=info -e NIXL_LOG_LEVEL=INFO -e VLLM_LOGGING_CONFIG_PATH=/root/.cache/pd-logging.json --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model --served-model-name deepseek-v4-flash --host 0.0.0.0 --port 31249 --trust-remote-code --enable-auto-tool-choice --no-enable-prefix-caching --no-enable-expert-parallel --enable-chunked-prefill --jit-monitor-verbose --async-scheduling --tensor-parallel-size 4 --pipeline-parallel-size 1 --data-parallel-size 1 --max-model-len 16384 --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8_e4m3 --block-size 256 --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' --moe-backend auto --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' --kernel-config '{"enable_flashinfer_autotune":false}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,12,16,24,32],"max_cudagraph_capture_size":32}' --seed 0 --distributed-executor-backend mp
```

</details>

<details>
<summary>普通48服务</summary>

```bash
docker run -d --pull never --name sb-pd-calibration-20260920-ordinary-decode --label io.serving-bench.run=pd-calibration-20260920-ordinary --network host --ipc host --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/vllm/67b1eb1b36bade5f682b:/root/.cache:rw --cpuset-cpus 32-47 --cpuset-mems 2 --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --cap-add SYS_NICE --device /dev/infiniband -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_NIXL_SIDE_CHANNEL_HOST=10.90.1.48 -e VLLM_NIXL_SIDE_CHANNEL_PORT=56300 -e UCX_NET_DEVICES=mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1 -e UCX_TLS=rc,cuda_copy,cuda_ipc,sm,self -e UCX_IB_GID_INDEX=3 -e UCX_LOG_LEVEL=info -e NIXL_LOG_LEVEL=INFO -e VLLM_LOGGING_CONFIG_PATH=/root/.cache/pd-logging.json --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model --served-model-name deepseek-v4-flash --host 0.0.0.0 --port 31249 --trust-remote-code --enable-auto-tool-choice --no-enable-prefix-caching --no-enable-expert-parallel --enable-chunked-prefill --jit-monitor-verbose --async-scheduling --tensor-parallel-size 4 --pipeline-parallel-size 1 --data-parallel-size 1 --max-model-len 16384 --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8_e4m3 --block-size 256 --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' --moe-backend auto --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' --kernel-config '{"enable_flashinfer_autotune":false}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,12,16,24,32],"max_cudagraph_capture_size":32}' --seed 0 --distributed-executor-backend mp
```

</details>

<details>
<summary>45代理</summary>

```bash
docker run -d --pull never --name pd-calibration-20260920-ordinary-proxy --label io.serving-bench.run=pd-calibration-20260920-ordinary --network host --cpuset-cpus 16-19 --cpuset-mems 1 -v /home/enhui/llm-serving-benchmarks/src:/src:ro -e PYTHONPATH=/src --entrypoint python3 sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 -m serving_bench.pd_proxy --port 31250 --timeout 600 --ordinary http://10.90.1.46:31249 http://10.90.1.48:31249 --ordinary-policy least-inflight
```

</details>

<details>
<summary>第1轮客户端（其余轮次仅产物目录和容器名变化）</summary>

```bash
docker run --pull never --name pd-calibration-20260920-ordinary-client-measurement-01 --label io.serving-bench.run=pd-calibration-20260920-ordinary --network host -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-pd-calibration/results/ordinary-01/benchmark/measurement-01:/results:rw -v /home/enhui/llm-serving-benchmarks/src/serving_bench/clients/jsonl_dataset.py:/jsonl_dataset.py:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-pd-calibration/data/govreport-near8k.jsonl:/dataset.jsonl:ro --cpuset-cpus 20-23 --cpuset-mems 1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 --entrypoint python3 sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 -c 'import re, json
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
    exec("import runpy\nfrom pathlib import Path\nimport vllm.benchmarks.serve as serve\nrunpy.run_path('"'"'/jsonl_dataset.py'"'"')['"'"'install'"'"'](serve, {'"'"'name'"'"': '"'"'jsonl'"'"', '"'"'path'"'"': '"'"'data/govreport-near8k.jsonl'"'"', '"'"'max_input_tokens'"'"': 9216, '"'"'output_tokens'"'"': 1024, '"'"'sha256'"'"': '"'"'33bdbd36a8b335300974e1b07aa66b2c24c88cc4c4b55a21350aa3564a945d53'"'"'}, Path('"'"'/results'"'"'), 0, 1)\nfrom vllm.benchmarks.serve import add_cli_args, main; from vllm.utils.argparse_utils import FlexibleArgumentParser; p=FlexibleArgumentParser(); add_cli_args(p); main(p.parse_args())")
finally:
    record_affinity('"'"'end'"'"')
' --backend vllm --base-url http://127.0.0.1:31250 --endpoint /v1/completions --model /model --tokenizer /model --served-model-name deepseek-v4-flash --num-prompts 128 --max-concurrency 32 --request-rate inf --num-warmups 0 --seed 0 --temperature 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 --save-result --result-dir /results --result-filename raw.json --dataset-name custom --dataset-path /dataset.jsonl --custom-output-len 1024 --skip-chat-template --disable-shuffle --no-oversample --ignore-eos --trust-remote-code
```

</details>
