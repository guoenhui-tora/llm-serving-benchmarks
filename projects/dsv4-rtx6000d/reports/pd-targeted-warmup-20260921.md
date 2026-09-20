# 定向预热通过；同八卡1P1D比普通双TP4低11.21%（2026-09-21）

**已解决受测配置中已定位的mHC启动预热遗漏：普通双TP4和1P1D均在一轮完整HTTP预热后，连续取得三轮无已知编译事件的测量。** 普通1029.25±1.58、PD913.91±1.66 output tok/s，PD低11.21%。因此旧quick中PD相对682普通对照的表面优势不成立；当前4卡P＋4卡D在GovReport总C32下没有吞吐或平均延迟优势。

这是固定vLLM 0.29.0、DSV4 Flash NVFP4、RTX6000D、两节点各四卡TP4、EP/DSpark off、GovReport前128条近8K输入/1024输出的结果。两组均为同次启动的三轮clean，非独立重启重复或stable协议；不能推广到所有请求长度、并发、P/D配比或DSpark。前批失败/quick原记录保留，不合并统计。

## 同资源正式对照

| 模式 | GPU服务数×每服务GPU | 输出tok/s，均值±样本SD | CV | Mean TTFT秒 | Mean TPOT毫秒 | Mean E2E秒 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 普通双TP4，least-inflight | 2×4＝8 | 1029.25±1.58 | 0.15% | 4.002 | 27.085 | 31.710 |
| 1P1D，P/D各TP4 | 2×4＝8 | 913.91±1.66 | 0.18% | 4.226 | 28.905 | 33.796 |

客户端计时包含45代理、跨节点HTTP、P、KV传输、D及排队。总C32不是每服务C32；普通动态分派、PD同一外部请求先P后D。统计取预先协议接纳的最先三轮，未挑最快。P95如从CSV使用，为每轮请求分位数，不能把多个P95均值当作合并分位数。

| 模式 | 阶段/轮次 | 输出tok/s | 已知编译事件行 |
| --- | --- | ---: | ---: |
| 普通 | 独立HTTP预热 | 1028.48 | 16 |
| 普通 | 正式1 | 1027.50 | 0 |
| 普通 | 正式2 | 1029.71 | 0 |
| 普通 | 正式3 | 1030.56 | 0 |
| PD | 独立HTTP预热 | 914.35 | 0 |
| PD | 正式1 | 913.97 | 0 |
| PD | 正式2 | 912.22 | 0 |
| PD | 正式3 | 915.53 | 0 |

普通HTTP预热的16条事件来自首次使用的其他Triton内核，mHC事件为0；PD HTTP预热和全部正式轮次事件均为0。没有修改gate、压低日志级别或隐藏warning。日志窗口含客户端初始化及刷新，因此比纯benchmark计时略宽；事件数为日志行，不等于独立kernel数。

## 预热遗漏的原因与适配

固定镜像的`model_executor/warmup/deepseek_v4_mhc_warmup.py`确有mHC预热，但`_find_first_mhc_layer`要求层上存在`hc_pre/hc_post`。当前NVIDIA `vllm.models.deepseek_v4.nvidia.model.DeepseekV4DecoderLayer`没有这两个属性，forward直接调用broadcast和fused-post-pre TileLang路径，且融合RMSNorm，因此旧筛选提前返回。旧运行日志也未见该mHC预热启动记录。仅增加旧token列表不能解决这个入口错配。

新增[项目专用Worker适配](../patches/mhc-startup-warmup/README.md)，经固定镜像支持的`--worker-cls`加载；不替换forward、kernel实现、权重或数值选项。它在原worker启动预热前，以已加载真实层参数调用NVIDIA的`mhc_pre_broadcast_tilelang`和`mhc_fused_post_pre_tilelang`，随后完整执行父类的正常预热、图捕获和随机状态重置。临时输入使用BF16零张量；逐次检查输出有限，没有运行attention/MoE或改写模型参数。

预热按实际156 SM、hidden4096、hc_mult4和8192上限，枚举1–8192 token的分派组合，选各组合首尾及图捕获尺寸，共25组、55个尺寸。broadcast的n_splits覆盖1–16；with-norm覆盖1–15、17、19、22、26、31、39、52、64。每个worker均实际调用两遍，第二遍两个主mHC函数的进程内缓存key不增加，所需split全部覆盖。16个worker均完成，临时张量释放后的已分配显存增量均为0；这不表示CUDA allocator保留内存完全不变。

启动时校验worker、NVIDIA模型及两份mHC实现的SHA256；模型、TP/PP/DP、DSpark和token范围不匹配即拒绝。候选当前只验证TP4/PP1/DP1、DSpark off、max-batched-tokens≤8192，不能直接用于DP/PP/DSpark或更大预填充预算。

| 服务 | 每worker定向段秒，范围 | 所在两服务联合READY秒 |
| --- | ---: | ---: |
| 普通46 | 39.17–39.36 | 142.80 |
| 普通48 | 40.56–40.95 | 142.80 |
| P，46 | 150.70–151.12 | 462.42 |
| D，48 | 360.86–361.43 | 462.42 |

各worker/两节点并行，不能把这些耗时相加当总启动时间。定向段第二遍约几十毫秒，第一遍包含编译和加载；不同服务继承的旧缓存覆盖程度不同，不能视作同冷启动条件对比。D完整覆盖8192的成本较高，后续可验证只覆盖实际远端命中解码范围的方案，本轮未缩范围。启动中共享内存广播60秒等待提示保留，现场只读进程检查发现nvcc/cicc/ptxas，后续全部完成，未以等待提示冒充服务故障或忽略挂死。

## KV传输及真实工作量

每模式一轮独立HTTP预热＋三轮正式，每轮128成功、0失败，输入1,042,149、输出131,072 tokens。合计1024条外部请求，输入8,337,192、输出1,048,576；其中正式768条，预热256条。代理日志两模式各512 start/done，没有额外初始生成请求。PD另有512个内部P请求，不能重复算作外部完成量。

PD三轮正式384条对应1,536次rank传输、88,349,460,480 bytes、3,126,447远端命中tokens；NIXL传输/通知失败及KV过期计数增量均0，D的computed-prefill增量0。这里统计的0不表示完全没有末prompt-token计算；固定实现的完整命中路径通常重算末token。P计算prefill为3,126,447；普通两副本相加同为3,126,447。两模式均无抢占。联合证据排除整段输入重算的假PD。

含HTTP预热的PD512条对应2,048次rank传输、117,799,280,640 bytes、4,168,596远端命中tokens，D computed-prefill仍0。本轮没有重做完整语义等价或错配KV负向测试；此前有限功能screen边界继续适用，不能据此宣称生产完整兼容。

普通分流日志验证满总C32时每副本16；按在途数动态路由，完成数量未强制各半，不声称固定请求分片。原31/1失衡未复现。

## 性能差距的线索

正式NIXL单rank传输计时均值22.86毫秒，不能当作完整请求的传输关键路径或四rank求和时间。P代理阶段均值3.882秒，包含网络、队列和计算；每轮首32条均值约12.510秒，后96条约1.006秒，最大约23.450秒。首批排队和流水线填充对固定128条负载明显，不能把所有TTFT都归因于RDMA带宽。

代理从首请求至末响应的窗口普通约127.17–127.54秒，PD约143.14–143.66秒。由请求elapsed积分计算平均在途量，普通约31.87；PD外部总量约30.16，其中P阶段约3.46、P完成后至D响应约26.69。这些是HTTP阶段的在途数，包含等待和传输，不等于GPU活动序列数。

只取正式请求实际窗口的低频GPU采样，普通两端平均利用率约99.38%/97.31%，PD P/D约75.04%/94.02%；包含客户端初始化和轮间间隔的完整阶段则为83.06%/79.88%及62.87%/80.04%。D更忙、P首批队列明显，说明固定1:1分配、P批处理和有限请求的流水线填充值得研究；这些计数不能替代kernel profiler，也不能证明单一原因解释全部11.21%。

下一步优先只验证P端批处理预算是否能缩短首批排队，例如保留4卡P＋4卡D和同负载，仅将P的prefill预算8192改为16384。**这是候选，未实施。** 必须先扩展预热范围与覆盖验证、检查实际显存/CLI，在新recipe中测量；D保持原配置，并保留本批基线。若收益不足，再依据分解考虑D侧拓扑或增加D份额；不能把普通混合推理最优拓扑直接当纯P/D最优，也不直接铺矩阵或启用DSpark。

## 环境、执行与复现

执行提交9847a29，先提交[计划](pd-targeted-warmup-plan.md)及候选再启动。普通→条件PD各一次启动，没有失败重启或追加轮次。每worker定向段480秒上限、服务READY900秒、完整HTTP预热480秒、正式最多5轮/1200秒；均在原预算内完成。两个jit_clean均PASS，三轮CV均低于预定5%停止阈值；不是stable协议的2%窗口验收。

45统一控制，46/48各GPU4–7、服务CPU32–47/NUMA2；45代理CPU16–19/NUMA1、客户端CPU20–23/NUMA1。模型`/data/models/DeepSeek-V4-Flash-0731-NVFP4`；固定镜像`vllm/vllm-openai:v0.29.0`、ID `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`。两端镜像及模型元数据/tokenizer身份与前批一致，未全量重算权重内容哈希。

共同选项为TP4/PP1/DP1、EP/DSpark off、FP8 KV（实际fp8_ds_mla）、block256、上下文16384、prefill8192、maxseq32、显存0.9、V2/async/FULL_DECODE_ONLY，prefix cache和autotune关闭。服务HTTP31249，45代理31250；PD NIXL侧通道56300，mlx5_4–7、UCX/GID3及/dev/infiniband访问沿用前批。未换镜像、改精度、清JIT缓存或改宿主配置。

新recipe新增worker-cls，因此缓存key改变；各服务从对应旧缓存完整复制后验证文件内容清单哈希一致，再放入独立预热模块目录，未假称空目录为热缓存。四个源目录全部保留。复制/内容核对单服务约0.4–0.5秒，不含准备容器启动等外部开销。实际模块前后SHA256一致，运行期间源码/config及脚本均无变化。

正式阶段整机CPU忙碌率，普通两端约12.96%/12.94%，PD约15.53%/16.18%；服务CPU集合分别34.47%/33.26%和47.51%/59.15%。平均GPU时钟约2418–2422 MHz。既有DOCA背景保留，cpuset不是CPU独占；顺序运行及低频遥测不能排除全部短时干扰，未据此作吞吐补偿。

生命周期北京时间00:40:15至01:12:13，共31.96分钟，含启动、两轮HTTP预热、六轮正式及清理。普通/PD正式协议各458.64/507.90秒，包含客户端初始化和日志检查。两控制器STOPPED，changed_files/changed_scripts/cleanup_errors均为空；最终SSH核实46/48无GPU计算进程、45/46/48无运行容器，模型、镜像、缓存保留。

[逐轮CSV](../data/pd-targeted-warmup-20260921.csv)、[证据JSON](../data/pd-targeted-warmup-20260921-evidence.json)包含完整轮次、预热、分流、资源与计数。工作区`experiments/dsv4-pd-warmup/`及缓存仅本机可用；`reports/offline-plan.json`、`cache-preparation.json`、`final-module-check.json`、`image-source/`、各result下command/worker-warmup/protocol/server/proxy日志为复现证据。离线2项覆盖测试、固定镜像Worker导入通过；16 worker实机验证、完整负载及传输计数进一步确认适配，仅限本报告范围。

## 实际启动与客户端命令

命令均直接从本次command.json导出。缓存目录下的模块准备见[适配说明](../patches/mhc-startup-warmup/README.md)；普通模式目录名prefill/decode不代表服务只执行其中一个阶段。

<details>
<summary>ordinary / 46服务</summary>

```bash
docker run -d --pull never --name sb-pd-warmup-20260921-ordinary-prefill --label io.serving-bench.run=pd-warmup-20260921-ordinary --network host --ipc host --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/vllm/f1eaa4457a3e31d4eec2:/root/.cache:rw --cpuset-cpus 32-47 --cpuset-mems 2 --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --cap-add SYS_NICE --device /dev/infiniband -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONPATH=/root/.cache/dsv4-mhc-warmup -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_NIXL_SIDE_CHANNEL_HOST=10.90.1.46 -e VLLM_NIXL_SIDE_CHANNEL_PORT=56300 -e UCX_NET_DEVICES=mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1 -e UCX_TLS=rc,cuda_copy,cuda_ipc,sm,self -e UCX_IB_GID_INDEX=3 -e UCX_LOG_LEVEL=info -e NIXL_LOG_LEVEL=INFO -e VLLM_LOGGING_CONFIG_PATH=/root/.cache/pd-logging.json --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model --served-model-name deepseek-v4-flash --host 0.0.0.0 --port 31249 --trust-remote-code --enable-auto-tool-choice --no-enable-prefix-caching --no-enable-expert-parallel --enable-chunked-prefill --jit-monitor-verbose --async-scheduling --tensor-parallel-size 4 --pipeline-parallel-size 1 --data-parallel-size 1 --max-model-len 16384 --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8_e4m3 --block-size 256 --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' --moe-backend auto --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' --kernel-config '{"enable_flashinfer_autotune":false}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,12,16,24,32],"max_cudagraph_capture_size":32}' --seed 0 --distributed-executor-backend mp --worker-cls dsv4_mhc_worker.Worker
```

</details>

<details>
<summary>ordinary / 48服务</summary>

```bash
docker run -d --pull never --name sb-pd-warmup-20260921-ordinary-decode --label io.serving-bench.run=pd-warmup-20260921-ordinary --network host --ipc host --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/vllm/c9835f9daa398fe78954:/root/.cache:rw --cpuset-cpus 32-47 --cpuset-mems 2 --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --cap-add SYS_NICE --device /dev/infiniband -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONPATH=/root/.cache/dsv4-mhc-warmup -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_NIXL_SIDE_CHANNEL_HOST=10.90.1.48 -e VLLM_NIXL_SIDE_CHANNEL_PORT=56300 -e UCX_NET_DEVICES=mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1 -e UCX_TLS=rc,cuda_copy,cuda_ipc,sm,self -e UCX_IB_GID_INDEX=3 -e UCX_LOG_LEVEL=info -e NIXL_LOG_LEVEL=INFO -e VLLM_LOGGING_CONFIG_PATH=/root/.cache/pd-logging.json --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model --served-model-name deepseek-v4-flash --host 0.0.0.0 --port 31249 --trust-remote-code --enable-auto-tool-choice --no-enable-prefix-caching --no-enable-expert-parallel --enable-chunked-prefill --jit-monitor-verbose --async-scheduling --tensor-parallel-size 4 --pipeline-parallel-size 1 --data-parallel-size 1 --max-model-len 16384 --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8_e4m3 --block-size 256 --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' --moe-backend auto --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' --kernel-config '{"enable_flashinfer_autotune":false}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,12,16,24,32],"max_cudagraph_capture_size":32}' --seed 0 --distributed-executor-backend mp --worker-cls dsv4_mhc_worker.Worker
```

</details>

<details>
<summary>ordinary / 45代理</summary>

```bash
docker run -d --pull never --name pd-warmup-20260921-ordinary-proxy --label io.serving-bench.run=pd-warmup-20260921-ordinary --network host --cpuset-cpus 16-19 --cpuset-mems 1 -v /home/enhui/llm-serving-benchmarks/src:/src:ro -e PYTHONPATH=/src --entrypoint python3 sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 -m serving_bench.pd_proxy --port 31250 --timeout 600 --ordinary http://10.90.1.46:31249 http://10.90.1.48:31249 --ordinary-policy least-inflight
```

</details>

<details>
<summary>ordinary / 正式第1轮客户端</summary>

```bash
docker run --pull never --name pd-warmup-20260921-ordinary-client-measurement-01 --label io.serving-bench.run=pd-warmup-20260921-ordinary --network host -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-pd-warmup/results/ordinary-01/benchmark/measurement-01:/results:rw -v /home/enhui/llm-serving-benchmarks/src/serving_bench/clients/jsonl_dataset.py:/jsonl_dataset.py:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-pd-warmup/data/govreport-near8k.jsonl:/dataset.jsonl:ro --cpuset-cpus 20-23 --cpuset-mems 1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 --entrypoint python3 sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 -c 'import re, json
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

<details>
<summary>pd / 46服务</summary>

```bash
docker run -d --pull never --name sb-pd-warmup-20260921-pd-prefill --label io.serving-bench.run=pd-warmup-20260921-pd --network host --ipc host --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/vllm/5f8dd1544fc3b1fedc3e:/root/.cache:rw --cpuset-cpus 32-47 --cpuset-mems 2 --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --cap-add SYS_NICE --device /dev/infiniband -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONPATH=/root/.cache/dsv4-mhc-warmup -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_NIXL_SIDE_CHANNEL_HOST=10.90.1.46 -e VLLM_NIXL_SIDE_CHANNEL_PORT=56300 -e UCX_NET_DEVICES=mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1 -e UCX_TLS=rc,cuda_copy,cuda_ipc,sm,self -e UCX_IB_GID_INDEX=3 -e UCX_LOG_LEVEL=info -e NIXL_LOG_LEVEL=INFO -e VLLM_LOGGING_CONFIG_PATH=/root/.cache/pd-logging.json --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model --served-model-name deepseek-v4-flash --host 0.0.0.0 --port 31249 --trust-remote-code --enable-auto-tool-choice --no-enable-prefix-caching --no-enable-expert-parallel --enable-chunked-prefill --jit-monitor-verbose --async-scheduling --tensor-parallel-size 4 --pipeline-parallel-size 1 --data-parallel-size 1 --max-model-len 16384 --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8_e4m3 --block-size 256 --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' --moe-backend auto --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' --kernel-config '{"enable_flashinfer_autotune":false}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,12,16,24,32],"max_cudagraph_capture_size":32}' --seed 0 --distributed-executor-backend mp --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda","kv_load_failure_policy":"fail","kv_connector_extra_config":{"backends":["UCX"],"enforce_handshake_compat":true}}' --worker-cls dsv4_mhc_worker.Worker
```

</details>

<details>
<summary>pd / 48服务</summary>

```bash
docker run -d --pull never --name sb-pd-warmup-20260921-pd-decode --label io.serving-bench.run=pd-warmup-20260921-pd --network host --ipc host --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/vllm/9d9a9762f7c5c900d8dd:/root/.cache:rw --cpuset-cpus 32-47 --cpuset-mems 2 --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --cap-add SYS_NICE --device /dev/infiniband -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONPATH=/root/.cache/dsv4-mhc-warmup -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_NIXL_SIDE_CHANNEL_HOST=10.90.1.48 -e VLLM_NIXL_SIDE_CHANNEL_PORT=56300 -e UCX_NET_DEVICES=mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1 -e UCX_TLS=rc,cuda_copy,cuda_ipc,sm,self -e UCX_IB_GID_INDEX=3 -e UCX_LOG_LEVEL=info -e NIXL_LOG_LEVEL=INFO -e VLLM_LOGGING_CONFIG_PATH=/root/.cache/pd-logging.json --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model --served-model-name deepseek-v4-flash --host 0.0.0.0 --port 31249 --trust-remote-code --enable-auto-tool-choice --no-enable-prefix-caching --no-enable-expert-parallel --enable-chunked-prefill --jit-monitor-verbose --async-scheduling --tensor-parallel-size 4 --pipeline-parallel-size 1 --data-parallel-size 1 --max-model-len 16384 --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8_e4m3 --block-size 256 --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' --moe-backend auto --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' --kernel-config '{"enable_flashinfer_autotune":false}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,12,16,24,32],"max_cudagraph_capture_size":32}' --seed 0 --distributed-executor-backend mp --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cuda","kv_load_failure_policy":"fail","kv_connector_extra_config":{"backends":["UCX"],"enforce_handshake_compat":true}}' --worker-cls dsv4_mhc_worker.Worker
```

</details>

<details>
<summary>pd / 45代理</summary>

```bash
docker run -d --pull never --name pd-warmup-20260921-pd-proxy --label io.serving-bench.run=pd-warmup-20260921-pd --network host --cpuset-cpus 16-19 --cpuset-mems 1 -v /home/enhui/llm-serving-benchmarks/src:/src:ro -e PYTHONPATH=/src --entrypoint python3 sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 -m serving_bench.pd_proxy --port 31250 --timeout 600 --prefill http://10.90.1.46:31249 --decode http://10.90.1.48:31249
```

</details>

<details>
<summary>pd / 正式第1轮客户端</summary>

```bash
docker run --pull never --name pd-warmup-20260921-pd-client-measurement-01 --label io.serving-bench.run=pd-warmup-20260921-pd --network host -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-pd-warmup/results/pd-01/benchmark/measurement-01:/results:rw -v /home/enhui/llm-serving-benchmarks/src/serving_bench/clients/jsonl_dataset.py:/jsonl_dataset.py:ro -v /home/enhui/llm-serving-benchmarks/experiments/dsv4-pd-warmup/data/govreport-near8k.jsonl:/dataset.jsonl:ro --cpuset-cpus 20-23 --cpuset-mems 1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 --entrypoint python3 sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 -c 'import re, json
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
