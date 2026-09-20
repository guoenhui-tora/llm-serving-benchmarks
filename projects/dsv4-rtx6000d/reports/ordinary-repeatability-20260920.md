# 普通路径重复性诊断：长输入概率变化无需NIXL也可复现（2026-09-20）

**已把首轮输出差异缩小到普通长输入的计算结果变化：关闭NIXL connector后，相同8631-token输入仍产生不同的首token概率分布和输出。短输入的三次输出及返回logprobs完全一致。** 这排除了“必须有PD传输才会出现该现象”，但尚未锁定具体kernel、压缩缓存状态或调度原因，也不能排除PD另外引入问题。

本轮仅完成有界定位，不是数值正确性验收；[首轮1P1D](pd-1p1d-20260920.md)的严格输出检查FAIL及正确性INCONCLUSIVE保持不变。没有启动P、代理、并发压测或DSpark。

## 核心结果

| 检查（每项3次） | 原D配置，connector开启但普通请求 | 移除connector |
| --- | --- | --- |
| 原始长输入，256输出 | 3种不同输出 | 3种不同输出 |
| 长输入，32输出＋top5 logprobs | 3种不同输出，首token改变 | 3种不同输出，首token改变 |
| 相同长输入＋固定8-token前缀，输出1 | token均为“ before”，概率不同 | token均为“ before”，概率不同 |
| 5-token短输入，32输出＋top5 logprobs | 输出、token序列、返回logprobs三次完全一致 | 输出、token序列、返回logprobs三次完全一致 |
| 返回logprobs中选中非最高概率token | 0/195 | 0/195 |
| 远端缓存命中 / 实际KV传输 | 0 / 0 bytes | 0 / 未启用connector |
| 抢占计数 | 0 | 0 |

每组三次重复的请求SHA256一致，包含相同token IDs、温度及seed。原请求不带logprobs时已经复现，因此并非只有增加logprobs观测后才出现差异。相同输入在第一个输出token处就分叉，不能只解释为后续自回归误差逐步累积；固定前缀的下一token分布变化进一步说明，仅看最后文本是否相同会漏掉数值差异。

首token的两个共同候选为token 666（`**`）与9544（`Here`）。下表为 `log p(**) − log p(Here)`，正数偏向前者，负数偏向后者；使用概率比避免把归一化常数变化误当成候选排序变化。

| 配置 | 重复1 | 重复2 | 重复3 |
| --- | ---: | ---: | ---: |
| connector开启 | +1.75 | +0.75 | −2.00 |
| connector移除 | +4.00 | −0.50 | +1.25 |

这种候选排序反转不能仅用显示精度或末位舍入解释，但仍不足以归因到某个算子，或判断摘要语义质量损失。没有据此改精度、切镜像或放宽gate。

两组共26条生成请求，其中2条预热、24条诊断；累计172,698输入、2,438输出tokens，全部HTTP成功且token数符合约定，无OOM或服务traceback。每组computed-prefill总数86,349，等于全部输入；connector组传输字节与外部命中始终为0，证实测的是本地普通路径。

小型逐请求数据见[CSV](../data/ordinary-repeatability-20260920.csv)，哈希、首token top5、固定前缀、计数和清理结果见[证据JSON](../data/ordinary-repeatability-20260920-evidence.json)。CSV中的wall time只用于诊断，不是吞吐或性能重复结果。

## 协议与适用范围

本轮针对首轮PD检查中“同一D普通请求也不同”的反例，不测吞吐、不修改旧gate。node48独占本轮GPU4–7，TP4/PP1/DP1、EP off、DSpark off、NVFP4权重、FP8 KV、V2 runner、async、FULL_DECODE_ONLY、prefill预算8192、prefix cache与autotune关闭；服务CPU32–47、内存NUMA2。客户端在45以Python urllib串行发请求，未额外绑定CPU或内存。后台负载未隔离，保留CPU时间序列；cpuset不等于CPU独占。

先使用首轮原D recipe，但只直连普通请求、从不提供远端KV元数据；复现差异后，在同节点至多再启动一次移除kv-transfer-config的对照。两次启动间保持镜像、权重精度、GPU组和模型计算参数。生成的CLI差异除容器归属名称外，是移除connector选项和recipe对应缓存目录改变；没有删除或清空旧缓存。缓存命名空间差异及独立启动意味着不能用两组耗时解释connector性能开销。

每组C1、1条独立预热＋12条诊断请求，无重试：

| 阶段 | 输入 | 输出/请求 | 请求数 | 用途 |
| --- | --- | ---: | ---: | --- |
| warmup | GovReport首条8631 token IDs | 256 | 1 | 预热，单独保存 |
| original | 完全相同的原请求 | 256 | 3 | 无logprobs仪表的原现象复现 |
| logprobs | 同一8631 token IDs | 32 | 3 | logprobs=5，返回token IDs |
| fixed-prefix | 同一输入＋固定8个输出token，共8639 | 1 | 3 | 固定历史后比较下一token概率 |
| short | `The capital of France is`，5 tokens | 32 | 3 | 短输入参照 |

固定前缀来自connector组logprobs-00的前8个token，两组复用同一序列。所有请求为非流式、temperature=0、seed=0、ignore_eos=true、add_special_tokens=false；同组重复的序列化请求SHA256完全一致。每次保存请求全文、响应、时间和前后metrics；检查输入/输出计数与返回的logprobs有限性。生成请求以外仅有health/models和tokenize核查。

每次启动上限20分钟，请求阶段10分钟，收尾预留5分钟；HTTP失败、OOM、计数错误、非有限logprobs、服务异常或超时即停止。输出差异是本诊断测量的现象，按预定三次保存全部结果；DIAGNOSIS_COMPLETE仅表示诊断执行完毕，不表示模型正确性PASS。

## 源码核查

从固定镜像提取CompletionRequest、SamplingStates、V2 sampler及gumbel实现。显式temperature=0不会被generation_config默认temperature=1覆盖；greedy路径不叠加Gumbel随机噪声。固定镜像内仅CPU执行请求参数转换，也得到temperature=0、seed=0和greedy枚举。

实际logprobs逐token核查中，选中token概率均等于当次top5中的最高值。因此观察到的是相同长输入对应的概率分布变化，不是API文本渲染差异或已经观察到的“选中非argmax token”。这仍不证明所有采样/调度状态正确。

固定DSV4 compressor源码包含历史PDL读写竞争的注释，但该处已显式launch_pdl=False；不能仅根据相似症状认定本次就是该问题。未修改镜像内源码或增加模型补丁。

## 下一步建议（尚未执行）

后续用户选择优先推进PD，未执行下述分块定位；已按[补充计划](pd-screen-plan.md)完成[相对普通波动的有限功能检查与同资源quick对照](pd-screen-20260920.md)。普通非确定性的根因仍未确定，不再要求先消除它才开始PD性能研究。

保持原长输入和FP8/NVFP4，优先对照prefill预算8192与16384：8631-token输入在前者至少跨两个调度步，后者预算允许单步prefill；须由日志/调度证据确认实际分块。先看同一前缀下一token的重复logprobs是否随分块方式改变，再考虑压缩缓存状态、长输入kernel或并行计算的进一步定位。该对照改变执行形状，不能直接认定chunked prefill是原因。

本轮当时保留首轮PD正确性未定，未开展C32性能测试。后续采用独立的有限功能检查支持性能探索；旧FAIL和普通非确定性warning仍保留，不以“非确定性正常”为由直接忽略缓存错误。

## 运行身份、启动命令与原始证据

执行Git版本为 `34aa690`，没有公共源码修改。固定镜像 `vllm/vllm-openai:v0.29.0`，image ID `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`。两次运行的模型config、tokenizer及权重索引哈希与首轮一致，未全量重算权重内容哈希。

connector组启动110.20秒、生命周期147.07秒；移除connector组启动316.45秒、生命周期371.40秒。两组均保留预热期间的JIT warning，不声称首次预热消除了所有编译或不同执行形状。两次运行都在预算内主动结束，changed_files与cleanup_errors均为空；收尾SSH确认本轮两容器均不存在，48无GPU计算进程，缓存保留。

以下命令从各run-root的 `decode/command.json` 导出，作为实际运行快照；环境检查、日志文件准备和owner清理仍由实验控制器及外围诊断脚本完成，不能只复制命令替代实验协议。

### connector

```bash
docker run -d --pull never --name sb-pd-repeat-20260920-connector-decode --label io.serving-bench.run=pd-repeat-20260920-connector --network host --ipc host --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/vllm/a8a97ce074e2ef4f67e7:/root/.cache:rw --cpuset-cpus 32-47 --cpuset-mems 2 --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --cap-add SYS_NICE --device /dev/infiniband -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_NIXL_SIDE_CHANNEL_HOST=10.90.1.48 -e VLLM_NIXL_SIDE_CHANNEL_PORT=56300 -e UCX_NET_DEVICES=mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1 -e UCX_TLS=rc,cuda_copy,cuda_ipc,sm,self -e UCX_IB_GID_INDEX=3 -e UCX_LOG_LEVEL=info -e NIXL_LOG_LEVEL=INFO -e VLLM_LOGGING_CONFIG_PATH=/root/.cache/pd-logging.json --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model --served-model-name deepseek-v4-flash --host 0.0.0.0 --port 31249 --trust-remote-code --enable-auto-tool-choice --no-enable-prefix-caching --no-enable-expert-parallel --enable-chunked-prefill --jit-monitor-verbose --async-scheduling --tensor-parallel-size 4 --pipeline-parallel-size 1 --data-parallel-size 1 --max-model-len 16384 --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8_e4m3 --block-size 256 --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' --moe-backend auto --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' --kernel-config '{"enable_flashinfer_autotune":false}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,12,16,24,32],"max_cudagraph_capture_size":32}' --seed 0 --distributed-executor-backend mp --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cuda","kv_load_failure_policy":"fail","kv_connector_extra_config":{"backends":["UCX"],"enforce_handshake_compat":true}}'
```

### ordinary

```bash
docker run -d --pull never --name sb-pd-repeat-20260920-ordinary-decode --label io.serving-bench.run=pd-repeat-20260920-ordinary --network host --ipc host --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/vllm/6aa44ea147613be806d5:/root/.cache:rw --cpuset-cpus 32-47 --cpuset-mems 2 --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --cap-add SYS_NICE --device /dev/infiniband -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_NIXL_SIDE_CHANNEL_HOST=10.90.1.48 -e VLLM_NIXL_SIDE_CHANNEL_PORT=56300 -e UCX_NET_DEVICES=mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1 -e UCX_TLS=rc,cuda_copy,cuda_ipc,sm,self -e UCX_IB_GID_INDEX=3 -e UCX_LOG_LEVEL=info -e NIXL_LOG_LEVEL=INFO -e VLLM_LOGGING_CONFIG_PATH=/root/.cache/pd-logging.json --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model --served-model-name deepseek-v4-flash --host 0.0.0.0 --port 31249 --trust-remote-code --enable-auto-tool-choice --no-enable-prefix-caching --no-enable-expert-parallel --enable-chunked-prefill --jit-monitor-verbose --async-scheduling --tensor-parallel-size 4 --pipeline-parallel-size 1 --data-parallel-size 1 --max-model-len 16384 --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8_e4m3 --block-size 256 --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' --moe-backend auto --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' --kernel-config '{"enable_flashinfer_autotune":false}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,12,16,24,32],"max_cudagraph_capture_size":32}' --seed 0 --distributed-executor-backend mp
```

### 复现入口

本机探索脚本 `experiments/dsv4-pd-repeatability/scripts/diagnose.py` 调用公共 `serving_bench.pd_pair`，启动一套远端服务并顺序发普通请求，最后写STOP清理。每组保存了当时脚本副本及SHA256；第二组复用第一组保存的固定前缀，避免两组各自生成不同前缀。该脚本使用本轮固定目录，复现时须建立新的run-root，不能覆盖已有结果。

以下内容仅在45本机 `/home/enhui/llm-serving-benchmarks/experiments/dsv4-pd-repeatability/` 保留，不随Git分发：

- `results/connector-01/`、`results/ordinary-01/`：完整请求/响应、脚本快照、解析配置、command、镜像和模型身份、日志、metrics、CPU/GPU/NIC遥测与清理状态。
- `configs/`、两份deployment JSON、`scripts/`：本轮探索配置和执行脚本。
- `reports/`：事先预算、离线plan、固定镜像源码、CPU参数转换核查、完整统计及控制日志。

普通模型数值质量仍需单独验证；本诊断不重新评价既有DSpark或random吞吐结果，也不把性能工作量验收等同于模型正确性。
