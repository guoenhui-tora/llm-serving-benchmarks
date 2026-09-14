# RTX 6000D 实机验证记录

日期：2026-09-14。两组均使用本工程完整的 `bench run` 流程，结果为 PASS。

| 主机 | 服务端 | 模型 | HTTP 就绪耗时 | 测量请求 | 输出 token 总数 | Kernel gate |
|---|---|---|---:|---|---:|---|
| 10.90.1.46 | vLLM 0.29.0 | GLM-5.2-NVFP4 | 176.17 秒 | 2 成功 / 0 失败 | 64 | PASS |
| 10.90.1.48 | SGLang 0.5.19-cu130 | DeepSeek-V4-Flash-0731-NVFP4 | 450.36 秒 | 2 成功 / 0 失败 | 64 | PASS |

就绪耗时从服务容器启动后等待 health 开始计算，不包含 preflight；48 包含首次 SM120 JIT / autotune。
这是 smoke 功能验证，未进行容量校准或正式性能测试，不能用这两组不同模型的结果比较引擎速度。

## 实际验证内容

- 两台均为 8 × RTX 6000D / SM120，配置中的 GPU、路径、镜像 ID 与实际环境匹配。
- health、models 和 chat 均成功。相同问题“1加1等于几”：GLM 回复“1加1等于2。”；DeepSeek 回复“2。”。
- 同一个 vLLM 0.29.0 CPU 客户端通过 OpenAI completions 接口测试两个服务端。
- random 输入 128 / 输出 32，C1；1 个预热请求，测量 2 个请求。
- 两边第一轮预热、第一轮测量均无已知编译日志事件，没有发生测量重试。
- 两边真实输入 token 总数均为 256，输出总数均为 64，固定长度检查通过。
- vLLM 显式关闭 prefix caching；SGLang 显式关闭 radix cache。
- 各自结束后服务端/客户端容器自动删除，GPU 计算进程为空，JIT 缓存保留。
- 结果已从两台服务器取回，并由本地 `bench report` 成功读取、分别分组。
- 最终代码在本机 Python 3.12.14、两台服务器 Python 3.12.3 上均通过 44 项测试，PyYAML 均为 6.0.3。

## 实机中修复的问题

1. Docker 29 的小写 `error: no such object` 被误认为清理失败。现在只把明确的“对象/容器不存在”视为已清理，仍拒绝权限错误等其他异常。
2. vLLM 0.29 顶层 CLI 会提前构造服务端配置，无 GPU 时连 benchmark/help 都可能报设备检测错误。
   服务端 CLI preflight 可见一张 GPU但不加载模型；CPU 客户端直接调用官方 `vllm.benchmarks.serve` 的参数解析及主函数。
3. 增加对应回归测试。服务镜像、模型权重和实际推理参数无需为这两个编排问题打补丁。

## Kernel 与警告

46 日志确认 `GlmMoeDsaForCausalLM`、`FLASHINFER_MLA_SPARSE_SM120`、`fp8_ds_mla` 和 `FLASHINFER_CUTLASS` NVFP4 MoE。
48 日志确认 `DeepseekV4ForCausalLM`、`DeepseekV4AttnBackend`、`ModelOptNvFp4FusedMoEMethod` 和 `flashinfer_cutlass` MoE runner。
检查仅证明配置的日志证据存在；没有新增 GPU profiler 或完整数值精度评测。

原有 PCIe/SM120 通信限制、FP8 KV scaling、tokenizer 等警告完整保留在 server.log 中，没有隐藏。
48 还报告缺少设置 NUMA affinity 的权限；本次 smoke 不要求 NUMA 绑定，性能测试前应评估 target 的 `cap_add: [SYS_NICE]`。
CPU 客户端的可选 GPU 扩展导入警告不阻碍 HTTP 测量，不能据此推断服务端模型 kernel 回退。
特别是 FP8 KV scaling 警告，短请求通过不等于长上下文数值精度已经验证。

## 工件位置

两台工程均在 `/home/enhui/llm-serving-benchmarks`。

- 46：`results/46-glm52-vllm-smoke/real-smoke-20260914-01/`
- 48：`results/48-dsv4-sglang-smoke/real-smoke-20260914-01/`

每份结果包含原始配置快照、源代码指纹、实际命令、镜像/模型/GPU 环境信息、请求响应、完整日志与原始/标准化指标。
验证后只更新 smoke recipe 的说明文字；实际启动参数不变，缓存目录随说明指纹变动平移以保留本次 JIT 缓存。
48 额外导入 vLLM 镜像作为 CPU benchmark 客户端。镜像经 46→48 内网传输，ID 与本地固定配置一致；临时传输服务和 8.1GB 中转文件已清理。

## 尚未验证

calibration/performance 候选 recipe、CUDA graphs/其他默认优化的完整覆盖、更长上下文和更高并发、完整精度、工具调用，以及另外两个模型/引擎交叉组合。
下一步应分别运行小规模 calibration，再决定正式测量的 recipe 和负载范围。
