# 本项目已确认的问题与处理

以下记录用于复现 GLM 基线。保留历史证据，不把其他机器或 DSV4 的现象当作本机结论。

| 问题 | 本轮处理与证据 | 适用边界 |
| --- | --- | --- |
| SGLang 共享专家加载报 128 / 256 维度不匹配 | 关闭 `shared-experts-fusion` 后加载与正式测试通过；vLLM 对应融合路径未启用 | 固定实现的兼容问题，不是已证明权重量化有误；未测试其他镜像的融合收益 |
| vLLM FlashInfer autotune 阻塞 | 关闭 autotune，保留 FlashInfer backend、CUDA Graph 和正常 JIT，完成三次重复 | 根因未完整定位；不能推断升级必然解决或 NVFP4 不适合 RTX6000D |
| JIT 日志和持久化不足 | vLLM 开启 `jit-monitor-verbose`；两引擎保留 FlashInfer/TileLang 输出，SGLang 配置 INFO logger，TileLang 缓存显式挂载 | 固定 FlashInfer 保留 `FLASHINFER_JIT_DEBUG=0`，避免 verbose 意外启用 debug 编译；未测日志零开销，也未完整覆盖所有 JIT |
| fallback 建议文字被误判为调优 | 识别真实 AutoTuner 开始事件，保留 `No tuned config` 警告；不改预热次数或 gate 条件 | fallback 不等于新一轮 autotune，后续日志安静也不证明没有回退 |
| SGLang C32 容量不足 | 最终显存比例 0.86，KV 池 310016 tokens；vLLM 0.90 对应 321280，均高于 32×(8192+1024) | 最终每个 C32 窗口达到 32 活动请求；比例只适用于这套权重和配置，不能按 DSV4 数值照搬 |
| NUMA 权限、PCIe 通信回退 | target 保留 `SYS_NICE`；保留 custom all-reduce、multimem/multicast 等限制提示 | 无 NVLink，未修改宿主全局设置；权限存在不等于全部线程及内存页绑定已经验证 |

三组共 18 次正式测量和 36 轮预热，全部通过工作量与日志 gate；没有测量重试、额外预热、OOM 或超时。已知编译事件在预热/正式窗口均为零，但日志静默不是性能收敛的证明。六组吞吐 CV 为 0.20%–0.97%，未做跨日交错复验。

SGLang OFF 的 C16 首轮预热有 16 条延迟加载 kernel、显存余量仅约 0.11 GiB 的警告，后续正式窗口未出现。它没有触发 OOM，仍需在扩大长度或容量时重新评估余量。

SGLang ON 的 8 个 rank 在 Graph 捕获前完成 autotune，每个约 6.75–6.80 秒，保存 18 个操作/形状条目。全日志有 80 条未覆盖形状的 fallback 警告，其中 C32 第二次正式测量有 16 条，采用默认 MoE tactic；该次样本完整保留。warning 按操作/签名去重，因此不能用后续无警告推断全部 shape 已调优。

ON 的实测收益已包含这些回退，额外显存占用约 1.86 GiB/卡，KV 池未增加。缓存条目中的 `trtllm` 操作标签不意味着顶层 MoE backend 被切换，日志仍为 `flashinfer_cutlass`。FP8 KV scale=1.0、tokenizer/接口及可选 metrics 提示均保留；中文探测正确不等于完整模型质量评测，也未进行逐算子 profiler 验证。

完整历史证据位置见 [复现说明](reproduction.md) 与 [provenance.json](../data/provenance.json)。早期短负载、未对齐比较和诊断配置保留在旧目录，不纳入本项目基线。
