# top-k 与 DSpark 输入内核启动覆盖

已完成三组完整模型验证：DSpark 1P1D、DSpark普通双服务、16K普通六TP4，共9轮正式测量均0已知编译事件。详见[实测报告](../../reports/input-kernel-warmup-20260921.md)。固定vLLM0.29.0、NVIDIA DSV4 Flash NVFP4，补齐已有mHC预热以外的两类Triton分派。只运行启动阶段的私有scratch张量，不替换forward、内核、权重、精度或日志规则。

- top-k从实际attention metadata builder及block table读取C4/C128索引宽度、容量步长、块大小；覆盖混合prefill/decode切片的0..15偏移（int32映射及bool有效位的三个对齐组合）。不把token数量相同当作布局相同。
- DSpark从实际speculator读取K、anchor模式、token ID和三组draft表参数，调用原`prepare_dflash_inputs`，覆盖每个BLOCK_SIZE区间的首尾跨度，包括64/128。检查拒绝后上下文、空块/SWA slot、query及sample位置、context上限及padding。
- 两遍相同覆盖要求Triton进程内cache key集合不变、临时allocated显存归零，再执行原mHC预热、Graph捕获和RNG重置。源码SHA不匹配或实际模式超出范围直接停止。

2026-09-23新增每DP引擎S96、budget16384、K5的GPU输入测试，并完成四卡D192完整启动与七档21轮正式测量，均0已知正式JIT，见[大容量验收](../../reports/decode-16k-d192-20260923.md)。扩大序列容量还必须另核对Graph覆盖；输入内核覆盖通过不代表Graph尺寸足够。

小型GPU测试通过独立整数参考检查输出，并用额外区间内部跨度回放确认无新key；不替代模型质量验证或完整HTTP验收。范围为PP1、上下文16384/32768，现有TP4/off或TP2DP2 EP on K5；helper另做K3合成检查，不能据此称K3完整服务已实测。不支持CP扩展。

接入时保留[原mHC扩展模块](../mhc-startup-warmup-extended/README.md)，将本目录Python/JSON文件放入该服务缓存的`dsv4-input-warmup/`，在原PYTHONPATH前添加`/root/.cache/dsv4-input-warmup`，改`--worker-cls dsv4_input_worker.Worker`。原DSpark量化/DP profiling补丁仍必须保留。每worker必须有`DSV4_INPUT_KERNEL_WARMUP ... COMPLETE`记录。

保留原缓存；新recipe复制原缓存并核对完整哈希，再放入模块。之后按该次协议执行完整输入/输出长度的HTTP预热，再固定三轮quick，记录全部事件与真实KV传输，不追加轮次或挑最后一轮。本轮为2C条完整生成预热、每轮正式4C条；历史报告中的完整N预热另按原协议解释。报告分别标记GPU内核测试、完整服务验证及未验证范围。

## 内核测试复测入口

在上述固定镜像内，将本目录只读挂载到`/patch`，指定一张空闲GPU并挂载持久Triton缓存后执行：

```bash
PYTHONPATH=/patch VLLM_USE_V2_MODEL_RUNNER=1 python3 -m unittest -v test_input_warmup
```

这是复测入口示例；完整服务本轮实际命令见[命令快照](../../reports/input-kernel-warmup-commands-20260921.md)。测试不加载模型，GPU测试检查真实固定内核的输出及cache回放，builder回归检查不会把indexer误当成attention builder。首次GPU测试约29秒（含导入）；完整模型仍需独立验收。

2026-09-23组合mHC拓扑扩展后，新增TP2×DP2 EP off、K5/S32的四worker完整启动与16K/1 HTTP验收；输入几何仍从实际builder/speculator读取，本模块未修改。on/off扫描结果与警告见[单机prefill报告](../../reports/prefill-ep-16k-20260923.md)。
