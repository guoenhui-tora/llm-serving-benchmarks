# 夜间扩展的固定镜像启动覆盖快照

这些是夜间实测实际使用的独立worker快照，保留原[TP4/8192模块](../mhc-startup-warmup/README.md)。只覆盖启动阶段mHC调用，不替换forward、权重或精度；仍需要完整HTTP预热及逐轮事件检查。

| 目录 | 已实测范围 | 结果边界 |
| --- | --- | --- |
| tp4-off | TP4/DP1/PP1、DSpark off；8K/16K真实输入，P预算≤16384；C32及8K/16K C64 | B1/B2/B3A/B4三轮0事件；C64共12组，其中11组三轮0事件，16K六副本普通正式4/4/0条top-k事件 |
| dp-off | TP2×DP2 EP on、PP1、DSpark off；8K、C32 | C1 PD三轮0事件；C2普通0事件但CV5.73% |
| dspark | TP2×DP2 EP on、PP1、K5对称P/D或普通；8K、C32 | C3H/C4完成quick，正式仍有草稿输入准备Triton事件；不是无JIT基线 |

代码允许的其他参数（例如TP4 K3、16K DSpark、DP off的16K预算）不等于已经GPU实测。DSpark两个模式每次八个worker均完成target/draft覆盖，三层draft SWA cache均完成PD注册检查；真实KV和各DP引擎draft/accepted在客户端协议中另验收。该检查不是完整模型质量证明。

将选定子目录中的`dsv4_mhc_worker.py`、`warmup_plan.py`和`pinned-source.json`完整复制到该服务独立缓存的`dsv4-mhc-warmup/`，设置`PYTHONPATH=/root/.cache/dsv4-mhc-warmup`、`--worker-cls dsv4_mhc_worker.Worker`。DSpark还必须按[原兼容补丁](../../reports/dspark-compatibility.md)加载native MXFP4分派及DP profiling回移，PYTHONPATH同时包含两模块目录。固定镜像image ID仍为`sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`；worker核对固定物理源码SHA，任何不匹配直接失败。

每服务复制原缓存并核对原文件哈希，保留来源，不能从空缓存开始却称为热缓存。不将所有子目录同时加进PYTHONPATH，以免导入同名模块。实际命令由每次报告command.json导出；这些快照尚未自动应用到精选baseline recipe。

target覆盖使用实际SM数、prefill预算、Graph capture sizes，逐worker两遍、有限值和cache key检查。DSpark另用真实三层draft及target auxiliary ids40/41/42，覆盖1..max(max-num-seqs×(K+1),maxcapture)（上限512）；两遍不新增cache key、临时allocated显存回到原值，再完整执行原worker预热、Graph捕获和RNG重置。

**不能承诺quick必然无JIT。** TP4扩展在16K六副本普通中完整HTTP预热88条top-k事件，正式仍4/4/0，来自`_compute_global_topk_indices_and_lens_kernel`；每进程实际分派可能覆盖不足，未修改规则或追加轮次。8K 2P2D三轮0事件但吞吐/TTFT漂移，不能据此称性能已稳定。 C3H正式事件4/2/0，C4为4/0/2，均来自`_prepare_dflash_inputs_kernel`；mHC覆盖通过未覆盖这一不同内核。普通DP off预热无事件但吞吐明显更低的反例也说明，0事件不能替代完整预热或稳定性验收。详见[DP报告](../../reports/pd-dp-off-20260921.md)、[HTTP失败与有界修正](../../reports/pd-dspark-http-failure-20260921.md)及[持续结果总览](../../reports/pd-overnight-results-20260921.md)。

离线测试：在固定镜像CPU容器将选定子目录挂载到`/patch:ro`、`PYTHONPATH=/patch`，执行`python3 -m unittest discover -s /patch -p 'test_*.py'`。这些测试核对固定kernel分派公式和预算边界，不替代真实GPU覆盖、NIXL传输或HTTP负载检查。
