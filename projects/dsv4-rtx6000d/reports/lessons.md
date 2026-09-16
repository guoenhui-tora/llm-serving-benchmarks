# 本项目已踩过的坑

这份记录用于避免重复探索，不替代完整日志。通用方法见 [实验规范](../../../docs/benchmark-methodology.md)。

| 问题 | 已确认的处理或证据 | 仍需注意 |
| --- | --- | --- |
| vLLM FlashInfer autotune 停滞 | 固定镜像关闭 autotune 后完成正式测试 | 根因未完整定位；不能归因于 NVFP4 质量差，也不能保证升级修复 |
| 安静预热后仍出现抖动 | 开启 vLLM JIT verbose 后捕获 TileLang 活动，保留缓存并按原规则预热 | 日志静默不等于收敛，cache miss 不一定实际重编译 |
| SGLang verbose 环境变量不足以保证可观测 | 使用自定义 INFO logger、FlashInfer/TileLang 输出，文件放入容器可见路径 | 不声称覆盖所有编译或 kernel |
| fallback 建议被误判为 autotune 事件 | 收紧事件起始匹配，回归测试覆盖真实开始与 fallback | warning 仍保留，不修改预热次数或通过条件来迎合结果 |
| 直接比较显存比例容易失真 | 0.90 / 0.89 对应 KV 51.75 / 51.38 GiB 每卡 | 不能把该比例推广到其他模型；KV/SWA 逻辑计数不同 |
| smoke 中 NUMA 亲和性权限警告 | 性能 target 使用 SYS_NICE；正式 SGLang 日志显示对应 GPU 的 NUMA 亲和性 | 权限存在不证明任意新配置都绑定正确 |
| PCIe/SM120 通信优化回退 | TP8 custom all-reduce、部分 SymmMem/multicast 路径不可用，保留日志 | 属平台支持边界，不等于模型不能运行 |
| 量化入口名称容易误读 | 日志入口含 fp8，专家实际分发有 NVFP4 证据 | SGLang FP8 KV scale=1.0 警告保留；未做质量评测 |

正式 18 次结果全部通过工作量与 gate 验收。vLLM C32 首次尝试完成请求，但因 16 条 TileLang 日志被拒绝；重新按既定上限预热后通过，未按速度挑样本。有效测量未观察到 OOM 或请求超时。

SGLang autotune on 的 8 个 worker 完成调优，每个保存 30 条配置；全日志保留 80 条形状未覆盖 warning，归为 10 种操作/形状。开启 autotune 不意味着所有 shape 被优化。

早期诊断、短负载、未对齐比较和容量 64 实验保留在本机 results/reports 中，不计入当前基线。
