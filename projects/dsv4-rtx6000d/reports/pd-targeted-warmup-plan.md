# DSV4定向mHC预热与PD复测计划（2026-09-21）

目标是在保留固定镜像和推理路径的条件下，排除已定位的mHC编译组合首次使用，取得可信的普通双TP4与1P1D同负载对照。前批普通第7轮1033.08仅一个clean样本，原FAIL/停止记录保留，不补成三轮结果。

## 固定版本发现

镜像内确有`deepseek_v4_mhc_warmup`，但其层筛选要求`hc_pre/hc_post`属性；当前`vllm.models.deepseek_v4.nvidia.model.DeepseekV4DecoderLayer`没有这些属性，故返回而未执行该mHC预热。NVIDIA forward直接调用broadcast和fused-post-pre函数，并融合RMSNorm。适配需覆盖这些真实入口，不只扩大旧函数token列表。

[启动预热候选](../patches/mhc-startup-warmup/README.md)使用自定义Worker子类在原启动预热前执行，两遍调用相同真实内核，检查输出有限、缓存key覆盖和第二遍不新增key，再完整执行父类启动逻辑。每worker上限480秒，所有八个worker均须记录COMPLETE；模型配置与固定源码哈希不匹配即停止。它不是修改n_splits取值或关闭warning来通过验收。

## 预算与顺序

先普通，成功后PD；46/48各GPU4–7，两个GPU服务、每服务TP4四卡，总八卡，EP/DSpark off。普通45代理采用least-inflight，PD为46P→48D，45控制。CPU/NUMA、NIC、模型、镜像ID、FP8 KV、上下文16384、block256、prefill8192、maxseq32及其他推理选项沿用前批。

每模式一次服务启动，启动READY上限900秒，含定向预热。先做独立1轮GovReport前128条HTTP预热（总C32、1024输出，480秒上限），再jit_clean最先三轮、最多5轮或1200秒，单轮480秒。每模式最多768请求，两模式最多1536；每轮输入1,042,149、输出131,072 tokens。HTTP预热、启动预热和测量分别落盘；本轮不重跑完整功能筛查。

普通未取得三轮clean，或吞吐CV>5%，不启动PD；PD使用相同验收和预算。CV门槛仅为停止排查阈值，不等价于stable协议。OOM、错误、真实工作量不符、预热验证失败、明显不稳定时立即取证停止。若只是本次worker接入实现缺陷，允许修正后以新run-root重启一次，保留失败；不为性能/JIT不达标重启或扩预算。

新recipe仅增加worker-cls并另建ID，旧recipe不改；新缓存目录完整继承各自旧缓存，复制前后核对哈希。对应模块放在新缓存的独立子目录并记录SHA256，运行中冻结。既有DOCA背景保留并记录，不修改宿主配置，不清JIT缓存。

## 验收与产物

固定镜像无GPU导入已确认Worker子类和CLI字段可用；离线测试直接比对固定源码的分派函数，覆盖1–8192 token及模型范围拒绝。GPU启动须进一步确认实际调用和缓存覆盖；日志安静不自动等价于性能稳定。

测量保持前128条GovReport、seed0、temperature0、ignore_eos，全系统C32，计时含45代理、P及传输。记录全部轮次、普通分流计数、P/D联合日志、NIXL传输/失败/远端命中/计算prefill增量，以及CPU/GPU/NIC背景。先判断PD是否仍落后，再选单个下一步，不扩配比或DSpark矩阵。

本机工作区`experiments/dsv4-pd-warmup/`：`reports/offline-plan.json`为解析命令，`reports/cache-preparation.json`为缓存复制证据，`image-source/`为固定源码只读快照，`patches/`为实际模块，`scripts/run.py`按普通→条件PD执行。运行入口为仓库根目录`PYTHONPATH=src python3 experiments/dsv4-pd-warmup/scripts/run.py`，新run-root为`results/ordinary-01`和`results/pd-01`。这些原始产物仅本机可用；精选结论另行归档，实际CLI从command.json导出。
