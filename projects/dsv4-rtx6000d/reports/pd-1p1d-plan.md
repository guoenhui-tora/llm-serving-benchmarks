# 跨节点1P1D首轮计划（2026-09-20）

**先验证两节点各四卡TP4、DSpark off的真实KV传输，再在相同八卡资源、GovReport总C32下比较普通双副本与PD。** 本文件是实施前计划，不是性能结果。用户已授权45统一控制四节点及实施本轮；节点角色按启动前资源核查选择。

## 固定条件与范围

- 起始研究总结：`c651fe4`，见[项目总结](../README.md)。不恢复已结束的DP补测任务。
- 模型：`/data/models/DeepSeek-V4-Flash-0731-NVFP4`；镜像：`vllm/vllm-openai:v0.29.0`，ID `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`。不换镜像、权重精度，不清JIT缓存。
- P和D各一个GPU服务，各TP4/PP1/DP1、EP off、DSpark off，每服务4卡，总8卡，分布在两个节点。45的代理、客户端和控制进程不占GPU；控制机也可承担P或D，身份不强制角色。
- 对照：同两节点、同八卡部署两个普通TP4完整推理副本。两种模式共用代理入口，全系统最大在途请求32，不能每服务各发C32后相加。
- 保留FP8 KV、block256、prefill8192、上下文16384、每服务活动容量32、显存比例0.90、V2/async/正常Graph，关闭prefix cache和FlashInfer autotune。
- GPU4–7优先配CPU32–47/NUMA2及近端mlx5_4–7；前四卡可对应CPU0–15/NUMA0及mlx5_0–3。实际节点、CPU/NIC选择须记录后台负载、SMT兄弟及设备映射，cpuset不代表独占CPU。

## 版本依据与待验证项

镜像构建标签指向vLLM `98dff2a81d747d1dba01a47f939f48c3526d4206`；草稿加载器及DFlash模块的该提交源码哈希与项目记录的镜像原模块一致。静态检查不等于实际PD验证。

- [该构建NIXL说明](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/docs/features/nixl_connector_usage.md)：NixlConnector为pull/READ，P/D使用producer/consumer；显式设置KV加载失败策略fail，保持握手兼容校验。UCX设备配置不由NCCL环境变量替代。
- [该构建兼容表](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/docs/features/nixl_connector_compatibility.md)：FP8两端格式一致；packed内联scale有传输支持线索，不能遗漏动态scale。DSV4实际为fp8_ds_mla，包含压缩KV、SWA、indexer和压缩器状态；128窗口、4/128压缩边界、256块边界均需验证。
- NIXL专用toy proxy能转交kv_transfer_params，缺失元数据必须改为失败；通用disagg_proxy_demo不能直接用于本次NIXL验收。
- 普通推理DSpark兼容补丁不等于PD兼容；DSpark明确不支持PP。首轮不启用DSpark、PP、DP或异构TP。后续分别研究纯P/D拓扑，不照搬混合推理排名。
- [互联报告](interconnect.html)的NCCL GDR/带宽结果不能代替本次NIXL路径验证。

## 功能验收

1. 客户端、代理、P和D以请求ID关联。P返回有效engine/request/block元数据，代理完整转交；缺失或错误立即失败，无自动普通推理回退。
2. 保存消费rank的NIXL完成记录、字节和描述符、NIC计数增量及实际UCX路径；只有握手或网络流量不算通过。按真实缓存区域核查传输量。
3. 核对远端命中token数和D实际调度计算量。API usage.prompt_tokens不是计算量；该版本完整命中后重算最后一个prompt token正常，整段近8K重算不通过。
4. C1比较普通路径和PD的greedy输出，覆盖非对齐输入、窗口/压缩/block边界，生成256 tokens观察后续状态。输出差异先定位，不凭文本通顺判定通过。
5. 缺失元数据和一次受控KV加载失败必须报错；失败探针单独保存。真实KV传输与GDR/RoCE分别判定，TCP/主机暂存不可冒充GDR。

## 测量、预算与停止条件

正式负载使用[GovReport前128条](../data/govreport-near8k.md)，固定1024输出，总C32；每轮128成功、0失败，输入1,042,149、输出131,072 tokens。原样流式/v1/completions、不重套模板，seed0、temperature0、ignore_eos。历史random结果不进入本轮收益计算。

P请求输出上限1，其响应token不额外拼到客户端输出；D返回完整1024。P计算仍计入端到端成本。客户端计时覆盖代理、P、KV传输和D，不相加独立服务吞吐，不跨机器直接相减未校准时钟。

| 阶段 | 固定预算 | 时间上限 |
| --- | --- | ---: |
| PD启动 | P/D各一次，可并行，实际CLI/镜像/设备预检 | 30分钟 |
| 功能与容量 | C1：8条PD+8条普通路径，输出256；C8：16条输出256；C32：32条输出1024；另最多2条失败探针 | 15分钟 |
| PD quick | 1轮完整128条预热+固定3轮128条正式测量，总C32 | 20分钟 |
| 普通双副本启动 | 同两节点、同GPU，各一次 | 30分钟 |
| 普通quick | 与PD相同预热、三轮及负载 | 20分钟 |
| 收尾取证 | 汇总、仅清理本次所属容器，保留缓存 | 5分钟 |

实施准备时间单计；运行总预算2小时。PD功能通过后复用同次启动。全部正式轮次保留、标记JIT及资源变化；quick不冒充clean/stable，未定结果不自动追加测量。

OOM、非法访问、布局/握手不兼容、状态缺失、整段重算、输出异常或真实请求失败时停止性能阶段取证。确定性不支持不重试；明确端口/地址错误最多修正重启一次且计入预算。资源被占用不终止他人任务。达到预算记PARTIAL/FAIL，不放宽gate或只选最快轮。

## 实施及产物

工作区为`experiments/dsv4-pd-1p1d/`，从本项目configs续接。公共适配放src：可配置监听/endpoint、RDMA透传、薄SSH控制、严格NIXL代理和联合日志验收；不建设全矩阵调度平台。必要模拟测试覆盖元数据、失败传播、流式计数和所有权清理。运行期间冻结源码/配置。

运行时显式指定新的工作区内--run-root。命令由解析计划导出，保存实际镜像源码/CLI、模型身份、节点映射、绑定、全部服务及代理日志、请求和指标；原始产物仅本机保存在工作区。结果另写报告并从此处链接。公共实现、实验配置和结果归档分开提交，不push。

链路及同资源对照通过后，另定预算研究TP4 DSpark K3、P/D独立拓扑和配比；不补历史PARTIAL，不自动扩展矩阵。命名按实例数：3P1D且每实例TP4表示4个GPU服务、每服务4卡、总16卡，节点数另列。
