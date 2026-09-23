# 固定 vLLM 0.30：PP2 PD 的 Mooncake 路由与观测

本目录保存2026-09-24实测的MXFP4草稿修复、Mooncake计数观测和独立代理。**模型forward、权重、KV布局和Mooncake传输算法保持镜像实现**；不是通用PP补丁或新版vLLM兼容层。

服务镜像固定 `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`。模型、量化模块及草稿loader的校验沿用[原始MXFP4补丁](../v030-dspark-mxfp4/README.md)，`manifest.json`、`dspark_native_mxfp4.py`与原补丁一致。`sitecustomize.py`在原hook后增加两个源码哈希固定的观测hook：

- 保留原始`clone_and_reset`语义，记录每批成功传输字节及传输/接收/过期失败计数。
- 记录worker无事件时的零计数心跳，以及producer每条请求的TP/PP rank完成记录；不将“没有日志”视为零失败。
- Mooncake的成功transfer计数统计批次，不等于请求数。实验要求每条请求在同一个P服务的TP0/1×PP0/1都有一次成功完成记录，并同时核对D远端命中、零整段重算、请求成功和DSpark活动。

将三个运行文件`sitecustomize.py`、`dspark_native_mxfp4.py`、`manifest.json`放入**本次recipe对应缓存**的`dsv4-v030-mxfp4/`目录，设置`PYTHONPATH=/root/.cache/dsv4-v030-mxfp4`。完整实际Docker命令、缓存路径、Graph和预算见对应实验报告；不得更新正在运行服务挂载的文件。保留JIT缓存，不加载旧mHC/输入kernel自定义worker。

`pd_proxy.py`使用aiohttp。代理要求P为TP2/PP2/DP1，启动时通过`MOONCAKE_BOOTSTRAP_MAP`中的各P bootstrap `/query`核对完整rank注册；每次请求生成独立transfer ID，P只生成一个token并保留KV，D按相同ID和真实engine ID拉取KV。拒绝客户端注入传输元数据，没有隐藏重试或回退。D可内部使用DP2。

代理环境`MOONCAKE_BOOTSTRAP_MAP`为“P API URL→其bootstrap URL”的JSON映射，CLI使用`--prefill`列出三个P、`--decode`列出D及`--port`设置代理端口。服务端设置`VLLM_HOST_IP`、`VLLM_MOONCAKE_BOOTSTRAP_PORT`和`MC_GID_INDEX=3`，本批已核对GID3为RoCE v2。`device_name`为对应GPU所在NUMA的`mlx5_0`或`mlx5_4`；不能假设其他节点沿用这些NIC/GID/端口即可运行。NIXL的`UCX_IB_GID_INDEX`不能代替Mooncake的`MC_GID_INDEX`。

离线检查无需GPU，在固定镜像内以`python3 /candidate/test_observer.py`运行；`test_proxy.py`可使用本批固定v0.29客户端镜像。把本目录只读挂载为`/candidate`即可。测试覆盖正字节、全部失败计数、零事件worker、reset语义、源码变化拒绝、唯一传输ID、P→D元数据、缺失rank/重复engine和客户端元数据拒绝。真实CUDA/KV兼容性仍以报告中的GPU功能和正式轮次为证。

PP组改用Mooncake，而DP2基线用NIXL；跨组差异包含connector、代理及观测开销，不能宣称纯拓扑因果收益。本批没有移植开放PR的prefill-only模型改动。

归档代理仅更新了文件头说明（从历史NIXL名称改为Mooncake）；其余可执行AST与本批冻结代理一致。观测hook与实测文件逐字节一致。
