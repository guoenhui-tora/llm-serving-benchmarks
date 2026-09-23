# 固定 v0.30 镜像的 DSpark 原生 MXFP4 草稿修复

这是 45 节点 TP2 PP2、EP on、K5 成功运行版本的原样归档，面向 NVIDIA `DeepSeek-V4-Flash-0731-NVFP4` 当前 checkpoint。仅修复草稿专家量化分派并核验加载对象，不转换权重、不修改 forward、不加入定向 kernel 预热。结果与适用边界见[实测文档](../../data/v030-pp2-16k-c32-20260923.md)。

## 运行依赖与准备

运行文件是 `sitecustomize.py`、`dspark_native_mxfp4.py`、`manifest.json`，必须放在同一目录。固定镜像 image ID 为 `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`。manifest 固定原 DSpark loader、V4/V4.1 量化实现及注册表哈希，Python 模块固定模型 config/index 哈希，并检查全部 2304 个草稿专家矩阵头。

本次服务把宿主 `/home/enhui/.cache/serving-bench/vllm/674035353284ec2c293a` 挂载为 `/root/.cache`，上述三个文件放在该宿主缓存的 `dsv4-v030-mxfp4/` 子目录，通过 `PYTHONPATH=/root/.cache/dsv4-v030-mxfp4` 自动加载。新节点应准备自己的缓存目录并相应修改挂载，保留已有 JIT 缓存，不能假定 Git 会带走 45 的缓存。不要把旧 worker 目录加入本次 PYTHONPATH。

按[实测命令快照](../../data/v030-pp2-16k-c32-20260923-commands.md)启动前先完成文件准备；本目录本身不会自动生效。补丁使用断言进行部分校验，保持镜像默认 Python 执行方式，不启用 `python -O`／`PYTHONOPTIMIZE`。镜像或权重哈希不匹配时应重新审计，不绕过校验。

每个末级 worker 应输出 `LOCAL_DSPARK_FORMAT_FIX` 和 `LOCAL_DSPARK_LOADED`；加载对象核验要求 3 个 `Mxfp4MoEMethod` 及已初始化 kernel。本次实际后端是 `DEEPGEMM_MXFP4`，主模型保持 `FLASHINFER_CUTLASS` NVFP4。仍须检查完整服务日志中的 scale 错配和异常，不能仅凭这两个标记放行。

## 原有验证与归档检查

- `test_dispatch.py`：实际分派器与 import hook，目标 NVFP4 不变，草稿 MXFP4，源哈希变化拒绝；不运行 GPU kernel。
- `test_loader_config.py`：实际 CUDA 注册表及 EngineArgs／DSpark loader 配置链，在分配完整模型前停止；覆盖 CPU fallback 与真实 CUDA 类型不一致的问题。
- `test_loaded_evidence.py`：运行时对象观察器的正例及错误类型、缺 kernel、缺层反例。

测试脚本依赖固定镜像、`/model` 权重挂载和包含本目录的 PYTHONPATH，不能用宿主未安装 vLLM 的 Python 代替运行。实验阶段配置链／分派及观察器测试已通过，最终观察器将 INFO 改为显式输出后在两个末级 GPU worker 验证通过。本次仅归档，逐文件核对与成功实验版本一致并检查 Python 语法，没有重新加载模型或重跑性能测试。

最终文件 SHA256 见[归档比较证据的 patch_sha256](../../data/v030-pp2-16k-c32-20260923-comparison.json)。本补丁不是通用上游修复，也不支持直接套用其他镜像或其他发布者的同名权重。
