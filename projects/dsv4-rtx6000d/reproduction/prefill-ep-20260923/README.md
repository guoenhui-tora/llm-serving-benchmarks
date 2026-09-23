# 16K/1 EP扫描复现依赖

这是2026-09-23实测的两份固定配置和自动运行脚本快照，不是新一轮实验结果。C8至64共享一次模型启动，各64条预热＋256条quick正式、显式1轮；两组启动及功能全部通过后放行，随后独立扫描，失败不重试。结果见[报告](../../reports/prefill-ep-16k-20260923.md)与[实际命令](../../reports/prefill-ep-16k-20260923-commands.md)。

外部输入必须先准备好：46节点的固定权重 `/data/models/DeepSeek-V4-Flash-0731-NVFP4`、镜像 `vllm/vllm-openai:v0.29.0`（ID见runtime配置），以及原 `govreport-16k-prefix-1024-v2.jsonl`（SHA256 `025246b18d8ba7e1bc4d98c1a20b32b570860bb06a0baf0cea4ae97403d982e7`）。本次1024条精确16384-token核验见[原数据验收](dataset-original-verification.json)。大数据、权重、镜像和编译缓存不随Git同步，不以8K或随机数据替代。

配置文件保持内部相对引用；数据相对工作区为 `data/govreport-16k-prefix-1-v1.jsonl`。准备脚本校验原数据后仅派生输出长度，校验派生SHA，不改原文件；将归档配置复制到新的`experiments/<名称>/`并重新validate/plan。模型代码仍复用仓库`src/`，启动补丁复用[DSpark](../../patches/dspark-native-mxfp4/manifest.json)、[mHC](../../patches/mhc-startup-warmup-extended/README.md)及[输入覆盖](../../patches/input-kernel-warmup/README.md)。

以下从仓库根执行；数据/缓存路径是本次46路径，迁移机器必须显式提供匹配输入，不能假设存在：

```bash
python3 projects/dsv4-rtx6000d/reproduction/prefill-ep-20260923/prepare_workspace.py \
  --workspace experiments/dsv4-prefill-ep-reproduction \
  --source-data experiments/dsv4-prefill-ep-20260923/data/govreport-16k-prefix-1024-v2.jsonl \
  --cache-source /home/enhui/.cache/serving-bench/vllm/63379b30bb3f7cb7c690
```

脚本拒绝覆盖已有工作区。从只读源缓存派生两份独立副本；保留继承补丁快照后安装本Git版本的完整依赖，检查既有`pd-logging.json`、实际CLI和空闲GPU，不加载模型。`--offline-only`跳过缓存复制及Docker/GPU预检，仅验证数据和配置，不能据此称运行准备通过。本次归档已在 `experiments/dsv4-prefill-reproduction-check-20260923/`离线验证两组。

重新运行前按项目README确认CPU背景、NUMA/SMT、端口、日志、镜像/权重与补丁测试。固定镜像单token流式测试脚本为[scripts/test_single_token_stream.py](scripts/test_single_token_stream.py)，无GPU、loopback HTTP即可；原mHC、native草稿/DP profiling、输入GPU测试入口在对应patch目录。保持DSpark、Graph和budget。

准备脚本把新的4小时截止写入`evidence/authorized-budget.json`；过期就停止，不静默延长。确认新一次运行预算与资源后，在长时会话执行：

```bash
python3 experiments/dsv4-prefill-ep-reproduction/scripts/run_study.py \
  --run-root experiments/dsv4-prefill-ep-reproduction/results/scan-01
python3 experiments/dsv4-prefill-ep-reproduction/reports/analysis.py
```

运行脚本复用`docker`生命周期、`runner.phase`、`protocols.collect`、官方vLLM客户端和轻量`Recorder`。客户端观察钩子复用已有AST计时点定位，在完整窗口之外采集Prometheus；不改变请求生成、采样和计时表达式。每阶段保存请求记录、原始统计、HTTP窗口、日志事件、DP计数和工作量验收；全程5秒GPU/CPU及DP指标采样。

run-root拒绝覆盖，各组独立owner与状态文件。两组成功启动/功能后写`RELEASE`；组故障时保存`FAILED`、退出码和原始证据，扫描中另一组无共享故障可继续。全部结束写`FINISHED`与精简`summary.json`，无需逐轮查看。后处理拒绝未结束运行，默认输出目录已存在时也拒绝覆盖，可用`--output-dir`指定新的工作区reports子目录。

原始运行脚本保持实测版本；早期准备脚本的失败尝试仅在46本机证据中。归档准备入口是迁移工具，其完整缓存复制/模型运行路径未另行启动验证，已验证的是数据派生、配置依赖、validate/plan和脚本语法。不得把离线复现检查算作第三次模型启动。
