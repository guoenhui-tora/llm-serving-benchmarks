# DSpark 草稿量化加载问题与本地补丁

**固定 vLLM 0.29.0 镜像加载 NVIDIA Flash-0731 内置 DSpark 草稿时，会错误地将主模型的 NVFP4 专家配置用于 MXFP4 草稿。当前采用本地加载补丁，保留原权重及其精度。** 本机 TP4 功能检查已确认草稿分派恢复、scale 不匹配警告消失；这不代表完成模型质量评测。后续性能与Graph覆盖修正另见[TP4 DSpark对照报告](dspark-tp4-20260919.md)。

## 适用范围与问题

- 模型：[nvidia/DeepSeek-V4-Flash-0731-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4)，本机目录 `/data/models/DeepSeek-V4-Flash-0731-NVFP4`。
- 镜像：`vllm/vllm-openai:v0.29.0`，实测 image ID 与 RepoDigest 中的 SHA256 均为 `c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`。
- 平台：RTX6000D、SM120；已验证 TP4/PP1/DP1、EP off。固定版本的 DSpark 加载器明确不支持 PP，不能直接用于 TP2×PP2。
- 本次检查日期：2026-09-19；原运行仓库 commit：`ef6c3e4d5c29efb31b1c9d5ef9cd838223068c46`。

这份 checkpoint 的主模型 routed experts 使用 NVFP4，但内置草稿的三层、共768个专家仍为 MXFP4：packed I8 权重、E8M0 的 group32 scale，缺少 NVFP4 所需的全局 weight/input scale。镜像已为草稿创建独立 quant_config，却为层元数据保留了主模型上下文；延迟解析专家格式时，又读取了主模型的全局 `moe_quant_algo=NVFP4`，从而选错加载方法。

原路径可以启动，但出现 `w1_weight_scale_2 must match w3_weight_scale_2. Accuracy may be affected.`，伴随很低的草稿接受率。相同输入下，仅修正草稿格式分派后接受率明显恢复，警告消失。证据指向混合格式的加载兼容问题，不能据此归咎于 NVFP4 量化质量或 RTX6000D 不支持该模型。

## 补丁做了什么

代码位于 [patches/dspark-native-mxfp4](../patches/dspark-native-mxfp4/)，[change.patch](../patches/dspark-native-mxfp4/change.patch)给出对镜像原模块的最小差异。

1. 在 DSpark 加载器取得草稿 quant_config 后，调用独立辅助函数，复制配置并固定草稿 MoE 为原生 MXFP4 分派。
2. 保留主模型 NVFP4、草稿 FP8 线性层处理和目标层元数据；不转换权重，不更改主模型 quant_config。
3. 通过 `PYTHONPATH` 中的 `sitecustomize.py` 覆盖一个加载模块，原镜像不变。结果应标记为“vLLM 0.29.0＋本地补丁”。

使用前校验原模块 SHA256、补丁文件哈希、模型 config/index 哈希及全部草稿专家的 tensor header。哈希见[manifest.json](../patches/dspark-native-mxfp4/manifest.json)，模型约束见[辅助函数](../patches/dspark-native-mxfp4/dspark_native_mxfp4.py)。这些检查不等于对全部权重内容做哈希；本补丁依赖该版本私有字段，不能直接跨镜像或 checkpoint 复用。运行 Python 时不要使用 `-O` 或 `PYTHONOPTIMIZE`，以免关闭其中的断言检查。

离线分派测试确认：主模型修复前后均选择 NVFP4，草稿三层选择 MXFP4，FP8 配置保持一致。此前 GPU 实测日志显示主模型 `FLASHINFER_CUTLASS` NVFP4、草稿 `DEEPGEMM_MXFP4`，两者 Graph 捕获完成。未完整验证所有 kernel 或端到端数值精度。

## 后续实验如何使用

从[公共 TP4 基线](../configs/campaigns/tp4.yaml)复制配置到本地 `experiments/<项目>/configs/`，创建新的 recipe/campaign，保留原基线。新 recipe 在原有配置上增加：

```yaml
options:
  # 与原 options 合并，不覆盖其他参数。
  compilation-config:
    cudagraph_mode: FULL_DECODE_ONLY
    cudagraph_capture_sizes: [5,6,10,12,20,24,40,48,60,72,80,96,120,144,160,192]
    max_cudagraph_capture_size: 192
  speculative-config:
    method: dspark
    num_speculative_tokens: 5
    draft_sample_method: greedy
    rejection_sample_method: standard
    enable_adaptive_verification: false
environment:
  # 保留原 environment；容器已有 /root/.cache 持久化挂载。
  PYTHONPATH: /root/.cache/dspark-native-mxfp4
```

以上K5、C32配置已完成后续性能对照，不是最优参数。Graph尺寸按token计，目标验证需覆盖192、草稿需160，不能沿用普通TP4上限32；其他K或容量需重新核对覆盖范围。保持原有日志规则，并在 `checks.required` 中追加 `LOCAL_DSPARK_FORMAT_FIX: validated 768 native MXFP4 draft experts` 和 `Mxfp4 MoE backend`，确认补丁实际加载。

在仓库根目录，使用 Python 环境中的项目依赖运行：

```bash
# 将示例路径换成刚创建的本地 campaign。
python3 projects/dsv4-rtx6000d/patches/dspark-native-mxfp4/prepare.py \
  experiments/<项目>/configs/campaigns/<新campaign>.yaml
./bench validate experiments/<项目>/configs/campaigns/<新campaign>.yaml
./bench plan experiments/<项目>/configs/campaigns/<新campaign>.yaml
```

`prepare.py` 只把补丁放到解析配置对应的用户缓存目录，不启动服务。重复运行会核对一致性，遇到已有不同文件则退出；加 `--check` 只检查。改变 recipe 后应重新 prepare，因为缓存目录随 recipe 指纹变化。正式启动仍按仓库流程执行 preflight、功能检查和所选压测协议，本报告不提供性能结论。

回退时使用原先不带补丁的 recipe/campaign，并去掉新增的 `PYTHONPATH` 与补丁 required 日志规则；不需要删除缓存、权重或修改镜像。

### 不加载模型的补丁测试

下面只在固定镜像中检查量化分派及 checkpoint header，不申请 GPU、不启动推理服务；MoE 构造器由测试替身代替，不能验证实际 CUDA kernel。

```bash
docker run --rm --pull never --network none \
  -v "$PWD/projects/dsv4-rtx6000d/patches/dspark-native-mxfp4:/audit:ro" \
  -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro \
  -e PYTHONPATH=/audit -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint python3 \
  sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 \
  /audit/test_dispatch.py
```

## 为什么暂不换草稿或镜像

以下来源于2026-09-19的检索，状态以当日为准：

- [vLLM PR #49133](https://github.com/vllm-project/vllm/pull/49133)直接描述 NVFP4 目标配置误用于 MXFP4 草稿的问题，但已关闭、未合并。本补丁是针对固定镜像的本地适配，没有原样套用旧 PR；当时未找到可直接采用的完整上游修复。
- [NVIDIA Flash-nvfp4-DSpark](https://huggingface.co/nvidia/DeepSeek-V4-Flash-nvfp4-DSpark)是早期 Flash 主模型与草稿合并包。模型卡在 B300 上验证，不能仅凭同架构认定其草稿匹配当前0731权重。
- [0731模型讨论 #2](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4/discussions/2)及[社区转换方案](https://github.com/takashito/DeepSeek-V4-Flash-0731-NVFP4-mtpfix)提供将同一0731草稿转为 NVFP4 的另一条路线。作者报告权重反量化值一致，但另加了未单独校准的 input scale；不能将其等同于端到端数值无变化。本机未采用或验证此方案。
- [社区 SM120 分派修复记录](https://github.com/jasl/vllm/issues/35)也报告保留 MXFP4 草稿后接受率恢复，但使用定制镜像，不能替代当前镜像的本机验证。

因此当前保留已验证的原权重＋加载补丁；后续性能研究已单独归档到[对照报告](dspark-tp4-20260919.md)。完整探索日志、配置和数据仍保存在节点48的 `experiments/dsv4-rtx6000d/`，仅本机可用，不随 Git 分发。
