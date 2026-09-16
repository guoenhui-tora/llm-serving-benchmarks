# GLM-5.2-NVFP4 在 RTX6000D 上的推理优化

**选用 `lmsysorg/sglang:v0.5.19-cu130`，作为当前场景的部署和后续调优起点。** 在单机 8×RTX6000D、8192 输入 / 1024 输出的对齐实验中，两边关闭 FlashInfer autotune 时，SGLang 在 C16 / C32 的输出吞吐领先 vLLM **6.94% / 7.27%**，平均 TTFT 和 TPOT 也更低。

偏重吞吐和生成速度时选 SGLang autotune on；更重视 C16 首 token 响应时保留 off。开启后吞吐进一步提高 **5.08% / 2.77%**，但 C16 平均 TTFT 从 14.98 秒增至 16.07 秒，显存约增加 1.86 GiB/卡。两档 TPOT 都未达到此前期望的 33 ms。

本项目保留 2026-09-15 至 09-16 在 `gpu-6000d-46` 完成的三组最终实验。权重和 tokenizer 均为 `/data/models/GLM-5.2-NVFP4`。每组 C16 / C32 各三次有效重复，共 **18 次测量、1728 成功请求、0 失败**。

| 并发 | 引擎 / autotune | 输出吞吐 tok/s | Mean TTFT 秒 | Mean TPOT ms |
| --- | --- | ---: | ---: | ---: |
| C16 | vLLM / off | 178.16 ± 0.61 | 15.80 | 74.40 |
| C16 | SGLang / off | 190.53 ± 1.85 | 14.98 | 69.36 |
| C16 | SGLang / on | 200.20 ± 0.56 | 16.07 | 64.23 |
| C32 | vLLM / off | 209.93 ± 0.54 | 22.31 | 130.62 |
| C32 | SGLang / off | 225.20 ± 0.79 | 20.97 | 121.57 |
| C32 | SGLang / on | 231.44 ± 0.47 | 20.92 | 117.80 |

吞吐为三次均值 ± 样本标准差，延迟为三次对应指标的均值。吞吐 CV 为 0.20%–0.97%；这是本轮短期重复的波动，不代表长期线上稳定性。完整 P95、ITL、端到端延迟、请求/token 数和各指标标准差见选型报告。

| 内容 | 入口 |
| --- | --- |
| 镜像 digest、参数对齐和完整结果 | [镜像选型](reports/image-selection.md) |
| 18 次测量的逐次指标 | [baseline-samples.csv](data/baseline-samples.csv) |
| 环境准备、运行命令和迁移说明 | [基线复现](reports/reproduction.md) |
| 加载、autotune、JIT 和容量问题 | [排查记录](reports/lessons.md) |
| 历史结果路径、哈希和配置映射 | [provenance.json](data/provenance.json) |

CSV 的 `case` 区分三组配置，`C` 为并发，`R` 为重复序号；延迟单位为 ms，`duration` 为秒，输出吞吐单位为 tokens/s。

只保留 **3 份 recipe、2 个 workload、1 个 campaign** 及其完整依赖，不迁入早期探索配置。从新仓库根目录执行：

```bash
./bench validate projects/glm52-rtx6000d/configs/campaigns/46-glm52-aligned-final-c16-c32.yaml
./bench plan projects/glm52-rtx6000d/configs/campaigns/46-glm52-aligned-final-c16-c32.yaml
```

继续实验时先将本项目 configs 复制到 `experiments/glm52-rtx6000d/configs/`，步骤见 [基线复现](reports/reproduction.md)。该 campaign 顺序执行三套配置，各测 C16 / C32 三次；target 固定为 46 节点。实际运行前按 [基线复现](reports/reproduction.md) 准备日志文件并做 preflight。

**历史结果已经实测；迁移后的入口仅做离线验证，未启动新实验。** 推理参数与归档一致，日志路径及编译事件规则按新框架接口适配。完整日志、旧配置快照和 JIT 缓存仍保留在本机旧目录，不随 Git 分发。
