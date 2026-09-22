# D96 decode扫描：Graph覆盖遗漏与单轮修正验证

**8K共享前缀下，D96/C80扩展Graph后，Mean TPOT从35.86降到10.06ms，完整批次吞吐从约1984升到6181 output tok/s。** 固定镜像源码和实际配置表明，旧192-token上限未覆盖K5大batch的目标验证与草稿形状。结果强烈支持Graph覆盖遗漏是本次拐点的主要原因，不能据此认定GPU存在64请求的固有性能上限。修正后只测了C80一轮；真实16K PD的恢复幅度尚未验证。

## 结果与适用范围

单节点单服务、4卡、TP2×DP2 EP on、DSpark K5。D96指每个DP engine的`max-num-seqs=48`，服务合计容量约96；C是客户端全服务并发。每条8192输入／1024输出，复用同一个共享前缀，不是历史真实GovReport各请求独立文本的PD负载。

| Graph上限，tokens | C | 测量方式 | output tok/s | Mean TPOT，ms |
| ---: | ---: | --- | ---: | ---: |
| 192 | 64 | 原扫描4批 | 4290.12 | 9.57 |
| 192 | 80 | 原扫描4批 | 1983.99 | 35.86 |
| 192 | 96 | 原扫描4批 | 2366.89 | 35.68 |
| **288** | **80** | **完整生成预热1轮＋正式1轮** | **6180.75** | **10.06** |

旧C80逐批TPOT为38.31、36.48、33.22、35.43ms，新一轮10.06ms；新预热轮也为10.11ms，但预热不进入正式结果。新旧吞吐约3.12倍、TPOT下降72%，属于单轮诊断与历史扫描比较，未进行同批交错A/B或三轮稳定性确认，不报告提升的置信区间。

原扫描各C独立reset＋C条1-token prime，然后4批各C条完整生成，没有独立完整生成预热。旧吞吐用输出总量除以四批时长之和；每批时长取最长单请求，未精确覆盖线程发起偏移，会略高估旧吞吐。新实验用`run_batch`外层实际墙钟，包含发起、残余prefill、首token等待与收尾。两者均不是纯GPU decode计算时间，也不能与完整PD吞吐直接比较。旧C64首批附近含采样JIT，不称为clean基线。

## 为什么192只够每引擎32条K5目标验证

本次固定镜像K5关闭adaptive verification，目标验证每请求为5个草稿token＋1个bonus token；anchor草稿每请求5个query。Graph尺寸按执行tokens而非请求数计：

| 每DP engine请求数B | 目标验证6×B | anchor草稿5×B | 旧Graph最大192 |
| ---: | ---: | ---: | --- |
| 32 | 192 | 160 | 覆盖 |
| 40 | 240 | 200 | 两条路径均超出 |
| 48 | 288 | 240 | 两条路径均超出 |

固定vLLM0.29.0镜像内源码：`vllm/v1/worker/gpu/model_runner.py:415`计算目标`decode_query_len`；`model_states/interface.py:236`的bonus默认为1；`spec_decode/dspark/speculator.py:52`设置anchor草稿query长度；`cudagraph_utils.py:252`跳过超过捕获上限的形状，`dispatch`在409–436行无匹配时返回`CUDAGraphMode.NONE`。

旧列表里包含40、48、96并不能证明相应请求batch已经覆盖。实际每步batch、DP同步与padding仍需逐步分派记录；本次未加入dispatch探针，没有实测Graph命中率。缺少Graph不代表请求被拆成两批，也不要求发生新的JIT：内核已编译仍可能通过非Graph路径执行。

历史PD D96的`N22P13-01/d0/command.json`同样为S48、Graph最大192；D64的N22P12-01/N22P14-01则为S32、Graph最大192。**历史性能数字保留，但“D96天然慢、因此容量应固定64”的解释不成立；它们只代表旧Graph覆盖下的结果。**

## 实际修改与验证证据

服务argv仅修改`compilation-config`，另改容器名称／label：在原捕获列表增加200、240、288，最大值192→288。实际列表：

```json
[5, 6, 10, 12, 20, 24, 40, 48, 60, 72, 80, 96, 120, 144, 160, 192, 200, 240, 288]
```

其余保持固定：image ID `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`；模型`/data/models/DeepSeek-V4-Flash-0731-NVFP4`；FP8 E4M3 KV；budget16384；max-model-len32768；显存比例0.90；greedy draft／standard rejection；V2、async、原定向预热和兼容补丁。45节点GPU4–7，服务CPU32–47／NUMA2；HTTP客户端运行于控制进程，未另设CPU绑定，不能套用其他实验的客户端亲和性。实际命令由产物导出，见[命令快照](decode-graph-coverage-20260923-commands.md)。

新实验每阶段独立reset＋80条1-token prime，再80条完整生成；预热1轮、正式1轮，无重试。启动上限900秒，每阶段300秒。实际启动126.17秒，正式批次13.254秒，启动到清理164.86秒。用户明确要求单轮快速验证，不是默认三轮quick。

- 正式80成功／0失败，655360输入、81920输出tokens，每条输出1024且finish reason为length。
- 输入预热模块4个worker全部完成；目标Graph捕获13→16个，草稿14→17个。每worker Graph池约0.85→1.12GiB，日志逻辑KV容量61,019→60,685 tokens。
- 正式窗口未识别JIT／编译事件，0抢占，无OOM或请求错误。原coordinator out-of-order warning保留。
- 两DP engine分别处理40条请求，采样running各达到40；有限采样不是逐步精确batch。
- 测量前后前缀命中96.875%，各engine残余prefill为10240 tokens，即每条256；它是高前缀命中的decode参照，不是完全零prefill。
- 草稿接受率约93.96%，旧C80约92%；共享前缀局部性、接受率与批次调度限制了向真实16K PD的外推。

## JIT预热的处理结论

三个采样内核`_compute_local_logits_stats_kernel`、`_rejection_kernel`、`_resample_kernel`在本次完整生成预热各出现4条首次事件，正式轮为0。这不是decode-only专用缺口，[此前报告](input-kernel-warmup-20260921.md)已记录普通与PD功能／HTTP预热中的同类事件。

现有定向模块覆盖top-k和草稿输入准备，不覆盖所有采样形状；1-token prime也不能替代完整生成预热。先保留“启动覆盖＋完整HTTP预热＋正式测量”，只有完整预热后仍新增specialization再扩展定向模块。本轮未改预热补丁、forward、精度或事件规则。**0已知JIT与Graph覆盖充分是两项独立验收。**

## 复现与原始产物

[逐批CSV](../data/decode-graph-coverage-20260923.csv)归档原扫描12批及新预热／正式各1批；CSV明确阶段、计时方式及来源，不将预热混入正式统计。命令、GPU／镜像／Git身份、HTTP逐请求记录、metrics、CPU/GPU采样及日志保存在下列工作区，**仅45节点本机可用**：

- [原扫描报告](../../../experiments/dsv4-decode-batch-d96/reports/scan.md)及其`results/c64`、`c80`、`c96`。
- [Graph288单轮报告](../../../experiments/dsv4-decode-graph288-c80/report.md)、[正式逐请求结果](../../../experiments/dsv4-decode-graph288-c80/results/measurement/summary.json)、[完整服务日志](../../../experiments/dsv4-decode-graph288-c80/server.log)。
- 原共享前缀池、客户端及日志配置也只在工作区，命令快照不能脱离这些依赖直接复现。

两次实验容器均已清理，JIT缓存保留。修正后的本地C64/C96及真实16K PD未测；下一步优先验证补齐Graph的16K 3P1D、D96/C96，再评估D容量与配比。
