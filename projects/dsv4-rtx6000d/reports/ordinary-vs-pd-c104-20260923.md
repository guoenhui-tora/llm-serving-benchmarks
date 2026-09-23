# C104：普通四服务与3P1D的同16卡对照

**PD相对普通的完整窗口吞吐变化+20.60%，goodput变化+38.17%；Mean TTFT变化+72.51%，Mean TPOT变化-32.93%。** 双方真实16384输入/1024输出、全系统C104、TP2×DP2 EP on、DSpark K5、相同16张GPU，每档208条完整预热＋三轮各416条正式。普通每个服务独立完成prefill和decode，PD为三个P加一个D。

| 指标 | 普通四个完整服务，16卡 | 3P1D，16卡 | PD相对变化 |
| --- | ---: | ---: | ---: |
| 输出吞吐，tok/s | 2256.61±19.32 | 2721.58±30.78 | +20.60% |
| Mean TTFT，s | 5.487 | 9.466 | +72.51% |
| Mean TPOT，ms | 39.209 | 26.298 | -32.93% |
| goodput，req/s | 1.514±0.058 | 2.091±0.027 | +38.17% |
| 联合SLO达标率 | 68.67% | 78.69% | +10.02个百分点 |


SLO为逐请求TTFT≤10秒且TPOT≤50ms；goodput为每轮达标请求数/完整HTTP窗口再取三轮均值。双方均经过同版本代理，吞吐包含输入计算、排队及生成；PD还包含KV传输。Mean TTFT低于10秒不等于绝大多数请求满足SLO。普通组为后续独立启动补测，未与PD同时运行，保留不同时间段的资源背景限制。

## 普通组三轮

| 轮次 | 输出tok/s | Mean TTFT，s | Mean TPOT，ms | goodput req/s | 联合SLO |
| --- | ---: | ---: | ---: | ---: | ---: |
| measurement-01 | 2263.58 | 5.458 | 39.189 | 1.536 | 69.47% |
| measurement-02 | 2271.47 | 5.479 | 38.822 | 1.557 | 70.19% |
| measurement-03 | 2234.77 | 5.523 | 39.617 | 1.448 | 66.35% |

普通组1248条中857条联合达标，TTFT超标216条，TPOT超标196条，同时超标21条。对应PD的逐轮结果和传输验收见[C104/D112报告](pd-16k-3p1d-c104-d112-20260923.md)。没有插值普通C112或其他未测配置。

## 固定配置与验收

- 普通四服务均4GPU，TP2×DP2、EP on、K5，每DP引擎S32、Graph192、budget16384；PD的P同为S32/Graph192，D为每DP引擎S56、服务总容量112、Graph336，budget同为16384。容量不同是已选服务配置的一部分，不把客户端C理解成每个服务运行104条。
- 普通r0/r1/r2分别位于46/47/48后四卡，r3位于46前四卡。与PD使用相同GPU集合和对应CPU/NUMA绑定；45代理和客户端保持原绑定。普通代理为least-inflight分流，PD代理选择P再调用D。
- 固定vLLM0.29.0镜像ID `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`；模型 `/data/models/DeepSeek-V4-Flash-0731-NVFP4`，NVFP4权重、FP8 KV、DSpark K5 greedy草稿设置不变，prefix caching关闭。
- 数据 `govreport-16k-prefix-1024-v2.jsonl`，SHA256 `025246b18d8ba7e1bc4d98c1a20b32b570860bb06a0baf0cea4ae97403d982e7`；固定前416条真实前缀，预热前208条，temperature0、ignore_eos=true、request_rate=inf。
- 继承并逐文件校验历史普通O128缓存，覆盖已验证的native MXFP4、DP profiling回移和target/input/draft预热模块；普通模型实际CLI保持历史基线不变，没有改公共代码或kernel补丁。16个worker均两遍cache一致、临时allocated显存增量0。
- 一次启动、24条功能、208条完整预热、三轮各416正式，无性能重试。普通正式1248成功/0失败，每轮真实计算输入6815744 tokens、输出425984 tokens；四个服务及每服务两个DP引擎均参与。无抢占，正式已知JIT共0条，全部轮次保留。
- PD对应三轮全部D远端输入命中、整段prefill重算0，详见原报告。两组均保留完整窗口，没有剔除首批或收尾来改善指标。
- case预算5400秒，协议预算3600秒；节点有既有CPU常驻负载，保存前置进程与运行期CPU/GPU遥测，未更改宿主功耗/时钟。脚本自动推进并清理；运行期源码/config未变化，结束后所有本次服务和观察进程已清理，保留JIT缓存。

## 复现与证据

[对照JSON](../data/ordinary-vs-pd-c104-20260923.json)、[六轮CSV](../data/ordinary-vs-pd-c104-20260923.csv)、[普通组实际命令](ordinary-vs-pd-c104-20260923-commands.md)。普通完整工作区仅45本机：`experiments/dsv4-ordinary-c104-20260923/`，原始结果 `results/O104-01/`；PD工作区 `experiments/dsv4-pd-c104-20260923/results/P104-01/`。准备入口为 `scripts/prepare104.py`，运行入口为 `scripts/run_case.py O104`；复现须使用新owner/run-root并重新检查空闲资源和缓存，不覆盖已有结果。
