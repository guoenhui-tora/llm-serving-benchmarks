# PD夜间位置对照进度（2026-09-21）

**同48八卡、GovReport C32下，普通双TP4为1026.74±1.52，修正UCX可达性的1P1D为911.15±1.36 output tok/s，PD低11.26%。** 两模式正式三轮无已知编译事件，工作量通过。当前是位置阶段的部分结果；跨节点A3/A4、16K、DP/DSpark和配比仍在执行队列，不能据此结束PD研究。

| case | GPU服务数×每服务GPU | 部署 | output tok/s（三轮均值±样本SD） | Mean TTFT秒 | Mean TPOT毫秒 |
| --- | --- | --- | ---: | ---: | ---: |
| A1 | 普通2×4，总8 | 48前后四卡 | 1026.74±1.52 | 3.992 | 27.115 |
| A2 | P1×4+D1×4，总8 | 同48，原UCX限制 | 功能失败，未测性能 | — | — |
| A2T | P1×4+D1×4，总8 | 同48，允许TCP/all接口 | 911.15±1.36 | 4.274 | 28.909 |

固定vLLM0.29.0、模型NVFP4及FP8 DSV4 KV；TP4/PP1/DP1、EP/DSpark off、context16384、prefill8192、max-num-seqs32、原decode Graph及已验证mHC启动预热。45代理CPU16–19/NUMA1、客户端20–23/NUMA1；48前/后服务CPU0–15/NUMA0及32–47/NUMA2、GPU0–3及4–7。普通least-inflight，PD同一请求先P后D，C32为全系统总量。

每轮原GovReport8K前128条，输入1,042,149、输出131,072 tokens。quick固定一轮完整128请求预热、再测三轮；PD另有一条显式C1功能请求。所有预热与失败保留；不是从最快轮选三次。计时包含45代理、HTTP、P、KV及D。A1预热16条其他JIT事件，正式三轮0；A2T预热及正式均0。

## 真实传输与失败反例

A2T三轮共384请求、1536次rank传输、88,349,460,480 bytes；外部命中3,126,447，D computed-prefill0，P计算同量输入。传输/通知失败、KV过期与抢占均0。结合计数排除了D重算整个输入；不宣称完成全面数值质量验证。

A2原配置在C1遇到UCX no active messages transport/no route to本机另一NIC地址，四rank NIXL失败、传输0bytes。**HTTP仍返回1024输出，远端命中也记8631；这不代表KV成功。** 联合验收拒绝该case并清理，未放宽规则。固定base_worker._handle_failed_transfer仍有HMA失败处理TODO，仅非HMA才填invalid_block_ids；DSV4的混合KV布局触及此边界，不能仅依赖kv_load_failure_policy=fail。

唯一修复候选A2T将容器UCX_TLS增加tcp、UCX_NET_DEVICES改all，未改宿主路由。实际NIC0–3接收计数每轮约29.90GB，与KV体量相符，发送计数0的驱动/loopback语义未确定；不称纯CUDA IPC或零网卡开销。单rank NIXL观测时间约23ms，含轮询/调度观察，不能作为线速或请求关键路径。

## 资源与复现

A1三轮路由均64/64，代理记录在途峰值32。GPU频率约2419MHz，温度约56°C；前后GPU平均利用率约94%–99%，无明显持续降频。服务CPU集合约34%–40%忙碌，整机约16%–17%；这不是CPU独占证明。A2T正式窗口代理/客户端一次CPU抽查约单核8%/11%，不是全窗口上界。

实际服务/代理命令及每轮协议/计数见[执行快照](../data/pd-overnight-placement-progress-20260921.json)，均直接从运行产物导出。完整原始产物仅45本机：`experiments/dsv4-pd-overnight/results/{A1,A2,A2T}-01/`，含resolved/command/preflight/affinity、worker-warmup、server/proxy日志、每轮raw/metrics/计数、资源时间序列与冻结检查；`reports/`保存资源复核和缓存继承哈希。A1/A2/A2T收尾均无代码配置变动、无清理错误。

下一步与预算见[夜间计划](pd-overnight-plan-20260921.md)。本页为约03:50的阶段记录，后续完整报告优先于此进度页。
