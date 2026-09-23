# 固定版本coordinator与API统计warning核查

## 结论与适用范围

本轮两类warning本身不足以推翻已经通过的HTTP 256请求、16384输入、1输出与服务计数一致性验收。`Received stats for out-of-order step`会涉及DP负载均衡快照，可能影响路由、排队和吞吐，不能淡化为完全无性能影响的打印。`api_server_count more than 1; disabling stats logging`仅关闭默认控制台统计logger，未关闭Prometheus。保留所有warning，不改gate、不追补性能轮。

两组完整日志的coordinator warning分别为EP on 721条、EP off 708条，依据各组保存的`kernel-checks.json`。这些是完整服务生命周期计数，不是正式benchmark窗口事件计数。

核查限于实际固定镜像`sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`。仅运行无GPU、无网络、`--rm --pull never`的临时源码读取容器，未修改运行配置、公共源码、缓存或验收标准。

## coordinator warning的路径

1. `vllm/v1/engine/core.py:2161–2176`：各DP engine在request counts变化时，把waiting/running/KV及本地step/wave送入`client_index=-1`输出队列。
2. `vllm/v1/engine/core.py:1839–1847`：`client_index=-1`消息送独立coordinator socket，普通请求结果另走API socket（1855–1858）。
3. `vllm/v1/engine/coordinator.py:378–419`：warning在390–415比较**跨engine共享**的最近wave/step；某engine报告落后于另一engine最近报告即可触发，不必然证明同一engine消息自身逆序。warning之后416–419仍写入该engine的waiting/running/KV快照，并设置`stats_changed`；没有因此抛异常、丢弃请求或改变实际scheduler队列。
4. `vllm/v1/engine/coordinator.py:259–282`：coordinator按间隔发布load快照；lockstep模式试图通过step边界快照和至少50 ms等待对齐。上述警告意味着不能假定所有快照均来自统一step。
5. `vllm/v1/engine/core_client.py:1384–1405`：API侧读最新发布消息并更新`lb_engines`。`1472–1517`使用waiting/running/KV给新请求选择DP engine，其中分数下界为`client_count × 本API在该engine的在途请求数`，加上KV压力相关waiting惩罚。该本地在途下界能缓解陈旧快照影响，但不能证明影响为零。

因此，警告可能通过新请求路由间接改变实际工作分布与排队，且warning输出也可能有开销。本轮没有量化这些影响；不能据此认定全部EP差异来自模型计算或通信，也不能把这条warning直接解释为请求结果损坏。

wave完成/启动处理位于`coordinator.py:421–453`，与warning后的计数更新是独立分支；该warning本身没有重写wave，也没有调用引擎内部token调度器。

## Prometheus与控制台统计的区别

- `vllm/v1/engine/async_llm.py:806–816`：API直接把engine输出携带的scheduler stats与请求iteration stats传给logger manager；不读取coordinator的`request_counts`缓存。
- `vllm/v1/metrics/loggers.py:1338–1350`：`client_count>1`时不注册默认`LoggingStatLogger`，以免控制台打印单API的不完整统计。`1370–1372`随后仍注册`PrometheusStatLogger`。
- `vllm/v1/metrics/prometheus.py:39–50`：设置多进程目录时通过`MultiProcessCollector`构造registry。
- `vllm/v1/metrics/loggers.py:493–523`：running/waiting及waiting-by-reason gauge使用`multiprocess_mode="mostrecent"`。`1108–1123`直接由scheduler stats更新；waiting总数等于capacity加deferred。它们仍是异步观测，不是逐step同步轨迹。
- `vllm/v1/metrics/loggers.py:1221–1250`：请求成功counter、queue/prefill直方图、输入输出token直方图来自完成请求。HTTP结果独立经过客户端成功数、长度及正TTFT检查。本轮通过的客户端与服务计数交叉验收仍有意义。

瞬时gauge仍受API异步更新与约5秒采样限制；时间加权均值及采样最大值不能解释为连续真实峰值。原始warning需与完整研究报告共同保留，不能宣称路由完全同步或统计没有任何观测误差。

## “实际计算输入”的证据边界

`vllm/v1/metrics/loggers.py:1240–1245`定义`request_prefill_kv_computed_tokens`为`num_prompt_tokens - max(num_cached_tokens, 0)`。因此它是服务记账层面的未命中前缀计算量证据，与关闭prefix caching及零命中相互校验，**不是独立GPU硬件计算计数器**。本轮不能仅凭该指标宣称已经逐kernel验证每个输入token的物理计算工作。

## 本机源码证据与身份

源码提取自固定镜像内`/usr/local/lib/python3.12/dist-packages/vllm/`。完整源码与SHA清单位于46本机`experiments/dsv4-prefill-ep-20260923/reports/pinned-warning-source/manifest.json`；该原始证据仅46本机保存。文件按原源码保留行号。

| 镜像内相对路径 | 本机提取文件 | SHA256 |
| --- | --- | --- |
| v1/engine/coordinator.py | `v1__engine__coordinator.py` | `b2a9a2f32e84b5ce7b4979b41ab4749deb1fb20f157086a6a9e9185a03ad78d9` |
| v1/engine/core_client.py | `v1__engine__core_client.py` | `969832582303057621f5c3ced4561a6bcc2ecc73d4d79b4dd97da0e84cb43e6c` |
| v1/engine/core.py | `v1__engine__core.py` | `f990e260cebf0826d08cfd30955c0c7a85b9af00504841b91759ca5bc7b036f6` |
| v1/engine/async_llm.py | `v1__engine__async_llm.py` | `46b1df9cb7466552e9ebd52f896eecc7e9e81bbf966077171f320366b7fcbbad` |
| v1/metrics/loggers.py | `v1__metrics__loggers.py` | `d5a91626cf79118f98ae6438c842cb30fc44ca14463c7e2af5871e3da8e51bcf` |
| v1/metrics/prometheus.py | `v1__metrics__prometheus.py` | `a835583311820f2e374a9e42ae229de1195081b288abc40ccc08eb368d6dc228` |
