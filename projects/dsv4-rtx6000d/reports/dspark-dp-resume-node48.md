# 48节点：DP＋DSpark补丁后续测（2026-09-20）

**四组核心配置均已按预算执行，性能验收部分完成：2组PASS、2组PARTIAL。** TP2×DP2、K3、EP on/off的补丁加载、Graph和完整C32请求均完成验证，未复现原DP profiling断言。结果限定本机GPU4–7、固定vLLM 0.29.0＋指定本地补丁、GovReport前128条／1024输出及本次持久缓存和DOCA后台负载。

**EP off的完整同机对照中，K3输出吞吐为855.20 ± 4.22 tok/s，较off的735.77 ± 42.01提高16.23%。** Mean TTFT／TPOT分别降低35.87%／9.44%，但P95 TPOT增加15.13%，P95端到端延迟增加0.97%，不能称为所有延迟都改善。K3三轮CV为0.49%；这些是本批顺序启动的结果，不代表独立重启或长期无干扰性能。

EP on的off只有1轮clean，K3有2轮clean，不能建立完整的同EP收益对照。EP off的off取得3轮，但吞吐CV为5.71%，协议PASS不代表性能稳定。各组最终结果见下表；PARTIAL样本仅作描述性统计，不拼接旧批次、不算作完成的三轮性能基线。

核心已有PARTIAL，按执行规则没有启动K5扩展；没有追加其他K值、并发、双部署或独立重启复测。旧[48节点报告](dspark-k-sweep-node48.md)和[CSV](../data/dspark-k-sweep-node48.csv)保持原判定，没有并入本次数据。DP修复来源和加载边界见[兼容性说明](dspark-compatibility.md#dpdspark回移已合并的上游修复)。

## 结果与适用范围

完整精度见[本轮44列逐轮CSV](../data/dspark-dp-resume-node48.csv)。下表只统计协议选中的clean轮，均值±样本标准差；n=1不计算SD/CV。P95列是各轮P95的平均，不是合并全部请求后的P95。不同EP分开启动，不能将所有差异归因于EP。

| 配置 | 状态 / clean n | output tok/s，均值±SD | CV | 相对同EP off | Mean / P95 TTFT ms | Mean / P95 TPOT ms | 草稿接受率% / 接受长度 |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| EP on / off | PARTIAL / 1 | 759.04（n=1） | — | —（参照未完成） | 4656.56 / 11218.23 | 34.269 / 36.968 | — / — |
| EP on / K3 | PARTIAL / 2 | 961.61 ± 7.82 | 0.81% | —（样本未齐） | 3396.92 / 11495.78 | 28.362 / 38.783 | 66.60 / 1.9979 |
| EP off / off | PASS / 3 | 735.77 ± 42.01 | 5.71% | 参照 | 5869.66 / 14434.37 | 35.792 / 39.016 | — / — |
| EP off / K3 | PASS / 3 | 855.20 ± 4.22 | 0.49% | +16.23% | 3764.00 / 13700.48 | 32.414 / 44.921 | 65.60 / 1.9680 |

接受长度只计接受的草稿token，不含目标补充token。表中接受率/长度为逐轮比率的均值；分子分母汇总后的加权值另见下文。DSpark的ITL是流式事件间隔，不等同逐token计算时间；不能用其ITL直接替代TPOT。

选中clean轮的完整汇总（各列n见上表）：

| 指标 | EP on / off | EP on / K3 | EP off / off | EP off / K3 |
| --- | ---: | ---: | ---: | ---: |
| 请求/s | 0.741（n=1） | 0.939 ± 0.008 | 0.719 ± 0.041 | 0.835 ± 0.004 |
| Mean TTFT ms | 4656.560（n=1） | 3396.916 ± 92.356 | 5869.664 ± 478.048 | 3764.000 ± 151.623 |
| P95 TTFT ms | 11218.226（n=1） | 11495.785 ± 234.797 | 14434.374 ± 948.243 | 13700.484 ± 11.068 |
| Mean TPOT ms | 34.269（n=1） | 28.362 ± 0.399 | 35.792 ± 0.408 | 32.414 ± 0.404 |
| P95 TPOT ms | 36.968（n=1） | 38.783 ± 1.756 | 39.016 ± 0.449 | 44.921 ± 1.385 |
| Mean ITL ms | 34.269（n=1） | 84.929 ± 0.480 | 35.792 ± 0.408 | 96.086 ± 0.321 |
| P95 ITL ms | 28.393（n=1） | 617.779 ± 2.407 | 27.131 ± 0.456 | 769.451 ± 1.505 |
| Mean E2EL ms | 39713.780（n=1） | 32411.384 ± 315.471 | 42485.332 ± 883.019 | 36923.314 ± 270.049 |
| P95 E2EL ms | 46112.371（n=1） | 44550.226 ± 19.392 | 50889.227 ± 1419.443 | 51382.615 ± 1616.068 |
| 客户端计时 s | 172.680（n=1） | 136.309 ± 1.108 | 178.527 ± 10.116 | 153.266 ± 0.758 |
| 草稿接受率 % | — | 66.598 ± 0.845 | — | 65.601 ± 0.958 |
| 接受草稿tokens / draft | — | 1.998 ± 0.025 | — | 1.968 ± 0.029 |

## 全部轮次与时间成本

每完整轮均要求128成功、0失败、输入1,042,149／输出131,072 tokens；下列计数不含独立功能探测。`accepted=true`只表示该轮被协议选中，PARTIAL组仍不进入完整基线。事件数是日志匹配行数，不是独立编译次数。

| 配置 | 轮次 | output tok/s | 事件行数 | 接纳 | 累计clean |
| --- | ---: | ---: | ---: | --- | ---: |
| EP on / off | 1 | 566.93 | 34 | 否 | 0 |
| EP on / off | 2 | 687.79 | 20 | 否 | 0 |
| EP on / off | 3 | 693.35 | 14 | 否 | 0 |
| EP on / off | 4 | 759.04 | 0 | 是 | 1 |
| EP on / off | 5 | 638.26 | 22 | 否 | 1 |
| EP on / off | 6 | 652.61 | 14 | 否 | 1 |
| EP on / off | 7 | 754.27 | 4 | 否 | 1 |
| EP on / off | 8 | 742.90 | 6 | 否 | 1 |
| EP on / off | 9 | 752.30 | 4 | 否 | 1 |
| EP on / off | 10 | 744.68 | 4 | 否 | 1 |
| EP on / off | 11 | 721.91 | 4 | 否 | 1 |
| EP on / off | 12 | 747.10 | 4 | 否 | 1 |
| EP on / K3 | 1 | 548.43 | 92 | 否 | 0 |
| EP on / K3 | 2 | 799.83 | 26 | 否 | 0 |
| EP on / K3 | 3 | 801.56 | 12 | 否 | 0 |
| EP on / K3 | 4 | 794.74 | 14 | 否 | 0 |
| EP on / K3 | 5 | 938.13 | 4 | 否 | 0 |
| EP on / K3 | 6 | 738.69 | 18 | 否 | 0 |
| EP on / K3 | 7 | 972.23 | 4 | 否 | 0 |
| EP on / K3 | 8 | 971.41 | 10 | 否 | 0 |
| EP on / K3 | 9 | 949.96 | 2 | 否 | 0 |
| EP on / K3 | 10 | 956.08 | 0 | 是 | 1 |
| EP on / K3 | 11 | 937.41 | 4 | 否 | 1 |
| EP on / K3 | 12 | 967.14 | 0 | 是 | 2 |
| EP off / off | 1 | 505.63 | 52 | 否 | 0 |
| EP off / off | 2 | 770.57 | 8 | 否 | 0 |
| EP off / off | 3 | 618.03 | 22 | 否 | 0 |
| EP off / off | 4 | 677.21 | 16 | 否 | 0 |
| EP off / off | 5 | 759.33 | 6 | 否 | 0 |
| EP off / off | 6 | 711.69 | 6 | 否 | 0 |
| EP off / off | 7 | 701.46 | 4 | 否 | 0 |
| EP off / off | 8 | 731.17 | 0 | 是 | 1 |
| EP off / off | 9 | 690.27 | 2 | 否 | 1 |
| EP off / off | 10 | 685.65 | 8 | 否 | 1 |
| EP off / off | 11 | 779.90 | 0 | 是 | 2 |
| EP off / off | 12 | 696.25 | 0 | 是 | 3 |
| EP off / K3 | 1 | 492.84 | 124 | 否 | 0 |
| EP off / K3 | 2 | 737.57 | 22 | 否 | 0 |
| EP off / K3 | 3 | 863.81 | 10 | 否 | 0 |
| EP off / K3 | 4 | 856.23 | 8 | 否 | 0 |
| EP off / K3 | 5 | 855.31 | 2 | 否 | 0 |
| EP off / K3 | 6 | 861.99 | 4 | 否 | 0 |
| EP off / K3 | 7 | 873.37 | 4 | 否 | 0 |
| EP off / K3 | 8 | 851.30 | 4 | 否 | 0 |
| EP off / K3 | 9 | 858.88 | 0 | 是 | 1 |
| EP off / K3 | 10 | 856.14 | 0 | 是 | 2 |
| EP off / K3 | 11 | 850.59 | 0 | 是 | 3 |

全部47个完整轮共 **6,016成功、0失败，输入48,981,003／输出6,160,384 tokens**。全部轮次含事件的诊断统计如下，不能作为clean性能结果：

| 配置 | 完整轮数 | 全轮输出 tok/s，均值±SD | CV |
| --- | ---: | ---: | ---: |
| EP on / off | 12 | 705.09 ± 59.81 | 8.48% |
| EP on / K3 | 12 | 864.63 ± 130.99 | 15.15% |
| EP off / off | 12 | 693.93 ± 74.40 | 10.72% |
| EP off / K3 | 11 | 814.37 ± 112.83 | 13.85% |

| 配置 | run内预检 s | wait_ready至就绪 s | 协议 s | run总耗时 s / min | 停止原因 |
| --- | ---: | ---: | ---: | ---: | --- |
| EP on / off | 44.964 | 245.194 | 2556.480 | 2854.152 / 47.569 | round_budget |
| EP on / K3 | 44.139 | 330.416 | 2180.038 | 2558.750 / 42.646 | round_budget |
| EP off / off | 44.664 | 247.369 | 2608.765 | 2904.806 / 48.413 | target_reached |
| EP off / K3 | 44.468 | 340.525 | 2106.859 | 2495.980 / 41.600 | target_reached |

核心首个run开始至最后run结束共199.42分钟，四个run耗时之和180.23分钟；差额19.19分钟为组间空闲、观察及准备间隔。北京时间区间2026-09-20 12:06:27–15:25:52。run含内部预检、启动、探测、协议和清理；外部配置准备与preflight不计入run。协议包含客户端初始化、测量和检查，拒绝轮同样计入成本。

## 草稿接受统计与容量

接受统计使用约1秒一次的原始`/metrics`快照，在每轮`started_at`至`finished_at`内取前后计数器增量；窗口含客户端初始化和收尾。逐rank检查无重置、成功请求增量合计128后，累加两个DP rank的分子与分母。字段为`spec_decode_num_draft_tokens_total`、`spec_decode_num_accepted_tokens_total`、`spec_decode_num_drafts_total`，标签`engine=0/1`，不重复计算TP worker。逐位置计数使用`spec_decode_num_accepted_tokens_per_pos_total`，原始各rank计数保存在CSV JSON字段及本机分析JSON中。

接受率＝Σ接受草稿tokens／Σ提出草稿tokens；接受长度＝Σ接受草稿tokens／Σdrafts，**不加目标补充token**。固定镜像的`SpecDecodingStats.observe_draft`逐draft累计这些计数；vLLM日志显示的惯例长度会加1，与本报告口径不同。快照缺失、重置或请求窗口无法核实时字段留空，不用K推算。off不适用，留空不填0。本轮47个完整窗口的成功请求增量均为128，没有计数器重置；窗口内最大快照间隔为1.0003秒，首末快照均处于对应轮次内，独立功能探测未计入。

| K3组 / 样本范围 | 提出草稿tokens | 接受草稿tokens | drafts | 加权接受率 % | 加权接受长度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EP on / K3 / 全部完整轮（n=12） | 1,583,592 | 1,045,219 | 527,864 | 66.0030 | 1.980091 |
| EP on / K3 / 选中clean轮（n=2） | 262,368 | 174,722 | 87,456 | 66.5942 | 1.997827 |
| EP off / K3 / 全部完整轮（n=11） | 1,460,448 | 955,141 | 486,816 | 65.4005 | 1.962016 |
| EP off / K3 / 选中clean轮（n=3） | 397,551 | 260,771 | 132,517 | 65.5944 | 1.967831 |

| 配置 | KV容量 tokens，DP0 / DP1 | KV使用率峰值%，DP0 / DP1 | 每rank最大活动请求 | 抢占增量总和 |
| --- | ---: | ---: | ---: | ---: |
| EP on / off | 78,014 / 78,014 | 23.698 / 23.779 | 16.0 / 16.0 | 0 |
| EP on / K3 | 65,099 / 65,099 | 28.376 / 28.335 | 16.0 / 16.0 | 0 |
| EP off / off | 78,506 / 78,506 | 23.505 / 23.468 | 16.0 / 16.0 | 0 |
| EP off / K3 | 65,618 / 65,618 | 28.089 / 28.170 | 16.0 / 16.0 | 0 |

CSV容量取两个rank最小值，使用率取rank采样峰值，抢占为无重置窗口的计数增量之和。DSV4容量为压缩缓存的等效token数，不可当作普通全注意力KV容量。采样峰值不是连续精确峰值；本负载未见OOM或抢占压力。

## 日志与资源背景

四组kernel必需日志检查均PASS；K3两个EP下均有四个worker的`LOCAL_DSPARK_FORMAT_FIX`和`UPSTREAM_DSPARK_DP_PROFILE_FIX: PR #54856 facd9a74a1`标记。主模型保留NVFP4／`FLASHINFER_CUTLASS`，草稿为原生MXFP4／`DEEPGEMM_MXFP4`；attention为`FLASHINFER_MLA_SPARSE_DSV4`，FP8 KV、V2、async及关闭autotune均在实际日志确认。两个DP rank、EP设置、Graph捕获和C32请求完成均有证据；这些标记不等于完整kernel或数值质量验证。

保留SM120 SymmMem不可用、TP2 FlashInfer All Reduce不可用、FP8 scale、NVFP4实验性、FP4 indexer参数弃用、共享内存等待和DP统计乱序等警告及提示。事件窗口含真实TileLang编译开始/完成以及JIT monitor记录，后者不能一律解释为实际CPU重编译；没有放宽gate或隐藏warning。

| 配置 | warning匹配行数 | kernel检查 |
| --- | ---: | --- |
| EP on / off | 605 | PASS |
| EP on / K3 | 259 | PASS |
| EP off / off | 579 | PASS |
| EP off / K3 | 273 | PASS |

CPU/GPU约每10秒采样。下表CPU为每个clean阶段首末`/proc/stat`增量，包含初始化/收尾及后台任务，不是服务进程独占率。服务CPU32–47/NUMA2，客户端48–51/NUMA3；SMT兄弟分别96–111及112–115。

| 配置 / clean轮 | 整机% | 服务% | 服务SMT% | 客户端% | 客户端SMT% | DOCA逻辑CPU当量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EP on / off / 4 | 13.99 | 36.72 | 0.22 | 26.70 | 0.16 | 12.00 |
| EP on / K3 / 10 | 13.79 | 35.57 | 0.23 | 25.92 | 0.17 | 12.00 |
| EP on / K3 / 12 | 13.79 | 34.94 | 0.20 | 19.04 | 35.34 | 12.00 |
| EP off / off / 8 | 14.17 | 36.46 | 0.16 | 24.82 | 4.41 | 12.00 |
| EP off / off / 11 | 14.05 | 36.71 | 0.13 | 21.99 | 11.91 | 12.00 |
| EP off / off / 12 | 13.97 | 37.46 | 0.18 | 18.67 | 8.06 | 12.00 |
| EP off / K3 / 9 | 13.92 | 35.08 | 0.14 | 25.65 | 2.64 | 12.00 |
| EP off / K3 / 10 | 13.96 | 35.13 | 0.13 | 27.70 | 0.21 | 12.00 |
| EP off / K3 / 11 | 13.86 | 35.94 | 0.12 | 40.29 | 0.13 | 12.00 |

12个DOCA忙线程持续存在，逐进程/cgroup增量及整机/逐核时间线均保存。客户端和SMT背景有变化，不能称为无干扰对照，也不按假设比例修正吞吐。尚未隔离后台竞争、调度shape和吞吐波动的因果关系。

| 配置 | GPU温度°C | SM MHz | 功耗W | 显存MiB |
| --- | ---: | ---: | ---: | ---: |
| EP on / off | 39–62 | 2295–2430 | 80.02–333.45 | 75659–80091 |
| EP on / K3 | 39–62 | 2295–2430 | 78.74–334 | 75687–78355 |
| EP off / off | 39–62 | 2295–2430 | 80.03–316.5 | 75213–79835 |
| EP off / K3 | 39–61 | 2295–2430 | 78.57–323.51 | 75235–78773 |

上述范围覆盖所有测量阶段并含空闲区间，未见持续热降频证据，低频采样不能排除瞬时干扰。没有修改锁频、功耗或全局NUMA设置。服务/客户端Docker绑定和线程允许集合检查通过，但允许集合不证明全部映射页本地驻留。

EP on / off的独立页快照中，worker映射驻留页约81.31%–81.50%在NUMA2，约17.67%–17.86%在NUMA3。含文件/共享映射，不能跨worker相加或视为测量期持续分布。

EP on / K3的独立页快照中，worker映射驻留页约81.34%–81.63%在NUMA2，约17.61%–17.88%在NUMA3。含文件/共享映射，不能跨worker相加或视为测量期持续分布。

EP off / off未取得容器内实际页分布快照；宿主观察器未能读取root worker页映射，保留此证据缺项。

EP off / K3的独立页快照中，worker映射驻留页约81.16%–81.35%在NUMA2，约17.87%–18.05%在NUMA3。含文件/共享映射，不能跨worker相加或视为测量期持续分布。

## 固定身份、配置与复现

执行前`git pull`返回Already up to date，`git merge-base --is-ancestor e8c1da1 HEAD`通过；运行HEAD为`e8c1da1ce0aafe3331a7ff1dc68cad4980642796`。没有在运行中pull、修改源码/配置、访问其他节点或更换镜像。每组保存Git差异、解析配置、实际argv及源码哈希；收尾核对冻结哈希一致。

runner源码指纹`f6e873002b8cf61963bd221d63b0fe0a520e0bc6118f4ca63acfa54fc6726d52`。服务与客户端镜像均为`vllm/vllm-openai:v0.29.0`，固定ID`sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`；驱动580.159.04、RTX6000D/SM120。权重`/data/models/DeepSeek-V4-Flash-0731-NVFP4`，模型元数据/tokenizer及分片大小身份`1e1b7e54442bcf7f15ef55192641b5c5da360bcaa608236aa0d06b273aedf8b4`，未对全部权重内容重算哈希。数据SHA256为`33bdbd36a8b335300974e1b07aa66b2c24c88cc4c4b55a21350aa3564a945d53`；补丁manifest SHA256为`2e30906bd35789d8e7cd5940f8e28d5b03eb10aff9a014814a23f780b9a5f368`。

TP2、DP2、PP1、local DP2；每rank活动容量16、prefill8192，总C32，聚合prefill上限16384。上下文16384、显存比例0.90、FP8 E4M3 KV、block size256，V2＋async、FULL_DECODE_ONLY，prefix cache与FlashInfer autotune关闭。K3为greedy、standard rejection、adaptive verification关闭。off不带speculative-config和补丁环境；on同时启用量化分派和DP修复。只变指定EP及K/Graph；不运行replica_targets双部署。

客户端为固定镜像内vLLM原生bench，流式`/v1/completions`、全服务C32、GovReport固定前128条、每请求输出1024、seed0、temperature0、ignore_eos、request_rate=inf。每轮由实际tokenizer核验输入总量。短中文chat探测关闭thinking，独立于测量；无额外预热请求。`jit_clean`累计最先三轮clean，最多12轮/2700秒，每轮客户端上限900秒且受剩余预算限制，服务启动上限1800秒。同配置只启动一次，没有因JIT重启。

工作区续用`experiments/dspark-k-sweep-node48/`。候选由[公共DP正式campaign](../configs/campaigns/dspark-dp-clean.yaml)、[公共DP K5 recipe](../configs/recipes/tp2-dp2-dspark-k5.yaml)及[普通DP recipe](../configs/recipes/dual-tp2-dp2-epon.yaml)派生，固定公共model/runtime/client/workload和rear target；只使用其中一套四卡服务。全部四组核心及两组未启动的K5候选在首组启动前完成validate/plan、日志准备、preflight；on逐配置prepare及`--check`通过。探索配置不重复归档。

| configuration | EP原生开关 | DSpark | Graph捕获列表 / 上限 | 实测缓存目录尾段 |
| --- | --- | --- | --- | --- |
| tp2-dp2-epon-off | `enable-expert-parallel` | off | `[1, 2, 4, 8, 12, 16]` / 16 | `5523bf5440d44d81492d` |
| tp2-dp2-epon-k3 | `enable-expert-parallel` | 3 | `[3, 4, 6, 8, 12, 16, 24, 32, 36, 48, 64]` / 64 | `3598e4b546c34d2f05da` |
| tp2-dp2-epoff-off | `no-enable-expert-parallel` | off | `[1, 2, 4, 8, 12, 16]` / 16 | `85747b3279a84feb6a8c` |
| tp2-dp2-epoff-k3 | `no-enable-expert-parallel` | 3 | `[3, 4, 6, 8, 12, 16, 24, 32, 36, 48, 64]` / 64 | `777c2b11056c60cf2115` |

各candidate使用独立可写缓存。off分别从本机上轮同EP普通DP缓存复制已核对的`triton`、`tilelang`、`flashinfer`子目录；K3从同EP的修正版K5功能验证缓存复制同三类编译缓存，核对镜像、模型、其他服务参数一致。没有复制`vllm`缓存或旧`dspark-native-mxfp4`目录；本轮新补丁保持原样。来源run、源/目标目录和逐文件哈希见本机`evidence/dpfix/cache-seeding.json`，旧缓存未删除，各K不共享可写目录。缓存已存在不保证本进程覆盖全部运行shape。

下面完整服务命令原样取自本轮EP on／K3的`command.sh`。其他核心配置按上表差异及off/on环境规则由plan生成；重跑生成新的owner/容器名，不复用本次run-root。服务命令本身不能替代上面的客户端负载与协议。

```bash
docker run -d --pull never --name sb-78f14a0ffcea4ad9-server --label io.serving-bench.run=78f14a0ffcea4ad9 --network host --ipc host --gpus '"device=4,5,6,7"' -v /data/models/DeepSeek-V4-Flash-0731-NVFP4:/model:ro -v /home/enhui/.cache/serving-bench/vllm/3598e4b546c34d2f05da:/root/.cache:rw --cpuset-cpus 32-47 --cpuset-mems 2 --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --cap-add SYS_NICE -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e NCCL_DEBUG=WARN -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e VLLM_USE_V2_MODEL_RUNNER=1 -e PYTHONPATH=/root/.cache/dspark-native-mxfp4 -e SERVING_BENCH_DSPARK_DP_PROFILE_FIX=1 --entrypoint vllm sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 serve /model --served-model-name deepseek-v4-flash --host 127.0.0.1 --port 31249 --trust-remote-code --enable-auto-tool-choice --no-enable-prefix-caching --enable-expert-parallel --enable-chunked-prefill --jit-monitor-verbose --async-scheduling --tensor-parallel-size 2 --pipeline-parallel-size 1 --data-parallel-size 2 --max-model-len 16384 --max-num-seqs 16 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8_e4m3 --block-size 256 --attention-config '{"use_fp4_indexer_cache":false,"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' --moe-backend auto --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' --kernel-config '{"enable_flashinfer_autotune":false}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[3,4,6,8,12,16,24,32,36,48,64],"max_cudagraph_capture_size":64}' --seed 0 --distributed-executor-backend mp --data-parallel-size-local 2 --speculative-config '{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":false}'
```

本轮四次实际bench调用如下，由保存的campaign和run目录身份导出；这些结果目录已经存在，重跑须换新`--run-root`。

```bash
./bench run experiments/dspark-k-sweep-node48/configs/campaigns/tp2-dp2-epon-off.yaml --run-root experiments/dspark-k-sweep-node48/results/dpfix-tp2-dp2-epon-off-01
./bench run experiments/dspark-k-sweep-node48/configs/campaigns/tp2-dp2-epon-k3.yaml --run-root experiments/dspark-k-sweep-node48/results/dpfix-tp2-dp2-epon-k3-01
./bench run experiments/dspark-k-sweep-node48/configs/campaigns/tp2-dp2-epoff-off.yaml --run-root experiments/dspark-k-sweep-node48/results/dpfix-tp2-dp2-epoff-off-01
./bench run experiments/dspark-k-sweep-node48/configs/campaigns/tp2-dp2-epoff-k3.yaml --run-root experiments/dspark-k-sweep-node48/results/dpfix-tp2-dp2-epoff-k3-01
```

## 归档与本机证据

原始产物**仅本机可用**：`experiments/dspark-k-sweep-node48/results/dpfix-*/`保存解析配置、命令、完整服务日志、镜像/模型身份、绑定、探测、逐轮raw/metrics/dataset manifest及protocol；同级`*-launch/`保存每秒原始metrics、CPU/GPU/cgroup/进程资源序列、缓存起点、Git及源码哈希。`evidence/dpfix/`保存准备/preflight、缓存来源、补丁校验、页分布及最终资源/冻结核对；`reports/dpfix-analysis.json`保存从所有轮次重算的统计和各rank计数器增量，`tools/*resume.py`保存本地观察/归档代码。

收尾确认本次容器已全部清理、GPU空闲；模型、镜像、历史结果和缓存保留。仅提交本节点新报告、44列CSV及README中本轮48的进度行，不修改其他节点/总表或提交工作区配置。归档前从CSV重算汇总、核对原始指标/接纳/计数器/单位、命令和相对链接，并运行`git diff --check`；没有重跑额外性能实验或修改公共runner/gate。提交不push。
