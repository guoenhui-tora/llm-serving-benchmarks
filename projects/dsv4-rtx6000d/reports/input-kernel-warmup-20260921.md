# top-k / DSpark 输入内核启动覆盖修复

**已补齐本轮定位的两类JIT分派缺口：DSpark 1P1D、DSpark普通双服务及16K普通六TP4，三组共9轮正式测量均为0已知编译事件。** 40个worker全部通过启动覆盖、第二遍缓存回放和临时allocated显存检查。修复只调用原内核处理私有scratch张量，不改变forward、模型权重、精度、采样算法或日志规则；不保证其他配置或所有内核均已穷尽，也不替代性能稳定性与模型质量验证。

## 固定环境与改动

镜像`vllm/vllm-openai:v0.29.0`，image ID `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`；模型`/data/models/DeepSeek-V4-Flash-0731-NVFP4`，FP8 E4M3 KV。45统一控制46/47/48，RTX6000D。源码从固定镜像读取并校验SHA，未套用最新文档；原mHC启动覆盖、DSpark量化与DP profiling补丁继续保留。

| 内核 | 夜间反例 | 新增启动覆盖 |
| --- | --- | --- |
| `_compute_global_topk_indices_and_lens_kernel` | 16K普通六副本预热有88条总事件；正式仍有4/4/0条top-k事件 | 从实际attention builder与block table读取C4/C128宽度、buffer步长、块大小；覆盖有效宽度变化和0..15切片偏移 |
| `_prepare_dflash_inputs_kernel` | DSpark PD正式4/2/0、普通4/0/2，残留BLOCK_SIZE64/128 | 从实际speculator读取参数，调用原wrapper，覆盖`min(256,nextpow2(单请求最大scheduled tokens+query tokens))`的各区间首尾 |

int32请求映射与bool有效位同时切片，产生三个指针对齐组合；C128有效宽度与底层容量步长也可能不同。固定请求长度不能固定这些布局。DSpark跨度按单请求计算，不能用整批token数或仅用Graph捕获尺寸替代；本轮K5跨度为`1,3,4,11,12,27,28,59,60,123,124,8192`。

新`dsv4_input_worker.Worker`继承原mHC worker：先对新内核执行两遍覆盖，按独立整数参考检查输出、第二遍Triton进程内key集合不增加、allocated显存恢复基线，再完整执行原mHC预热、Graph捕获与RNG重置。记录完整specialization keys及SHA。保持实际请求buffer、模型KV及sampling状态；不清理磁盘或显存编译缓存。

## 验证结果

每组一次启动，**一轮完整HTTP预热＋固定三轮quick**。不追加轮次、不选最快结果。吞吐为三轮均值±样本标准差，仅作修复回归记录，不与夜间样本合并或据此重排拓扑。

| case／服务 | GPU总数 | 真实负载／全系统并发／每轮请求 | 预热事件 | 正式事件 | output tok/s | 吞吐CV |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| JSPR2：46 P＋48 D，各TP2×DP2 EP on、K5 | 2服务×4卡＝8 | 近8K／C32／N128 | 0 | 0/0/0 | 1503.25±21.52 | 1.43% |
| JSOR2：46/48普通双服务，各TP2×DP2 EP on、K5 | 2服务×4卡＝8 | 近8K／C32／N128 | 24 | 0/0/0 | 1425.76±28.85 | 2.02% |
| JTO6R2：46/47/48各两个普通TP4，DSpark off | 6服务×4卡＝24 | 精确16K／C64／N256 | 0 | 0/0/0 | 1877.08±0.62 | 0.033% |

DSpark一套四卡服务内部含两个DP引擎与两个API进程，不把它们另算作多个P或D服务。各组均输出1024 tokens；近8K每轮实际输入1,042,149／输出131,072 tokens，16K每轮输入4,194,304／输出262,144 tokens。全部成功、0失败，所有普通服务及DP引擎均实际承接请求。三组均关闭prefix cache及FlashInfer autotune，保留正常Graph；prefill budget均8192，上下文上限分别16384/32768。请求集、tokenizer和采样继承对应夜间C3H/C4/D16O6，完整参数见命令快照。

JSPR2另有8条功能请求，其中12条首次事件来自拒绝采样的`_compute_local_logits_stats_kernel`、`_rejection_kernel`、`_resample_kernel`（各4条）。JSOR2没有额外功能请求，因此同三类内核的24条首次事件落在HTTP预热（各8条）。**这两类待修内核在三组全部HTTP请求中均未再出现事件；其他内核仍说明完整HTTP预热不可省略。** 日志事件不是独立CPU重编译次数，未修改识别器或隐藏warning。

| 实际启动覆盖 | worker数 | 每worker缓存组合 | 新增覆盖耗时（两遍合计） |
| --- | ---: | --- | --- |
| 两组DSpark | 8＋8 | top-k 6，draft inputs 6 | 0.181–0.235秒 |
| 16K普通六TP4 | 24 | top-k 9 | 0.136–0.152秒 |

这些耗时在继承完整缓存的条件下测得，不代表全新冷缓存编译成本；每个worker的第二遍key集合均不增加，allocated显存差为0。8K实际top-k几何是`(width,row_stride,block_stride,block_size)=(128,128,64,2)/(512,512,64,64)`；16K组是`(128,256,128,2)/(256,256,128,2)/(512,512,128,64)`。DSpark内部三个draft KV group的表在本轮具有相同几何，去重后用实际参数覆盖，并非漏掉两组表。

### KV、资源与失败检查

JSPR2每个正式轮次D端远端命中全部1,042,149个输入tokens，prefill新增计算0；256次TP-rank传输、14,724,910,080 bytes。零传输/通知失败、零过期、零抢占；每DP引擎均有请求、传输、草稿提出与接受。NIXL使用后四卡对应mlx5_4–7，端到端计时经过45代理、P及传输；没有用D重算输入冒充PD成功。

后四卡服务绑定CPU32–47/NUMA2，前四卡服务CPU0–15/NUMA0；45代理CPU16–19、客户端CPU20–23，均NUMA1。正式期间GPU平均SM频率约2418–2422 MHz、最高温度61°C；45代理与客户端无cgroup CPU节流，服务/客户端及SMT背景采样保留。完整服务日志无ERROR/Traceback，原有通信能力、实验格式及DP统计顺序warning保留。采样不能排除短时背景干扰；普通DSpark即使0事件仍有2.02%吞吐CV，未称stable通过。

首次JSP在HTTP请求前启动失败：indexer builder也有`compress_ratio=4`，原补丁误选它并访问不存在的`topk_tokens`。修正为精确选择`DeepseekV4SparseMLAMetadataBuilder`，仍要求实际C4/C128均存在；固定镜像CPU回归18.35秒通过，验证indexer被排除、正确几何及缺少attention builder时仍拒绝启动。失败使用0个HTTP请求，旧补丁、日志及缓存保留，未放宽gate。

此前小型GPU测试29.36秒通过：五类top-k几何、K5/K3草稿参数，按独立参考检查slot、SWA空块、拒绝后context、query/sample位置、上下文截断与padding；15个top-k keys、7个draft keys，第二遍及额外内部跨度回放无新增key。K3只做合成内核测试，不代表K3完整服务已验证。

## 预算、复用与证据

工作区从项目configs及原组完整依赖续接；新候选各自完整复制原角色缓存、核对哈希后放入补丁，失败缓存不覆盖。共4次服务部署尝试（1失败＋3完成），2056条外部请求全部在原预算内；小型GPU测试上限600秒，READY900秒，单轮600秒；DSpark协议1800/case3300秒，六TP4协议2400/case4200秒。未扩展性能矩阵。

三组运行期间源码、配置、脚本及补丁冻结检查全部通过，控制器均STOPPED、cleanup errors为空，遥测已结束。未更换镜像或权重，未改宿主功耗/锁频，全部历史缓存保留，仅清理本次owner容器。

- [补丁及接入步骤](../patches/input-kernel-warmup/README.md)：保留对应mHC扩展和DSpark原补丁，增加模块挂载与PYTHONPATH，使用新worker。当前项目精选baseline配置**尚未自动启用**，复用需按步骤接入。
- [逐轮CSV](../data/input-kernel-warmup-20260921.csv)：3轮预热与9轮正式数据分别记录。
- [结构化验证证据](../data/input-kernel-warmup-20260921.json)：每worker完整key、源码/补丁哈希、缓存继承、测试日志、逐轮计数、资源和失败状态。
- [实际服务／代理／客户端命令](input-kernel-warmup-commands-20260921.md)及[完整26组argv快照](../data/input-kernel-warmup-20260921-commands.json)，从实测`command.json`导出。

原始产物仅45本机`experiments/dsv4-jit-coverage-fix/results/{JSP,JSPR2,JSOR2,JTO6R2}-01/`、`reports/`；失败补丁在`reports/failed-JSP-patches-01/`。基于提交`d46e9d2`执行，历史`c651fe4`已在本地历史中。此前夜间报告保留原事件和数字，不追改为无JIT。

适用范围限定固定镜像、NVIDIA DSV4、PP1，以及本轮上述TP4/off与TP2×DP2 EP on K5配置。其他K、并发、预算、上下文、CP或新模型必须重新核对分派与容量；本次没有进行完整输出等价或摘要质量评测。推荐流程仍是“启动覆盖＋完整HTTP预热＋固定三轮quick并检查事件和波动”。
