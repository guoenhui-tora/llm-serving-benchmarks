# C3H：一次HTTP连接边界修正复核

C3功能8条真实KV/逐DP接受量均通过；完整N128预热126成功/2失败，未进入正式轮。请求92在P返回metadata后约1ms遇到ServerDisconnectedError，D日志无该request id；请求106在P HTTP开始约0.7ms断开，P日志无该id。没有引擎ERROR/Traceback；92未被D读取的P KV随后到期回收，失败和过期均保留。不能把126成功部分的1443.78 tok/s视作有效性能。

当前假设是上游keepalive复用关闭竞争，尚未通过TCP抓包唯一证明。代理改为TCPConnector(force_close=True)：每次P/D或普通上游POST独立新连接，不改变正文/metadata，不重试POST。新增测试验证连续请求不复用transport、主动断开只收到一次POST且下一个请求可成功，全部11项HTTP/配置测试通过。6项runner gate测试通过，新增API route/completion逐边验收。

C3H是预先允许的一次具体集成修正复核；复用C3完全相同的模型配置和保留缓存，记录已在失败尝试学习到的kernel。总八卡、8K前128/C32、功能8、1full预热+3quick、READY900/case3300/round600/protocol1800不变。再次HTTP/计算/KV失败停止DSpark DP主线，不无限修复重试。C4普通参照也使用相同新连接代理；旧C1/C2代理不同，精确off/on效应有此额外边界，不冒充仅DSpark一个变量的比较。

精选证据：[失败请求、原始预热汇总及注册记录](../data/pd-dspark-http-failure-20260921.json)。原始产物仅45本机`experiments/dsv4-pd-overnight/results/C3-01/`。C3H于06:27前已启动；修正是否足够尚待完整协议验收。代理修复commit `0befaf7`，前一多池实现为`8ea0f65`。
