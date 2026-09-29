# 历史 JIT 排障经验（vLLM v0.29）

**范围：这是 v0.29 的历史排障方法，不是当前 v0.30 PD 的安装步骤。** 当时镜像自带的 mHC 启动预热按 `hc_pre/hc_post` 属性寻找层，而所测 NVIDIA DSV4 实现没有这两个属性；模型实际 forward 使用 broadcast 和 fused post/pre TileLang 路径，受测配置因此出现重复 JIT。定向 worker 的验证只适用于当时的镜像、形状和拓扑。

对实际触发的 mHC kernel 与分派形状进行定向启动预热后，限定的普通双 TP4 和 1P1D 经完整 HTTP 预热取得连续三轮 0 已知正式编译事件。随后 top-k、DSpark 输入准备等 kernel 仍各自可能触发 JIT；后续针对真实 builder/speculator 几何的定向预热也不能穷尽采样 kernel。不能把所有 JIT 都归因于 mHC，也不能省略真实请求预热。

0 条**已知日志匹配**并不证明 CUDA Graph 覆盖正确或吞吐稳定：缺少 K5 大 batch 捕获形状时，即使没有新 JIT 也可能落在非 Graph 路径；低 CV、无已知 JIT 和资源无干扰亦是不同验收条件。不要为 PASS 放宽日志 gate，保留其他 warning。

旧实现与完整适用边界仍见 [`../patches/mhc-startup-warmup/`](../patches/mhc-startup-warmup/README.md)、[`../patches/mhc-startup-warmup-extended/`](../patches/mhc-startup-warmup-extended/README.md) 和 [`../patches/input-kernel-warmup/`](../patches/input-kernel-warmup/README.md)。[当前 v0.30 启动](pd-startup.md)依赖的是**不同的** [DSpark MXFP4 草稿加载修复](../patches/v030-dspark-mxfp4/README.md)，不加载这些 v0.29 worker。
