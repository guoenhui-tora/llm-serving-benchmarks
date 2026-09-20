# 固定vLLM 0.29.0 NVIDIA DSV4启动预热候选

只扩展启动阶段，不修改forward、kernel或权重。GPU验证状态见[预热计划](../../reports/pd-targeted-warmup-plan.md)及后续报告；提交时已通过固定镜像导入和离线分派覆盖检查。

固定image ID为`sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`。worker启动时对四个实现文件核对SHA256，版本不匹配直接失败。适用NVIDIA DSV4 Flash、hidden4096、hc_mult4、TP4/PP1/DP1、DSpark off、max-batched-tokens≤8192。

复制`dsv4_mhc_worker.py`、`warmup_plan.py`和`pinned-source.json`到该服务缓存挂载下的`dsv4-mhc-warmup/`，配置`PYTHONPATH=/root/.cache/dsv4-mhc-warmup`和`--worker-cls dsv4_mhc_worker.Worker`。新增recipe会改变框架缓存key：必须从对应旧缓存完整复制并核对哈希，不用新空目录替代已有缓存；原缓存保留。测试文件不参与服务导入。

通过原worker的compile_or_warm_up_model启动入口，先在已加载的真实层参数上调用broadcast和fused-post-pre路径，然后执行原有预热、图捕获及随机状态重置。候选覆盖函数按固定版本compute_num_split枚举可达分派组合，取每组首尾token数和图捕获尺寸。每worker两遍、480秒上限，检查输出有限、第二遍mHC缓存key不增加、所需n_splits全部已加载。日志`DSV4_TARGETED_WARMUP`保存PID/rank、参数范围、耗时和覆盖证据。

测试：在固定镜像内将本目录挂载到`/patch:ro`，设置`PYTHONPATH=/patch`，执行`python3 -m unittest discover -s /patch -p 'test_*.py'`；不需要GPU。测试直接读取镜像compute_num_split函数，比对1–8192范围及80/156/188 SM规格，并拒绝超出模型和token范围的输入。这不替代真实GPU调用和正式负载验收。
