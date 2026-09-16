# 基线数据与原始证据

`baseline-samples.csv` / `.json` 包含 2026-09-15 的 18 次有效测量，所有数值沿用已完成实验。仅将 `source_path` 从旧宿主绝对路径转换为仓库根目录相对路径，便于查找原始尝试。

`baseline-statistics.json` 原样保留六组统计。输出吞吐、requests/s 使用每次完整客户端测量窗口；延迟字段单位为 ms。标准差为三次重复的样本标准差，CV 为百分数；P95/P99 的重复均值不是合并请求后的分位数。

正式总量：1728 成功、0 失败，输入 14,155,776 tokens、输出 1,769,472 tokens。预热、客户端初始连通性请求及被 gate 拒绝的尝试不在正式总量中。

## 原始产物仅本机保留

原宿主：gpu-6000d-48，仓库 `/home/enhui/llm-serving-benchmarks`。

- 原始运行：`results/dsv4-aligned-final-20260915/run-01`。
- 额外遥测、冻结快照与最终审计：`results/dsv4-aligned-final-20260915/`。
- 原始完整汇总：`reports/dsv4-aligned-final-20260915/`。
- 本次迁移前归档：`artifacts/repository-layout-20260916/before-layout.tar.gz`，同目录有逐文件 SHA-256 清单。

以上目录不随 Git 分发。新 clone 可以读取本项目的全部精选表格与说明；复核完整日志时需要另行取得原始归档。历史 JSON、源码快照和命令保留原路径及身份，不回写成新目录。

Git tag `baseline-before-project-layout-20260916` 保存整理前的正式配置与必要修复；完整探索文件在本机归档中。正式实验运行时使用旧 commit 加本地差异，不能把这个后来建立的 tag 误称为当时运行的 commit，详见 `provenance.json`。
