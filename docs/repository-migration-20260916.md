# 项目目录迁移记录 · 2026-09-16

本次整理将项目配置与精选成果放到 `projects/`，公共执行器和配置格式不变。没有启动 GPU 实验、访问其他节点或推送 Git。原始结果、镜像、模型和 JIT 缓存没有清理或改写。

## 新入口

| 原位置 / 用途 | 新位置 |
| --- | --- |
| 根 configs 中的 DSV4 最终配置 | [DSV4 configs](../projects/dsv4-rtx6000d/configs/) |
| 根 configs 中的初始 GLM 配置 | [GLM 项目](../projects/glm52-rtx6000d/README.md) |
| DSV4 镜像选型文档 | [image-selection.md](../projects/dsv4-rtx6000d/reports/image-selection.md) |
| RTX6000D 互联 HTML | [interconnect.html](../projects/dsv4-rtx6000d/reports/interconnect.html)，附阅读摘要 |
| 用户提供的外部 PDF | DSV4 项目 references，标明证据边界 |
| 通用测试所需历史配置 | tests/fixtures/configs，与项目演进解耦 |
| 创建项目的参考 | [projects/_template](../projects/_template/README.md) |

DSV4 项目只保留三套正式 recipe 和复现依赖，新增 C32-only、短功能验证入口均已离线校验，未作为新的实测结果。初始 GLM smoke 与候选保留，不冒充后来其他机器的研究成果。

## 如何保护旧结果

迁移前已把源码、配置、文档、results 和 reports 共 4,075 个文件归档到原宿主的 `artifacts/repository-layout-20260916/before-layout.tar.gz`，并逐文件核对 SHA-256；同目录保存原 Git 状态、diff 和哈希清单。它是同机独立归档，不是异地备份，也不随 Git 分发。

本地 tag `baseline-before-project-layout-20260916` 对应提交 `31ef3ee`，保存旧目录的正式配置、必要代码修复和精选资料。这个 tag 建立于实验结束后，不替代原始运行的 commit 与本地差异。早期未提交的探索配置保留在完整归档和本机 `artifacts/repository-layout-20260916/legacy-untracked-configs/`，不全部写入 Git 历史。

需要查看旧代码时，在另外一个不存在的目录建立 worktree，不回退当前工作区：

```bash
git worktree add --detach /tmp/llm-serving-benchmarks-before-layout baseline-before-project-layout-20260916
```

旧 worktree 不包含被忽略的原始数据。如需完整历史配置，可在独立目录解开本机归档；不要覆盖现在的配置、结果或缓存。

## 本次验证

- 原有 45 项测试通过；迁移后含日志准备、项目解析和打包范围检查的 49 项测试通过。
- 正式 campaign 的完整解析配置、服务端/客户端命令、缓存目录与迁移前一致。
- runner 源码指纹仍为 `13471c026a1801fa2e3fdab8842035b26579a2765c0438baf8185a8cc9b0badd`；预热、gate 和重试上限未改。
- 18 次有效结果的数值与原统计一致，仅将精选样本的 source_path 改为仓库相对路径。原始 JSON 保留不变。
- 新增日志准备脚本，为新 clone 补齐 SGLang 文件依赖；不启动容器，不复制或删除 JIT 缓存，不覆盖不同的既有日志配置。
- 源码打包改为明确选择框架与项目内容，排除根 reports/artifacts 等本地文件。原脚本遗漏了这些目录的排除。

具体版本和数据来源见 [DSV4 provenance](../projects/dsv4-rtx6000d/data/provenance.json)。目录迁移验证不等于重新完成了模型功能或性能测试。
