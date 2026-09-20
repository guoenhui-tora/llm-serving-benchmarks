# GovReport原始分片下载：为近16K真实请求做准备

**在控制机45的home目录建立`~/hf_env` Python虚拟环境，通过`https://hf-mirror.com`下载固定版本的一个训练分片到`~/datasets/govreport/`。** 本文件只提供用户手动执行的下载命令，本轮未联网验证镜像可达性、未下载文件，也未生成16K请求集。

该分片与[现有8K子集](govreport-near8k.md)来源完全相同，先下载它即可，不需要整个数据仓库或模型。能否筛出128篇完整近16K报告，须下载后用固定DSV4 tokenizer实际统计；不预先保证数量，不自动追加其他分片。数据公开，命令不要求HF登录、不发送本地HF token。虚拟环境只用于下载，不改仓库的Python环境。

| 项目 | 固定值 |
| --- | --- |
| dataset repo | `ccdv/govreport-summarization` |
| revision | `4e21184e01ae8017e2c036e180fe5e541fef60a0` |
| repo内文件 | `document/train-00000-of-00002.parquet` |
| 本地文件 | `~/datasets/govreport/document/train-00000-of-00002.parquet` |
| SHA256 | `8b02abb5691b3fa1ba95f76331cadc8bb3e80195c626b4656b480d36e0c92242` |

直接在45的bash终端执行以下整段。需要系统已有`python3`及venv支持；安装依赖失败时命令结束，不自动安装系统软件。只在临时子shell内设置HF endpoint和取消常见代理环境变量，不调用`proxy on`、不修改shell启动文件；子shell结束后恢复原终端环境。这里取消的是进程环境代理，不改变系统网络。pip安装Python库使用默认包索引，HF数据请求使用指定镜像。

```bash
python3 -m venv "$HOME/hf_env" && source "$HOME/hf_env/bin/activate" && (
    set -e
    unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
    export HF_ENDPOINT=https://hf-mirror.com
    export HF_HUB_DISABLE_XET=1

    python -m pip install huggingface_hub
    python - <<'PY'
import hashlib
from pathlib import Path
from huggingface_hub import hf_hub_download

path = Path(hf_hub_download(
    repo_id="ccdv/govreport-summarization",
    repo_type="dataset",
    revision="4e21184e01ae8017e2c036e180fe5e541fef60a0",
    filename="document/train-00000-of-00002.parquet",
    local_dir=Path.home() / "datasets" / "govreport",
    token=False,
))
digest = hashlib.sha256()
with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
expected = "8b02abb5691b3fa1ba95f76331cadc8bb3e80195c626b4656b480d36e0c92242"
if digest.hexdigest() != expected:
    raise SystemExit(f"SHA256 mismatch: {path}\nactual={digest.hexdigest()}")
print(f"SHA256 OK\n{path}")
PY
)
```

看到`SHA256 OK`及绝对文件路径即完成。同一命令可重跑，使用HF下载元数据复用已完成文件；不要为了重试删除缓存。若镜像不可达/缺少固定revision，保留报错，不自动切换付费代理，也不要改成浮动main版本后当作相同数据。虚拟环境仍在当前终端激活，需要退出时运行`deactivate`。

后续由实验准备步骤读取本地Parquet，沿用8K的指令、编码与tokenizer身份，从完整报告中筛选最终输入约15K–17K的固定128条；保存来源行号、选择规则、实际token数、JSONL哈希和许可说明。每条输出1024，普通/PD与DSpark off/on复用同一个新请求集。16K prompt加1K输出需验证上下文32768；数据文件下载成功不代表模型长度、PD或DSpark已通过验证。

来源仍为GovReport、CC BY 4.0；原始分片和完整下载缓存留在home目录，不提交Git。精选16K JSONL和生成说明待校验后归档，详见[PD规划](../reports/pd-overnight-plan-20260921.md)。
