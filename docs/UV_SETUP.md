# 使用 uv 运行项目（Windows）

项目使用 `.python-version` 指定 Python 3.12，虚拟环境位于 `.venv`。
`uv.lock` 固定 Python 依赖版本。Windows 使用 PyTorch 2.9.1 的 CUDA 12.6 构建；
音频解码使用 TorchCodec 和 FFmpeg 的共享库。

## 日常使用

在仓库根目录运行，无需先激活环境：

```powershell
uv run --extra torch-runtime --extra finetune python infer.py
uv run --extra torch-runtime --extra finetune python app.py
```

运行 `infer.py` 前，需要准备模型权重，并设置其中的模型及音频路径。
Gradio 的模型路径通过 `$env:MOSS_AUDIO_MODEL_ID` 设置。

也可以激活后直接使用 Python：

```powershell
.\.venv\Scripts\Activate.ps1
python infer.py
```

如果 PowerShell 执行策略禁止激活脚本，直接使用上面的 `uv run` 即可。
编辑器的 Python 解释器选择 `.venv\Scripts\python.exe`。

请保留两个 `--extra` 参数：裸 `uv sync` 或 `uv run` 可能移除已安装的可选依赖。
已同步环境也可通过 `uv run --no-sync python ...` 使用。
Windows 环境不安装 FlashAttention；训练时使用 `--attn_implementation eager`。

## 重建 Python 环境

```powershell
uv python install 3.12
uv sync --locked --extra torch-runtime --extra finetune
uv pip check
```

## FFmpeg

本机的 FFmpeg 可执行文件和共享 DLL 已安装在 `.venv\Scripts`，没有修改系统 PATH。
通过 `uv run` 或激活环境启动时，该目录会进入进程 PATH。
FFmpeg 是外部二进制依赖，不由 `uv.lock` 管理；删除 `.venv` 后需要重新安装。

下面从 BtbN 的发布资产下载 FFmpeg 8.1 LGPL shared 构建，校验发布方给出的 SHA256，
并安装到已经创建的虚拟环境中。TorchCodec 需要 shared DLL，仅有静态 ffmpeg.exe 不够。

```powershell
$asset = (Invoke-RestMethod 'https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/tags/latest').assets |
    Where-Object { $_.name -eq 'ffmpeg-n8.1-latest-win64-lgpl-shared-8.1.zip' }
if (-not $asset -or -not $asset.digest) { throw 'FFmpeg asset or checksum missing' }
Invoke-WebRequest $asset.browser_download_url -OutFile '.venv/ffmpeg-shared.zip'
$digest = 'sha256:' + (Get-FileHash '.venv/ffmpeg-shared.zip' -Algorithm SHA256).Hash.ToLower()
if ($digest -ne $asset.digest) { throw 'FFmpeg checksum mismatch' }
Expand-Archive '.venv/ffmpeg-shared.zip' -DestinationPath '.venv/ffmpeg' -Force
Get-ChildItem '.venv/ffmpeg' -Directory | ForEach-Object {
    Copy-Item (Join-Path $_.FullName 'bin/*') -Destination '.venv/Scripts' -Force
}
uv run --no-sync ffmpeg -version
```

该地址随发布更新；上面的过程会验证下载的当前资产，而不是固定某个 FFmpeg 补丁版本。
FFmpeg 的许可证与说明保留在 `.venv\ffmpeg` 的解压目录内。

## 本机验证结果（2026-09-06）

- Python 3.12.13；PyTorch / torchaudio 2.9.1+cu126；Transformers 4.57.1；TorchCodec 0.9.1。
- CUDA 可用，RTX 3060 Laptop GPU 上的矩阵乘法及同步执行成功。
- 三段测试 MP3 均通过项目的 `load_audio` 成功读取并重采样到 16kHz。
- 中文样例提取出 `[128, 400]`、bfloat16 的 Mel 特征。
- 核心模型模块、微调模块及 PEFT 导入成功，Gradio 界面构建成功。
- `uv pip check` 通过；没有下载模型权重或验证完整模型推理/训练。
