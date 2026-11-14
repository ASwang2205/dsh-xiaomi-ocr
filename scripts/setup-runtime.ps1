# Build a private Python runtime for dsh-xiaomi-ocr.
#
#   pwsh scripts/setup-runtime.ps1 -Python "C:\Python313\python.exe"
#   pwsh scripts/setup-runtime.ps1 -Python "C:\Python313\python.exe" -TorchIndex "https://download.pytorch.org/whl/cu124"
#
# Creates %USERPROFILE%\.dsh\xiaomi-ocr\venv and installs everything the OCR
# service needs.  It never touches any other Python environment.
param(
    [string]$Python = "",
    [string]$TorchIndex = "https://download.pytorch.org/whl/rocm6.4",
    [switch]$SkipTorch
)

$ErrorActionPreference = "Stop"
$root = Join-Path $env:USERPROFILE ".dsh\xiaomi-ocr"
$venv = Join-Path $root "venv"
$exe = Join-Path $venv "Scripts\python.exe"

if (-not $Python) {
    foreach ($candidate in @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"),
        "C:\Python313\python.exe",
        "C:\Program Files\Python313\python.exe"
    )) {
        if (Test-Path $candidate) { $Python = $candidate; break }
    }
}
if (-not $Python -or -not (Test-Path $Python)) {
    throw "找不到基础 Python。请用 -Python 指定 python.exe（3.11+）。"
}

Write-Host "==> 基础解释器: $Python"
if (-not (Test-Path $exe)) {
    Write-Host "==> 创建 venv: $venv"
    New-Item -ItemType Directory -Force -Path $root | Out-Null
    & $Python -m venv $venv
} else {
    Write-Host "==> 复用 venv: $venv"
}

& $exe -m pip install --upgrade pip wheel | Out-Null

if (-not $SkipTorch) {
    Write-Host "==> 安装 torch / torchvision（索引 $TorchIndex，约 2-3 GB，耐心等）"
    & $exe -m pip install --index-url $TorchIndex torch torchvision
}

Write-Host "==> 安装 transformers / pillow / huggingface_hub / pypdfium2"
& $exe -m pip install "transformers>=5" pillow huggingface_hub pypdfium2

Write-Host "==> 自检"
& $exe -c "import torch, transformers, PIL, pypdfium2; print('python ok'); print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('transformers', transformers.__version__)"

$model = Join-Path $root "models\Xiaomi-OCR-0\model.safetensors"
if (-not (Test-Path $model)) {
    Write-Host ""
    Write-Host "模型未下载。执行："
    Write-Host "  & `"$exe`" -c `"from huggingface_hub import snapshot_download as s; s('xiaomi-research/Xiaomi-OCR-0', local_dir=r'$root\models\Xiaomi-OCR-0', allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','processor_config.json','chat_template.jinja'])`""
} else {
    Write-Host "模型已就绪: $model"
}

Write-Host ""
Write-Host "完成。在 设置 -> 本地 OCR -> 运行时 里把 Python 路径设为："
Write-Host "  $exe"
