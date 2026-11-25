# 一键启动（标准命令路径，Docker 之外的选择）：.\scripts\run.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$py = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "py" }

if (-not (Test-Path ".venv")) {
    Write-Host "[1/3] 创建虚拟环境 ..."
    & $py -m venv .venv
}

Write-Host "[2/3] 安装依赖 ..."
$venvPy = Join-Path (Get-Location) ".venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip --quiet
# setuptools<81：langchain 依赖 pkg_resources，新版 setuptools 已移除
& $venvPy -m pip install "setuptools<81" --quiet
& $venvPy -m pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "[提示] 已生成 .env，请填入 DASHSCOPE_API_KEY 后重跑" -ForegroundColor Yellow
    exit 1
}

Write-Host "[3/3] 启动服务： http://127.0.0.1:7860"
& $venvPy main.py
