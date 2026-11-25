#!/usr/bin/env bash
# 一键启动（标准命令路径，Docker 之外的选择）
# 用法：./scripts/run.sh          # macOS / Linux
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v python3 >/dev/null 2>&1; then
  echo "[错误] 未找到 python3，请先安装 Python 3.10+" >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "[1/3] 创建虚拟环境 ..."
  python3 -m venv .venv
fi

echo "[2/3] 安装依赖 ..."
# shellcheck disable=SC1091
. .venv/bin/activate
pip install --upgrade pip >/dev/null
# setuptools<81：langchain 依赖 pkg_resources，新版 setuptools 已移除
pip install "setuptools<81" >/dev/null
pip install -r requirements.txt

if [ ! -f .env ]; then
  echo "[提示] 未找到 .env，已从 .env.example 复制，请填入 DASHSCOPE_API_KEY 后重跑" >&2
  cp .env.example .env
  exit 1
fi

echo "[3/3] 启动服务： http://127.0.0.1:7860"
exec python main.py
