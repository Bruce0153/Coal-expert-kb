#!/usr/bin/env bash
# 功能：执行不访问外部模型、API 或 Elasticsearch 的完整仓库验收。
# 运行目录：可在仓库任意目录调用；脚本会自动定位仓库根目录。
# 外部工具：安装 requirements/ci.txt 后即可运行。
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

export PYTHONPATH=${PYTHONPATH:-src}
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cleanup() {
  find src scripts tests -type d -name __pycache__ -prune -exec rm -rf {} +
  find src scripts tests -type f -name "*.pyc" -delete
  rm -rf .pytest_tmp
}
trap cleanup EXIT

python -m compileall -q src/coal_kb scripts tests
python scripts/quality/check_internal_imports.py
python scripts/quality/check_dependencies.py
python -m pip check
ruff check src/coal_kb scripts tests
MYPYPATH="$REPO_ROOT/src" mypy src/coal_kb
mkdir -p .pytest_tmp
pytest tests --basetemp=.pytest_tmp

# 运行命令：bash scripts/quality/check_repository.sh
