#!/usr/bin/env bash
# 功能：执行不访问外部模型、API 或 Elasticsearch 的离线仓库验收。
# 运行目录：可在仓库任意目录调用；脚本会自动定位仓库根目录。
# 外部工具：安装 requirements/ci.txt 中的 Python、ruff、mypy 与 pytest。
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/config.sh"
cd "$REPO_ROOT"

"$PYTHON_BIN" -m compileall -q "${SOURCE_PATHS[@]}"
find "$REPO_ROOT/src" "$REPO_ROOT/scripts" "$REPO_ROOT/tests" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$REPO_ROOT/src" "$REPO_ROOT/scripts" "$REPO_ROOT/tests" -type f -name "*.pyc" -delete
"$RUFF_BIN" check "${SOURCE_PATHS[@]}" --select E,F,I,B --ignore E501,B905

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MYPYPATH="$REPO_ROOT/src" "$MYPY_BIN" --follow-imports=skip --ignore-missing-imports "${MYPY_TARGETS[@]}"
rm -rf "$REPO_ROOT/.pytest_tmp"
mkdir -p "$REPO_ROOT/.pytest_tmp"
timeout "$TEST_TIMEOUT_SECONDS" "$PYTEST_BIN" -q --basetemp="$REPO_ROOT/.pytest_tmp/architecture" -o cache_dir="$REPO_ROOT/.pytest_cache/architecture" "${ARCHITECTURE_TESTS[@]}"
timeout "$TEST_TIMEOUT_SECONDS" "$PYTEST_BIN" -q --basetemp="$REPO_ROOT/.pytest_tmp/foundation" -o cache_dir="$REPO_ROOT/.pytest_cache/foundation" "${FOUNDATION_TESTS[@]}"
timeout "$TEST_TIMEOUT_SECONDS" "$PYTEST_BIN" -q --basetemp="$REPO_ROOT/.pytest_tmp/context" -o cache_dir="$REPO_ROOT/.pytest_cache/context" "${RAG_TESTS[@]}"
rm -rf "$REPO_ROOT/.pytest_tmp"

# 运行命令：bash scripts/quality/check_repository.sh
