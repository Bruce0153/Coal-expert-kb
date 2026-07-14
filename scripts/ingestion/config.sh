#!/usr/bin/env bash
# 功能：定义文档摄入 Shell 入口实际使用的路径与解释器。
# 运行目录：任意目录；路径由脚本位置自动解析。
# 外部工具编译方式：无；Python 依赖使用 pip install -e . 安装。
set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
INGEST_SCRIPT="$PROJECT_ROOT/scripts/ingest.py"
# 使用方式：source scripts/ingestion/config.sh
