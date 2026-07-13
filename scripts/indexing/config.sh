#!/usr/bin/env bash
# 功能：定义索引管理 Shell 入口实际使用的路径与解释器。
# 运行目录：任意目录；路径由脚本位置自动解析。
# 外部工具编译方式：无；Elasticsearch 需由部署环境预先提供。
set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
INDEX_SCRIPT="$PROJECT_ROOT/scripts/index.py"
VALIDATE_INDEX_SCRIPT="$PROJECT_ROOT/scripts/validate_index.py"
# 使用方式：source scripts/indexing/config.sh
