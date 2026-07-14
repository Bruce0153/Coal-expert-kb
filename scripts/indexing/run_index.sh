#!/usr/bin/env bash
# 功能：调用 Python 索引构建、切换或回滚脚本。
# 运行目录：任意目录；建议在仓库根目录运行。
# 外部工具编译方式：无；Elasticsearch 需由部署环境预先提供。
set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/config.sh"
exec "$PYTHON_BIN" "$INDEX_SCRIPT" "$@"
# 运行命令：bash scripts/indexing/run_index.sh build
