#!/usr/bin/env bash
# 功能：调用 Python 脚本生成确定性的 LoRA 训练与验证数据。
# 运行目录：任意目录；建议在仓库根目录运行。
# 外部工具编译方式：无；Python 依赖使用 pip install -e . 安装。
set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/config.sh"
exec "$PYTHON_BIN" "$BUILD_LORA_DATASET_SCRIPT" "$@"
# 运行命令：bash scripts/data_preparation/run_build_lora_dataset.sh
