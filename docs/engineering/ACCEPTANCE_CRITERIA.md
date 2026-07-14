# 仓库清理验收标准

本阶段不建立 GitHub Actions，也不执行可能调用外部模型或消耗 Token 的全量测试。合并依据是可重复的本地离线验收。

## 必须满足

1. **旧结构清零**
   - 已废弃的顶层模块目录、单文件 facade、备份源码、IDE 配置、构建产物和 `egg-info` 不存在。
   - `src/`、`scripts/`、`tests/` 中不存在旧模块 import。
   - `tests/test_no_legacy_modules.py` 通过。

2. **静态正确性**
   - `python -m compileall` 通过。
   - Ruff 的 `E/F/I/B` 检查通过。
   - canonical 核心包的定向 mypy 通过。

3. **离线行为回归**
   - 运行清单中的单元、架构、API contract、检索、上下文、回答、摄入和持久化测试。
   - 测试不得访问外部 LLM、Embedding API 或真实 Elasticsearch。
   - 不执行全量 `pytest`，不以未运行的测试作为通过依据。

4. **脚本契约**
   - 可执行 Python 脚本保留中文单行 docstring、`process()` 入口和尾部运行命令。
   - Shell 入口遵循 `set -e`、`SCRIPT_DIR` 和 `config.sh` 约束。

## 执行命令

```bash
bash scripts/quality/run_acceptance.sh
```

该命令成功退出即表示达到本阶段提交标准；真实模型、真实索引和端到端质量评估留到具备 Token 与服务环境时单独执行。
