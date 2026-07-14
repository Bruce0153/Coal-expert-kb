# 仓库清理验收标准

GitHub Actions 与本地检查共用同一套离线验收命令，不访问外部 LLM、Embedding API 或真实 Elasticsearch。

## 必须满足

1. **旧结构清零**
   - 已废弃的顶层模块目录、单文件 facade、备份源码、IDE 配置、构建产物和 `egg-info` 不存在。
   - `src/`、`scripts/`、`tests/` 中不存在旧模块 import。
   - `tests/test_repository_conventions.py` 通过。

2. **静态正确性**
   - `python -m compileall` 通过。
   - Ruff 的 `E/F/I/B` 检查通过。
   - canonical 核心包的定向 mypy 通过。

3. **离线行为回归**
   - 运行清单中的单元、架构、API contract、检索、上下文、回答、摄入和持久化测试。
   - 测试不得访问外部 LLM、Embedding API 或真实 Elasticsearch。
   - 需要 Token 或真实索引的端到端评估单独执行。

4. **脚本契约**
   - 可执行 Python 脚本保留中文单行 docstring、`process()` 入口和尾部运行命令。
   - Shell 入口遵循 `set -e`、`SCRIPT_DIR` 和 `config.sh` 约束。

## 执行命令

```bash
bash scripts/quality/check_repository.sh
```

该命令成功退出即表示达到本阶段提交标准；真实模型、真实索引和端到端质量评估留到具备 Token 与服务环境时单独执行。
