# Coal Expert KB 代码规范

## Python 脚本

- 可执行脚本首行使用单行中文模块 docstring，文件末尾写 `# 运行命令：...`。
- 需要保存状态或支持多态的步骤使用与脚本同名的 CamelCase 类，统一入口为 `process()`。
- 无实例状态的纯函数组合使用模块级函数，不创建空壳类。
- 文件或记录批处理先扫描得到 `total`，再使用 `tqdm(..., total=total, desc=self.__class__.__name__)`。
- 公共函数按职责放入 `utils/`；仅当前步骤使用的辅助函数放在类内并使用 `_` 前缀。
- 每个功能模块维护实际使用的 `config.py`，通过 `from xxx import config` 和 `config.XXX` 访问。
- 随机采样在 `config.py` 定义 `SAMPLE_SEED`，并使用 `random.Random(config.SAMPLE_SEED)` 创建独立实例。
- 新增模块必须包含中文 docstring 或中文注释说明职责和约束。

## Shell 脚本

- Shell 只负责调用 Python 或外部工具，数据读取、转换和写入由 Python 完成。
- 头部说明功能、运行目录和外部工具安装方式。
- 使用 `set -e`、`SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)` 和 `source "$SCRIPT_DIR/config.sh"`。
- 路径和常量来自模块 `config.sh`，文件末尾保留运行命令示例。

## 类型和定义顺序

- 被调用的模块级函数写在调用方之前。
- 类型注解必须与实际数据类型一致。
- 重写父类方法时保留父类参数名；未使用参数使用 `# type: ignore[override]`。
- `T | None` 经外部 guard 后，使用前仍通过 `assert value is not None` 显式收窄。

## 修改和回归

- 跨文件修改同步检查调用方、测试、脚本和配置。
- 不允许恢复已删除的兼容模块或重复实现路径。
## 文件命名

- Python、Shell、配置、数据和文档文件使用小写 `snake_case`。
- GitHub Actions 工作流使用小写 `kebab-case.yml`。
- `README.md`、`LICENSE`、`Dockerfile`、`pyproject.toml` 等行业标准名称可以保留。
- 文件名不得包含迁移阶段号或状态后缀。
- 样例数据使用 `_sample` 后缀。

- 合并前执行 `bash scripts/quality/check_repository.sh`。
