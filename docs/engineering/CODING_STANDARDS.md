# Coal Expert KB 代码规范

## Python 脚本

- 可执行脚本首行使用单行中文模块 docstring，文件末尾写 `# 运行命令：...`。
- 有状态或需要多态的处理步骤使用与脚本同名的 CamelCase 类，统一入口为 `process()`；保留 `main()` 作为 CLI 兼容入口。
- 无实例状态的逻辑使用模块级函数，不为纯函数组合创建空壳类。
- 文件或记录批处理先完整扫描得到 `total`，再使用 `tqdm(..., total=total, desc=self.__class__.__name__)`。
- 可复用公共函数放在按职责拆分的 `utils/` 文件中；仅当前步骤使用的函数放入类内，并使用 `_` 前缀的静态方法、类方法或实例方法。
- 每个 Python 功能模块维护 `config.py`，只保存该模块实际引用的变量。代码使用 `from xxx import config`，通过 `config.XXX` 访问。
- 涉及随机采样时，固定在模块 `config.py` 中定义 `SAMPLE_SEED`，并使用 `random.Random(config.SAMPLE_SEED)` 创建独立随机实例，禁止修改全局随机状态。
- 新增模块必须包含中文 docstring 或中文注释，说明职责或关键约束。

## Shell 脚本

- Shell 只负责调用 Python、Docker、Elasticsearch 等外部工具；数据读取、转换和写入逻辑放入 Python。
- 头部注释必须说明功能、运行目录和外部工具安装或编译方式。
- 使用 `set -e`，并按以下形式加载模块配置：

```bash
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/config.sh"
```

- 每个存在 Shell 入口的模块维护 `config.sh`，只保存对应 Shell 脚本实际引用的路径和常量。
- 路径和常量不得在运行脚本中硬编码；文件末尾写 `# 运行命令：...`。

## 类型和定义顺序

- 被调用的模块级函数必须写在调用方之前，避免 Pylance 前向解析错误。
- 类型注解必须与真实数据一致。例如 `df.to_dict("records")` 的元素类型是 `dict[str, Any]`，不能标注为 `pd.Series`。
- 重写父类方法时保留父类参数名；参数未使用时添加 `# type: ignore[override]`，不通过改名规避检查。
- `T | None` 经外部 guard 函数检查后，使用前仍应写 `assert value is not None` 显式收窄类型。

## 修改和回归

- 一个改动影响多个文件时，同步检查调用方、兼容导入、测试、脚本和配置。
- 结构重构保留旧入口作为兼容 facade，不修改检索、切分、Prompt、数据库 schema 和 API contract。
- 每个阶段必须通过编译、完整 pytest、脚本规范测试和兼容性测试后再提交。
