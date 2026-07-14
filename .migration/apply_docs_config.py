"""统一文档、架构说明和 embedding 配置入口。"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path.cwd()


def _write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _replace(path: str, pairs: list[tuple[str, str]]) -> None:
    target = ROOT / path
    if not target.exists():
        return
    text = target.read_text(encoding="utf-8")
    for old, new in pairs:
        text = text.replace(old, new)
    target.write_text(text, encoding="utf-8")


def process() -> None:
    _write(
        "README.md",
        """# Coal Expert KB

面向煤炭热解与气化文献的可追溯 RAG 系统。系统强调确定性摄取、可解释检索、证据编排、引用约束回答和离线可复现评估。

## 核心能力

- PDF、Markdown、Word、表格等多格式摄取；
- 分层、章节感知和语义切块；
- Chroma 与 Elasticsearch 检索后端；
- Dense、Sparse、Hybrid 和父子结构检索；
- 查询规划、约束过滤、放宽策略和可选重排序；
- 有界上下文、稳定证据编号、Claim 与引用输出；
- CLI、FastAPI、静态 Web UI 和会话历史；
- 无模型 Token、无外部服务的 GitHub Actions 质量门禁。

## 运行链路

```text
原始文档
  → ingestion：加载、解析、清洗、元数据、切块
  → indexing：索引构建、Manifest、验证
  → retrieval/query：问题理解与 QueryPlan
  → recall：Dense / Sparse / Fusion / Parent-Child
  → retrieval/service：约束、放宽、多样性、邻接和 Trace
  → reranking：候选重排序
  → context：去重、预算、证据编号和来源卡片
  → answering：Claim、引用、置信度和拒答
  → application：Ask / Chat / Admin 用例
  → interfaces：CLI / FastAPI / Web
```

## 仓库结构

```text
configs/                     应用配置、领域 Schema 和 Prompt
scripts/                     摄取、索引、问答、评估和质量入口
src/coal_kb/
  core/                      核心协议与领域模型
  infra/                     配置、Provider、持久化、日志和安全
  ingestion/                 文档摄取流水线
  tokenization/              Token 统计
  indexing/                  索引生命周期
  retrieval/                 查询理解、约束和检索编排
  recall/                    Dense、Sparse、融合和结构召回
  reranking/                 重排序服务
  context/                   证据编排
  answering/                 回答、Claim、引用和置信度
  records/                   结构化实验记录
  conversation/              会话模型、历史和持久化
  evaluation/                数据集、指标、Runner 和报告
  application/               Ask、Chat、Admin 用例
  interfaces/                CLI、API 和 Web
  operations/                健康检查
  utils/                     公共无状态工具
tests/                       单元、架构和离线验收测试
```

## 配置

正式配置文件为 `configs/app.yaml`。Embedding 只有一个入口：

```yaml
embeddings:
  provider: dashscope
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  api_key_env: DASHSCOPE_API_KEY
  model: text-embedding-v4
  dimensions: 1024
```

环境变量覆盖：

```text
COAL_KB_EMBEDDINGS_MODEL
COAL_KB_LLM_MODEL
COAL_KB_CONFIG
```

查询阶段使用的 embedding 必须与索引构建阶段保持相同模型、维度和版本。

## 常用命令

```bash
python scripts/ingest.py
python scripts/index.py build
python scripts/validate_index.py --index coal_kb_chunks_current
python scripts/ask.py --show-plan --save-trace
python scripts/serve.py
bash scripts/quality/check_repository.sh
```

## Evaluation

评估数据、检索指标、回答指标、失败归因和多基线对比统一放在 `coal_kb.evaluation`。需要模型 Token 或真实 Elasticsearch 的端到端实验不进入默认 PR 门禁。

## 工程约束

- 每项职责只有一个正式实现路径；
- 不创建重复模块、迁移壳层或状态后缀文件；
- Python、Shell、配置、测试和文档重命名必须同步更新调用方；
- 所有 Claim 必须映射到 ContextPackage 中存在的证据编号；
- 合并前执行 `bash scripts/quality/check_repository.sh`。

详细文档：

- `docs/architecture/system_overview.md`
- `docs/architecture/application_interfaces.md`
- `docs/architecture/retrieval_answering_layers.md`
- `docs/architecture/evaluation_operations.md`
- `docs/engineering/coding_standards.md`
- `docs/engineering/acceptance_criteria.md`
""",
    )
    _write(
        "docs/architecture/system_overview.md",
        """# 系统架构总览

Coal Expert KB 使用单向分层架构。接口层只调用应用层；应用层编排领域服务；领域服务通过 `infra` 中的协议实现访问模型与存储。

## 主数据流

```text
Raw documents
  → ingestion
  → indexing
  → retrieval.query
  → recall
  → retrieval.service
  → reranking
  → context
  → answering
  → application
  → interfaces
```

## 模块边界

- `core/`：Embedding、LLM、Reranker 协议和 QueryPlan 等核心模型。
- `infra/`：配置加载、Provider、Chroma、Elasticsearch、SQLite、日志和安全工具。
- `ingestion/`：加载、解析、清洗、元数据、切块和摄取流水线。
- `indexing/`：索引构建、Manifest、验证、切换和回滚。
- `retrieval/query/`：问题标准化、约束解析和 QueryPlan 构建。
- `retrieval/constraints/`：约束执行与放宽规则。
- `recall/`：Dense、Sparse、融合和父子结构候选生成。
- `retrieval/service.py`：执行 QueryPlan，完成过滤、放宽、多样性和 Trace。
- `reranking/`：候选重排序，不负责召回和回答。
- `context/`：证据去重、预算裁剪、稳定编号和来源卡片。
- `answering/`：Claim、引用、置信度、拒答和回答生成。
- `evaluation/`：评估数据、指标、Runner、报告和失败归因。
- `application/`：Ask、Chat、Admin 用例编排。
- `interfaces/`：CLI、FastAPI 和 Web 传输适配。

## 依赖方向

```text
interfaces → application → domain services → core contracts ← infra implementations
```

禁止：

- `application` 依赖 FastAPI 或 Web 资源；
- `retrieval` 依赖 `answering`；
- `evaluation` 依赖接口层；
- 为同一职责重新建立平行模块。

## Replay 与调试

- `python scripts/ask.py --show-plan`：查看 QueryPlan；
- `python scripts/ask.py --save-trace`：保存检索 Trace、引用和运行指标；
- `trace_id`：关联查询计划、候选、上下文和回答；
- `bash scripts/quality/check_repository.sh`：执行离线工程验收。
""",
    )

    app_path = ROOT / "configs/app.yaml"
    app_text = app_path.read_text(encoding="utf-8")
    app_text = re.sub(r"\nembedding:\n(?:  .*\n)+?(?=\nmodel_versions:)", "\n", app_text)
    app_text = app_text.replace("  embedding_backend: dashscope\n", "  embedding_backend: configured\n")
    app_path.write_text(app_text, encoding="utf-8")

    models_path = ROOT / "src/coal_kb/infra/config/models.py"
    models_text = models_path.read_text(encoding="utf-8")
    models_text = re.sub(
        r"\nclass LocalEmbeddingConfig\(BaseModel\):\n(?:    .*\n)+?(?=\nclass ChunkingProfile)",
        "\n",
        models_text,
    )
    models_text = models_text.replace("    embedding: LocalEmbeddingConfig = Field(default_factory=LocalEmbeddingConfig)\n", "")
    models_path.write_text(models_text, encoding="utf-8")

    env_path = ROOT / "src/coal_kb/infra/config/env.py"
    env_text = env_path.read_text(encoding="utf-8")
    env_text = env_text.replace("    embed_model: Optional[str] = None\n", "")
    env_text = env_text.replace("    emb_model: Optional[str] = None\n", "    embeddings_model: Optional[str] = None\n")
    env_path.write_text(env_text, encoding="utf-8")

    loader_path = ROOT / "src/coal_kb/infra/config/loader.py"
    loader_text = loader_path.read_text(encoding="utf-8")
    loader_text = loader_text.replace("    if env.embed_model:\n        cfg.embedding.model_name = env.embed_model\n", "")
    loader_text = loader_text.replace("    if env.emb_model:\n        cfg.embeddings.model = env.emb_model\n", "    if env.embeddings_model:\n        cfg.embeddings.model = env.embeddings_model\n")
    loader_path.write_text(loader_text, encoding="utf-8")

    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")) + list((ROOT / "tests").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        text = text.replace("cfg.embedding.model_name", "cfg.embeddings.model")
        path.write_text(text, encoding="utf-8")

    _write(
        "tests/test_config_consistency.py",
        """\"\"\"验证 embedding 配置只有一个正式入口。\"\"\"

from pathlib import Path

from coal_kb.infra.config import AppConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_embedding_config_has_single_public_entry() -> None:
    cfg = AppConfig()
    assert hasattr(cfg, "embeddings")
    assert not hasattr(cfg, "embedding")
    assert cfg.embeddings.model


def test_yaml_does_not_define_duplicate_embedding_section() -> None:
    text = (ROOT / "configs/app.yaml").read_text(encoding="utf-8")
    assert "\nembedding:\n" not in text
    assert text.count("\nembeddings:\n") == 1


def test_environment_override_uses_embeddings_model(monkeypatch) -> None:
    monkeypatch.setenv("COAL_KB_EMBEDDINGS_MODEL", "test-embedding-model")
    load_config.cache_clear()
    cfg = load_config()
    assert cfg.embeddings.model == "test-embedding-model"
    load_config.cache_clear()
""",
    )

    quality = ROOT / "scripts/quality/config.sh"
    quality_text = quality.read_text(encoding="utf-8")
    quality_text = quality_text.replace(
        '  "$REPO_ROOT/tests/test_registry.py"\n',
        '  "$REPO_ROOT/tests/test_registry.py"\n  "$REPO_ROOT/tests/test_config_consistency.py"\n',
    )
    quality.write_text(quality_text, encoding="utf-8")


if __name__ == "__main__":
    process()
