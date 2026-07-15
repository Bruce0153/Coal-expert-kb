# Coal Expert KB

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
- `docs/architecture/complex_science_qa.md`
- `docs/engineering/coding_standards.md`
- `docs/engineering/acceptance_criteria.md`
