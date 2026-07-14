# 系统架构总览

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
