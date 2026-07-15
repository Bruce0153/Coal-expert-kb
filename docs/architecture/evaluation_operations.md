# Evaluation、Observability、Security 与 Operations 分层

## Evaluation Pipeline

```text
evaluation dataset JSONL
  → EvaluationCase validation
  → retrieval callback
  → optional answer callback
  → retrieval / citation / claim / abstention metrics
  → failure attribution
  → metrics.json / case_results.jsonl / failures.jsonl / summary.md / manifest.json
```

`coal_kb.evaluation` 是唯一评估实现路径：

- `models.py`：案例、证据、检索结果、Claim、回答和案例结果；
- `datasets.py`：JSONL 加载、验证和稳定写出；
- `metrics.py`：Recall、Precision、MRR、nDCG、来源/页召回、引用、Claim 和拒答指标；
- `attribution.py`：RECALL、EVIDENCE_COVERAGE、QUERY_PLAN、CITATION、GENERATION 和 ABSTENTION 归因；
- `pipeline.py`：统一执行入口；
- `reporting.py`：版本化产物输出。

旧的引用数量启发式和独立检索评估器已删除。评估层不得依赖 FastAPI、Web 或 CLI；`scripts/evaluate.py` 只负责运行时组装。

## Observability

`infra/observability` 保存日志、耗时与 Trace 摘要。评估 Manifest 必须记录后端、Embedding 模型、Embedding 版本、K 值和是否启用回答。

## Security 与 Operations

- `infra/security` 负责上传文件名和目标路径安全；
- `operations` 负责健康状态；
- `/health` 返回 `{"status": "ok"}`；
- 默认 CI 评估不得访问外部 LLM、Embedding API 或真实 Elasticsearch。
