# Evaluation、Observability、Security 与 Operations 分层

本阶段只调整工程职责，不改变评估公式、上传行为和健康检查协议。

```text
evaluation
  datasets / retrieval / faithfulness

infra/observability
  logging / timing / trace summary

infra/security
  upload filename / target path

operations
  health status
```

兼容路径继续保留：

- `coal_kb.eval.datasets`
- `coal_kb.eval.eval_retrieval`
- `coal_kb.eval.eval_faithfulness`

行为约束：

- 轻量 faithfulness 仍按数字引用数量除以 6 计算。
- RetrievalEvaluator 仍按来源子串和可选页码判定命中。
- 上传仍使用 `Path(filename).name`，重名时追加六位 UUID。
- `/health` 仍返回 `{"status": "ok"}`。
