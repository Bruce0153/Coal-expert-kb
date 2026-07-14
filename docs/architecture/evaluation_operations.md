# Evaluation、Observability、Security 与 Operations 分层

工程职责统一如下：

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

行为约束：

- 轻量 faithfulness 按数字引用数量除以 6 计算。
- `RetrievalEvaluator` 按来源子串和可选页码判定命中。
- 上传使用安全文件名，重名时追加六位 UUID。
- `/health` 返回 `{"status": "ok"}`。
- 评估层不得依赖 FastAPI 或应用编排层。
