# Retrieval 与回答链分层

本阶段只调整模块归属和依赖方向，不改变召回、约束放宽、软排序、重排、上下文预算、引用编号、Prompt 和置信度规则。

```text
retrieval query / constraints
            ↓
recall (dense / sparse / fusion / parent-child)
            ↓
retrieval service (soft rank / diversity / trace)
            ↓
reranking service
            ↓
context (dedup / budget / citations / source cards)
            ↓
answering (confidence / prompt / claims / rendered citations)
```

旧入口继续保留为兼容 facade：

- `coal_kb.retrieval.retriever`
- `coal_kb.retrieval.bm25`
- `coal_kb.retrieval.constraint_policy`
- `coal_kb.retrieval.filter_parser`
- `coal_kb.context.builder`
- `coal_kb.context.types`
- `coal_kb.generation.answerer`

新的业务代码应使用 canonical 路径，不再依赖上述旧入口。
