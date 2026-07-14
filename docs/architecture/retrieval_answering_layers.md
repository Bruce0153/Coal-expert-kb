# Retrieval 与回答链分层

召回、检索编排、上下文和回答已经统一到单一实现路径。

```text
retrieval/query + retrieval/constraints
                    ↓
recall (dense / sparse / fusion / parent-child)
                    ↓
retrieval/service (soft rank / diversity / trace)
                    ↓
reranking/service
                    ↓
context (dedup / budget / citations / source cards)
                    ↓
answering (confidence / prompt / claims / citations)
```

canonical 入口：

- 查询理解：`coal_kb.retrieval.query`
- 约束策略：`coal_kb.retrieval.constraints`
- 召回：`coal_kb.recall`
- 检索编排：`coal_kb.retrieval.service`
- 重排序：`coal_kb.reranking`
- 上下文：`coal_kb.context`
- 回答：`coal_kb.answering`

任何新增代码不得重新创建同职责的平行模块。
