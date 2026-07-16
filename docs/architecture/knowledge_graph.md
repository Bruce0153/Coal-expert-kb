# Graph Schema、抽取与 Graph route

Graph route 使用版本化 `coal-knowledge-graph.v1` 协议。图谱只由标准路线已经召回的 `Document` 构建，不扩大检索范围，也不依赖图数据库。

## 节点

- `evidence`：标准证据块，保存 `chunk_id`、来源文件、章节、页码、父块和文本预览；
- `entity`：由显式 `metadata.entities` 或确定性文本规则抽取并规范化的实体；
- `claim`：包含因果、促进、抑制、增减或相关性标记的可审计句子。

## 关系

- 证据结构：`same_parent`、`same_source`、`same_section`、`shared_terms`；
- 语义关系：`mentions`、`co_occurs`、`supports`、`about`。

每条关系包含权重、置信度和 provenance。关系端点在写入时验证，图谱序列化时再次检查 Schema 版本和悬空关系。

## 抽取

`KnowledgeGraphExtractor` 不会隐式调用外部模型。它优先读取上游提供的实体元数据，并使用可复现规则补充英文实体、中文术语和 Claim。抽取结果可通过以下命令独立生成：

```bash
PYTHONPATH=src python scripts/extract_knowledge_graph.py \
  --input documents.jsonl \
  --output data/artifacts/knowledge_graph.json
```

输入 JSONL 每行必须包含 `page_content`、`content` 或 `text`，并可提供 `metadata`。

## Graph route

Graph route 先保留原始排名，再执行两类受控增益：

1. 从头部种子沿证据结构关系传播；
2. 对与种子共享实体的证据提供实体连接增益。

Claim 数只提供较小的局部证据增益。Trace 同时保留旧版兼容字段和完整 typed graph，包括 Schema 版本、节点、关系、统计、配置、得分和最终顺序。
