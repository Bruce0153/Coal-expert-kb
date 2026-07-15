# Milestone C：复杂科学问答

## 统一运行链

```text
用户问题
  → QueryPlanner
  → ComplexQuestionSpec
  → ComplexQuestionService
  → Comparison / Multi-hop / Aggregation / Table / Cross-document
  → Document 证据
  → ContextBuilder
  → Answerer
```

复杂路线不会建立平行的 Context 或 Answering 实现。所有执行器都返回标准 `Document`，因此引用、Token 预算、Reranker 和 UI 继续使用现有正式链路。

## C0 复杂问答评估集

`EvaluationCase` 支持：

- `expected_subqueries`：预期子问题；
- `expected_operation`：预期聚合操作；
- `expected_min_sources`：跨文档最少来源数；
- `expected_table_ids`：预期表格；
- 比较、多跳、聚合、表格、跨文档和不可回答类型。

样例文件为 `data/eval/complex_science_sample.jsonl`。样例仅用于格式和离线测试，正式实验必须替换为人工核验的真实文献标注。

## C1 比较问题

每个比较对象独立生成子查询和检索预算。证据通过 `comparison_entity` 和 `complex_role` 标记，避免只有一侧证据时形成伪比较。

## C2 多跳问题

最多执行配置中的受限步骤。每一跳保存查询、依赖、桥接术语和命中数量；禁止无限检索循环。

## C3 统计聚合

聚合只读取 SQLite 结构化实验记录，由 Python 执行 `count`、`sum`、`average`、`median`、`min`、`max`、`group_by` 和 `top_k`。LLM 只解释程序结果，不负责心算。

## C4 表格问题

标准表格资产使用 JSONL：

```json
{"table_id":"table_001","source_file":"paper.pdf","page":7,"caption":"...","headers":["T_K","H2"],"rows":[{"T_K":1200,"H2":42.1}],"nearby_text":"..."}
```

表格路线返回带 `table_id`、页码和命中行的证据。没有表格资产时才回退到文档检索。

## C5 跨文档综合

分别检索支持、冲突和条件差异证据，并限制每个来源的最大证据数。Trace 记录真实来源数量及是否满足最少来源要求。

## C6 统一路由

路由类型：

```text
fact
comparison
multi_hop
aggregation
table
cross_document
unanswerable
```

第一版使用可解释规则；每个计划保存置信度、路由原因、子问题和结构化操作，便于回放和评估。

## 配置

`configs/app.yaml` 的 `complex_qa` 控制子问题数量、多跳步数、聚合记录数、表格路径、跨文档来源数和上下文预算。
