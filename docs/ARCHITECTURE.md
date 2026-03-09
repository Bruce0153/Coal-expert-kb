# Modern RAG Architecture (Planner/Executor/ContextBuilder)

## Data Flow
1. `QueryPlanner.build_plan(question, config)` 产出可序列化 `QueryPlan`
2. `ExpertRetriever.execute(plan)` 只执行检索（parent->child）
3. `ContextBuilder.build(plan, docs)` 进行证据分组、去重、预算裁剪、引用编号
4. `Answerer.answer(plan, context_package)` 按引用生成答案/拒答

## Module Boundary
- `src/coal_kb/query/*`: 负责“理解问题 + 决策计划”
- `src/coal_kb/retrieval/retriever.py`: 负责“执行计划”
- `src/coal_kb/context/*`: 负责“证据编排”
- `src/coal_kb/generation/*`: 负责“回答生成 + 不确定性处理”

## Replay / Debug
- `scripts/ask.py --show-plan` 查看当次计划
- `scripts/ask.py --save-trace` 在 registry 的 query log 保存 plan + 检索统计 + citations
- 使用 `trace_id` 定位 badcase 做回放分析
