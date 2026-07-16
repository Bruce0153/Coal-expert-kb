# 研究实验、Manifest 与消融运行

研究实验继续复用统一 `EvaluationPipeline`，不维护第二套指标实现。单次运行使用 `ExperimentSpec` 与 `ResearchExperiment`；批量实验使用版本化 `research-suite.v1` 配置和 `AblationSuiteRunner`。

## 配置协议

配置固定数据集、输出目录、Recall 截断值、重复次数、随机种子和实验变体。每个变体选择一条正式研究路线，并可设置白名单消融参数。配置文件路径中的数据集和输出目录均相对配置文件解析。

当前白名单参数包括：

- `graph.seed_count`
- `graph.max_edges`
- `multimodal.requested_boost`
- `multimodal.secondary_boost`
- `agent.max_steps`

未知参数会直接报错，避免拼写错误被静默忽略。

## 运行产物

套件输出：

- `suite_manifest.json`：配置版本、配置摘要、Git SHA、Python 与平台信息以及全部稳定运行 ID；
- 每个运行目录中的 `experiment.json` 和 `run_status.json`；
- `suite_results.json`：逐运行状态和按变体聚合的均值、标准差与原始值；
- `summary.md`：适合人工审阅的汇总。

`resume: true` 时，已有 `experiment.json` 的运行不会重复执行。失败运行被记录后继续下一个变体；`fail_fast: true` 可改为首次失败立即停止。

## 运行命令

```bash
PYTHONPATH=src python scripts/run_ablation_suite.py \
  --config configs/research/milestone_d_ablation.yaml
```
