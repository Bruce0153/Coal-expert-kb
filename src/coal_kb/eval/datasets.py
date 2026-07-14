"""兼容旧评估数据导入路径。"""

from coal_kb.evaluation.datasets import EvalItem, load_eval_set, save_eval_template

__all__ = ["EvalItem", "load_eval_set", "save_eval_template"]
