"""兼容旧回答可审计性评估导入路径。"""

from coal_kb.evaluation.faithfulness import FaithfulnessEvaluator, simple_faithfulness_check

__all__ = ["FaithfulnessEvaluator", "simple_faithfulness_check"]
