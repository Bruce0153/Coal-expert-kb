"""导出 Milestone C 复杂科学问答的正式接口。"""

from coal_kb.complex_qa.planning import build_complex_spec
from coal_kb.complex_qa.router import route_question
from coal_kb.complex_qa.service import ComplexQuestionService

__all__ = ["ComplexQuestionService", "build_complex_spec", "route_question"]
