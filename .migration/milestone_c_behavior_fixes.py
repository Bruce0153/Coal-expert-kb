"""修正复杂问答失败归因、聚合优先级和跨文档测试桩。"""

from pathlib import Path


def process() -> None:
    attribution_path = Path("src/coal_kb/evaluation/attribution.py")
    attribution_text = attribution_path.read_text(encoding="utf-8")
    old_retrieval = '''    if retrieval.get("hit_at_10", retrieval.get("hit_at_5", 0.0)) == 0.0 and case.expected_evidence:
        return "RECALL"
    if retrieval.get("recall_at_10", retrieval.get("recall_at_5", 1.0)) < 1.0:
        return "EVIDENCE_COVERAGE"
'''
    new_retrieval = '''    hit_values = [value for key, value in retrieval.items() if key.startswith("hit_at_")]
    recall_values = [value for key, value in retrieval.items() if key.startswith("recall_at_")]
    if case.expected_evidence and (not hit_values or max(hit_values) == 0.0):
        return "RECALL"
    if case.expected_evidence and recall_values and max(recall_values) < 1.0:
        return "EVIDENCE_COVERAGE"
'''
    if old_retrieval not in attribution_text:
        raise ValueError("未找到检索失败归因代码块")
    attribution_path.write_text(attribution_text.replace(old_retrieval, new_retrieval), encoding="utf-8")

    planning_path = Path("src/coal_kb/complex_qa/planning.py")
    planning_text = planning_path.read_text(encoding="utf-8")
    old_operation = '''    if re.search(r"平均|均值|average|mean", lowered):
        operation = "average"
    elif re.search(r"中位数|median", lowered):
        operation = "median"
    elif re.search(r"最高|最大|max", lowered):
        operation = "max"
    elif re.search(r"最低|最小|min", lowered):
        operation = "min"
    elif re.search(r"前\\s*\\d+|top\\s*\\d+|排名", lowered):
        operation = "top_k"
    elif re.search(r"按.+统计|分组|group", lowered):
        operation = "group_by"
'''
    new_operation = '''    if re.search(r"平均|均值|average|mean", lowered):
        operation = "average"
    elif re.search(r"中位数|median", lowered):
        operation = "median"
    elif re.search(r"前\\s*\\d+|top\\s*\\d+|排名", lowered):
        operation = "top_k"
    elif re.search(r"最高|最大|max", lowered):
        operation = "max"
    elif re.search(r"最低|最小|min", lowered):
        operation = "min"
    elif re.search(r"按.+统计|分组|group", lowered):
        operation = "group_by"
'''
    if old_operation not in planning_text:
        raise ValueError("未找到聚合操作路由代码块")
    planning_path.write_text(planning_text.replace(old_operation, new_operation), encoding="utf-8")

    test_path = Path("tests/test_complex_question_service.py")
    test_text = test_path.read_text(encoding="utf-8")
    old_fake = '''        if "co2" in query or "二氧化碳" in query:
            source = "co2.pdf"
        elif "相反" in query or "冲突" in query:
            source = "conflict.pdf"
        elif "条件差异" in query:
            source = "conditions.pdf"
        elif "蒸汽" in query:
            source = "steam.pdf"
        else:
            source = f"{abs(hash(query)) % 3}.pdf"
'''
    new_fake = '''        if "支持性证据" in query:
            source = "support.pdf"
        elif "相反结果" in query or "冲突证据" in query:
            source = "conflict.pdf"
        elif "实验条件差异" in query:
            source = "conditions.pdf"
        elif "co2" in query or "二氧化碳" in query:
            source = "co2.pdf"
        elif "蒸汽" in query:
            source = "steam.pdf"
        else:
            source = "default.pdf"
'''
    if old_fake not in test_text:
        raise ValueError("未找到跨文档测试 Retriever 代码块")
    test_path.write_text(test_text.replace(old_fake, new_fake), encoding="utf-8")


if __name__ == "__main__":
    process()
