"""修正迁移生成的配置测试字符串转义。"""

from pathlib import Path

path = Path("tests/test_config_consistency.py")
text = path.read_text(encoding="utf-8")
text = text.replace('assert "\nembedding:\n" not in text', 'assert "\\nembedding:\\n" not in text')
text = text.replace('text.count("\nembeddings:\n")', 'text.count("\\nembeddings:\\n")')
path.write_text(text, encoding="utf-8")
