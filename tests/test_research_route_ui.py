"""验证网页加载研究路线控件并向请求注入路线。"""

from pathlib import Path


STATIC = Path("src/coal_kb/interfaces/web/static")


def test_research_route_ui_is_loaded() -> None:
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "research_routes.js").read_text(encoding="utf-8")
    assert '/static/research_routes.js' in index
    for route in ("standard", "graph", "multimodal", "agent"):
        assert f'["{route}"' in script
    assert "payload.research_route = activeRoute" in script
    assert "coal-kb-research-route" in script
