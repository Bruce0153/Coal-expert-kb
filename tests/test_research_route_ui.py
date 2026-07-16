"""验证网页在主脚本中加载研究路线并向请求注入路线。"""

from pathlib import Path


STATIC = Path("src/coal_kb/interfaces/web/static")


def test_research_route_ui_is_integrated_without_duplicate_script() -> None:
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert '/static/app.js' in index
    assert '/static/research_routes.js' not in index
    assert not (STATIC / "research_routes.js").exists()
    assert 'id="setting-research-route"' in index
    for route in ("standard", "graph", "multimodal", "agent"):
        assert f'{route}:' in script
    assert "research_route: settings.researchRoute" in script
