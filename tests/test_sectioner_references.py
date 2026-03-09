from __future__ import annotations

from coal_kb.chunking.sectioner import infer_section, is_reference_like


def test_references_detection():
    text = """
    References
    1. Smith J. Fuel 2019, 123, 45-56. doi:10.1016/j.fuel.2019.01.001
    2. Zhang L. Energy & Fuels 2020, 34, 100-110. doi:10.1021/ef2c00001
    3. Wang Q. J. Anal. Appl. Pyrolysis 2018, 130, 200-210.
    4. Lee K. Fuel 2017, 205, 77-88.
    """
    assert is_reference_like(text) is True
    assert infer_section(text) == "references"
