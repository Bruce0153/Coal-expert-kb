from coal_kb.utils.text_clean import find_common_header_footer_lines, remove_common_header_footer


def test_header_footer_removal():
    pages = [
        "Journal of X\nPaper Title\n1200 K results...\nPage 1",
        "Journal of X\nPaper Title\nMore results...\nPage 2",
        "Journal of X\nPaper Title\nDiscussion...\nPage 3",
    ]
    common = find_common_header_footer_lines(pages, head_n=2, tail_n=1, min_ratio=0.9)
    assert "Journal of X" in common
    assert "Paper Title" in common

    cleaned = [remove_common_header_footer(p, common, head_n=2, tail_n=1) for p in pages]
    assert all("Journal of X" not in c for c in cleaned)
    assert all("Paper Title" not in c for c in cleaned)
