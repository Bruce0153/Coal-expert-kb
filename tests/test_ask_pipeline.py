from coal_kb.qa.ask_pipeline import normalize_query, parse_command


def test_normalize_query_collapses_whitespace():
    assert normalize_query("  steam   gasification   NH3  ") == "steam gasification NH3"


def test_parse_command_supports_debug_and_exit():
    assert parse_command("debug") == "debug"
    assert parse_command("quit") == "exit"
    assert parse_command("steam gasification") is None
