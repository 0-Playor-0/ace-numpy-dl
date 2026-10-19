from tests.gradcheck import RESULTS, format_table


def pytest_terminal_summary(terminalreporter):
    if RESULTS:
        terminalreporter.section("gradcheck: max rel_err per op")
        terminalreporter.write_line(format_table())
