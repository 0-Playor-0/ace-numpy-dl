import json
import pathlib
import platform
import time

import numpy as np

from tests.gradcheck import EPS, RESULTS, TOL, format_table

GRADCHECK_JSON = pathlib.Path(__file__).resolve().parent.parent / "runs" / "gradcheck.json"
T0 = time.time()


def pytest_terminal_summary(terminalreporter):
    stats = {k: len(v) for k, v in terminalreporter.stats.items() if k in ("passed", "failed", "error", "skipped")}
    if RESULTS:
        terminalreporter.section("gradcheck: max rel_err per op")
        terminalreporter.write_line(format_table())
    # the README's gradcheck/test tables are built from this file; only a full, green run overwrites it
    if len(RESULTS) >= 20 and not stats.get("failed") and not stats.get("error"):
        GRADCHECK_JSON.write_text(json.dumps({
            "max_rel_err": dict(RESULTS), "tolerance": TOL, "eps": EPS, "dtype": "float64",
            "pytest": stats, "wall_s": round(time.time() - T0, 1),
            "python": platform.python_version(), "numpy": np.__version__}, indent=2))
