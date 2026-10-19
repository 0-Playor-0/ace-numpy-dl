import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
FORBIDDEN = ("torch", "tensorflow", "jax", "autograd", "scipy", "sklearn", "cv2")
PATTERN = re.compile(r"^\s*(?:import|from)\s+(" + "|".join(FORBIDDEN) + r")\b", re.MULTILINE)


def test_no_forbidden_imports():
    hits = []
    for path in ROOT.rglob("*.py"):
        if ".venv" in path.parts:
            continue
        hits += [f"{path.relative_to(ROOT)}: {m.group(0).strip()}" for m in PATTERN.finditer(path.read_text())]
    assert not hits, hits
