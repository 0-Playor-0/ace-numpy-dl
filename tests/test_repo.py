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


LINK = re.compile(r"\]\(([^)#\s]+)")
CODE_PATH = re.compile(r"`((?:runs|docs|experiments|acenet|scripts|tests)/[^`\s]+)`")


def test_readme_and_docs_links_exist():
    """Every markdown link/image target and every backticked repo path in README.md and docs/*.md exists.
    Paths with placeholders ({a,b}, <name>, *) are skipped; the hidden-set outputs do not exist before release."""
    missing = []
    for md in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        text = md.read_text()
        targets = [(md.parent / t, t) for t in LINK.findall(text) if "://" not in t]
        targets += [(ROOT / t, t) for t in CODE_PATH.findall(text)]
        for path, t in targets:
            if any(c in t for c in "{}<>*") or "hidden_eval_hidden" in t:
                continue
            if not path.exists():
                missing.append(f"{md.relative_to(ROOT)}: {t}")
    assert not missing, missing
