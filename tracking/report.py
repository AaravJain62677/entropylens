# tracking/report.py

import json
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts/runs")

def list_runs() -> list[str]:
    return sorted(f.name for f in ARTIFACTS_DIR.glob("run_*.json"))

def load_run(filename: str) -> dict:
    return json.loads((ARTIFACTS_DIR / filename).read_text())

def diff_runs(file_a: str, file_b: str) -> dict:
    a, b = load_run(file_a), load_run(file_b)
    return {"run_a": a, "run_b": b}