import json
from datetime import datetime
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts/runs")

def save_run(prompt: str, model_name: str, results: dict) -> str:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"run_{timestamp}.json"

    # strip raw entropy lists to keep file size small; keep mean only
    clean_results = {}
    for strategy, data in results.items():
        clean_results[strategy] = {k: v for k, v in data.items() if k != "entropies"}

    artifact = {
        "timestamp": timestamp,
        "model": model_name,
        "prompt": prompt,
        "results": clean_results,
    }
    (ARTIFACTS_DIR / filename).write_text(json.dumps(artifact, indent=2))
    return filename