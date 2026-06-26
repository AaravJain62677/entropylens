import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent

def load_config(path: str = "config/settings.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())