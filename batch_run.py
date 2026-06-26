import json
from pathlib import Path
from inference.loader import load_model_and_tokenizer
from services.state import model_state
from services.comparator import run_comparison
from display.terminal import print_comparison_table
from tracking.writer import save_run
from utils.loader import load_config
from rich.console import Console

console = Console()
config = load_config()

prompts = json.loads(Path("data/prompts.json").read_text())

model_name = config["inference"]["default_model"]
model_cfg = config["models"][model_name]
model_state.load(model_name, model_cfg["path"], config["inference"]["device"])

for i, prompt in enumerate(prompts):
    console.rule(f"[bold cyan]Prompt {i+1}: {prompt[:50]}...")
    results = run_comparison(
        prompt,
        ["greedy", "top_k", "top_p", "beam"],
        model_state.model,
        model_state.tokenizer,
        config["inference"]["max_new_tokens"],
    )
    print_comparison_table(results)
    fname = save_run(prompt, model_name, results)
    console.print(f"[dim]Saved → artifacts/runs/{fname}[/dim]\n")