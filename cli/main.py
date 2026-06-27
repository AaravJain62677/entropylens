import typer
from services.state import model_state
from services.comparator import run_comparison
from display.terminal import print_comparison_table, print_entropy_bars
from tracking.writer import save_run
from tracking.report import list_runs, load_run, diff_runs
from utils.loader import load_config
from strategies.registry import list_strategies
from rich import print_json
from rich.console import Console
console = Console()
import json

BANNER = """
  ███████╗███╗   ██╗████████╗██████╗  ██████╗ ██████╗ ██╗   ██╗██╗     ███████╗███╗   ██╗███████╗
  ██╔════╝████╗  ██║╚══██╔══╝██╔══██╗██╔═══██╗██╔══██╗╚██╗ ██╔╝██║     ██╔════╝████╗  ██║██╔════╝
  █████╗  ██╔██╗ ██║   ██║   ██████╔╝██║   ██║██████╔╝ ╚████╔╝ ██║     █████╗  ██╔██╗ ██║███████╗
  ██╔══╝  ██║╚██╗██║   ██║   ██╔══██╗██║   ██║██╔═══╝   ╚██╔╝  ██║     ██╔══╝  ██║╚██╗██║╚════██║
  ███████╗██║ ╚████║   ██║   ██║  ██║╚██████╔╝██║        ██║   ███████╗███████╗██║ ╚████║███████║
  ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝        ╚═╝   ╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝
                    UNCERTAINITY BEHIND EVERY TOKEN MADE VISIBLE 
"""

def show_banner():
    console.print(f"[bold cyan]{BANNER}[/bold cyan]")

app = typer.Typer(name="EntropyLens", help="EntropyLens")
config = load_config()

def _ensure_model_loaded(model_name: str):
    if not model_state.is_loaded() or model_state.active_model_name != model_name:
        model_cfg = config["models"][model_name]
        typer.echo(f"Loading {model_name}...")
        model_state.load(model_name, model_cfg["path"], config["inference"]["device"])

@app.command()
def compare(
    prompt: str = typer.Option(..., "-p", help="Prompt to compare strategies on"),
    model: str = typer.Option(config["inference"]["default_model"], "-m"),
    strategies: list[str] = typer.Option(list_strategies(), "-s"),
    ):
    show_banner()
    _ensure_model_loaded(model)
    results = run_comparison(
        prompt, strategies,
        model_state.model, model_state.tokenizer,
        config["inference"]["max_new_tokens"],
    )
    print_comparison_table(results)
    print_entropy_bars(results)
    fname = save_run(prompt, model, results)
    typer.echo(f"\nRun saved → artifacts/runs/{fname}")
@app.command("list")
def list_cmd():
    runs = list_runs()
    for r in runs:
        typer.echo(r)

@app.command()
def show(filename: str = typer.Argument(...)):
    run = load_run(filename)
    print_json(json.dumps(run))

@app.command()
def diff(
    file_a: str = typer.Option(..., "-a"),
    file_b: str = typer.Option(..., "-b"),
):
    result = diff_runs(file_a, file_b)
    print_comparison_table(result["run_a"]["results"])
    typer.echo("---")
    print_comparison_table(result["run_b"]["results"])

if __name__ == "__main__":
    app()