# display/terminal.py

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

HIGH_ENTROPY = 6.0
MID_ENTROPY = 3.0

def print_comparison_table(results: dict):
    table = Table(title="Decoding Strategy Comparison", box=box.ROUNDED)
    table.add_column("Strategy", style="bold")
    table.add_column("Output tokens", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Tokens/sec", justify="right")
    table.add_column("Mean entropy", justify="right")
    table.add_column("Repetition rate", justify="right")

    for name, data in results.items():
        table.add_row(
            name,
            str(data["num_output_tokens"]),
            str(data["latency"]["latency_ms"]),
            str(data["latency"]["tokens_per_sec"]),
            f"{data['mean_entropy']:.3f}",
            f"{data['repetition_rate']:.3f}",
        )
    console.print(table)

def print_entropy_bars(results: dict, max_tokens: int = 40):
    console.print("\n[bold]Per-token entropy (■ = high uncertainty)[/bold]\n")
    for name, data in results.items():
        console.print(f"[bold cyan]{name}[/bold cyan]")
        entropies = data["entropies"][:max_tokens]
        for i, H in enumerate(entropies):
            bar_len = min(int(H * 4), 40)
            if H > HIGH_ENTROPY:
                color = "red"
            elif H > MID_ENTROPY:
                color = "yellow"
            else:
                color = "green"
            bar = "■" * bar_len
            console.print(f"  t{i+1:02d} [{color}]{bar}[/{color}] {H:.2f}")
        console.print()