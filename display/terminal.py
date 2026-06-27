from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

HIGH_ENTROPY = 6.0
MID_ENTROPY = 3.0
SPARK_CHARS = "▁▂▃▄▅▆▇█"


def entropy_to_spark(H: float, max_H: float = 8.0) -> str:
    idx = int((H / max_H) * (len(SPARK_CHARS) - 1))
    idx = max(0, min(idx, len(SPARK_CHARS) - 1))
    return SPARK_CHARS[idx]


def print_comparison_table(results: dict):
    console.print("\n[bold]Generated Outputs[/bold]")
    for name, data in results.items():
        console.print(f"\n[bold cyan]{name}[/bold cyan]")
        console.print(data["text"])
    console.print()

    table = Table(title="ENTROPYLENS", box=box.ROUNDED)
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


def print_entropy_bars(results: dict, verbose: bool = False):
    console.print("\n[bold]Per-token entropy[/bold]\n")
    for name, data in results.items():
        entropies = data["entropies"]
        spark = "".join(entropy_to_spark(H) for H in entropies)
        mean_H = data["mean_entropy"]
        peak_H = max(entropies) if entropies else 0.0
        console.print(
            f"[bold cyan]{name:8}[/bold cyan]  "
            f"[yellow]{spark}[/yellow]  "
            f"mean [green]{mean_H:.3f}[/green]  "
            f"peak [red]{peak_H:.3f}[/red]"
        )

        if verbose:
            for i, H in enumerate(entropies):
                bar_len = min(int(H * 4), 40)
                color = "red" if H > HIGH_ENTROPY else "yellow" if H > MID_ENTROPY else "green"
                bar = "■" * bar_len
                console.print(f"  t{i+1:03d} [{color}]{bar}[/{color}] {H:.2f}")
        else:
            console.print(f"  [dim]Top uncertainty positions:[/dim]")
            top5 = sorted(enumerate(entropies), key=lambda x: x[1], reverse=True)[:5]
            top5 = sorted(top5, key=lambda x: x[0])  
            for i, H in top5:
                bar_len = min(int(H * 4), 40)
                color = "red" if H > HIGH_ENTROPY else "yellow" if H > MID_ENTROPY else "green"
                bar = "■" * bar_len
                console.print(f"  t{i+1:03d} [{color}]{bar}[/{color}] {H:.2f}")

        console.print()