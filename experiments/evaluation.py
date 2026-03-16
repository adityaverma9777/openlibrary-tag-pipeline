import json
import statistics
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def load_results() -> dict:
    result_path = OUTPUT_DIR / "pipeline_result.json"
    if not result_path.exists():
        console.print(f"[bold red]Error:[/] Result file not found at {result_path}")
        console.print("Please run `python demo.py` first to generate results.")
        exit(1)
    
    with open(result_path, "r") as f:
        return json.load(f)


def compute_metrics(results: dict):
    stats = results.get("stats", {})
    clusters = results.get("clusters", [])
    classified = results.get("classified_clusters", [])

    total_tags = stats.get("unique_after_cleaning", 0)
    
    cluster_sizes = [c["size"] for c in clusters]
    avg_cluster_size = statistics.mean(cluster_sizes) if cluster_sizes else 0
    noise_count = sum(1 for c in clusters if c["size"] == 1)
    noise_percentage = (noise_count / total_tags * 100) if total_tags else 0
    
    confidences = [c.get("confidence", 0) for c in classified]
    mean_confidence = statistics.mean(confidences) if confidences else 0
    
    nli_count = sum(1 for c in classified if c.get("method") == "nli")
    keyword_count = sum(1 for c in classified if str(c.get("method")).startswith("keyword"))
    
    return {
        "total_tags": total_tags,
        "total_clusters": len(clusters),
        "avg_cluster_size": avg_cluster_size,
        "noise_count": noise_count,
        "noise_percentage": noise_percentage,
        "mean_confidence": mean_confidence,
        "nli_count": nli_count,
        "keyword_count": keyword_count,
        "category_counts": stats.get("categories_assigned", {}),
    }


def print_evaluation(metrics: dict):
    console.print(Panel("[bold cyan]Pipeline Evaluation Metrics[/]", border_style="bright_blue"))
    
    console.print(f"[bold]Clustering Performance:[/]")
    console.print(f"  Total Unique Tags:      [bold cyan]{metrics['total_tags']}[/]")
    console.print(f"  Clusters Formed:        [bold green]{metrics['total_clusters']}[/]")
    console.print(f"  Average Cluster Size:   [bold yellow]{metrics['avg_cluster_size']:.2f}[/]")
    console.print(f"  Noise Tags (Size 1):    [bold red]{metrics['noise_count']} ({metrics['noise_percentage']:.1f}%)[/]\n")
    
    console.print(f"[bold]Classification Performance:[/]")
    console.print(f"  Mean Confidence:        [bold magenta]{metrics['mean_confidence']:.3f}[/]")
    console.print(f"  Zero-Shot NLI Matches:  [bold blue]{metrics['nli_count']}[/]")
    console.print(f"  Keyword Fallbacks:      [bold yellow]{metrics['keyword_count']}[/]\n")
    
    table = Table(title="Category Distribution", box=box.SIMPLE_HEAVY, header_style="bold magenta")
    table.add_column("Category")
    table.add_column("Count", justify="right")
    
    for cat, count in sorted(metrics['category_counts'].items(), key=lambda x: x[1], reverse=True):
        table.add_row(cat.title(), str(count))
        
    console.print(table)


def main():
    results = load_results()
    metrics = compute_metrics(results)
    print_evaluation(metrics)


if __name__ == "__main__":
    main()
