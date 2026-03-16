import json
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from tag_pipeline import TagPipeline


console = Console()

OUTPUT_DIR = Path(__file__).parent / "output"


def load_sample_tags() -> list[str]:
    data_path = Path(__file__).parent / "data" / "sample_tags.json"
    with open(data_path) as f:
        return json.load(f)


def save_structured_output(result):
    OUTPUT_DIR.mkdir(exist_ok=True)

    classified_path = OUTPUT_DIR / "classified_clusters.json"
    with open(classified_path, "w") as f:
        json.dump(result.classified_clusters, f, indent=2)

    structured_path = OUTPUT_DIR / "structured_categories.json"
    with open(structured_path, "w") as f:
        json.dump(result.structured_categories, f, indent=2)

    full_path = OUTPUT_DIR / "pipeline_result.json"
    with open(full_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    return classified_path, structured_path, full_path


def print_header():
    header = Text()
    header.append("Open Library ", style="bold white")
    header.append("Tag Normalization & Classification Pipeline", style="bold cyan")
    console.print()
    console.print(Panel(header, border_style="bright_blue", padding=(1, 2)))
    console.print()


def print_raw_tags(tags: list[str]):
    console.print("[bold yellow]Stage 0:[/] Raw Input Tags")
    console.print(f"  Total: [bold]{len(tags)}[/] tags\n")

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Raw Tag", style="white")

    for i, tag in enumerate(tags, 1):
        table.add_row(str(i), tag)

    console.print(table)
    console.print()


def print_cleaning_results(result):
    console.print("[bold yellow]Stage 1:[/] Tag Normalization")
    console.print(f"  Input:  [bold]{len(result.raw_tags)}[/] tags")
    console.print(f"  Output: [bold]{len(result.cleaned_tags)}[/] unique normalized tags")
    console.print(
        f"  Removed: [bold red]{result.stats['duplicates_removed']}[/] duplicates\n"
    )

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    table.add_column("Normalized Tag", style="bold green")
    table.add_column("Original Variants", style="dim white")

    for tag in result.cleaned_tags:
        originals = result.raw_to_clean_map.get(tag, [])
        variants = ", ".join(f'"{o}"' for o in originals if o.lower().strip() != tag)
        table.add_row(tag, variants if variants else "—")

    console.print(table)
    console.print()


def print_similar_pairs(result):
    if not result.similar_pairs:
        return

    console.print("[bold yellow]Stage 2:[/] Semantic Similarity (top pairs)")

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    table.add_column("Tag A", style="cyan")
    table.add_column("Tag B", style="cyan")
    table.add_column("Similarity", style="bold white", justify="right")

    for a, b, score in result.similar_pairs[:15]:
        bar_len = int(score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        table.add_row(a, b, f"{score:.3f} {bar}")

    console.print(table)
    console.print()


def print_clusters(result):
    console.print("[bold yellow]Stage 3:[/] Tag Clusters")
    console.print(f"  Clusters formed: [bold]{result.stats['clusters_formed']}[/]")
    console.print(f"  Largest cluster: [bold]{result.stats['largest_cluster']}[/] members")
    console.print(f"  Singletons:      [bold]{result.stats['singletons']}[/]\n")

    colors = [
        "bright_cyan", "bright_green", "bright_yellow", "bright_magenta",
        "bright_red", "bright_blue", "bright_white", "cyan", "green",
        "yellow", "magenta", "red", "blue",
    ]

    for i, cluster in enumerate(result.clusters):
        color = colors[i % len(colors)]
        rep = cluster["representative"]
        members = cluster["members"]

        member_text = ", ".join(
            f"[bold]{m}[/]" if m == rep else m for m in members
        )

        console.print(
            f"  [{color}]Cluster {i + 1}[/] "
            f"[dim]({cluster['size']} tags)[/]  "
            f"→  {member_text}"
        )

    console.print()


def print_classification(result):
    if not result.classified_clusters:
        return

    console.print("[bold yellow]Stage 4:[/] Semantic Category Classification\n")

    category_styles = {
        "genre": "bold bright_cyan",
        "theme": "bold bright_green",
        "setting": "bold bright_yellow",
        "mood": "bold bright_magenta",
        "audience": "bold bright_red",
    }

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    table.add_column("Cluster Label", style="white", min_width=20)
    table.add_column("Category", min_width=10)
    table.add_column("Confidence", justify="right", min_width=10)
    table.add_column("Method", style="dim", min_width=8)
    table.add_column("Members", style="dim white")

    for item in result.classified_clusters:
        cat = item["category"]
        style = category_styles.get(cat, "white")
        conf = item["confidence"]
        bar_len = int(conf * 15)
        bar = "█" * bar_len + "░" * (15 - bar_len)

        members_str = ", ".join(item["members"][:5])
        if len(item["members"]) > 5:
            members_str += f" (+{len(item['members']) - 5})"

        table.add_row(
            item["cluster_label"],
            f"[{style}]{cat}[/]",
            f"{conf:.3f} {bar}",
            item["method"],
            members_str,
        )

    console.print(table)
    console.print()


def print_structured_output(result):
    if not result.structured_categories:
        return

    console.print("[bold yellow]Structured Output:[/]\n")

    display = {}
    for category, entries in result.structured_categories.items():
        display[category] = [e["label"] for e in entries]

    formatted = json.dumps(display, indent=2)
    console.print(Panel(formatted, title="Structured Categories", border_style="bright_green"))
    console.print()


def print_timing(result):
    timing = result.timing
    if not timing:
        return

    console.print(Panel(
        f"[bold]Pipeline Runtime Summary[/]\n"
        f"  Normalization:    [cyan]{timing.get('normalization', 0):.3f}s[/]\n"
        f"  Embedding:        [cyan]{timing.get('embedding', 0):.3f}s[/]\n"
        f"  Clustering:       [cyan]{timing.get('clustering', 0):.3f}s[/]\n"
        f"  Classification:   [cyan]{timing.get('classification', 0):.3f}s[/]\n"
        f"  [bold]Total:            [bright_green]{timing.get('total', 0):.3f}s[/]",
        title="⏱ Timing",
        border_style="bright_blue",
    ))

    cache = result.cache_stats
    if cache:
        hit_rate = cache.get("hit_rate", 0)
        console.print(Panel(
            f"[bold]Embedding Cache[/]\n"
            f"  Total processed:  [cyan]{cache.get('total_processed', 0)}[/]\n"
            f"  Cache hits:       [green]{cache.get('cache_hits', 0)}[/]\n"
            f"  New computed:     [yellow]{cache.get('new_computed', 0)}[/]\n"
            f"  Hit rate:         [bold bright_green]{hit_rate:.1f}%[/]",
            title="💾 Cache",
            border_style="bright_yellow",
        ))

    stats = result.stats
    before_agglom = stats.get("clusters_before_agglom_merge", "?")
    after_agglom = stats.get("clusters_after_agglom_merge", "?")
    before_cat = stats.get("clusters_before_category_merge", "?")
    final = stats.get("clusters_final", "?")

    console.print(Panel(
        f"[bold]Cluster Merge Pipeline[/]\n"
        f"  After agglomerative:     [cyan]{before_agglom}[/] clusters\n"
        f"  After centroid merge:    [green]{after_agglom}[/] clusters\n"
        f"  After classification:    [yellow]{before_cat}[/] clusters\n"
        f"  After category merge:    [bold bright_green]{final}[/] clusters",
        title="🔗 Merge Stats",
        border_style="bright_magenta",
    ))
    console.print()


def print_summary(result, paths: tuple):
    cat_counts = result.stats.get("categories_assigned", {})
    cat_line = ", ".join(f"{k}: {v}" for k, v in cat_counts.items())
    total_time = result.timing.get("total", 0)
    final_clusters = result.stats.get("clusters_final", result.stats.get("clusters_formed", 0))

    console.print(
        Panel(
            f"[bold]Pipeline Complete[/]\n"
            f"  {result.stats['input_tags']} raw tags → "
            f"{result.stats['unique_after_cleaning']} unique → "
            f"{final_clusters} clusters\n"
            f"  Categories: {cat_line}\n"
            f"  Time: {total_time:.2f}s\n\n"
            f"  [dim]Output saved:[/]\n"
            f"  [dim]  {paths[0]}[/]\n"
            f"  [dim]  {paths[1]}[/]\n"
            f"  [dim]  {paths[2]}[/]",
            border_style="bright_green",
            title="Summary",
        )
    )


def main():
    print_header()

    tags = load_sample_tags()
    print_raw_tags(tags)

    console.print("[dim]Initializing pipeline (loading models)...[/]\n")
    pipeline = TagPipeline()

    result = pipeline.process(tags)

    print_cleaning_results(result)
    print_similar_pairs(result)
    print_clusters(result)
    print_classification(result)
    print_structured_output(result)
    print_timing(result)

    paths = save_structured_output(result)
    print_summary(result, paths)


if __name__ == "__main__":
    main()
