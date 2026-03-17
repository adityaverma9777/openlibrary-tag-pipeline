import json
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

    hierarchy_path = OUTPUT_DIR / "hierarchy.json"
    with open(hierarchy_path, "w") as f:
        json.dump(result.hierarchy, f, indent=2)

    full_path = OUTPUT_DIR / "pipeline_result.json"
    with open(full_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    return classified_path, structured_path, hierarchy_path, full_path


def print_header():
    header = Text()
    header.append("Open Library ", style="bold white")
    header.append("Tag Normalization & Classification Pipeline", style="bold cyan")
    console.print()
    console.print(Panel(header, border_style="bright_blue", padding=(1, 2)))
    console.print()


def print_pipeline_summary(result):
    stats = result.stats
    input_count = stats.get("input_tags", len(result.raw_tags))
    normalized_count = stats.get("unique_after_cleaning", len(result.cleaned_tags))
    before_taxonomy = stats.get(
        "clusters_before_taxonomy",
        stats.get("clusters_before_category_merge", len(result.classified_clusters)),
    )
    after_taxonomy = stats.get("clusters_final", len(result.classified_clusters))
    singletons = stats.get("singletons_remaining", stats.get("singletons", 0))
    rejected_low_conf = stats.get("rejected_low_confidence", 0)
    rejected_low_consistency = stats.get("rejected_low_consistency", 0)
    rejected_domain_mismatch = stats.get("rejected_domain_mismatch", 0)

    console.print("[bold yellow]Pipeline Summary[/]")
    console.print(f"Input tags: {input_count}")
    console.print(f"Normalized tags: {normalized_count}")
    console.print(f"Clusters before taxonomy merge: {before_taxonomy}")
    console.print(f"Clusters after taxonomy merge: {after_taxonomy}")
    console.print(f"Singleton clusters: {singletons}\n")
    console.print(f"Rejected merges (low confidence): {rejected_low_conf}")
    console.print(f"Rejected merges (low consistency): {rejected_low_consistency}")
    console.print(f"Rejected merges (domain mismatch): {rejected_domain_mismatch}\\n")


def print_before_after_taxonomy(result):
    stats = result.stats

    clusters_before = stats.get(
        "clusters_before_taxonomy",
        stats.get("clusters_before_category_merge", 0),
    )
    clusters_after = stats.get("clusters_final", len(result.classified_clusters))

    singletons_before = stats.get(
        "singletons_before_taxonomy",
        stats.get("singletons", 0),
    )
    singletons_after = stats.get(
        "singletons_remaining",
        stats.get("singletons", 0),
    )

    cluster_reduction = 0.0
    if clusters_before > 0:
        cluster_reduction = ((clusters_before - clusters_after) / clusters_before) * 100

    singleton_reduction = 0.0
    if singletons_before > 0:
        singleton_reduction = ((singletons_before - singletons_after) / singletons_before) * 100

    console.print("[bold yellow]Before vs After Taxonomy Merge[/]")
    console.print("Before Taxonomy Merge:")
    console.print(f"Clusters: {clusters_before}")
    console.print(f"Singleton clusters: {singletons_before}\n")

    console.print("After Taxonomy Merge:")
    console.print(f"Clusters: {clusters_after}")
    console.print(f"Singleton clusters: {singletons_after}\n")

    console.print("Cluster Reduction:")
    console.print(f"{clusters_before} -> {clusters_after} ({cluster_reduction:.1f}% reduction)\n")

    console.print("Singleton Reduction:")
    console.print(f"{singletons_before} -> {singletons_after} ({singleton_reduction:.1f}% reduction)\n")


def print_normalization_examples(result, limit: int = 10):
    console.print(f"[bold yellow]Normalization Examples (first {limit})[/]")

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    table.add_column("Normalized Tag", style="bold green")
    table.add_column("Original Variants", style="dim white")

    for tag in result.cleaned_tags[:limit]:
        originals = result.raw_to_clean_map.get(tag, [])
        variants = ", ".join(dict.fromkeys(originals))
        table.add_row(tag, variants if variants else "—")

    console.print(table)
    console.print()


def print_parent_genre_distribution(result):
    distribution = result.stats.get("parent_genre_distribution", {})
    if not distribution:
        return

    console.print("[bold yellow]Parent Genre Distribution[/]")
    sorted_items = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    for genre, count in sorted_items:
        label = " ".join(part.capitalize() for part in genre.split())
        console.print(f"{label} ({count} tags)")
    console.print()


def print_top_largest_clusters(result, top_n: int = 10, max_members_display: int = 12):
    if not result.classified_clusters:
        return

    console.print(f"[bold yellow]Top {top_n} Largest Clusters[/]")
    clusters = sorted(
        result.classified_clusters,
        key=lambda c: len(c.get("members", [])),
        reverse=True,
    )

    for cluster in clusters[:top_n]:
        label = cluster.get("cluster_label", "unknown")
        members = sorted(cluster.get("members", []))
        console.print(f"[bold cyan]{label}[/bold cyan]")

        shown_members = members[:max_members_display]
        for member in shown_members:
            console.print(member)

        remaining = len(members) - len(shown_members)
        if remaining > 0:
            console.print(f"[dim]+{remaining} more[/]")
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
            f"  [dim]  {paths[2]}[/]\n"
            f"  [dim]  {paths[3]}[/]",
            border_style="bright_green",
            title="Summary",
        )
    )


def main():
    print_header()

    tags = load_sample_tags()
    console.print(f"[bold yellow]Loaded input tags:[/] {len(tags)}\n")

    console.print("[dim]Initializing pipeline (loading models)...[/]\n")
    pipeline = TagPipeline()

    result = pipeline.process(tags)

    print_pipeline_summary(result)
    print_before_after_taxonomy(result)
    print_normalization_examples(result, limit=10)
    print_parent_genre_distribution(result)
    print_top_largest_clusters(result, top_n=10, max_members_display=12)

    paths = save_structured_output(result)
    print_summary(result, paths)


if __name__ == "__main__":
    main()
