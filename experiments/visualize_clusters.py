import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap

from rich.console import Console

# Suppress UMAP numba warnings
warnings.filterwarnings('ignore', category=UserWarning, module='umap')

console = Console()
OUTPUT_DIR = Path(__file__).parent.parent / "output"
PLOTS_DIR = Path(__file__).parent.parent / "plots"


def load_results() -> dict:
    result_path = OUTPUT_DIR / "pipeline_result.json"
    if not result_path.exists():
        console.print(f"[bold red]Error:[/] Result file not found at {result_path}")
        console.print("Please run `python demo.py` first to generate results.")
        exit(1)
        
    with open(result_path, "r") as f:
        return json.load(f)


def extract_embeddings_and_labels(results: dict):
    """
    To visualize accurately, we need the embeddings for all tags.
    Since pipeline_result.json only stores clusters, we will re-embed the normalized tags
    just for the visualization.
    """
    console.print("[dim]Loading embedding model to re-encode tags for visualization...[/]")
    from tag_pipeline.embedder import TagEmbedder
    from tag_pipeline.config import PipelineConfig
    
    embedder = TagEmbedder(PipelineConfig())
    tags = results.get("cleaned_tags", [])
    
    if not tags:
        console.print("[bold red]Error:[/] No cleaned tags found in results.")
        exit(1)
        
    embeddings = embedder.encode(tags)
    
    # Map tags to their assigned cluster ID (index)
    tag_to_cluster = {}
    cluster_representatives = {}
    
    clusters = results.get("clusters", [])
    for idx, cluster in enumerate(clusters):
        rep = cluster.get("representative", "")
        if rep:
            cluster_representatives[idx] = rep
        for member in cluster.get("members", []):
            tag_to_cluster[member] = idx
            
    # Assign labels to each tag
    labels = [tag_to_cluster.get(t, -1) for t in tags]
    
    return tags, embeddings, labels, cluster_representatives


def plot_umap(tags, embeddings, labels, representatives):
    console.print(f"[dim]Reducing dimensions with UMAP ({len(tags)} points)...[/]")
    
    # UMAP dimensionality reduction
    reducer = umap.UMAP(
        n_neighbors=min(15, len(tags) - 1),
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Setup plot
    plt.figure(figsize=(14, 10))
    sns.set_theme(style="whitegrid")
    
    # Define a color palette suitable for the number of clusters
    unique_labels = list(set(labels))
    n_clusters = len(unique_labels)
    palette = sns.color_palette("husl", n_clusters)
    
    # Map labels to colors (handle noise if label == -1)
    colors = [palette[unique_labels.index(l)] if l != -1 else (0.7, 0.7, 0.7) for l in labels]
    
    # Scatter plot
    scatter = plt.scatter(
        embedding_2d[:, 0], 
        embedding_2d[:, 1], 
        c=colors, 
        s=60, 
        alpha=0.7, 
        edgecolors='w', 
        linewidth=0.5
    )
    
    # Annotate representative tags
    plotted_reps = set()
    for i, tag in enumerate(tags):
        cluster_id = labels[i]
        if cluster_id != -1 and cluster_id in representatives:
            rep_tag = representatives[cluster_id]
            if tag == rep_tag and rep_tag not in plotted_reps:
                plt.annotate(
                    rep_tag,
                    (embedding_2d[i, 0], embedding_2d[i, 1]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
                plotted_reps.add(rep_tag)
    
    plt.title("Semantic Tag Clusters (UMAP Projection)", fontsize=16, pad=15)
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    
    # Save the plot
    PLOTS_DIR.mkdir(exist_ok=True)
    plot_path = PLOTS_DIR / "tag_clusters_umap.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    console.print(f"[bold green]Success:[/] Saved cluster visualization to {plot_path}")
    
    # Show it if requested, but generally saving is safer in CI/agent modes
    # plt.show()


def main():
    console.print(Panel("[bold cyan]Tag Cluster Visualization[/]", border_style="bright_blue"))
    results = load_results()
    tags, embeddings, labels, reps = extract_embeddings_and_labels(results)
    plot_umap(tags, embeddings, labels, reps)


if __name__ == "__main__":
    main()
