# Open Library Tag Processing Pipeline

A scalable, Python-based pipeline for normalizing, clustering, and semantically classifying messy book subject tags, developed as part of a Google Summer of Code proposal for Open Library.

## Features

- **Normalization**: Unicode cleaning, synonym mapping, and deduplication
- **Semantic Similarity**: State-of-the-art sentence embeddings (`all-MiniLM-L6-v2`) via `sentence-transformers`
- **Density-Based Clustering**: HDBSCAN with noise recovery
- **Zero-Shot Classification**: Fast inference mapped to Genre, Theme, Setting, Mood, and Audience using `facebook/bart-large-mnli`

## Installation

```bash
pip install -r requirements.txt
pip install faiss-cpu joblib umap-learn matplotlib seaborn
```

## Running the Pipeline

```bash
python demo.py
```

Results are saved to the `output/` directory:
- `structured_categories.json` — Clean grouped mappings
- `classified_clusters.json` — Full cluster objects with confidence scores
- `pipeline_result.json` — Complete run snapshot with timing and cache metrics

## Performance Optimizations

### GPU Acceleration

The pipeline automatically detects CUDA availability via `torch.cuda.is_available()`. When a GPU is present:
- **SentenceTransformer** embeddings run on GPU with configurable `batch_size`
- **Zero-shot NLI model** (`facebook/bart-large-mnli`) runs on `device=0` with batched inference

When no GPU is available, everything falls back to CPU transparently.

### Persistent Embedding Cache

Embeddings are stored in `cache/embeddings_cache.npz` using a compressed NumPy format. On each run:
1. The cache is loaded at startup
2. Tags already in the cache are fetched instantly (no model inference)
3. Only new/unseen tags are encoded
4. The cache is updated and saved after encoding

This means the second run on the same dataset completes embedding in near-zero time. Cache statistics (hits, misses, hit rate) are printed in the runtime summary.

### FAISS Approximate Nearest Neighbor

Instead of computing an O(N²) pairwise similarity matrix, the pipeline uses **FAISS IndexHNSWFlat** to find the top-K nearest neighbors for each tag. This scales to hundreds of thousands of tags. FAISS is optional — if not installed, the system falls back to brute-force cosine similarity.

### Batch Neural Inference

All neural model calls are batched:
- **Embeddings**: `model.encode(tags, batch_size=64)` processes tags in chunks
- **NLI Classification**: All cluster representatives are classified in a single batched call rather than one-by-one

### Parallel CPU Operations

Lightweight CPU operations use `joblib` for parallel execution:
- Tag normalization runs across all available cores
- The number of workers is configurable via `config.num_workers` (-1 = all cores)

### Reproducibility

Set `random_seed` in `PipelineConfig` for deterministic clustering and experiments. All configurable parameters live in `tag_pipeline/config.py`.

### Expected Performance

| Dataset Size | First Run | Cached Run | Speedup |
|---|---|---|---|
| 65 tags | ~7min (CPU) | ~30s | ~14x |
| 1,300 tags | ~45min (CPU) | ~2min | ~22x |
| 1,300 tags (GPU) | ~5min | ~30s | ~10x |

*Times are approximate and depend on hardware.*

## Running Experiments

1. Generate summary metrics:
   ```bash
   python experiments/evaluation.py
   ```

2. Visualize tag clusters using UMAP:
   ```bash
   python experiments/visualize_clusters.py
   ```
   Saves a high-res scatter plot to `plots/tag_clusters_umap.png`.

## Project Structure

```
coderobot/
├── tag_pipeline/
│   ├── __init__.py
│   ├── config.py            # all tunable parameters
│   ├── tag_cleaner.py       # parallel normalization
│   ├── embedder.py          # cached embeddings + FAISS
│   ├── clusterer.py         # HDBSCAN clustering
│   ├── classifier.py        # batch NLI + keyword fallback
│   └── pipeline.py          # orchestrator with timing
├── experiments/
│   ├── evaluation.py        # metrics computation
│   └── visualize_clusters.py # UMAP scatter plots
├── data/
│   └── sample_tags.json
├── cache/                   # auto-generated embedding cache
├── output/                  # pipeline JSON outputs
├── plots/                   # generated visualizations
├── demo.py
├── README.md
└── requirements.txt
```
