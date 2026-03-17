import time
from dataclasses import dataclass, field

import numpy as np

from tag_pipeline.config import PipelineConfig
from tag_pipeline.tag_cleaner import TagCleaner
from tag_pipeline.embedder import TagEmbedder
from tag_pipeline.clusterer import TagClusterer
from tag_pipeline.classifier import CategoryClassifier
from tag_pipeline.taxonomy_merger import TaxonomyMerger


@dataclass
class PipelineResult:
    raw_tags: list[str] = field(default_factory=list)
    cleaned_tags: list[str] = field(default_factory=list)
    raw_to_clean_map: dict[str, list[str]] = field(default_factory=dict)
    clusters: list[dict] = field(default_factory=list)
    classified_clusters: list[dict] = field(default_factory=list)
    structured_categories: dict = field(default_factory=dict)
    hierarchy: dict = field(default_factory=dict)
    similar_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    cache_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "stats": self.stats,
            "timing": self.timing,
            "cache_stats": self.cache_stats,
            "clusters": [
                {
                    "representative": c["representative"],
                    "members": c["members"],
                    "size": c["size"],
                }
                for c in self.clusters
            ],
            "classified_clusters": self.classified_clusters,
            "structured_categories": self.structured_categories,
            "hierarchy": self.hierarchy,
            "top_similar_pairs": [
                {"tag_a": a, "tag_b": b, "similarity": round(s, 4)}
                for a, b, s in self.similar_pairs[:20]
            ],
        }


class TagPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.cleaner = TagCleaner(self.config)
        self.embedder = TagEmbedder(self.config)
        self.clusterer = TagClusterer(self.config)
        self.classifier = CategoryClassifier(self.config)
        self.taxonomy = TaxonomyMerger(self.config, self.embedder)

    def process(self, tags: list[str]) -> PipelineResult:
        total_start = time.time()
        result = PipelineResult(raw_tags=list(tags))

        t0 = time.time()
        result.cleaned_tags, result.raw_to_clean_map = (
            self.cleaner.clean_with_mapping(tags)
        )
        result.timing["normalization"] = round(time.time() - t0, 3)

        if len(result.cleaned_tags) < 2:
            result.clusters = [
                {"representative": t, "members": [t], "size": 1}
                for t in result.cleaned_tags
            ]
            result.stats = self._build_stats(result)
            result.timing["total"] = round(time.time() - total_start, 3)
            return result

        t0 = time.time()
        embeddings = self.embedder.encode(result.cleaned_tags)
        result.timing["embedding"] = round(time.time() - t0, 3)
        result.cache_stats = dict(self.embedder.cache_stats)

        t0 = time.time()
        if self.config.faiss_enabled:
            result.similar_pairs = self.embedder.similarity_pairs_faiss(
                embeddings, result.cleaned_tags,
                top_k=self.config.faiss_top_k, threshold=0.4,
            )
        else:
            result.similar_pairs = self.embedder.pairwise_similarity(
                embeddings, result.cleaned_tags, threshold=0.4,
            )

        raw_clusters = self.clusterer.cluster(result.cleaned_tags, embeddings)
        result.clusters = TagClusterer.strip_internals(raw_clusters)
        result.timing["clustering"] = round(time.time() - t0, 3)

        t0 = time.time()
        result.classified_clusters = self.classifier.classify_clusters(
            result.clusters
        )

        clusters_before_cat_merge = len(result.classified_clusters)

        result.classified_clusters = self.clusterer.category_merge(
            result.classified_clusters, result.cleaned_tags, embeddings,
        )
        result.timing["classification"] = round(time.time() - t0, 3)

        t0 = time.time()
        clusters_before_taxonomy = len(result.classified_clusters)
        singletons_before_taxonomy = sum(
            1 for c in result.classified_clusters if len(c.get("members", [])) == 1
        )

        result.classified_clusters = self.taxonomy.merge(
            result.classified_clusters, result.cleaned_tags, embeddings,
        )

        result.hierarchy = self.taxonomy.build_hierarchy(
            result.classified_clusters, result.cleaned_tags, embeddings,
        )

        result.structured_categories = self.classifier.build_structured_output(
            result.classified_clusters
        )
        result.timing["taxonomy_merge"] = round(time.time() - t0, 3)

        result.timing["total"] = round(time.time() - total_start, 3)
        result.stats = self._build_stats(result)
        result.stats["clusters_before_agglom_merge"] = self.clusterer.stats["clusters_before_merge"]
        result.stats["clusters_after_agglom_merge"] = self.clusterer.stats["clusters_after_merge"]
        result.stats["clusters_before_category_merge"] = clusters_before_cat_merge
        result.stats["clusters_before_taxonomy"] = clusters_before_taxonomy
        result.stats["singletons_before_taxonomy"] = singletons_before_taxonomy
        result.stats["clusters_absorbed"] = self.taxonomy.stats["clusters_absorbed"]
        result.stats["singletons_remaining"] = self.taxonomy.stats["singletons_remaining"]
        result.stats["parent_genres_active"] = self.taxonomy.stats["parent_genres_active"]
        result.stats["parent_genre_distribution"] = self.taxonomy.stats["parent_genre_distribution"]
        result.stats["rejected_low_confidence"] = self.taxonomy.stats["rejected_low_confidence"]
        result.stats["rejected_low_consistency"] = self.taxonomy.stats["rejected_low_consistency"]
        result.stats["rejected_domain_mismatch"] = self.taxonomy.stats["rejected_domain_mismatch"]
        result.stats["allowed_despite_domain_mismatch"] = self.taxonomy.stats["allowed_despite_domain_mismatch"]
        result.stats["allowed_with_domain_score_below_one"] = self.taxonomy.stats["allowed_with_domain_score_below_one"]
        result.stats["clusters_final"] = len(result.classified_clusters)
        return result

    def _build_stats(self, result: PipelineResult) -> dict:
        category_counts = {}
        for item in result.classified_clusters:
            cat = item.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "input_tags": len(result.raw_tags),
            "unique_after_cleaning": len(result.cleaned_tags),
            "duplicates_removed": len(result.raw_tags) - len(result.cleaned_tags),
            "clusters_formed": len(result.classified_clusters),
            "largest_cluster": max(
                (len(c.get("members", [])) for c in result.classified_clusters), default=0
            ),
            "singletons": sum(
                1 for c in result.classified_clusters if len(c.get("members", [])) == 1
            ),
            "categories_assigned": category_counts,
        }
