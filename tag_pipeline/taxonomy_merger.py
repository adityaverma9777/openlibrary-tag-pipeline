import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from tag_pipeline.config import PipelineConfig


PARENT_GENRES = [
    "fantasy",
    "science fiction",
    "horror",
    "romance",
    "mystery",
    "thriller",
    "crime",
    "history",
    "nonfiction",
    "philosophy",
    "science",
    "technology",
    "politics",
    "religion",
    "sports",
    "animals",
    "art",
    "music",
    "education",
    "health",
    "psychology",
    "sociology",
    "economics",
    "law",
    "military",
    "nature",
    "cooking",
    "travel",
    "humor",
    "drama",
    "poetry",
    "biography",
    "adventure",
    "western",
    "children",
    "young adult",
    "self help",
    "business",
    "mathematics",
    "engineering",
    "medicine",
    "agriculture",
    "environment",
    "media",
    "communication",
    "transportation",
    "architecture",
    "fashion",
    "games",
    "crafts",
    "photography",
    "dance",
    "film",
    "theater",
    "linguistics",
]


class TaxonomyMerger:
    def __init__(self, config: PipelineConfig, embedder):
        self.config = config
        self.embedder = embedder
        self._parent_embeddings = None
        self._parent_list = list(PARENT_GENRES)
        self.stats = {
            "singletons_absorbed": 0,
            "parent_genres_active": 0,
            "clusters_before_taxonomy": 0,
            "clusters_after_taxonomy": 0,
        }

    def _get_parent_embeddings(self) -> np.ndarray:
        if self._parent_embeddings is None:
            self._parent_embeddings = self.embedder.encode(self._parent_list)
        return self._parent_embeddings

    def merge(self, classified_clusters: list[dict], tags: list[str], embeddings: np.ndarray) -> list[dict]:
        if not self.config.enable_hierarchical_merge:
            return classified_clusters

        self.stats["clusters_before_taxonomy"] = len(classified_clusters)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = embeddings / norms

        tag_to_idx = {t: i for i, t in enumerate(tags)}
        parent_embs = self._get_parent_embeddings()
        threshold = self.config.parent_genre_similarity_threshold

        parent_clusters: dict[str, dict] = {}
        unmatched = []

        for cluster in classified_clusters:
            rep = cluster["cluster_label"]
            rep_idx = tag_to_idx.get(rep)

            if rep_idx is None:
                unmatched.append(cluster)
                continue

            rep_vec = normed[rep_idx].reshape(1, -1)
            sims = cosine_similarity(rep_vec, parent_embs).flatten()
            best_parent_idx = int(np.argmax(sims))
            best_sim = float(sims[best_parent_idx])

            if rep in self._parent_list:
                parent_name = rep
                if parent_name not in parent_clusters:
                    parent_clusters[parent_name] = self._init_parent(parent_name, cluster)
                else:
                    self._absorb_into_parent(parent_clusters[parent_name], cluster)
            elif best_sim >= threshold:
                parent_name = self._parent_list[best_parent_idx]
                if parent_name not in parent_clusters:
                    parent_clusters[parent_name] = self._init_parent(parent_name, cluster)
                else:
                    self._absorb_into_parent(parent_clusters[parent_name], cluster)
                if len(cluster.get("members", [])) == 1:
                    self.stats["singletons_absorbed"] += 1
            else:
                unmatched.append(cluster)

        result = list(parent_clusters.values()) + unmatched
        result.sort(key=lambda c: len(c.get("members", [])), reverse=True)

        self.stats["parent_genres_active"] = len(parent_clusters)
        self.stats["clusters_after_taxonomy"] = len(result)

        return result

    def build_hierarchy(self, classified_clusters: list[dict], tags: list[str], embeddings: np.ndarray) -> dict:
        if not self.config.enable_hierarchical_merge:
            return {}

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = embeddings / norms

        tag_to_idx = {t: i for i, t in enumerate(tags)}
        parent_embs = self._get_parent_embeddings()
        threshold = self.config.parent_genre_similarity_threshold

        hierarchy: dict[str, list[str]] = {}

        for cluster in classified_clusters:
            rep = cluster["cluster_label"]
            rep_idx = tag_to_idx.get(rep)

            if rep_idx is None:
                hierarchy.setdefault("other", []).append(rep)
                continue

            rep_vec = normed[rep_idx].reshape(1, -1)
            sims = cosine_similarity(rep_vec, parent_embs).flatten()
            best_parent_idx = int(np.argmax(sims))
            best_sim = float(sims[best_parent_idx])

            if rep in self._parent_list:
                hierarchy.setdefault(rep, [])
            elif best_sim >= threshold:
                parent_name = self._parent_list[best_parent_idx]
                hierarchy.setdefault(parent_name, []).append(rep)
            else:
                hierarchy.setdefault("other", []).append(rep)

        hierarchy = {k: v for k, v in hierarchy.items() if k or v}
        return dict(sorted(hierarchy.items(), key=lambda x: len(x[1]), reverse=True))

    def _init_parent(self, parent_name: str, cluster: dict) -> dict:
        return {
            "cluster_label": parent_name,
            "members": list(cluster.get("members", [])),
            "category": cluster.get("category", "genre"),
            "confidence": cluster.get("confidence", 0.5),
            "method": cluster.get("method", "taxonomy"),
            "subclusters": [cluster["cluster_label"]],
        }

    def _absorb_into_parent(self, parent: dict, child: dict):
        child_members = child.get("members", [])
        existing = set(parent["members"])
        for m in child_members:
            if m not in existing:
                parent["members"].append(m)
                existing.add(m)
        parent["members"].sort()

        if child["cluster_label"] not in parent["subclusters"]:
            parent["subclusters"].append(child["cluster_label"])

        total = len(parent["members"])
        prev_weight = (total - len(child_members)) / total if total > 0 else 0.5
        parent["confidence"] = round(
            parent["confidence"] * prev_weight + child["confidence"] * (1 - prev_weight), 4
        )
