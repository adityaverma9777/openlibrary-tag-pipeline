import numpy as np
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity

from tag_pipeline.config import PipelineConfig


class TagClusterer:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def cluster(self, tags: list[str], embeddings: np.ndarray) -> list[dict]:
        if len(tags) == 0:
            return []
        if len(tags) == 1:
            return [self._make_cluster(tags, embeddings, [0])]

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.hdbscan_min_cluster_size,
            min_samples=self.config.hdbscan_min_samples,
            metric=self.config.hdbscan_metric,
            cluster_selection_method=self.config.hdbscan_cluster_selection_method,
        )
        labels = clusterer.fit_predict(embeddings)

        label_to_indices: dict[int, list[int]] = {}
        noise_indices: list[int] = []

        for idx, label in enumerate(labels):
            if label == -1:
                noise_indices.append(idx)
            else:
                label_to_indices.setdefault(label, []).append(idx)

        clusters = []
        for indices in label_to_indices.values():
            clusters.append(self._make_cluster(tags, embeddings, indices))

        for idx in noise_indices:
            merged = self._try_merge_noise(idx, tags, embeddings, clusters)
            if not merged:
                clusters.append(self._make_cluster(tags, embeddings, [idx]))

        clusters = self._post_merge(clusters, embeddings, tags)
        clusters.sort(key=lambda c: c["size"], reverse=True)
        return clusters

    def _make_cluster(
        self, tags: list[str], embeddings: np.ndarray, indices: list[int]
    ) -> dict:
        members = [tags[i] for i in indices]
        member_embeddings = embeddings[indices]
        centroid = np.mean(member_embeddings, axis=0)
        representative = self._pick_representative(members, member_embeddings, centroid)

        return {
            "representative": representative,
            "members": sorted(members),
            "size": len(members),
            "centroid": centroid,
            "_indices": indices,
        }

    def _pick_representative(
        self, members: list[str], embeddings: np.ndarray, centroid: np.ndarray
    ) -> str:
        if len(members) == 1:
            return members[0]

        centroid_2d = centroid.reshape(1, -1)
        similarities = cosine_similarity(embeddings, centroid_2d).flatten()
        best_idx = int(np.argmax(similarities))
        return members[best_idx]

    def _try_merge_noise(
        self,
        noise_idx: int,
        tags: list[str],
        embeddings: np.ndarray,
        existing_clusters: list[dict],
    ) -> bool:
        if not existing_clusters:
            return False

        noise_vec = embeddings[noise_idx].reshape(1, -1)
        best_sim = -1.0
        best_cluster_idx = -1

        for ci, cluster in enumerate(existing_clusters):
            centroid = cluster["centroid"].reshape(1, -1)
            sim = float(cosine_similarity(noise_vec, centroid)[0, 0])
            if sim > best_sim:
                best_sim = sim
                best_cluster_idx = ci

        if best_sim >= self.config.merge_threshold:
            cluster = existing_clusters[best_cluster_idx]
            cluster["members"].append(tags[noise_idx])
            cluster["members"].sort()
            cluster["size"] += 1
            cluster["_indices"].append(noise_idx)
            member_embeddings = embeddings[cluster["_indices"]]
            cluster["centroid"] = np.mean(member_embeddings, axis=0)
            cluster["representative"] = self._pick_representative(
                cluster["members"], member_embeddings, cluster["centroid"]
            )
            return True

        return False

    def _post_merge(
        self, clusters: list[dict], embeddings: np.ndarray, tags: list[str]
    ) -> list[dict]:
        if len(clusters) <= 1:
            return clusters

        merged = True
        while merged:
            merged = False
            centroids = np.array([c["centroid"] for c in clusters])
            sim_matrix = cosine_similarity(centroids)

            i, j = -1, -1
            best_score = -1.0
            n = len(clusters)

            for a in range(n):
                for b in range(a + 1, n):
                    if sim_matrix[a, b] > best_score:
                        best_score = sim_matrix[a, b]
                        i, j = a, b

            if best_score >= self.config.merge_threshold:
                merged_indices = clusters[i]["_indices"] + clusters[j]["_indices"]
                new_cluster = self._make_cluster(tags, embeddings, merged_indices)
                clusters = [c for k, c in enumerate(clusters) if k not in (i, j)]
                clusters.append(new_cluster)
                merged = True

        return clusters

    @staticmethod
    def strip_internals(clusters: list[dict]) -> list[dict]:
        return [
            {
                "representative": c["representative"],
                "members": c["members"],
                "size": c["size"],
            }
            for c in clusters
        ]
