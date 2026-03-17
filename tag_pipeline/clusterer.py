import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from tag_pipeline.config import PipelineConfig


class TagClusterer:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.stats = {
            "clusters_before_merge": 0,
            "clusters_after_merge": 0,
        }

    def cluster(self, tags: list[str], embeddings: np.ndarray) -> list[dict]:
        if len(tags) == 0:
            return []
        if len(tags) == 1:
            return [self._make_cluster(tags, embeddings, [0])]

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = embeddings / norms

        labels = self._cluster_labels(normed)

        label_to_indices: dict[int, list[int]] = {}
        noise_label_counter = max(int(np.max(labels)) + 1, 0) if len(labels) else 0
        for idx, label in enumerate(labels):
            # Keep unresolved noise as singletons; do not collapse all -1 items together.
            if label == -1:
                label = noise_label_counter
                noise_label_counter += 1
            label_to_indices.setdefault(label, []).append(idx)

        clusters = []
        for indices in label_to_indices.values():
            clusters.append(self._make_cluster(tags, normed, indices))

        self.stats["clusters_before_merge"] = len(clusters)

        clusters = self._centroid_merge(clusters, normed, tags)

        clusters.sort(key=lambda c: c["size"], reverse=True)
        self.stats["clusters_after_merge"] = len(clusters)

        return clusters

    def _cluster_labels(self, normed: np.ndarray) -> np.ndarray:
        if self.config.use_hdbscan:
            try:
                import hdbscan

                hdb = hdbscan.HDBSCAN(
                    min_cluster_size=max(2, self.config.hdbscan_min_cluster_size),
                    min_samples=max(1, self.config.hdbscan_min_samples),
                    metric="euclidean",
                    cluster_selection_method="eom",
                )
                labels = hdb.fit_predict(normed)

                if self.config.hdbscan_reassign_noise:
                    labels = self._reassign_noise_points(
                        normed,
                        labels,
                        threshold=self.config.hdbscan_noise_reassign_similarity,
                    )

                # If HDBSCAN degenerates into all noise, fall back gracefully.
                if np.all(labels == -1):
                    return self._cluster_labels_agglomerative(normed)

                return labels
            except Exception:
                return self._cluster_labels_agglomerative(normed)

        return self._cluster_labels_agglomerative(normed)

    def _cluster_labels_agglomerative(self, normed: np.ndarray) -> np.ndarray:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=self.config.clustering_distance_threshold,
        )
        return clustering.fit_predict(normed)

    def _reassign_noise_points(
        self, normed: np.ndarray, labels: np.ndarray, threshold: float
    ) -> np.ndarray:
        if len(labels) == 0:
            return labels

        unique = [int(l) for l in np.unique(labels) if int(l) >= 0]
        if not unique:
            return labels

        centroids = []
        for label in unique:
            idxs = np.where(labels == label)[0]
            centroid = np.mean(normed[idxs], axis=0)
            c_norm = np.linalg.norm(centroid)
            if c_norm > 0:
                centroid = centroid / c_norm
            centroids.append(centroid)

        centroids = np.array(centroids)
        noise_idxs = np.where(labels == -1)[0]

        for idx in noise_idxs:
            sims = np.dot(centroids, normed[idx])
            best_pos = int(np.argmax(sims))
            best_sim = float(sims[best_pos])
            if best_sim >= threshold:
                labels[idx] = unique[best_pos]

        return labels

    def category_merge(self, classified_clusters: list[dict], tags: list[str], embeddings: np.ndarray) -> list[dict]:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = embeddings / norms

        tag_to_idx = {t: i for i, t in enumerate(tags)}

        merged = True
        while merged:
            merged = False
            n = len(classified_clusters)
            if n <= 1:
                break

            best_i, best_j = -1, -1
            best_sim = -1.0

            for a in range(n):
                for b in range(a + 1, n):
                    if classified_clusters[a]["category"] != classified_clusters[b]["category"]:
                        continue

                    rep_a = classified_clusters[a]["cluster_label"]
                    rep_b = classified_clusters[b]["cluster_label"]

                    idx_a = tag_to_idx.get(rep_a)
                    idx_b = tag_to_idx.get(rep_b)
                    if idx_a is None or idx_b is None:
                        continue

                    sim = float(np.dot(normed[idx_a], normed[idx_b]))
                    if sim > best_sim:
                        best_sim = sim
                        best_i, best_j = a, b

            if best_sim >= self.config.category_merge_threshold and best_i >= 0:
                merged_item = self._merge_classified(
                    classified_clusters[best_i],
                    classified_clusters[best_j],
                    normed, tag_to_idx,
                )
                classified_clusters = [
                    c for k, c in enumerate(classified_clusters) if k not in (best_i, best_j)
                ]
                classified_clusters.append(merged_item)
                merged = True

        classified_clusters.sort(key=lambda c: len(c["members"]), reverse=True)
        return classified_clusters

    def _merge_classified(self, a: dict, b: dict, normed: np.ndarray, tag_to_idx: dict) -> dict:
        all_members = sorted(set(a["members"] + b["members"]))

        member_indices = [tag_to_idx[m] for m in all_members if m in tag_to_idx]
        if member_indices:
            member_embs = normed[member_indices]
            centroid = np.mean(member_embs, axis=0)
            sims = cosine_similarity(member_embs, centroid.reshape(1, -1)).flatten()
            best_idx = int(np.argmax(sims))
            representative = all_members[best_idx]
        else:
            representative = a["cluster_label"]

        avg_conf = (a["confidence"] * len(a["members"]) + b["confidence"] * len(b["members"])) / len(all_members)

        return {
            "cluster_label": representative,
            "members": all_members,
            "category": a["category"],
            "confidence": round(avg_conf, 4),
            "method": a["method"],
        }

    def _make_cluster(
        self, tags: list[str], embeddings: np.ndarray, indices: list[int]
    ) -> dict:
        members = [tags[i] for i in indices]
        member_embs = embeddings[indices]
        centroid = np.mean(member_embs, axis=0)
        representative = self._pick_representative(members, member_embs, centroid)

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

    def _centroid_merge(
        self, clusters: list[dict], embeddings: np.ndarray, tags: list[str]
    ) -> list[dict]:
        if len(clusters) <= 1:
            return clusters

        merged = True
        while merged:
            merged = False
            centroids = np.array([c["centroid"] for c in clusters])
            sim_matrix = cosine_similarity(centroids)

            best_i, best_j = -1, -1
            best_score = -1.0
            n = len(clusters)

            for a in range(n):
                for b in range(a + 1, n):
                    if sim_matrix[a, b] > best_score:
                        best_score = sim_matrix[a, b]
                        best_i, best_j = a, b

            if best_score >= self.config.cluster_merge_threshold:
                merged_indices = clusters[best_i]["_indices"] + clusters[best_j]["_indices"]
                new_cluster = self._make_cluster(tags, embeddings, merged_indices)
                clusters = [c for k, c in enumerate(clusters) if k not in (best_i, best_j)]
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
