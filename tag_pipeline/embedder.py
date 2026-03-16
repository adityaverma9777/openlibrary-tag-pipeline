from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from tag_pipeline.config import PipelineConfig


class TagEmbedder:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._model: SentenceTransformer | None = None
        self._cache_dir = Path(self.config.cache_dir)
        self._cache_path = self._cache_dir / "embeddings_cache.npz"
        self._cached_tags: list[str] = []
        self._cached_embs: np.ndarray | None = None
        self._tag_to_idx: dict[str, int] = {}

        self.cache_stats = {
            "total_processed": 0,
            "cache_hits": 0,
            "new_computed": 0,
            "hit_rate": 0.0,
        }

        self._load_cache()

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(
                self.config.embedding_model,
                device=self.config.device,
            )
        return self._model

    def _load_cache(self):
        if not self._cache_path.exists():
            return
        try:
            data = np.load(self._cache_path, allow_pickle=True)
            self._cached_tags = data["tags"].tolist()
            self._cached_embs = data["embeddings"]
            self._tag_to_idx = {t: i for i, t in enumerate(self._cached_tags)}
        except Exception:
            self._cached_tags = []
            self._cached_embs = None
            self._tag_to_idx = {}

    def _save_cache(self):
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        if self._cached_embs is not None and len(self._cached_tags) > 0:
            np.savez_compressed(
                self._cache_path,
                tags=np.array(self._cached_tags, dtype=object),
                embeddings=self._cached_embs,
            )

    def encode(self, tags: list[str]) -> np.ndarray:
        if not tags:
            return np.zeros((0, 384), dtype=np.float32)

        self.cache_stats["total_processed"] += len(tags)

        dim = self._cached_embs.shape[1] if self._cached_embs is not None else 384
        result = np.zeros((len(tags), dim), dtype=np.float32)

        missing_tags = []
        missing_positions = []

        for i, tag in enumerate(tags):
            if tag in self._tag_to_idx:
                self.cache_stats["cache_hits"] += 1
                result[i] = self._cached_embs[self._tag_to_idx[tag]]
            else:
                missing_tags.append(tag)
                missing_positions.append(i)

        if missing_tags:
            self.cache_stats["new_computed"] += len(missing_tags)

            new_embs = self.model.encode(
                missing_tags,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            new_embs = np.array(new_embs, dtype=np.float32)

            for idx, pos in enumerate(missing_positions):
                result[pos] = new_embs[idx]

            offset = len(self._cached_tags)
            self._cached_tags.extend(missing_tags)
            for idx, tag in enumerate(missing_tags):
                self._tag_to_idx[tag] = offset + idx

            if self._cached_embs is None:
                self._cached_embs = new_embs
            else:
                self._cached_embs = np.vstack([self._cached_embs, new_embs])

            self._save_cache()

        total = self.cache_stats["total_processed"]
        self.cache_stats["hit_rate"] = (
            self.cache_stats["cache_hits"] / total * 100
        ) if total > 0 else 0.0

        return result

    def similarity_pairs_faiss(
        self, embeddings: np.ndarray, tags: list[str], top_k: int = 10, threshold: float = 0.4
    ) -> list[tuple[str, str, float]]:
        try:
            import faiss
        except ImportError:
            return self.pairwise_similarity(embeddings, tags, threshold=threshold)

        n, dim = embeddings.shape
        k = min(top_k + 1, n)

        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 100
        index.add(embeddings.astype(np.float32))

        distances, indices = index.search(embeddings.astype(np.float32), k)

        seen = set()
        pairs = []

        for i in range(n):
            for j_pos in range(k):
                j = int(indices[i, j_pos])
                if j == i or j < 0:
                    continue

                pair_key = (min(i, j), max(i, j))
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim >= threshold:
                    pairs.append((tags[i], tags[j], sim))

        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    def similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        return np.dot(embeddings, embeddings.T)

    def pairwise_similarity(
        self, embeddings: np.ndarray, tags: list[str], threshold: float = 0.5
    ) -> list[tuple[str, str, float]]:
        sim_matrix = self.similarity_matrix(embeddings)
        n = len(tags)
        pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                score = float(sim_matrix[i, j])
                if score >= threshold:
                    pairs.append((tags[i], tags[j], score))

        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs
