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
    "spirituality",
]

SUBGENRE_MAP = {
    "grimdark": "fantasy",
    "portal fantasy": "fantasy",
    "isekai": "fantasy",
    "sword and sorcery": "fantasy",
    "litrpg": "fantasy",
    "gamelit": "fantasy",
    "wuxia": "fantasy",
    "xianxia": "fantasy",
    "progression fantasy": "fantasy",
    "gaslamp fantasy": "fantasy",
    "flintlock fantasy": "fantasy",
    "gunpowder fantasy": "fantasy",
    "mythic fantasy": "fantasy",
    "heroic fantasy": "fantasy",
    "science fantasy": "fantasy",
    "romantasy": "fantasy",

    "cyberpunk": "science fiction",
    "steampunk": "science fiction",
    "biopunk": "science fiction",
    "solarpunk": "science fiction",
    "nanopunk": "science fiction",
    "atompunk": "science fiction",
    "dieselpunk": "science fiction",
    "space opera": "science fiction",
    "mecha": "science fiction",
    "military scifi": "science fiction",
    "afrofuturism": "science fiction",
    "slipstream": "science fiction",
    "generation ship": "science fiction",
    "first contact": "science fiction",
    "cli fi": "science fiction",
    "climate fiction": "science fiction",

    "cozy mystery": "mystery",
    "hardboiled": "mystery",
    "police procedural": "mystery",
    "whodunit": "mystery",
    "howdunit": "mystery",
    "amateur sleuth": "mystery",
    "locked room mystery": "mystery",
    "inverted mystery": "mystery",
    "cat sleuth": "mystery",
    "noir": "mystery",
    "neo noir": "mystery",

    "splatterpunk": "horror",
    "lovecraftian": "horror",
    "cosmic horror": "horror",
    "slasher": "horror",
    "creature feature": "horror",
    "weird fiction": "horror",
    "bizarro fiction": "horror",
    "macabre": "horror",

    "heist": "thriller",
    "caper": "thriller",
    "espionage": "thriller",
    "spy fiction": "thriller",

    "archery": "sports",
    "fencing": "sports",
    "judo": "sports",
    "karate": "sports",
    "taekwondo": "sports",
    "boxing": "sports",
    "kickboxing": "sports",
    "wrestling": "sports",
    "swimming": "sports",
    "rugby": "sports",
    "cricket": "sports",
    "volleyball": "sports",
    "badminton": "sports",
    "gymnastics": "sports",
    "surfing": "sports",
    "snowboarding": "sports",
    "mountaineering": "sports",
    "marathon": "sports",
    "triathlon": "sports",
    "polo": "sports",
    "rodeo": "sports",
    "weightlifting": "sports",
    "fitness": "sports",
    "martial arts": "sports",
    "parkour": "sports",

    "mindfulness": "spirituality",
    "meditation": "spirituality",
    "yoga": "spirituality",
    "tarot": "spirituality",
    "new age": "spirituality",
    "mysticism": "spirituality",
    "esotericism": "spirituality",
    "astrology": "spirituality",
    "shamanism": "spirituality",
    "occult": "spirituality",

    "manga": "media",
    "manhwa": "media",
    "manhua": "media",
    "webcomic": "media",
    "graphic novel": "media",
    "comics": "media",
    "comic book": "media",
    "web novel": "media",
    "light novel": "media",
    "audiobooks": "media",

    "knitting": "crafts",
    "crochet": "crafts",
    "sewing": "crafts",
    "quilting": "crafts",
    "pottery": "crafts",
    "ceramics": "crafts",
    "origami": "crafts",
    "woodworking": "crafts",
    "metalworking": "crafts",
    "calligraphy": "crafts",

    "botany": "science",
    "zoology": "science",
    "paleontology": "science",
    "ecology": "science",
    "genetics": "science",
    "neuroscience": "science",
    "marine biology": "science",
    "ornithology": "science",

    "stoicism": "philosophy",
    "existentialism": "philosophy",
    "nihilism": "philosophy",
    "epistemology": "philosophy",
    "metaphysics": "philosophy",
    "absurdism": "philosophy",

    "samurai": "history",
    "viking": "history",
    "colonial": "history",
    "cold war": "history",
    "silk road": "history",

    "dogs": "animals",
    "cats": "animals",
    "horses": "animals",
    "birds": "animals",
    "wolves": "animals",
    "bears": "animals",
    "wildlife": "animals",
    "pets": "animals",
    "dinosaurs": "animals",
}


class TaxonomyMerger:
    def __init__(self, config: PipelineConfig, embedder):
        self.config = config
        self.embedder = embedder
        self._parent_embeddings = None
        self._parent_list = list(PARENT_GENRES)
        self.stats = {
            "clusters_before_taxonomy": 0,
            "clusters_after_taxonomy": 0,
            "clusters_absorbed": 0,
            "singletons_remaining": 0,
            "parent_genres_active": 0,
            "parent_genre_distribution": {},
        }

    def _get_parent_embeddings(self) -> np.ndarray:
        if self._parent_embeddings is None:
            self._parent_embeddings = self.embedder.encode(self._parent_list)
        return self._parent_embeddings

    def _assign_parent_explicit(self, cluster: dict) -> str | None:
        for member in cluster.get("members", []):
            mapped = SUBGENRE_MAP.get(member.lower())
            if mapped:
                return mapped

        rep = cluster.get("cluster_label", "").lower()
        mapped = SUBGENRE_MAP.get(rep)
        if mapped:
            return mapped

        return None

    def _assign_parent_keyword(self, cluster: dict) -> str | None:
        all_text = [cluster.get("cluster_label", "")] + cluster.get("members", [])

        for text in all_text:
            text_lower = text.lower()
            for parent in self._parent_list:
                if parent in text_lower and text_lower != parent:
                    return parent

        return None

    def _assign_parent_embedding(
        self, cluster: dict, tags: list[str], normed: np.ndarray, tag_to_idx: dict
    ) -> str | None:
        member_indices = []
        for m in cluster.get("members", []):
            idx = tag_to_idx.get(m)
            if idx is not None:
                member_indices.append(idx)

        if not member_indices:
            rep_idx = tag_to_idx.get(cluster.get("cluster_label", ""))
            if rep_idx is not None:
                member_indices = [rep_idx]

        if not member_indices:
            return None

        parent_embs = self._get_parent_embeddings()
        member_embs = normed[member_indices]
        sims = cosine_similarity(member_embs, parent_embs)
        avg_sims = np.mean(sims, axis=0)

        best_parent_idx = int(np.argmax(avg_sims))
        best_sim = float(avg_sims[best_parent_idx])

        if best_sim >= self.config.parent_genre_similarity_threshold:
            return self._parent_list[best_parent_idx]

        return None

    def _assign_parent(
        self, cluster: dict, tags: list[str], normed: np.ndarray, tag_to_idx: dict
    ) -> str | None:
        rep = cluster.get("cluster_label", "").lower()
        if rep in self._parent_list:
            return rep

        result = self._assign_parent_explicit(cluster)
        if result:
            return result

        result = self._assign_parent_keyword(cluster)
        if result:
            return result

        result = self._assign_parent_embedding(cluster, tags, normed, tag_to_idx)
        if result:
            return result

        return None

    def merge(
        self, classified_clusters: list[dict], tags: list[str], embeddings: np.ndarray
    ) -> list[dict]:
        if not self.config.enable_hierarchical_taxonomy_merge:
            return classified_clusters

        self.stats["clusters_before_taxonomy"] = len(classified_clusters)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = embeddings / norms

        tag_to_idx = {t: i for i, t in enumerate(tags)}
        max_absorb = self.config.taxonomy_max_cluster_size_to_absorb

        parent_clusters: dict[str, dict] = {}
        unmatched = []

        for cluster in classified_clusters:
            parent_name = self._assign_parent(cluster, tags, normed, tag_to_idx)

            if parent_name is None:
                unmatched.append(cluster)
                continue

            if parent_name not in parent_clusters:
                parent_clusters[parent_name] = self._init_parent(parent_name, cluster)
            else:
                cluster_size = len(cluster.get("members", []))
                rep = cluster.get("cluster_label", "").lower()

                if rep == parent_name:
                    self._absorb_into_parent(parent_clusters[parent_name], cluster)
                elif cluster_size <= max_absorb:
                    self._absorb_into_parent(parent_clusters[parent_name], cluster)
                    self.stats["clusters_absorbed"] += 1
                else:
                    self._absorb_into_parent(parent_clusters[parent_name], cluster)

        result = list(parent_clusters.values()) + unmatched
        result.sort(key=lambda c: len(c.get("members", [])), reverse=True)

        self.stats["parent_genres_active"] = len(parent_clusters)
        self.stats["clusters_after_taxonomy"] = len(result)
        self.stats["singletons_remaining"] = sum(
            1 for c in result if len(c.get("members", [])) == 1
        )

        distribution = {}
        for name, pc in parent_clusters.items():
            distribution[name] = len(pc.get("members", []))
        self.stats["parent_genre_distribution"] = dict(
            sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        )

        return result

    def build_hierarchy(
        self, classified_clusters: list[dict], tags: list[str], embeddings: np.ndarray
    ) -> dict:
        if not self.config.enable_hierarchical_taxonomy_merge:
            return {}

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = embeddings / norms

        tag_to_idx = {t: i for i, t in enumerate(tags)}

        hierarchy: dict[str, list[str]] = {}

        for cluster in classified_clusters:
            rep = cluster.get("cluster_label", "")
            parent_name = self._assign_parent(cluster, tags, normed, tag_to_idx)

            if parent_name is None:
                hierarchy.setdefault("other", []).append(rep)
            elif rep.lower() == parent_name:
                hierarchy.setdefault(parent_name, [])
            else:
                hierarchy.setdefault(parent_name, []).append(rep)

        hierarchy = {k: v for k, v in hierarchy.items() if k or v}
        return dict(sorted(hierarchy.items(), key=lambda x: len(x[1]), reverse=True))

    def _init_parent(self, parent_name: str, cluster: dict) -> dict:
        return {
            "cluster_label": parent_name,
            "members": list(cluster.get("members", [])),
            "category": cluster.get("category", "genre"),
            "confidence": cluster.get("confidence", 0.5),
            "method": "taxonomy",
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

        child_label = child.get("cluster_label", "")
        if child_label and child_label not in parent["subclusters"]:
            parent["subclusters"].append(child_label)

        total = len(parent["members"])
        prev_count = total - len(child_members)
        prev_weight = prev_count / total if total > 0 else 0.5
        parent["confidence"] = round(
            parent["confidence"] * prev_weight + child.get("confidence", 0.5) * (1 - prev_weight), 4
        )
