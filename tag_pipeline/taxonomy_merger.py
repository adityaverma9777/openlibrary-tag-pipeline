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

FICTION_PARENTS = {
    "fantasy",
    "science fiction",
    "horror",
    "romance",
    "mystery",
    "thriller",
    "crime",
    "humor",
    "drama",
    "poetry",
    "adventure",
    "western",
}

PARENT_DOMAIN_MAP = {
    "animals": "animals",
    "nature": "nature",
    "sports": "sports",
    "religion": "religion",
    "spirituality": "religion",
    "politics": "politics",
    "history": "history",
    "science": "science",
    "technology": "science",
    "medicine": "science",
    "engineering": "science",
    "mathematics": "science",
    "education": "abstract",
    "psychology": "abstract",
    "philosophy": "abstract",
    "sociology": "abstract",
    "economics": "abstract",
    "law": "abstract",
    "business": "abstract",
    "children": "audience",
    "young adult": "audience",
}

DOMAIN_KEYWORDS = {
    "animals": [
        "animal", "wildlife", "zoology", "dog", "cat", "bird", "wolf",
        "bear", "horse", "pet", "dinosaurs", "marine biology",
    ],
    "sports": [
        "sport", "football", "basketball", "baseball", "hockey", "tennis",
        "rugby", "cricket", "swimming", "gym", "fitness", "boxing",
        "martial", "athlete", "competition",
    ],
    "religion": [
        "religion", "spiritual", "faith", "church", "mosque", "temple",
        "theology", "scripture", "sacred", "ritual", "god", "divine",
    ],
    "politics": [
        "politic", "election", "government", "policy", "state", "democracy",
        "diplomacy", "geopolitic", "ideology",
    ],
    "history": [
        "history", "historical", "archaeolog", "ancient", "civilization",
        "artifact", "medieval", "bronze age", "iron age",
    ],
    "science": [
        "science", "technology", "biology", "physics", "chemistry", "genetics",
        "engineering", "medicine", "ai", "machine learning",
    ],
    "abstract": [
        "identity", "freedom", "justice", "morality", "ethics", "power",
        "love", "existential", "human nature", "philosophy",
    ],
}

INCOMPATIBLE_DOMAINS = {
    "animals": {"sports", "politics", "religion", "abstract"},
    "sports": {"animals", "politics", "religion", "history", "abstract"},
    "religion": {"sports", "animals", "politics", "abstract"},
    "politics": {"animals", "sports", "religion"},
}

LITERARY_CROSS_DOMAIN_ALLOWED = {
    ("science", "fiction"),
    ("fiction", "science"),
    ("politics", "history"),
    ("history", "politics"),
    ("religion", "abstract"),
    ("religion", "history"),
    ("animals", "nature"),
    ("nature", "animals"),
}

LITERARY_WEAK_DOMAIN_ALLOWED = {
    ("science", "history"),
    ("history", "science"),
    ("fiction", "history"),
    ("history", "fiction"),
    ("religion", "politics"),
    ("politics", "religion"),
}

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

CLEAR_GENRE_PARENT_RULES = {
    "romance": ["romance", "romantic", "rom-com", "romcom"],
    "fantasy": ["fantasy", "grimdark", "magic", "wizard", "witch"],
    "thriller": ["thriller", "suspense", "techno-thriller", "psychological thriller"],
    "mystery": ["mystery", "whodunit", "detective", "crime fiction"],
    "horror": ["horror", "slasher", "gothic", "haunting", "macabre"],
    "science fiction": ["science fiction", "scifi", "sci fi", "sci-fi", "cyberpunk", "space opera"],
}


class TaxonomyMerger:
    def __init__(self, config: PipelineConfig, embedder):
        self.config = config
        self.embedder = embedder
        self._parent_embeddings = None
        self._parent_list = list(PARENT_GENRES)
        self._parent_aliases = self._build_parent_aliases()
        self.stats = {
            "clusters_before_taxonomy": 0,
            "clusters_after_taxonomy": 0,
            "clusters_absorbed": 0,
            "singletons_remaining": 0,
            "parent_genres_active": 0,
            "parent_genre_distribution": {},
            "rejected_low_confidence": 0,
            "rejected_low_consistency": 0,
            "rejected_domain_mismatch": 0,
            "allowed_despite_domain_mismatch": 0,
            "allowed_with_domain_score_below_one": 0,
            "fallback_small_clusters_absorbed": 0,
        }

    def _build_parent_aliases(self) -> dict[str, set[str]]:
        aliases = {p: {p} for p in self._parent_list}
        for child, parent in SUBGENRE_MAP.items():
            if parent in aliases:
                aliases[parent].add(child)

        aliases.setdefault("science fiction", {"science fiction"}).update(
            {
                "scifi", "sci fi", "sci-fi", "sf", "cyberpunk", "steampunk",
                "space opera", "robotics", "artificial intelligence", "ai",
            }
        )
        aliases.setdefault("romance", {"romance"}).update({"romantic", "rom-com", "romcom"})
        aliases.setdefault("thriller", {"thriller"}).update({"suspense", "techno-thriller"})
        return aliases

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

    def _get_member_indices(self, cluster: dict, tag_to_idx: dict) -> list[int]:
        member_indices = []
        for m in cluster.get("members", []):
            idx = tag_to_idx.get(m)
            if idx is not None:
                member_indices.append(idx)

        if member_indices:
            return member_indices

        rep_idx = tag_to_idx.get(cluster.get("cluster_label", ""))
        if rep_idx is not None:
            return [rep_idx]
        return []

    def _confidence_thresholds(self) -> tuple[float, float, float]:
        similarity_threshold = self.config.parent_genre_similarity_threshold
        margin_threshold = self.config.parent_assignment_margin_threshold
        consistency_threshold = self.config.parent_semantic_consistency_threshold

        if self.config.taxonomy_strict_mode:
            similarity_threshold = max(
                similarity_threshold,
                self.config.strict_parent_genre_similarity_threshold,
            )
            margin_threshold = max(
                margin_threshold,
                self.config.strict_parent_assignment_margin_threshold,
            )
            consistency_threshold = max(
                consistency_threshold,
                self.config.strict_parent_semantic_consistency_threshold,
            )

        return similarity_threshold, margin_threshold, consistency_threshold

    def _extract_cluster_domains(self, cluster: dict) -> set[str]:
        detected = set()
        category = str(cluster.get("category", "")).lower()

        if category == "genre":
            detected.add("fiction")
        elif category == "audience":
            detected.add("audience")

        text = " ".join([cluster.get("cluster_label", "")] + cluster.get("members", [])).lower()
        for domain, keywords in DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    detected.add(domain)
                    break

        return detected

    def _canonical_parent_domain(self, parent_name: str) -> str:
        if parent_name in FICTION_PARENTS or parent_name == "science fiction":
            return "fiction"
        return PARENT_DOMAIN_MAP.get(parent_name, "other")

    def _domain_compatibility_score(self, cluster: dict, parent_name: str) -> float:
        parent_domain = self._canonical_parent_domain(parent_name)
        cluster_domains = self._extract_cluster_domains(cluster)
        if parent_domain is None or not cluster_domains:
            return 0.4

        if parent_domain in cluster_domains:
            return 1.0

        for d in cluster_domains:
            if (parent_domain, d) in LITERARY_CROSS_DOMAIN_ALLOWED:
                return 0.75
            if (parent_domain, d) in LITERARY_WEAK_DOMAIN_ALLOWED:
                return 0.4

        incompatible = INCOMPATIBLE_DOMAINS.get(parent_domain, set())
        if cluster_domains.intersection(incompatible):
            return 0.2

        strong_domains = {"animals", "sports", "religion", "politics", "history", "science"}
        strong_hits = cluster_domains.intersection(strong_domains)
        if strong_hits and parent_domain in strong_domains and parent_domain not in strong_hits:
            return 0.2

        return 0.2

    def _best_and_second_score(self, scores: np.ndarray) -> tuple[int, float, float]:
        if scores.size == 0:
            return -1, 0.0, 0.0

        order = np.argsort(scores)
        best_idx = int(order[-1])
        best_score = float(scores[best_idx])
        second_score = float(scores[order[-2]]) if scores.size > 1 else 0.0
        return best_idx, best_score, second_score

    def _keyword_score(self, cluster: dict, parent_name: str) -> float:
        aliases = self._parent_aliases.get(parent_name, {parent_name})
        texts = [cluster.get("cluster_label", "")] + cluster.get("members", [])
        if not texts:
            return 0.0

        score = 0.0
        max_local_score = 0.0
        for text in texts:
            text_lower = text.lower()
            local = 0.0
            for alias in aliases:
                if text_lower == alias:
                    local = max(local, 1.0)
                elif alias in text_lower:
                    local = max(local, 0.85)
                elif text_lower in alias and len(text_lower) >= 4:
                    local = max(local, 0.65)
            score += local
            max_local_score = max(max_local_score, local)

        avg_score = score / len(texts)
        blended = 0.6 * avg_score + 0.4 * max_local_score
        return float(min(1.0, blended))

    def _detect_clear_genre_parent(self, cluster: dict) -> str | None:
        texts = [cluster.get("cluster_label", "")] + cluster.get("members", [])
        texts = [t.lower() for t in texts if t]
        if not texts:
            return None

        for parent, keywords in CLEAR_GENRE_PARENT_RULES.items():
            if all(any(keyword in text for keyword in keywords) for text in texts):
                return parent

        return None

    def _strong_member_match(
        self, member_embs: np.ndarray, parent_embs: np.ndarray
    ) -> tuple[int, float]:
        member_scores = cosine_similarity(member_embs, parent_embs)
        max_by_parent = np.max(member_scores, axis=0)
        best_parent_idx = int(np.argmax(max_by_parent))
        return best_parent_idx, float(max_by_parent[best_parent_idx])

    def _score_parent_candidates(
        self,
        cluster: dict,
        member_embs: np.ndarray,
        rep_emb: np.ndarray,
        parent_embs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rep_scores = cosine_similarity(rep_emb.reshape(1, -1), parent_embs).flatten()
        member_avg_scores = np.mean(cosine_similarity(member_embs, parent_embs), axis=0)
        embedding_scores = (0.5 * rep_scores) + (0.5 * member_avg_scores)

        keyword_scores = np.array([
            self._keyword_score(cluster, parent_name) for parent_name in self._parent_list
        ], dtype=np.float32)

        domain_scores = np.array([
            self._domain_compatibility_score(cluster, parent_name) for parent_name in self._parent_list
        ], dtype=np.float32)

        hybrid_scores = (
            self.config.parent_embedding_weight * embedding_scores
            + self.config.parent_keyword_weight * keyword_scores
            + self.config.parent_domain_weight * domain_scores
        )

        domain_penalty_mask = domain_scores < self.config.parent_domain_soft_mismatch_threshold
        if np.any(domain_penalty_mask):
            hybrid_scores = hybrid_scores + (
                domain_penalty_mask.astype(np.float32) * self.config.parent_domain_mismatch_penalty
            )

        return embedding_scores, keyword_scores, domain_scores, hybrid_scores

    def _evaluate_assignment(
        self,
        cluster: dict,
        member_embs: np.ndarray,
        rep_emb: np.ndarray,
        parent_embs: np.ndarray,
    ) -> tuple[str | None, str | None, dict]:
        clear_parent = self._detect_clear_genre_parent(cluster)
        if clear_parent:
            domain_score = self._domain_compatibility_score(cluster, clear_parent)
            return clear_parent, None, {
                "domain_soft_mismatch": domain_score < 1.0,
                "domain_score": domain_score,
            }

        similarity_threshold, margin_threshold, consistency_threshold = self._confidence_thresholds()

        embedding_scores, keyword_scores, domain_scores, hybrid_scores = self._score_parent_candidates(
            cluster,
            member_embs,
            rep_emb,
            parent_embs,
        )

        best_idx, best_score, second_score = self._best_and_second_score(hybrid_scores)
        if best_idx < 0:
            return None, "low_confidence", {}

        margin = best_score - second_score
        parent_name = self._parent_list[best_idx]

        parent_rep_sim = float(embedding_scores[best_idx])
        strong_keyword = float(keyword_scores[best_idx]) >= self.config.parent_keyword_strong_match_threshold
        domain_score = float(domain_scores[best_idx])

        strong_parent_idx, strong_member_sim = self._strong_member_match(member_embs, parent_embs)
        strong_member_for_best = (
            strong_parent_idx == best_idx
            and strong_member_sim >= self.config.parent_member_strong_match_threshold
        )

        very_low_confidence_triplet = (
            parent_rep_sim < self.config.parent_very_low_similarity_threshold
            and float(keyword_scores[best_idx]) < self.config.parent_no_keyword_threshold
            and domain_score < self.config.parent_extremely_low_domain_threshold
        )
        if very_low_confidence_triplet:
            return None, "low_confidence", {}

        final_threshold = self.config.global_merge_threshold
        if best_score < final_threshold:
            return None, "low_confidence", {}

        member_scores = cosine_similarity(member_embs, parent_embs[best_idx].reshape(1, -1)).flatten()
        avg_member_score = float(np.mean(member_scores))
        required_consistency = max(
            self.config.parent_min_relaxed_consistency_threshold,
            consistency_threshold - self.config.parent_consistency_relax_delta,
        ) if (strong_keyword or strong_member_for_best) else consistency_threshold

        if avg_member_score < required_consistency and best_score < (final_threshold + 0.10):
            return None, "low_consistency", {}

        return parent_name, None, {
            "domain_soft_mismatch": domain_score < 1.0,
            "domain_score": domain_score,
            "final_score": best_score,
        }

    def _fallback_absorb_small_clusters(
        self,
        unmatched: list[dict],
        parent_clusters: dict[str, dict],
        normed: np.ndarray,
        tag_to_idx: dict[str, int],
        parent_embs: np.ndarray,
    ) -> list[dict]:
        if not unmatched:
            return []

        remaining = []
        for cluster in unmatched:
            members = cluster.get("members", [])
            cluster_size = len(members)
            if cluster_size == 0 or cluster_size > self.config.singleton_fallback_max_cluster_size:
                remaining.append(cluster)
                continue

            member_indices = self._get_member_indices(cluster, tag_to_idx)
            if not member_indices:
                remaining.append(cluster)
                continue

            member_embs = normed[member_indices]
            rep_label = cluster.get("cluster_label", "")
            rep_idx = tag_to_idx.get(rep_label)
            rep_emb = normed[rep_idx] if rep_idx is not None else np.mean(member_embs, axis=0)

            embedding_scores, keyword_scores, domain_scores, hybrid_scores = self._score_parent_candidates(
                cluster,
                member_embs,
                rep_emb,
                parent_embs,
            )

            best_idx, best_score, _ = self._best_and_second_score(hybrid_scores)
            if best_idx < 0:
                remaining.append(cluster)
                continue

            best_parent = self._parent_list[best_idx]
            best_embedding = float(embedding_scores[best_idx])
            best_keyword = float(keyword_scores[best_idx])
            best_domain = float(domain_scores[best_idx])

            should_absorb = (
                best_score >= self.config.singleton_fallback_merge_threshold
                and best_domain >= self.config.singleton_fallback_min_domain_score
                and (
                    best_keyword >= self.config.singleton_fallback_keyword_threshold
                    or best_embedding >= self.config.singleton_fallback_min_embedding_score
                )
            )

            if not should_absorb:
                remaining.append(cluster)
                continue

            if best_parent not in parent_clusters:
                parent_clusters[best_parent] = self._init_parent(best_parent, cluster)
            else:
                self._absorb_into_parent(parent_clusters[best_parent], cluster)

            self.stats["clusters_absorbed"] += 1
            self.stats["fallback_small_clusters_absorbed"] += 1

            if best_domain < 1.0:
                self.stats["allowed_despite_domain_mismatch"] += 1
                self.stats["allowed_with_domain_score_below_one"] += 1

        return remaining

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

        self.stats["clusters_absorbed"] = 0
        self.stats["rejected_low_confidence"] = 0
        self.stats["rejected_low_consistency"] = 0
        self.stats["rejected_domain_mismatch"] = 0
        self.stats["allowed_despite_domain_mismatch"] = 0
        self.stats["allowed_with_domain_score_below_one"] = 0
        self.stats["fallback_small_clusters_absorbed"] = 0

        self.stats["clusters_before_taxonomy"] = len(classified_clusters)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = embeddings / norms

        tag_to_idx = {t: i for i, t in enumerate(tags)}
        max_absorb = self.config.taxonomy_max_cluster_size_to_absorb
        parent_embs = self._get_parent_embeddings()

        parent_clusters: dict[str, dict] = {}
        unmatched = []

        for cluster in classified_clusters:
            member_indices = self._get_member_indices(cluster, tag_to_idx)
            if not member_indices:
                unmatched.append(cluster)
                continue

            member_embs = normed[member_indices]
            rep_label = cluster.get("cluster_label", "")
            rep_idx = tag_to_idx.get(rep_label)
            rep_emb = normed[rep_idx] if rep_idx is not None else np.mean(member_embs, axis=0)

            parent_name, reject_reason, details = self._evaluate_assignment(
                cluster,
                member_embs,
                rep_emb,
                parent_embs,
            )

            if parent_name is None:
                if reject_reason == "low_confidence":
                    self.stats["rejected_low_confidence"] += 1
                elif reject_reason == "low_consistency":
                    self.stats["rejected_low_consistency"] += 1
                elif reject_reason == "domain_mismatch":
                    self.stats["rejected_domain_mismatch"] += 1
                unmatched.append(cluster)
                continue

            if details.get("domain_soft_mismatch"):
                self.stats["allowed_despite_domain_mismatch"] += 1
                self.stats["allowed_with_domain_score_below_one"] += 1

            if parent_name not in parent_clusters:
                parent_clusters[parent_name] = self._init_parent(parent_name, cluster)
            else:
                cluster_size = len(cluster.get("members", []))
                rep = cluster.get("cluster_label", "").lower()
                final_score = float(details.get("final_score", 0.0))

                if rep == parent_name:
                    self._absorb_into_parent(parent_clusters[parent_name], cluster)
                elif cluster_size <= max_absorb:
                    self._absorb_into_parent(parent_clusters[parent_name], cluster)
                    self.stats["clusters_absorbed"] += 1
                elif final_score >= self.config.parent_large_cluster_merge_threshold:
                    self._absorb_into_parent(parent_clusters[parent_name], cluster)
                    self.stats["clusters_absorbed"] += 1
                else:
                    unmatched.append(cluster)

        unmatched = self._fallback_absorb_small_clusters(
            unmatched,
            parent_clusters,
            normed,
            tag_to_idx,
            parent_embs,
        )

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
        parent_embs = self._get_parent_embeddings()

        hierarchy: dict[str, list[str]] = {}

        for cluster in classified_clusters:
            rep = cluster.get("cluster_label", "")
            member_indices = self._get_member_indices(cluster, tag_to_idx)
            if not member_indices:
                hierarchy.setdefault("other", []).append(rep)
                continue

            member_embs = normed[member_indices]
            rep_idx = tag_to_idx.get(rep)
            rep_emb = normed[rep_idx] if rep_idx is not None else np.mean(member_embs, axis=0)

            parent_name, _, _ = self._evaluate_assignment(
                cluster,
                member_embs,
                rep_emb,
                parent_embs,
            )

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
