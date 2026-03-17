from dataclasses import dataclass, field

import torch


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class PipelineConfig:
    device: str = field(default_factory=detect_device)
    batch_size: int = 64
    num_workers: int = -1
    cache_dir: str = "cache"
    random_seed: int = 42

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 64

    clustering_distance_threshold: float = 0.38
    cluster_merge_threshold: float = 0.62
    category_merge_threshold: float = 0.52
    min_cluster_size: int = 1
    use_hdbscan: bool = True
    hdbscan_min_cluster_size: int = 2
    hdbscan_min_samples: int = 1
    hdbscan_reassign_noise: bool = True
    hdbscan_noise_reassign_similarity: float = 0.55

    enable_hierarchical_taxonomy_merge: bool = True
    taxonomy_strict_mode: bool = True
    parent_genre_similarity_threshold: float = 0.40
    parent_assignment_margin_threshold: float = 0.05
    parent_semantic_consistency_threshold: float = 0.33
    strict_parent_genre_similarity_threshold: float = 0.43
    strict_parent_assignment_margin_threshold: float = 0.05
    strict_parent_semantic_consistency_threshold: float = 0.34
    parent_embedding_weight: float = 0.50
    parent_keyword_weight: float = 0.30
    parent_domain_weight: float = 0.20
    global_merge_threshold: float = 0.40
    parent_domain_soft_mismatch_threshold: float = 0.30
    parent_domain_mismatch_penalty: float = -0.10
    parent_very_low_similarity_threshold: float = 0.22
    parent_no_keyword_threshold: float = 0.15
    parent_extremely_low_domain_threshold: float = 0.12
    parent_keyword_strong_match_threshold: float = 0.70
    parent_member_strong_match_threshold: float = 0.58
    parent_fallback_similarity_delta: float = 0.04
    parent_fallback_margin_delta: float = 0.03
    strict_fallback_similarity_delta: float = 0.02
    strict_fallback_margin_delta: float = 0.02
    parent_consistency_relax_delta: float = 0.10
    parent_min_relaxed_consistency_threshold: float = 0.20
    taxonomy_max_cluster_size_to_absorb: int = 12
    parent_large_cluster_merge_threshold: float = 0.52
    singleton_fallback_max_cluster_size: int = 3
    singleton_fallback_merge_threshold: float = 0.36
    singleton_fallback_keyword_threshold: float = 0.20
    singleton_fallback_min_domain_score: float = 0.40
    singleton_fallback_min_embedding_score: float = 0.38

    faiss_top_k: int = 10
    faiss_enabled: bool = True

    nli_model: str = "facebook/bart-large-mnli"
    nli_confidence_threshold: float = 0.45

    category_keywords: dict[str, list[str]] = field(default_factory=lambda: {
        "genre": [
            "fiction", "nonfiction", "mystery", "thriller", "romance", "fantasy",
            "horror", "comedy", "drama", "satire", "poetry", "memoir", "biography",
            "autobiography", "essay", "science fiction", "historical fiction",
            "detective", "crime", "western", "adventure", "fable", "mythology",
            "dystopian fiction", "cyberpunk", "steampunk", "space opera",
            "magical realism", "paranormal", "supernatural", "suspense",
            "graphic novel", "manga", "noir",
        ],
        "theme": [
            "artificial intelligence", "machine learning", "technology",
            "war", "peace", "love", "death", "identity", "freedom", "justice",
            "power", "corruption", "survival", "redemption", "betrayal",
            "family", "friendship", "loneliness", "isolation", "rebellion",
            "philosophy", "existentialism", "consciousness", "human nature",
            "morality", "ethics", "coming of age", "loss", "grief", "hope",
            "psychology", "self help",
        ],
        "setting": [
            "space", "outer space", "ocean", "sea", "island", "desert",
            "jungle", "forest", "mountain", "arctic", "tropical",
            "city", "urban", "rural", "medieval", "ancient", "victorian",
            "futuristic", "post apocalyptic", "dystopian",
            "mars", "moon", "galaxy", "underwater",
        ],
        "mood": [
            "dark", "gothic", "hopeful", "melancholy", "nostalgic",
            "suspenseful", "tense", "eerie", "haunting", "whimsical",
            "humorous", "tragic", "romantic", "inspiring", "grim",
            "mysterious", "serene", "chaotic", "intense", "dreamlike",
            "surreal", "somber", "atmospheric",
        ],
        "audience": [
            "young adult", "children", "middle grade", "adult",
            "academic", "professional", "beginner",
        ],
    })

    synonym_map: dict[str, str] = field(default_factory=lambda: {
        "scifi": "science fiction",
        "sci fi": "science fiction",
        "sci-fi": "science fiction",
        "sf": "science fiction",
        "ai": "artificial intelligence",
        "ml": "machine learning",
        "ya": "young adult",
        "wwi": "world war i",
        "wwii": "world war ii",
        "ww2": "world war ii",
        "ww1": "world war i",
        "bio": "biography",
        "autobio": "autobiography",
        "hist fic": "historical fiction",
        "hist": "history",
        "phil": "philosophy",
        "psych": "psychology",
        "nonfic": "nonfiction",
        "non fiction": "nonfiction",
        "non-fiction": "nonfiction",
    })
