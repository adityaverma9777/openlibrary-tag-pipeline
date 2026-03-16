import re
import unicodedata
from typing import Sequence

from joblib import Parallel, delayed

from tag_pipeline.config import PipelineConfig


class TagCleaner:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._synonym_map = dict(self.config.synonym_map)

    def clean(self, tags: Sequence[str]) -> list[str]:
        normalized = Parallel(n_jobs=self.config.num_workers, prefer="threads")(
            delayed(self._normalize)(t) for t in tags
        )

        seen: set[str] = set()
        cleaned = []
        for tag in normalized:
            if tag and tag not in seen:
                seen.add(tag)
                cleaned.append(tag)
        return cleaned

    def clean_with_mapping(self, tags: Sequence[str]) -> tuple[list[str], dict[str, list[str]]]:
        normalized = Parallel(n_jobs=self.config.num_workers, prefer="threads")(
            delayed(self._normalize)(t) for t in tags
        )

        cleaned = []
        seen: set[str] = set()
        raw_to_clean: dict[str, list[str]] = {}

        for raw, norm in zip(tags, normalized):
            if not norm:
                continue
            raw_to_clean.setdefault(norm, []).append(raw)
            if norm not in seen:
                seen.add(norm)
                cleaned.append(norm)

        return cleaned, raw_to_clean

    def _normalize(self, tag: str) -> str:
        tag = tag.strip()
        tag = unicodedata.normalize("NFKD", tag)
        tag = "".join(
            ch for ch in tag if not unicodedata.combining(ch)
        )
        tag = tag.lower()
        tag = tag.replace("-", " ").replace("_", " ")
        tag = re.sub(r"[^\w\s]", "", tag)
        tag = re.sub(r"\s+", " ", tag).strip()
        tag = self._apply_synonyms(tag)
        return tag

    def _apply_synonyms(self, tag: str) -> str:
        return self._synonym_map.get(tag, tag)

    def add_synonym(self, variant: str, canonical: str) -> None:
        self._synonym_map[variant.lower().strip()] = canonical.lower().strip()
