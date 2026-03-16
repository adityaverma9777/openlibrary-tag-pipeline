from joblib import Parallel, delayed
from transformers import pipeline as hf_pipeline

from tag_pipeline.config import PipelineConfig


class CategoryClassifier:
    CATEGORY_HYPOTHESES = {
        "genre": "This is a literary genre or type of book such as fiction, mystery, romance, or fantasy",
        "theme": "This is a thematic concept or idea explored in literature such as love, war, identity, or freedom",
        "setting": "This describes a physical location, time period, or world where a story takes place",
        "mood": "This describes an emotional tone, atmosphere, or feeling of a story",
        "audience": "This describes the target readership, age group, or demographic of a book",
    }

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._nli_pipeline = None
        self._keyword_map = self.config.category_keywords

    @property
    def nli(self):
        if self._nli_pipeline is None:
            device_id = 0 if self.config.device == "cuda" else -1
            self._nli_pipeline = hf_pipeline(
                "zero-shot-classification",
                model=self.config.nli_model,
                device=device_id,
                batch_size=self.config.batch_size,
            )
        return self._nli_pipeline

    def classify_clusters(self, clusters: list[dict]) -> list[dict]:
        representatives = [c["representative"] for c in clusters]

        nli_results = self._batch_nli_classify(representatives)

        classified = []
        for cluster, nli_result in zip(clusters, nli_results):
            rep = cluster["representative"]

            if nli_result["confidence"] >= self.config.nli_confidence_threshold:
                classified.append({
                    "cluster_label": rep,
                    "members": cluster["members"],
                    "category": nli_result["category"],
                    "confidence": nli_result["confidence"],
                    "method": "nli",
                })
                continue

            keyword_result = self._keyword_classify(rep)
            if keyword_result:
                classified.append({
                    "cluster_label": rep,
                    "members": cluster["members"],
                    "category": keyword_result["category"],
                    "confidence": keyword_result["confidence"],
                    "method": "keyword",
                })
                continue

            classified.append({
                "cluster_label": rep,
                "members": cluster["members"],
                "category": nli_result["category"],
                "confidence": nli_result["confidence"],
                "method": "nli-lowconf",
            })

        return classified

    def _batch_nli_classify(self, tags: list[str]) -> list[dict]:
        if not tags:
            return []

        categories = list(self.CATEGORY_HYPOTHESES.keys())
        hypotheses = list(self.CATEGORY_HYPOTHESES.values())

        results = self.nli(
            tags,
            candidate_labels=hypotheses,
            multi_label=False,
        )

        if isinstance(results, dict):
            results = [results]

        parsed = []
        for result in results:
            best_hypothesis = result["labels"][0]
            best_score = result["scores"][0]
            category = categories[hypotheses.index(best_hypothesis)]

            parsed.append({
                "category": category,
                "confidence": round(float(best_score), 4),
                "all_scores": {
                    categories[hypotheses.index(label)]: round(float(score), 4)
                    for label, score in zip(result["labels"], result["scores"])
                },
            })

        return parsed

    def _keyword_classify(self, tag: str) -> dict | None:
        tag_lower = tag.lower()
        best_match = None
        best_overlap = 0.0

        for category, keywords in self._keyword_map.items():
            for keyword in keywords:
                if keyword in tag_lower or tag_lower in keyword:
                    overlap = len(keyword) / max(len(tag_lower), 1)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = {
                            "category": category,
                            "confidence": round(min(0.75, 0.5 + overlap * 0.3), 4),
                            "matched_keyword": keyword,
                        }

        return best_match

    def _batch_keyword_classify(self, tags: list[str]) -> list[dict | None]:
        results = Parallel(n_jobs=self.config.num_workers, prefer="threads")(
            delayed(self._keyword_classify)(tag) for tag in tags
        )
        return results

    def build_structured_output(self, classified_clusters: list[dict]) -> dict:
        structured = {
            "genre": [],
            "theme": [],
            "setting": [],
            "mood": [],
            "audience": [],
        }

        for item in classified_clusters:
            category = item["category"]
            if category in structured:
                entry = {
                    "label": item["cluster_label"],
                    "members": item["members"],
                    "confidence": item["confidence"],
                }
                structured[category].append(entry)

        structured = {k: v for k, v in structured.items() if v}
        return structured
