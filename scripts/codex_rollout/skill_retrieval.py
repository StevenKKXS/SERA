#!/usr/bin/env python3
"""Small, dependency-free hybrid retriever for the frozen executor vocabulary."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMES = ("R0", "R1", "R2", "R3", "R4")


def words(value: str) -> list[str]:
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    return re.findall(r"[a-z0-9]+", value.lower())


def char_ngrams(value: str, size: int = 3) -> Counter[str]:
    compact = " " + re.sub(r"\s+", " ", value.lower()).strip() + " "
    return Counter(compact[index : index + size] for index in range(len(compact) - size + 1))


class ExecutorSkillRetriever:
    def __init__(self, vocabulary_path: Path) -> None:
        self.vocabulary_path = vocabulary_path.resolve()
        self.vocabulary = json.loads(self.vocabulary_path.read_text(encoding="utf-8"))
        self.entries = list(self.vocabulary["entries"])
        self.by_id = {entry["template_id"]: entry for entry in self.entries}
        if len(self.by_id) != len(self.entries):
            raise ValueError("duplicate template_id in executor vocabulary")
        self.documents = {entry["template_id"]: self._document(entry) for entry in self.entries}
        self.tokens = {key: words(value) for key, value in self.documents.items()}
        self.term_frequency = {key: Counter(value) for key, value in self.tokens.items()}
        self.avg_length = sum(map(len, self.tokens.values())) / len(self.tokens)
        document_frequency: Counter[str] = Counter()
        for value in self.tokens.values():
            document_frequency.update(set(value))
        count = len(self.entries)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        self.char_vectors = {key: char_ngrams(value) for key, value in self.documents.items()}
        char_df: Counter[str] = Counter()
        for value in self.char_vectors.values():
            char_df.update(value)
        self.char_idf = {
            gram: math.log((1 + count) / (1 + frequency)) + 1.0
            for gram, frequency in char_df.items()
        }

    @staticmethod
    def _document(entry: dict[str, Any]) -> str:
        goals = [
            goal
            for values in entry.get("task_goal_examples", {}).values()
            for goal in values
        ]
        return " ".join(
            [entry["normalized_template"], *entry.get("simple_labels", []), *entry["tasks"], *goals]
        )

    def _bm25(self, query: str, allowed: set[str] | None = None) -> dict[str, float]:
        query_terms = words(query)
        k1, b = 1.5, 0.75
        result: dict[str, float] = {}
        for template_id, frequencies in self.term_frequency.items():
            if allowed is not None and template_id not in allowed:
                continue
            length = len(self.tokens[template_id])
            score = 0.0
            for term in query_terms:
                frequency = frequencies[term]
                if not frequency:
                    continue
                denominator = frequency + k1 * (1 - b + b * length / self.avg_length)
                score += self.idf.get(term, 0.0) * frequency * (k1 + 1) / denominator
            result[template_id] = score
        return result

    def _char_cosine(self, query: str, allowed: set[str] | None = None) -> dict[str, float]:
        query_counts = char_ngrams(query)
        query_vector = {
            gram: count * self.char_idf.get(gram, 0.0)
            for gram, count in query_counts.items()
            if gram in self.char_idf
        }
        query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
        result: dict[str, float] = {}
        for template_id, counts in self.char_vectors.items():
            if allowed is not None and template_id not in allowed:
                continue
            vector = {gram: count * self.char_idf[gram] for gram, count in counts.items()}
            norm = math.sqrt(sum(value * value for value in vector.values()))
            dot = sum(query_vector.get(gram, 0.0) * value for gram, value in vector.items())
            result[template_id] = dot / (query_norm * norm) if query_norm and norm else 0.0
        return result

    @staticmethod
    def _rank(scores: dict[str, float]) -> list[str]:
        return [key for key, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]

    @staticmethod
    def _rrf(rankings: Iterable[list[str]], constant: int = 60) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        for ranking in rankings:
            for index, template_id in enumerate(ranking, start=1):
                scores[template_id] += 1.0 / (constant + index)
        return scores

    def retrieve(
        self,
        *,
        query: str,
        task: str | None,
        previous_template_id: str | None,
        recent_status: str,
        top_k: int = 5,
        scheme: str = "R4",
        transition_override: dict[str, dict[str, int]] | None = None,
    ) -> list[dict[str, Any]]:
        if scheme not in SCHEMES:
            raise ValueError(f"unknown retrieval scheme {scheme!r}")
        task_ids = {
            entry["template_id"] for entry in self.entries if task in entry.get("tasks", [])
        }
        allowed = task_ids if scheme in {"R0", "R3", "R4"} and task_ids else None
        frequency_scores = {
            entry["template_id"]: (
                entry.get("task_frame_counts", {}).get(task, 0)
                if task is not None
                else entry["training_frame_count"]
            )
            for entry in self.entries
            if allowed is None or entry["template_id"] in allowed
        }
        semantic = self._rank(self._bm25(query, allowed))
        char_ranking = self._rank(self._char_cosine(query, allowed))
        frequency = self._rank(frequency_scores)
        current: list[str] = []
        transitions: list[str] = []
        if previous_template_id in self.by_id:
            current = [previous_template_id]
            source = self.by_id[previous_template_id]
            if transition_override is not None:
                transition_counts = transition_override.get(previous_template_id, {})
            elif task and task in source.get("task_successors", {}):
                transition_counts = source["task_successors"][task]
            else:
                transition_counts = source.get("successors", {})
            transitions = [
                key
                for key, _ in sorted(
                    transition_counts.items(), key=lambda item: (-item[1], item[0])
                )
                if allowed is None or key in allowed
            ]
        if recent_status in {"progressing", "uncertain", "stagnant", "initial"}:
            status_rankings = [current, transitions]
        else:
            status_rankings = [transitions, current]

        if scheme == "R0":
            scores = self._rrf([*status_rankings, frequency])
        elif scheme == "R1":
            scores = self._rrf([semantic])
        elif scheme == "R2":
            scores = self._rrf([*status_rankings, semantic])
        elif scheme == "R3":
            scores = self._rrf([*status_rankings, semantic, frequency])
        else:
            scores = self._rrf([*status_rankings, semantic, char_ranking, frequency])
        ranking = self._rank(scores)[:top_k]
        return [
            {
                "rank": index,
                "retrieval_score": scores[template_id],
                **self.by_id[template_id],
            }
            for index, template_id in enumerate(ranking, start=1)
        ]
