"""Routage steering / nouveau sujet par similarité d'embedding (pas d'appel LLM)."""

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    topic_id: str | None
    """Sujet à steerer, ou None pour une nouvelle génération indépendante."""
    score: float | None
    """Meilleure similarité cosinus trouvée parmi les sujets actifs."""
    threshold: float


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def route(query: list[float], active_topics: Mapping[str, list[float]], threshold: float) -> RouteDecision:
    """`active_topics` associe chaque sujet actif à son centroïde d'embedding."""
    best_id: str | None = None
    best_score = -math.inf
    for topic_id, centroid in active_topics.items():
        score = cosine(query, centroid)
        if score > best_score:
            best_id, best_score = topic_id, score
    if best_id is not None and best_score >= threshold:
        return RouteDecision(best_id, best_score, threshold)
    return RouteDecision(None, best_score if best_id is not None else None, threshold)
