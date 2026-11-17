"""Contrats communs des fournisseurs LLM et d'embeddings."""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["user", "assistant"]
EmbedKind = Literal["document", "query"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class LLMChunk:
    """Fragment de flux. `usage` n'est renseigné que sur le dernier fragment, s'il est connu."""

    text: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMProvider(Protocol):
    name: str
    model: str

    def stream(self, system: str, messages: Sequence[ChatMessage]) -> AsyncIterator[LLMChunk]:
        """Diffuse la réponse. Doit être annulable (fermeture propre du flux)."""
        ...


class EmbeddingProvider(Protocol):
    name: str
    model: str
    dimension: int

    @property
    def key(self) -> str:
        """Identifiant stable (fournisseur, modèle, dimension) : les espaces vectoriels ne sont pas comparables."""
        ...

    async def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        """Embedde les textes. Les vecteurs renvoyés sont normalisés L2."""
        ...


def l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return [0.0 for _ in vector]
    return [x / norm for x in vector]


def embedding_key(provider: str, model: str, dimension: int) -> str:
    safe_model = "".join(c if c.isalnum() else "_" for c in model.lower())
    return f"{provider}_{safe_model}_{dimension}"
