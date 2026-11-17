"""Fournisseur Cohere (chat en flux et embeddings) via le SDK v2."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import cohere

from zeroctx.providers.base import ChatMessage, EmbedKind, LLMChunk, embedding_key, l2_normalize

EMBED_BATCH = 96


class CohereLLM:
    name = "cohere"

    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self._client = cohere.AsyncClientV2(api_key=api_key)

    async def stream(self, system: str, messages: Sequence[ChatMessage]) -> AsyncIterator[LLMChunk]:
        payload: list[Any] = [{"role": "system", "content": system}]
        payload += [{"role": m.role, "content": m.content} for m in messages]
        input_tokens: int | None = None
        output_tokens: int | None = None
        async for event in self._client.chat_stream(model=self.model, messages=payload):
            delta = getattr(event, "delta", None)
            if event.type == "content-delta" and delta is not None and delta.message and delta.message.content:
                text = delta.message.content.text
                if text:
                    yield LLMChunk(text=text)
            elif event.type == "message-end":
                tokens = getattr(getattr(delta, "usage", None), "tokens", None)
                if tokens is not None:
                    input_tokens = int(tokens.input_tokens or 0)
                    output_tokens = int(tokens.output_tokens or 0)
        yield LLMChunk(input_tokens=input_tokens, output_tokens=output_tokens)


class CohereEmbedder:
    name = "cohere"

    def __init__(self, api_key: str, model: str, dimension: int) -> None:
        self.model = model
        self.dimension = dimension
        self._client = cohere.AsyncClientV2(api_key=api_key)

    @property
    def key(self) -> str:
        return embedding_key(self.name, self.model, self.dimension)

    async def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        input_type = "search_document" if kind == "document" else "search_query"
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = list(texts[start : start + EMBED_BATCH])
            response = await self._client.embed(
                model=self.model,
                texts=batch,
                input_type=input_type,
                output_dimension=self.dimension,
                embedding_types=["float"],
            )
            floats = response.embeddings.float_ or []
            vectors.extend(l2_normalize(v) for v in floats)
        return vectors
