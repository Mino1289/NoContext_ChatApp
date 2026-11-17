"""Fournisseur Gemini (génération en flux et embeddings) via google-genai."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from google import genai
from google.genai import types

from zeroctx.providers.base import ChatMessage, EmbedKind, LLMChunk, embedding_key, l2_normalize


class GeminiLLM:
    name = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self._client = genai.Client(api_key=api_key)

    async def stream(self, system: str, messages: Sequence[ChatMessage]) -> AsyncIterator[LLMChunk]:
        contents = [
            types.Content(
                role="user" if m.role == "user" else "model",
                parts=[types.Part(text=m.content)],
            )
            for m in messages
        ]
        # Exécution de code Python : le modèle décide seul de s'en servir.
        tools = [types.Tool(code_execution=types.ToolCodeExecution())]
        config = types.GenerateContentConfig(system_instruction=system, tools=tools)
        stream = await self._client.aio.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        )
        input_tokens: int | None = None
        output_tokens: int | None = None
        async for chunk in stream:
            usage = chunk.usage_metadata
            if usage is not None:
                input_tokens = usage.prompt_token_count or input_tokens
                output_tokens = usage.candidates_token_count or output_tokens
            text = chunk.text or ""
            if text:
                yield LLMChunk(text=text)
        yield LLMChunk(input_tokens=input_tokens, output_tokens=output_tokens)


class GeminiEmbedder:
    name = "gemini"

    def __init__(self, api_key: str, model: str, dimension: int) -> None:
        self.model = model
        self.dimension = dimension
        self._client = genai.Client(api_key=api_key)

    @property
    def key(self) -> str:
        return embedding_key(self.name, self.model, self.dimension)

    async def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        task_type = "RETRIEVAL_DOCUMENT" if kind == "document" else "RETRIEVAL_QUERY"
        vectors: list[list[float]] = []
        # Un appel par texte : avec plusieurs contenus, l'API peut renvoyer moins de vecteurs que de textes.
        for text in texts:
            response = await self._client.aio.models.embed_content(
                model=self.model,
                contents=[text],
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.dimension,
                ),
            )
            vectors.extend(l2_normalize(e.values or []) for e in response.embeddings or [])
        if len(vectors) != len(texts):
            raise RuntimeError(f"Gemini : {len(vectors)} vecteurs pour {len(texts)} textes")
        return vectors
