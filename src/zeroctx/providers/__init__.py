"""Fabrique des fournisseurs LLM et d'embeddings à partir de la configuration."""

from __future__ import annotations

from zeroctx.config import Settings
from zeroctx.providers.base import EmbeddingProvider, LLMProvider

LLM_NAMES = ("gemini", "cohere")
EMBEDDER_NAMES = ("gemini", "cohere")


def make_llm(name: str, settings: Settings) -> LLMProvider:
    if name == "gemini":
        from zeroctx.providers.gemini import GeminiLLM

        return GeminiLLM(settings.gemini_api_key, settings.gemini_model)
    if name == "cohere":
        from zeroctx.providers.cohere import CohereLLM

        return CohereLLM(settings.cohere_api_key, settings.cohere_model)
    raise ValueError(f"Fournisseur LLM inconnu : {name}")


def make_embedder(name: str, settings: Settings) -> EmbeddingProvider:
    if name == "gemini":
        from zeroctx.providers.gemini import GeminiEmbedder

        return GeminiEmbedder(
            settings.gemini_api_key,
            settings.gemini_embedding_model,
            settings.gemini_embedding_dimension,
        )
    if name == "cohere":
        from zeroctx.providers.cohere import CohereEmbedder

        return CohereEmbedder(
            settings.cohere_api_key,
            settings.cohere_embedding_model,
            settings.cohere_embedding_dimension,
        )
    raise ValueError(f"Fournisseur d'embeddings inconnu : {name}")
