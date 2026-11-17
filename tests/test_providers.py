"""Adaptateurs fournisseurs testés avec des clients factices (aucun appel réseau)."""

from types import SimpleNamespace
from typing import Any

import pytest

from zeroctx.config import Settings
from zeroctx.providers import make_embedder, make_llm
from zeroctx.providers.base import embedding_key, l2_normalize
from zeroctx.providers.cohere import CohereEmbedder
from zeroctx.providers.gemini import GeminiEmbedder


def test_l2_normalize():
    assert l2_normalize([3.0, 4.0]) == pytest.approx([0.6, 0.8])
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_embedding_key_is_stable_and_safe():
    assert embedding_key("gemini", "gemini-embedding-002", 1536) == "gemini_gemini_embedding_002_1536"


def test_gemini_embedder_uses_task_type_dimension_and_normalizes():
    calls = []

    async def embed_content(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[3.0, 4.0]) for _ in kwargs["contents"]])

    emb: Any = GeminiEmbedder("k", "gemini-embedding-002", 2)
    emb._client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(embed_content=embed_content)))

    import asyncio

    vectors = asyncio.run(emb.embed(["a", "b"], "query"))
    assert vectors == [pytest.approx([0.6, 0.8]), pytest.approx([0.6, 0.8])]
    assert calls[0]["config"].task_type == "RETRIEVAL_QUERY"
    assert calls[0]["config"].output_dimensionality == 2


def test_gemini_llm_enables_code_execution_tool():
    import asyncio

    from zeroctx.providers.base import ChatMessage
    from zeroctx.providers.gemini import GeminiLLM

    calls = []

    async def generate_content_stream(**kwargs):
        calls.append(kwargs)
        return _empty_stream()

    async def _empty_stream():
        return
        yield  # pragma: no cover

    llm: Any = GeminiLLM("k", "gemini-test")
    llm._client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=generate_content_stream))
    )

    async def consume():
        async for _ in llm.stream("sys", [ChatMessage("user", "salut")]):
            pass

    asyncio.run(consume())
    tools = calls[0]["config"].tools
    assert len(tools) == 1
    assert tools[0].code_execution is not None


def test_cohere_embedder_uses_input_type_dimension_and_normalizes():
    calls = []

    async def embed(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(embeddings=SimpleNamespace(float_=[[0.0, 2.0] for _ in kwargs["texts"]]))

    emb: Any = CohereEmbedder("k", "embed-v5.0-pro", 2)
    emb._client = SimpleNamespace(embed=embed)

    import asyncio

    vectors = asyncio.run(emb.embed(["doc"], "document"))
    assert vectors == [pytest.approx([0.0, 1.0])]
    assert calls[0]["input_type"] == "search_document"
    assert calls[0]["output_dimension"] == 2


def test_dotenv_overrides_shell_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=cle-du-projet\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "cle-du-terminal")
    settings = Settings(_env_file=env_file)
    assert settings.gemini_api_key == "cle-du-projet"


def test_factories_build_both_providers_and_reject_unknown():
    settings = Settings(
        _env_file=None,
        gemini_api_key="fake-gemini",
        cohere_api_key="fake-cohere",
        gemini_embedding_dimension=1536,
        cohere_embedding_dimension=2048,
    )
    assert make_llm("gemini", settings).name == "gemini"
    assert make_llm("cohere", settings).name == "cohere"
    assert make_embedder("gemini", settings).dimension == 1536
    assert make_embedder("cohere", settings).dimension == 2048
    with pytest.raises(ValueError):
        make_llm("openai", settings)
