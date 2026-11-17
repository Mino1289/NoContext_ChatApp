"""Fixtures partagées : faux fournisseurs déterministes et moteur sur SQLite en mémoire."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Iterator, Sequence

import pytest

from zeroctx.config import Settings
from zeroctx.engine import Engine
from zeroctx.providers.base import ChatMessage, EmbedKind, LLMChunk
from zeroctx.store import Store

DIM = 4


class FakeLLM:
    name = "fake"
    model = "fake-model"

    def __init__(self, reply: str = "Réponse de test.", delay: float = 0.0, pieces: int = 1) -> None:
        self.reply = reply
        self.delay = delay
        self.pieces = pieces
        self.calls: list[tuple[str, list[ChatMessage]]] = []

    async def stream(self, system: str, messages: Sequence[ChatMessage]) -> AsyncIterator[LLMChunk]:
        self.calls.append((system, list(messages)))
        step = max(1, len(self.reply) // self.pieces)
        for start in range(0, len(self.reply), step):
            await asyncio.sleep(self.delay)
            yield LLMChunk(text=self.reply[start : start + step])
        yield LLMChunk(input_tokens=12, output_tokens=7)


class FakeEmbedder:
    """Vecteurs fixes par mot-clé : deux textes contenant le même mot-clé sont très proches."""

    name = "fake"
    model = "fake-embed"
    dimension = DIM

    VECTORS: dict[str, list[float]] = {
        "chat": [1.0, 0.0, 0.0, 0.0],
        "miaou": [0.98, 0.2, 0.0, 0.0],
        "météo": [0.0, 1.0, 0.0, 0.0],
        "pluie": [0.0, 0.98, 0.2, 0.0],
    }
    DEFAULT = [0.0, 0.0, 0.0, 1.0]

    def __init__(self) -> None:
        self.calls: list[tuple[EmbedKind, list[str]]] = []

    @property
    def key(self) -> str:
        return "fake_fake_embed_4"

    def vector_for(self, text: str) -> list[float]:
        lowered = text.lower()
        for keyword, vector in self.VECTORS.items():
            if keyword in lowered:
                return list(vector)
        return list(self.DEFAULT)

    async def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        self.calls.append((kind, list(texts)))
        await asyncio.sleep(0)
        return [self.vector_for(t) for t in texts]


def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Condition non atteinte dans le délai imparti")


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        steering_threshold=0.6,
        chunk_size=1024,
        chunk_overflow_ratio=0.10,
        rrf_k=60,
        search_candidates=20,
        context_chunks=8,
        context_char_budget=12_000,
    )


@pytest.fixture
def store() -> Iterator[Store]:
    s = Store(":memory:")
    yield s
    s.close()


@pytest.fixture
def engine(settings: Settings, store: Store) -> Iterator[Engine]:
    e = Engine(settings, store)
    yield e
    e.shutdown()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()
