"""Moteur de conversation zéro-contexte.

- Une boucle asyncio dédiée tourne dans un thread ; l'UI Streamlit lui soumet des messages
  (`submit`) et lit un instantané de la timeline (`snapshot`).
- Chaque génération est une `asyncio.Task`. Un sujet (`Topic`) porte au plus une génération
  en cours et le centroïde des embeddings de ses messages.
- Steering : si le message est assez proche du centroïde d'un sujet ayant une génération en cours,
  cette génération est annulée (sa réponse partielle est conservée, statut `interrupted`) et une
  nouvelle génération répond à tous les messages en attente du sujet.
"""

import asyncio
import logging
import threading
import time
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any

from zeroctx.chunking import chunk_text
from zeroctx.config import Settings
from zeroctx.providers.base import EmbeddingProvider, LLMProvider
from zeroctx.retrieval import Prompt, build_messages, build_system_prompt, select_context
from zeroctx.router import RouteDecision, route
from zeroctx.store import Hit, Store

log = logging.getLogger(__name__)

GEN_RUNNING = "running"
GEN_COMPLETE = "complete"
GEN_INTERRUPTED = "interrupted"
GEN_ERROR = "error"


@dataclass
class UserEntry:
    message_id: int
    topic_id: str
    text: str
    kind: str = "user"


@dataclass
class Generation:
    id: str
    topic_id: str
    provider: str
    answered_ids: list[int]
    status: str = GEN_RUNNING
    partial: str = ""
    error: str | None = None
    task: asyncio.Task[None] | None = None
    kind: str = "generation"
    usage: dict[str, int] = field(default_factory=dict)
    prompt: Prompt | None = None  # exactement la requête envoyée au fournisseur (immuable)
    started_at: float = 0.0  # time.perf_counter()
    first_token_at: float | None = None
    ended_at: float | None = None

    def view(self) -> Generation:
        """Copie figée pour l'UI (sans la tâche asyncio)."""
        return Generation(
            id=self.id,
            topic_id=self.topic_id,
            provider=self.provider,
            answered_ids=list(self.answered_ids),
            status=self.status,
            partial=self.partial,
            error=self.error,
            usage=dict(self.usage),
            prompt=self.prompt,
            started_at=self.started_at,
            first_token_at=self.first_token_at,
            ended_at=self.ended_at,
        )


@dataclass
class Topic:
    id: str
    index: int
    embedder_key: str
    centroid_sum: list[float]
    count: int
    pending_ids: list[int]
    running: Generation | None = None

    def centroid(self) -> list[float]:
        n = sum(x * x for x in self.centroid_sum) ** 0.5
        return [x / n for x in self.centroid_sum] if n else list(self.centroid_sum)

    def absorb(self, vector: list[float]) -> None:
        if not self.centroid_sum:
            self.centroid_sum = list(vector)
        else:
            self.centroid_sum = [a + b for a, b in zip(self.centroid_sum, vector, strict=True)]
        self.count += 1


@dataclass
class SubmitResult:
    message_id: int
    topic_index: int
    decision: RouteDecision
    generation_id: str


class Engine:
    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self._lock = threading.Lock()
        self._timeline: list[UserEntry | Generation] = []
        self._topics: dict[str, Topic] = {}
        self._topic_counter = 0
        self._background: set[asyncio.Task[Any]] = set()
        self._backfilling: set[str] = set()
        self._embedders: dict[str, EmbeddingProvider] = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="zeroctx-engine", daemon=True)
        self._thread.start()

    # ----- API appelée depuis l'UI (thread Streamlit) ----------------------------

    def submit(
        self,
        text: str,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        threshold: float,
    ) -> SubmitResult:
        """Enregistre le message, route, lance la génération. Bloque le temps de l'embedding
        et du routage seulement (la génération continue en arrière-plan)."""
        return self._call(self._submit(text, llm, embedder, threshold))

    def ensure_index(self, embedder: EmbeddingProvider) -> None:
        """Lance en tâche de fond le ré-embedding des chunks absents de cet embedder."""
        self._loop.call_soon_threadsafe(self._spawn_backfill, embedder)

    def snapshot(self) -> list[UserEntry | Generation]:
        with self._lock:
            return [e.view() if isinstance(e, Generation) else e for e in self._timeline]

    def topic_indexes(self) -> dict[str, int]:
        with self._lock:
            return {tid: t.index for tid, t in self._topics.items()}

    def shutdown(self) -> None:
        if self._loop.is_closed():
            return
        for topic in list(self._topics.values()):
            if topic.running and topic.running.task:
                self._loop.call_soon_threadsafe(topic.running.task.cancel)
        self._call(asyncio.sleep(0.05))
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    # ----- boucle asyncio --------------------------------------------------------

    def _call(self, coro: Coroutine[Any, Any, Any]) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _spawn(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        task = self._loop.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    async def _submit(
        self,
        text: str,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        threshold: float,
    ) -> SubmitResult:
        text = text.strip()
        if not text:
            raise ValueError("Message vide")

        self._embedders[embedder.key] = embedder
        query_vector = (await embedder.embed([text], "query"))[0]

        with self._lock:
            active = {
                t.id: t.centroid()
                for t in self._topics.values()
                if t.running is not None and t.embedder_key == embedder.key
            }
        decision = route(query_vector, active, threshold)

        if decision.topic_id is not None:
            topic = self._topics[decision.topic_id]
            message_id = self._store_user_message(text, topic)
            topic.absorb(query_vector)
            topic.pending_ids.append(message_id)
            old = topic.running
            assert old is not None and old.task is not None
            old_message_id = await self._steer(old)
            generation = self._start_generation(
                topic,
                llm,
                embedder,
                query_vector,
                partial_message_id=old_message_id,
                partial=old.partial,
            )
        else:
            topic = self._new_topic(embedder.key, query_vector)
            message_id = self._store_user_message(text, topic)
            topic.pending_ids.append(message_id)
            generation = self._start_generation(
                topic,
                llm,
                embedder,
                query_vector,
                partial_message_id=None,
                partial="",
            )

        self._spawn(self._ingest(message_id, "user", text, embedder))
        return SubmitResult(message_id, topic.index, decision, generation.id)

    async def _steer(self, old: Generation) -> int | None:
        """Annule la génération en cours et attend la sauvegarde de sa réponse partielle."""
        assert old.task is not None
        old.task.cancel()
        await asyncio.wait({old.task})
        if old.partial:
            return self._last_partial_id(old)
        return None

    def _last_partial_id(self, gen: Generation) -> int | None:
        row = self.store.conn.execute(
            "SELECT id FROM messages WHERE generation_id = ? AND role = 'assistant' ORDER BY id DESC LIMIT 1",
            (gen.id,),
        ).fetchone()
        return int(row[0]) if row else None

    def _new_topic(self, embedder_key: str, vector: list[float]) -> Topic:
        with self._lock:
            self._topic_counter += 1
            topic = Topic(
                id=uuid.uuid4().hex,
                index=self._topic_counter,
                embedder_key=embedder_key,
                centroid_sum=list(vector),
                count=1,
                pending_ids=[],
            )
            self._topics[topic.id] = topic
        return topic

    def _store_user_message(self, text: str, topic: Topic) -> int:
        message_id = self.store.add_message("user", text, topic_id=topic.id)
        with self._lock:
            self._timeline.append(UserEntry(message_id=message_id, topic_id=topic.id, text=text))
        return message_id

    def _start_generation(
        self,
        topic: Topic,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        query_vector: list[float],
        *,
        partial_message_id: int | None,
        partial: str,
    ) -> Generation:
        pending_ids = list(topic.pending_ids)
        pending_texts = [self.store.get_message_text(i) or "" for i in pending_ids]
        query_text = "\n".join(pending_texts)

        hits: list[Hit] = self.store.hybrid_search(
            embedder.key,
            embedder.dimension,
            query_text,
            query_vector,
            candidates=self.settings.search_candidates,
            limit=self.settings.context_chunks,
            rrf_k=self.settings.rrf_k,
            exclude_message_ids=set(pending_ids) | ({partial_message_id} if partial_message_id else set()),
            min_similarity=self.settings.context_min_similarity,
            min_bm25=self.settings.context_min_bm25,
            relative_score=self.settings.context_relative_score,
        )
        hits = select_context(hits, self.settings.context_char_budget)
        # Le prompt est construit une seule fois : c'est cet objet qui est envoyé au fournisseur ET affiché.
        prompt = Prompt(
            system=build_system_prompt(hits),
            messages=tuple(build_messages(pending_texts, partial or None)),
        )

        generation = Generation(
            id=uuid.uuid4().hex,
            topic_id=topic.id,
            provider=llm.name,
            answered_ids=pending_ids,
            prompt=prompt,
            started_at=time.perf_counter(),
        )
        topic.running = generation
        with self._lock:
            self._timeline.append(generation)

        generation.task = self._loop.create_task(self._run_generation(generation, topic, llm, prompt))
        return generation

    async def _run_generation(
        self,
        generation: Generation,
        topic: Topic,
        llm: LLMProvider,
        prompt: Prompt,
    ) -> None:
        status = GEN_ERROR
        try:
            try:
                async for chunk in llm.stream(prompt.system, list(prompt.messages)):
                    if chunk.text:
                        with self._lock:
                            if generation.first_token_at is None:
                                generation.first_token_at = time.perf_counter()
                            generation.partial += chunk.text
                    if chunk.input_tokens is not None or chunk.output_tokens is not None:
                        generation.usage = {
                            "input": chunk.input_tokens or 0,
                            "output": chunk.output_tokens or 0,
                        }
                status = GEN_COMPLETE
            except asyncio.CancelledError:
                status = GEN_INTERRUPTED
                raise
            except Exception as exc:
                status = GEN_ERROR
                generation.error = f"{type(exc).__name__}: {exc}"
                log.exception("Génération en erreur")
        finally:
            self._finish_generation(generation, topic, status)

    def _finish_generation(self, generation: Generation, topic: Topic, status: str) -> None:
        generation.status = status
        generation.ended_at = time.perf_counter()
        if topic.running is generation:
            topic.running = None
        if status == GEN_COMPLETE:
            answered = set(generation.answered_ids)
            topic.pending_ids = [i for i in topic.pending_ids if i not in answered]
        if generation.partial.strip():
            message_id = self.store.add_message(
                "assistant",
                generation.partial,
                topic_id=topic.id,
                generation_id=generation.id,
                status=status,
            )
            embedder = self._embedder_for_topic(topic)
            if embedder is not None:
                self._spawn(self._ingest(message_id, "assistant", generation.partial, embedder))

    def _embedder_for_topic(self, topic: Topic) -> EmbeddingProvider | None:
        return self._embedders.get(topic.embedder_key)

    async def _ingest(self, message_id: int, role: str, text: str, embedder: EmbeddingProvider) -> None:
        """Chunking, insertion FTS, puis embedding document et insertion vectorielle."""
        chunks = chunk_text(text, self.settings.chunk_size, self.settings.chunk_overflow_ratio)
        if not chunks:
            return
        chunk_ids = self.store.add_chunks(message_id, role, chunks)
        try:
            vectors = await embedder.embed(chunks, "document")
            self.store.store_vectors(embedder.key, embedder.dimension, chunk_ids, vectors)
        except Exception:
            log.exception("Échec de l'embedding d'ingestion ; chunks conservés sans vecteur")

    def _spawn_backfill(self, embedder: EmbeddingProvider) -> None:
        if embedder.key in self._backfilling:
            return
        self._backfilling.add(embedder.key)
        self._spawn(self._backfill(embedder))

    async def _backfill(self, embedder: EmbeddingProvider) -> None:
        try:
            while True:
                pending = self.store.pending_chunks(embedder.key, embedder.dimension, limit=64)
                if not pending:
                    break
                vectors = await embedder.embed([t for _, t in pending], "document")
                self.store.store_vectors(embedder.key, embedder.dimension, [i for i, _ in pending], vectors)
        except Exception:
            log.exception("Échec du ré-embedding de fond pour %s", embedder.key)
        finally:
            self._backfilling.discard(embedder.key)
