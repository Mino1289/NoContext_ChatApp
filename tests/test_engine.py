"""Moteur avec faux fournisseurs : pas d'appel réseau."""

from tests.conftest import FakeEmbedder, FakeLLM, wait_until
from zeroctx.engine import GEN_COMPLETE, GEN_INTERRUPTED, Generation
from zeroctx.store import Store


def _generations(engine) -> list[Generation]:
    return [e for e in engine.snapshot() if isinstance(e, Generation)]


def _all_done(engine) -> bool:
    gens = _generations(engine)
    return bool(gens) and all(g.status != "running" for g in gens)


def test_single_message_generates_and_ingests(engine, store: Store, embedder: FakeEmbedder):
    llm = FakeLLM(reply="Il fait beau.")
    result = engine.submit("Quel temps fait-il ? météo", llm, embedder, threshold=0.6)
    assert result.decision.topic_id is None
    wait_until(lambda: _all_done(engine))

    (gen,) = _generations(engine)
    assert gen.status == GEN_COMPLETE
    assert gen.partial == "Il fait beau."

    # Une seule génération : le prompt ne contient que le message courant (zéro historique).
    _, messages = llm.calls[0]
    assert [m.role for m in messages] == ["user"]
    assert messages[0].content == "Quel temps fait-il ? météo"

    # Message utilisateur et réponse persistés, chunks ingérés et vectorisés.
    rows = store.conn.execute("SELECT role, status FROM messages ORDER BY id").fetchall()
    assert rows == [("user", "complete"), ("assistant", "complete")]
    wait_until(lambda: store.chunk_count() == 2 and store.pending_chunks(embedder.key, embedder.dimension) == [])


def test_steering_cancels_keeps_partial_and_answers_all_pending(engine, store: Store, embedder: FakeEmbedder):
    llm = FakeLLM(reply="x" * 400, delay=0.01, pieces=40)
    first = engine.submit("Mon chat dort", llm, embedder, threshold=0.6)
    wait_until(lambda: any(g.partial for g in _generations(engine)))

    second = engine.submit("Et mon miaou ?", llm, embedder, threshold=0.6)
    assert second.decision.topic_id is not None
    assert second.topic_index == first.topic_index

    wait_until(lambda: _all_done(engine))
    gens = _generations(engine)
    assert sorted(g.status for g in gens) == [GEN_COMPLETE, GEN_INTERRUPTED]

    interrupted = next(g for g in gens if g.status == GEN_INTERRUPTED)
    assert interrupted.partial  # la réponse partielle est conservée
    assert interrupted.partial != "x" * 400

    # Une seule réponse finale : le second appel répond aux deux messages en attente.
    assert len(llm.calls) == 2
    _, messages = llm.calls[1]
    roles = [m.role for m in messages]
    assert roles == ["user", "assistant", "user"]
    assert messages[0].content.startswith("Message 1. Mon chat dort")
    assert "Message 2. Et mon miaou ?" in messages[0].content
    assert messages[1].content == interrupted.partial

    statuses = store.conn.execute("SELECT status FROM messages WHERE role = 'assistant' ORDER BY id").fetchall()
    assert [s for (s,) in statuses] == [GEN_INTERRUPTED, GEN_COMPLETE]


def test_two_subjects_run_in_parallel(engine, store: Store, embedder: FakeEmbedder):
    llm = FakeLLM(reply="y" * 200, delay=0.02, pieces=20)
    a = engine.submit("Question sur le chat", llm, embedder, threshold=0.6)
    b = engine.submit("Question sur la météo", llm, embedder, threshold=0.6)

    assert a.topic_index != b.topic_index
    assert a.decision.topic_id is None and b.decision.topic_id is None
    # Les deux générations tournent en même temps.
    wait_until(lambda: len([g for g in _generations(engine) if g.status == "running"]) == 2)

    wait_until(lambda: _all_done(engine))
    gens = _generations(engine)
    assert [g.status for g in gens] == [GEN_COMPLETE, GEN_COMPLETE]
    assert len(llm.calls) == 2


def test_prompt_is_stored_with_answer_and_matches_request(engine, store: Store, embedder: FakeEmbedder):
    llm = FakeLLM(reply="Il fait beau.")
    engine.submit("Quel temps fait-il ? météo", llm, embedder, threshold=0.6)
    wait_until(lambda: _all_done(engine))

    (gen,) = _generations(engine)
    assert gen.prompt is not None
    # Le prompt stocké est exactement celui qui a été envoyé au fournisseur.
    system, messages = llm.calls[0]
    assert gen.prompt.system == system
    assert list(gen.prompt.messages) == messages
    # Le prompt survit aux copies de l'UI (snapshot) et reste inchangé après la génération.
    assert _generations(engine)[0].prompt == gen.prompt


def test_prompt_contains_retrieved_chunks(engine, store: Store, embedder: FakeEmbedder):
    llm = FakeLLM(reply="Il dort beaucoup.")
    engine.submit("Mon chat dort", llm, embedder, threshold=0.6)
    wait_until(lambda: _all_done(engine))
    wait_until(lambda: store.chunk_count() == 2 and store.pending_chunks(embedder.key, embedder.dimension) == [])

    engine.submit("Question sur le chat", llm, embedder, threshold=0.6)
    wait_until(lambda: _all_done(engine) and len(_generations(engine)) == 2)

    second = _generations(engine)[1]
    assert second.prompt is not None
    assert "<contexte>" in second.prompt.system
    assert "Mon chat dort" in second.prompt.system
    assert second.prompt.system == llm.calls[1][0]


def test_prompt_is_kept_on_interruption(engine, store: Store, embedder: FakeEmbedder):
    llm = FakeLLM(reply="x" * 400, delay=0.01, pieces=40)
    engine.submit("Mon chat dort", llm, embedder, threshold=0.6)
    wait_until(lambda: any(g.partial for g in _generations(engine)))
    engine.submit("Et mon miaou ?", llm, embedder, threshold=0.6)
    wait_until(lambda: _all_done(engine))

    interrupted = next(g for g in _generations(engine) if g.status == GEN_INTERRUPTED)
    assert interrupted.prompt is not None
    assert interrupted.prompt.system == llm.calls[0][0]
    assert list(interrupted.prompt.messages) == llm.calls[0][1]

    # La réponse finale part avec la réponse partielle interrompue comme message en cours.
    complete = next(g for g in _generations(engine) if g.status == GEN_COMPLETE)
    assert complete.prompt is not None
    assert list(complete.prompt.messages) == llm.calls[1][1]
    assert complete.prompt.messages[1].content == interrupted.partial


def test_empty_message_is_rejected(engine, embedder: FakeEmbedder):
    import pytest

    with pytest.raises(ValueError):
        engine.submit("   ", FakeLLM(), embedder, threshold=0.6)


def test_provider_error_is_recorded_not_raised(engine, embedder: FakeEmbedder):
    class BrokenLLM(FakeLLM):
        async def stream(self, system, messages):
            raise RuntimeError("quota dépassé")
            yield  # pragma: no cover

    engine.submit("Bonjour", BrokenLLM(), embedder, threshold=0.6)
    wait_until(lambda: _all_done(engine))
    (gen,) = _generations(engine)
    assert gen.status == "error"
    assert gen.error is not None and "quota dépassé" in gen.error
