from zeroctx.providers.base import Role
from zeroctx.store import Store, reciprocal_rank_fusion


def test_sqlite_vec_is_loaded(store: Store):
    version = store.conn.execute("SELECT vec_version()").fetchone()[0]
    assert version.startswith("v")


def test_rrf_scores_items_present_in_both_lists_higher():
    scores = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]], k=60)
    assert scores[1] > scores[4]
    assert scores[3] > scores[2]  # 3 est rang 1 et rang 3 ; 2 est rang 2 seulement
    assert scores[1] == 1 / 61 + 1 / 62


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([[], []], k=60) == {}


def _index(store: Store, role: Role, text: str, vector: list[float]) -> int:
    mid = store.add_message(role, text, topic_id="t")
    (cid,) = store.add_chunks(mid, role, [text])
    store.store_vectors("k", 4, [cid], [vector])
    return mid


def test_hybrid_search_combines_vector_and_keyword(store: Store):
    store.ensure_vec_table("k", 4)
    m_pluie = _index(store, "user", "Il pleut beaucoup à Lyon", [0.0, 1.0, 0.0, 0.0])
    m_chat = _index(store, "user", "Le chat dort sur le canapé", [1.0, 0.0, 0.0, 0.0])
    _index(store, "assistant", "Les océans sont salés", [0.0, 0.0, 0.0, 1.0])

    hits = store.hybrid_search("k", 4, "chat", [0.9, 0.1, 0.0, 0.0], candidates=10, limit=3, rrf_k=60)
    assert hits[0].message_id == m_chat  # présent dans le KNN et dans BM25

    hits = store.hybrid_search("k", 4, "pluie", [0.0, 0.9, 0.1, 0.0], candidates=10, limit=3, rrf_k=60)
    assert hits[0].message_id == m_pluie


def test_hybrid_search_filters_weak_matches(store: Store):
    store.ensure_vec_table("k", 4)
    m_good = _index(store, "user", "Le chat dort", [1.0, 0.0, 0.0, 0.0])
    _index(store, "user", "Recette de cookies", [0.0, 0.0, 1.0, 0.0])
    # Seuil de similarité : le chunk éloigné (cosinus 0) est écarté.
    hits = store.hybrid_search("k", 4, "", [1.0, 0.0, 0.0, 0.0], candidates=10, limit=5, rrf_k=60, min_similarity=0.6)
    assert [h.message_id for h in hits] == [m_good]
    # Score relatif : un résultat trouvé seulement par mots-clés (score RRF moitié) est écarté.
    m_kw = _index(store, "user", "Le chat noir", [0.0, 1.0, 0.0, 0.0])
    hits = store.hybrid_search(
        "k", 4, "chat", [1.0, 0.0, 0.0, 0.0], candidates=10, limit=5, rrf_k=60, min_similarity=0.6, relative_score=0.9
    )
    assert m_kw not in [h.message_id for h in hits]
    assert hits[0].message_id == m_good


def test_hybrid_search_excludes_messages(store: Store):
    store.ensure_vec_table("k", 4)
    m_chat = _index(store, "user", "Le chat dort", [1.0, 0.0, 0.0, 0.0])
    m_other = _index(store, "user", "Le chien aboie", [0.9, 0.1, 0.0, 0.0])
    hits = store.hybrid_search(
        "k", 4, "chat", [1.0, 0.0, 0.0, 0.0], candidates=10, limit=5, rrf_k=60, exclude_message_ids={m_chat}
    )
    ids = {h.message_id for h in hits}
    assert m_chat not in ids
    assert m_other in ids


def test_fts_query_escapes_special_syntax(store: Store):
    store.ensure_vec_table("k", 4)
    _index(store, "user", "Question sur C++ et (parenthèses)", [1.0, 0.0, 0.0, 0.0])
    # Ne doit pas lever d'erreur de syntaxe FTS.
    hits = store.hybrid_search("k", 4, 'C++ "NEAR( OR AND', [1.0, 0.0, 0.0, 0.0], candidates=5, limit=5, rrf_k=60)
    assert hits


def test_pending_chunks_and_vector_backfill(store: Store):
    mid = store.add_message("user", "texte", topic_id="t")
    ids = store.add_chunks(mid, "user", ["un", "deux"])
    assert [i for i, _ in store.pending_chunks("k2", 4)] == ids
    store.store_vectors("k2", 4, ids, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    assert store.pending_chunks("k2", 4) == []
