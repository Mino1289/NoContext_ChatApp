from zeroctx.chunking import chunk_text


def test_empty_text_gives_no_chunk():
    assert chunk_text("   ") == []


def test_short_text_is_a_single_chunk():
    assert chunk_text("Bonjour le monde.") == ["Bonjour le monde."]


def test_chunks_respect_target_and_overflow_limit():
    text = "Une phrase assez courte qui se répète. " * 400  # ~15 000 caractères
    chunks = chunk_text(text, size=1024, overflow=0.10)
    assert len(chunks) > 10
    assert all(len(c) <= 1126 for c in chunks)


def test_chunk_cut_on_sentence_end_within_overflow():
    # La première fin de phrase après la cible (1024) doit être choisie, dans la limite de 1126.
    sentence = "a" * 98 + ". "  # 100 caractères, point suivi d'un espace à chaque bloc
    text = sentence * 30
    chunks = chunk_text(text, size=1024, overflow=0.10)
    assert chunks[0].endswith(".")
    assert len(chunks[0]) == 1099  # 1099 = 11e point (indice 10 * 100 + 99) : premier >= 1024


def test_hard_cut_on_space_when_no_sentence_end():
    words = ["mot"] * 1000  # séparateurs d'espace, aucune ponctuation de fin
    text = " ".join(words)
    chunks = chunk_text(text, size=1024, overflow=0.10)
    assert len(chunks[0]) <= 1024
    assert chunks[0].endswith("mot")
    assert " ".join(chunks).split() == words  # aucun mot perdu ni coupé


def test_no_separator_falls_back_to_hard_cut():
    text = "x" * 3000
    chunks = chunk_text(text, size=1024, overflow=0.10)
    assert [len(c) for c in chunks] == [1024, 1024, 952]
