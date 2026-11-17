"""Découpage des messages en chunks.

Cible `size` caractères. Un dépassement allant jusqu'à `overflow` (10 % par défaut,
soit 1126 caractères pour 1024) est toléré pour finir sur une fin de phrase.
Sinon, la coupe dure se fait au dernier espace avant la cible.
"""

import re

SENTENCE_END = re.compile(r"[.!?…]+[\"”»)]*(?=\s|$)")


def chunk_text(text: str, size: int = 1024, overflow: float = 0.10) -> list[str]:
    text = text.strip()
    if not text:
        return []
    max_len = int(size * (1 + overflow))
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    pos = 0
    while pos < len(text):
        if len(text) - pos <= max_len:
            chunks.append(text[pos:].strip())
            break
        end = _find_cut(text, pos, size, max_len)
        piece = text[pos:end].strip()
        if piece:
            chunks.append(piece)
        pos = end
        while pos < len(text) and text[pos].isspace():
            pos += 1
    return chunks


def _find_cut(text: str, start: int, size: int, max_len: int) -> int:
    """Position de coupe absolue pour un chunk débutant à `start`."""
    target = start + size
    window = text[start : start + max_len + 1]

    # 1. Fin de phrase dans la zone [cible, cible + 10 %].
    for match in SENTENCE_END.finditer(window):
        if match.end() >= size:
            return start + match.end()

    # 2. Coupe dure au dernier espace avant la cible.
    last_space = max(text.rfind(" ", start, target + 1), text.rfind("\n", start, target + 1))
    if last_space > start:
        return last_space

    # 3. Aucun séparateur : coupe brute.
    return target
