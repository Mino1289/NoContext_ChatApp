"""Construction du contexte récupéré et du prompt (zéro historique)."""

from collections.abc import Sequence
from dataclasses import dataclass

from zeroctx.providers.base import ChatMessage
from zeroctx.store import Hit

SYSTEM_PROMPT = (
    "Réponds dans la langue de l'utilisateur (français par défaut). "
    "Tu n'as aucun historique : ne prétends pas te souvenir d'autre chose que le contexte ci-dessous. "
    "Sois concis : va droit au but, sans introduction, sans reformuler la question, sans conclusion "
    "superflue, et sans titres ni listes sauf si on te les demande."
)

CONTINUATION_INSTRUCTION = (
    "Ta réponse précédente a été interrompue par de nouveaux messages. "
    "Reprends-la et complète-la pour répondre à l'ensemble des messages ci-dessus, "
    "sans répéter inutilement ce qui a déjà été dit."
)


@dataclass(frozen=True)
class Prompt:
    """Requête exacte envoyée à un fournisseur LLM : prompt système puis messages, dans l'ordre."""

    system: str
    messages: tuple[ChatMessage, ...]


def select_context(hits: Sequence[Hit], char_budget: int) -> list[Hit]:
    """Garde les hits dans l'ordre RRF tant que le budget de caractères le permet."""
    selected: list[Hit] = []
    used = 0
    for hit in hits:
        if used + len(hit.content) > char_budget and selected:
            break
        selected.append(hit)
        used += len(hit.content)
    return selected


def build_system_prompt(hits: Sequence[Hit]) -> str:
    if not hits:
        return SYSTEM_PROMPT
    blocks = [f"[{i}] ({h.role}) {h.content}" for i, h in enumerate(hits, start=1)]
    return f"{SYSTEM_PROMPT}\n\n<contexte>\n" + "\n\n".join(blocks) + "\n</contexte>"


def build_messages(pending: Sequence[str], partial: str | None) -> list[ChatMessage]:
    """Messages en cours du sujet (jamais d'historique) : tous les messages en attente,
    et, en cas de steering, la réponse partielle interrompue."""
    if len(pending) == 1:
        user_text = pending[0]
    else:
        user_text = "\n\n".join(f"Message {i}. {text}" for i, text in enumerate(pending, start=1))
    messages = [ChatMessage("user", user_text)]
    if partial:
        messages.append(ChatMessage("assistant", partial))
        messages.append(ChatMessage("user", CONTINUATION_INSTRUCTION))
    return messages
