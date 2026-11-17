"""Point d'entrée Streamlit : interface seule, toute la logique est dans `zeroctx`."""

import streamlit as st

from zeroctx.config import DB_PATH, get_settings
from zeroctx.engine import Engine, Generation, UserEntry
from zeroctx.providers import EMBEDDER_NAMES, LLM_NAMES, make_embedder, make_llm
from zeroctx.store import Store

PROVIDER_LABELS = {"gemini": "Gemini", "cohere": "Cohere"}
STATUS_LABELS = {
    "running": "en cours",
    "complete": "terminée",
    "interrupted": "interrompue (steering)",
    "error": "erreur",
}


@st.cache_resource
def get_engine() -> Engine:
    settings = get_settings()
    store = Store(DB_PATH)
    return Engine(settings, store)


@st.cache_resource
def get_llm(name: str):
    return make_llm(name, get_settings())


@st.cache_resource
def get_embedder(name: str):
    return make_embedder(name, get_settings())


def render_prompt(generation: Generation) -> None:
    """Affiche la requête exacte envoyée au modèle pour cette réponse (débogage, sans appel supplémentaire)."""
    prompt = generation.prompt
    if prompt is None:
        return
    with st.expander("Prompt envoyé"):
        st.caption("Prompt système")
        st.code(prompt.system, language=None, wrap_lines=True)
        for index, message in enumerate(prompt.messages, start=1):
            st.caption(f"Message {index} · rôle « {message.role} »")
            st.code(message.content, language=None, wrap_lines=True)


def render_metrics(generation: Generation) -> None:
    """TTFT, débit de sortie et tokens. Affiché seulement pour les réponses terminées."""
    if generation.status != "complete" or generation.first_token_at is None or generation.ended_at is None:
        return
    parts = [f"TTFT {generation.first_token_at - generation.started_at:.2f} s"]
    output = generation.usage.get("output", 0)
    stream_duration = generation.ended_at - generation.first_token_at
    if output and stream_duration > 0:
        parts.append(f"{output / stream_duration:.1f} tok/s")
    if generation.usage:
        parts.append(f"entrée {generation.usage['input']} tokens")
        parts.append(f"sortie {output} tokens")
    st.caption(" · ".join(parts))


def render_timeline(engine: Engine) -> None:
    entries = engine.snapshot()
    topics = engine.topic_indexes()
    if not entries:
        st.caption("Aucun message pour le moment. Posez une question ci-dessous.")
        return
    for entry in entries:
        if isinstance(entry, UserEntry):
            with st.chat_message("user"):
                st.caption(f"Sujet {topics.get(entry.topic_id, '?')}")
                st.write(entry.text)
        elif isinstance(entry, Generation):
            with st.chat_message("assistant"):
                label = PROVIDER_LABELS.get(entry.provider, entry.provider)
                status = STATUS_LABELS.get(entry.status, entry.status)
                st.caption(f"Sujet {topics.get(entry.topic_id, '?')} · {label} · {status}")
                if entry.error:
                    st.error(entry.error)
                st.markdown(entry.partial or "…")
                render_metrics(entry)
                render_prompt(entry)


def main() -> None:
    settings = get_settings()
    st.set_page_config(page_title="Chat zéro-contexte", layout="wide")

    with st.sidebar:
        st.header("Réglages")
        llm_name = st.selectbox(
            "Fournisseur de génération",
            LLM_NAMES,
            format_func=lambda n: PROVIDER_LABELS[n],
        )
        embedder_name = st.selectbox(
            "Fournisseur d'embeddings",
            EMBEDDER_NAMES,
            index=EMBEDDER_NAMES.index(llm_name),
            format_func=lambda n: PROVIDER_LABELS[n],
            help="Les embeddings servent au routage (steering) et à la recherche de contexte.",
        )
        threshold = st.slider(
            "Seuil de steering",
            min_value=0.0,
            max_value=1.0,
            value=settings.steering_threshold,
            step=0.01,
            help="Similarité cosinus minimale avec un sujet en cours pour le steerer. Valeur par défaut à calibrer.",
        )

    engine = get_engine()
    llm = get_llm(llm_name)
    embedder = get_embedder(embedder_name)

    if st.session_state.get("indexed_embedder") != embedder.key:
        engine.ensure_index(embedder)
        st.session_state["indexed_embedder"] = embedder.key

    st.title("Chat zéro-contexte")
    st.caption(
        "Chaque appel au modèle ne reçoit que le contexte récupéré et les messages en cours, jamais l'historique."
    )

    @st.fragment(run_every=0.5)
    def timeline() -> None:
        render_timeline(engine)

    timeline()

    prompt = st.chat_input("Votre message")
    if prompt:
        try:
            engine.submit(prompt, llm, embedder, threshold)
        except Exception as exc:  # affichage à l'utilisateur, pas de crash de l'app
            st.error(f"Échec de l'envoi : {exc}")


main()
