# zeroctx — chat sans historique

Chat Streamlit où le modèle ne voit **jamais l'historique** de la conversation. Chaque réponse est construite à partir :

1. des messages **en attente** du sujet (ceux auxquels la réponse doit répondre) ;
2. d'un **contexte récupéré** dans la base des messages passés (recherche hybride : BM25 + vecteurs).

Le fil reste ainsi court et le coût en tokens ne croît pas avec la conversation.

## Fonctionnement

- **Sujets.** Chaque message est embeddé et comparé au centroïde des sujets qui ont une génération en cours. Si la similarité cosinus dépasse `STEERING_THRESHOLD` (0.6 par défaut), le message est un **steering** : la génération en cours est annulée, sa réponse partielle est conservée, et une nouvelle génération répond à tous les messages en attente. Sinon, un nouveau sujet démarre.
- **Génération en parallèle.** Chaque sujet a au plus une génération active ; plusieurs sujets peuvent générer en même temps.
- **Ingestion.** Messages utilisateur et réponses sont découpés en chunks (1024 caractères), indexés en FTS5 et vectorisés avec sqlite-vec. Un ré-embedding de fond complète les chunks manquants si l'embedder change.
- **Recherche.** Fusion RRF des résultats BM25 et vectoriels, puis découpe au budget de caractères (12 000 par défaut).
- **Fournisseurs.** Génération : Gemini ou Cohere. Embeddings : Gemini ou Cohere. Une table vectorielle distincte est créée par couple (fournisseur, modèle, dimension) : les espaces ne sont jamais mélangés.
- **Debug.** Chaque réponse affiche le prompt exact envoyé au modèle (bloc « Prompt envoyé »).

## Structure

```
app.py                  Interface Streamlit (seule la UI)
src/zeroctx/
  engine.py             Boucle asyncio, sujets, steering, génération, ingestion
  router.py             Décision steering / nouveau sujet (cosinus, sans appel LLM)
  retrieval.py          Sélection du contexte, construction du prompt
  store.py              SQLite : messages, chunks, FTS5, sqlite-vec
  chunking.py           Découpage des textes
  config.py             Réglages (variables d'environnement et .env)
  providers/            Clients Gemini et Cohere (génération + embeddings)
tests/                  Tests pytest
data/                   Base SQLite locale (créée automatiquement, ignorée par git)
```

## Prérequis

- [uv](https://docs.astral.sh/uv/)
- Python 3.14 (uv l'installe automatiquement si besoin, cf. `.python-version`)
- Une clé API Gemini et/ou Cohere

## Installation

```bash
uv sync
cp .env.example .env
```

Renseigne ensuite au moins une clé dans `.env` :

```env
GEMINI_API_KEY=...
COHERE_API_KEY=...
```

Les autres variables sont optionnelles : si elles sont vides, les valeurs par défaut de `src/zeroctx/config.py` s'appliquent.

## Lancer l'app

```bash
uv run streamlit run app.py
```

Puis ouvre http://localhost:8501.

## Tests

```bash
uv run pytest
```

Lint et format (optionnel) :

```bash
uv run ruff check . && uv run ruff format .
```

## Configuration

| Variable | Défaut | Rôle |
|---|---|---|
| `GEMINI_API_KEY` / `COHERE_API_KEY` | vide | Clés API |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Modèle de génération Gemini |
| `COHERE_MODEL` | `command-a-plus-05-2026` | Modèle de génération Cohere |
| `GEMINI_EMBEDDING_MODEL` / `_DIMENSION` | `gemini-embedding-002` / `1536` | Embeddings Gemini |
| `COHERE_EMBEDDING_MODEL` / `_DIMENSION` | `embed-v5.0-pro` / `2048` | Embeddings Cohere |

Les réglages de chunking, de recherche (`rrf_k`, `search_candidates`, `context_chunks`, `context_char_budget`) et le seuil de steering (`steering_threshold`) se modifient dans `src/zeroctx/config.py`.

