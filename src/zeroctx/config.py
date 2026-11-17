"""Configuration lue depuis l'environnement et le fichier `.env` à la racine du projet."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "zeroctx.sqlite3"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # une variable vide (copie de .env.example) retombe sur le défaut
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        # Le .env prime sur les variables du terminal (une vieille clé exportée ne doit pas écraser le projet).
        return init_settings, dotenv_settings, env_settings, file_secret_settings

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_embedding_model: str = "gemini-embedding-002"
    gemini_embedding_dimension: int = 1536

    # Cohere
    cohere_api_key: str = ""
    cohere_model: str = "command-a-plus-05-2026"
    cohere_embedding_model: str = "embed-v5.0-pro"
    cohere_embedding_dimension: int = 2048

    # Chunking (en caractères)
    chunk_size: int = 512
    chunk_overflow_ratio: float = 0.10

    # Recherche
    rrf_k: int = 60
    search_candidates: int = 20
    context_chunks: int = 8
    context_char_budget: int = 12_000
    context_min_similarity: float = (
        0.6  # cosinus minimal d'un chunk vectoriel (Gemini : pertinent ~0.85, hors sujet ~0.57)
    )
    context_min_bm25: float = 1.0  # pertinence BM25 minimale d'un chunk trouvé seulement par mots-clés
    context_relative_score: float = 0.5  # score RRF minimal, en fraction du meilleur résultat

    # Steering
    steering_threshold: float = 0.6


@lru_cache
def get_settings() -> Settings:
    return Settings()
