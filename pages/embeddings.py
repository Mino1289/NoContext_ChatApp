"""Visualisation 2D (PCA) des embeddings stockés dans la base SQLite."""

import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from zeroctx.config import DB_PATH
from zeroctx.store import Store


@st.cache_resource
def get_store() -> Store:
    return Store(DB_PATH)


def vector_tables(store: Store) -> dict[str, int]:
    """Nom de chaque table vec0 -> dimension des vecteurs."""
    rows = store.conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND sql LIKE 'CREATE VIRTUAL TABLE%vec0%'"
    ).fetchall()
    tables: dict[str, int] = {}
    for name, sql in rows:
        match = re.search(r"float\[(\d+)\]", sql)
        if match:
            tables[name] = int(match.group(1))
    return tables


def load_vectors(store: Store, table: str, dimension: int) -> pd.DataFrame:
    rows = store.conn.execute(f"SELECT rowid, embedding FROM {table}").fetchall()
    if not rows:
        return pd.DataFrame(columns=["id", "role", "content", "x", "y"])
    ids = [int(r[0]) for r in rows]
    matrix = np.array([np.frombuffer(r[1], dtype=np.float32) for r in rows]).reshape(len(rows), dimension)

    marks = ",".join("?" * len(ids))
    chunks = pd.DataFrame(
        store.conn.execute(f"SELECT id, role, content FROM chunks WHERE id IN ({marks})", ids).fetchall(),
        columns=["id", "role", "content"],
    )
    df = pd.DataFrame({"id": ids}).merge(chunks, on="id", how="left")

    # PCA : projection sur les trois axes de plus grande variance.
    centered = matrix - matrix.mean(axis=0)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ components[:3].T
    df["x"], df["y"], df["z"] = coords[:, 0], coords[:, 1], coords[:, 2]
    df.attrs["explained"] = float((singular[:3] ** 2).sum() / (singular**2).sum())
    return df


st.title("Embeddings")
store = get_store()
tables = vector_tables(store)

if not tables:
    st.info("Aucun index vectoriel dans la base pour le moment.")
    st.stop()

table = st.selectbox("Index", list(tables), format_func=lambda t: f"{t} ({tables[t]} dimensions)")
df = load_vectors(store, table, tables[table])

if len(df) < 3:
    st.info(f"Pas assez de chunks vectorisés pour une projection ({len(df)} trouvés).")
    st.stop()

st.caption(
    f"{len(df)} chunks · PCA 3D : {df.attrs['explained']:.0%} de la variance conservée. "
    "Une projection 3D perd encore de l'information : les voisins proches sont fiables, les distances lointaines moins."
)
df["extrait"] = df["content"].fillna("").str.slice(0, 120)
fig = px.scatter_3d(
    df, x="x", y="y", z="z", color="role", hover_data={"id": True, "extrait": True, "x": False, "y": False, "z": False}
)
fig.update_traces(marker={"size": 4})
fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=600)
st.plotly_chart(fig, width="stretch")
st.dataframe(df[["id", "role", "content"]], width="stretch", hide_index=True)
