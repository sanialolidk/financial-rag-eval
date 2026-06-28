"""ChromaDB vector store for dense retrieval."""

from __future__ import annotations

from typing import Any

import chromadb

from app.config import CHROMA_DIR, settings
from app.models import Chunk
from app.retrieval.embeddings import embed_texts


class ChromaStore:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        name = settings.chroma_collection
        try:
            self._client.delete_collection(name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            texts = [c.text for c in batch]
            embeddings = embed_texts(texts)
            metadatas = [_flatten_metadata(c.metadata) for c in batch]
            self._collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

    def query(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        from app.retrieval.embeddings import embed_query

        if self.count == 0:
            return []
        q_emb = embed_query(query)
        results = self._collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k, self.count),
            include=["documents", "metadatas", "distances"],
        )
        out: list[tuple[Chunk, float]] = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            # cosine distance → similarity
            score = 1.0 - float(dist)
            out.append((Chunk(chunk_id=cid, text=doc, metadata=meta), score))
        return out


def _flatten_metadata(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Chroma metadata must be scalar types."""
    flat: dict[str, str | int | float | bool] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            flat[k] = v
        else:
            flat[k] = str(v)
    return flat