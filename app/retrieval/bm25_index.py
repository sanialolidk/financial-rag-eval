"""BM25 sparse index persisted alongside Chroma."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.config import BM25_DIR
from app.models import Chunk


def _tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self) -> None:
        self.chunk_ids: list[str] = []
        self.chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.chunk_ids = [c.chunk_id for c in chunks]
        corpus = [_tokenize_for_bm25(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        if not self._bm25 or not self.chunks:
            return []
        tokens = _tokenize_for_bm25(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.chunks[i], float(s)) for i, s in ranked if s > 0]

    def save(self, path: Path | None = None) -> Path:
        path = path or (BM25_DIR / "bm25_index.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        meta_path = path.with_suffix(".meta.json")
        payload = {
            "chunk_ids": self.chunk_ids,
            "chunks": [
                {"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata}
                for c in self.chunks
            ],
        }
        meta_path.write_text(json.dumps(payload), encoding="utf-8")
        with path.open("wb") as f:
            pickle.dump(self._bm25, f)
        return path

    def load(self, path: Path | None = None) -> bool:
        path = path or (BM25_DIR / "bm25_index.pkl")
        meta_path = path.with_suffix(".meta.json")
        if not path.exists() or not meta_path.exists():
            return False
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        self.chunk_ids = payload["chunk_ids"]
        self.chunks = [
            Chunk(chunk_id=c["chunk_id"], text=c["text"], metadata=c["metadata"])
            for c in payload["chunks"]
        ]
        with path.open("rb") as f:
            self._bm25 = pickle.load(f)
        return True