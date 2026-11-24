from pathlib import Path
import pickle
import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from src.retrieval.chunker import Chunk


def tokenize_indonesian(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"\b[a-z0-9\-_/]+\b", text)
    return tokens


class BM25Index:
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: List[str] = []
        self.chunks_data: List[Dict[str, Any]] = []

    def build(self, chunks: List[Chunk]) -> None:
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.chunks_data = [c.to_dict() for c in chunks]
        corpus_tokens = [tokenize_indonesian(c.text) for c in chunks]
        self.bm25 = BM25Okapi(corpus_tokens)

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "chunk_ids": self.chunk_ids,
                    "chunks_data": self.chunks_data,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, filepath: str | Path) -> "BM25Index":
        index = cls()
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            index.bm25 = data["bm25"]
            index.chunk_ids = data["chunk_ids"]
            index.chunks_data = data["chunks_data"]
        return index

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.bm25:
            return []

        query_tokens = tokenize_indonesian(query)
        scores = self.bm25.get_scores(query_tokens)

        # Apply metadata filters if provided
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked_indices:
            if len(results) >= top_k:
                break
            score = float(scores[idx])
            if score <= 0.0:
                continue

            chunk_meta = self.chunks_data[idx]

            if filters:
                match = True
                for k, v in filters.items():
                    if v is not None and chunk_meta.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            results.append({
                "chunk_id": self.chunk_ids[idx],
                "score": score,
                "metadata": chunk_meta,
            })

        return results
