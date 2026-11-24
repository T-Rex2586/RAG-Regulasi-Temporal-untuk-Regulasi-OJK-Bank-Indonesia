from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from src.retrieval.chunker import Chunk


def chunk_id_to_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class DenseIndexer:
    def __init__(
        self,
        host: Optional[str] = "127.0.0.1",
        port: int = 6334,
        prefer_grpc: bool = True,
        qdrant_path: Optional[str | Path] = None,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        vector_dim: int = 384,
        threads: int = 8,
    ):
        self.model_name = model_name
        self.vector_dim = vector_dim
        self.host = host
        self.port = port
        self.prefer_grpc = prefer_grpc

        if host:
            self.client = QdrantClient(
                host=host,
                port=port,
                prefer_grpc=prefer_grpc,
                check_compatibility=False,
            )
        else:
            self.qdrant_path = Path(qdrant_path or "index/qdrant")
            self.qdrant_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(self.qdrant_path))

        self.embed_model = TextEmbedding(model_name, threads=threads)

    def create_collection(self, collection_name: str, recreate: bool = False) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if collection_name in collections:
            if recreate:
                self.client.delete_collection(collection_name)
            else:
                return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_dim,
                distance=models.Distance.COSINE,
            ),
        )

        filter_fields = [
            "doc_id",
            "regulator",
            "jenis",
            "nomor",
            "tahun",
            "status",
            "pasal_nomor",
            "bab_nomor",
            "strategy",
        ]
        for field in filter_fields:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

    def index_chunks(
        self,
        collection_name: str,
        chunks: List[Chunk],
        batch_size: int = 256,
        recreate: bool = False,
        desc: str = "Embedding & Indexing",
    ) -> int:
        self.create_collection(collection_name, recreate=recreate)

        existing_count = self.client.count(collection_name).count
        total_points = len(chunks)

        if existing_count >= total_points:
            print(f"Collection '{collection_name}' already indexed with {existing_count} points. Skipping.")
            return existing_count

        # Resume from last committed batch. Chunk order is deterministic (files sorted by name
        # in run_index.py), so existing_count is a valid offset into the chunks list.
        start_idx = 0 if recreate else existing_count
        if start_idx > 0:
            print(f"Resuming '{collection_name}' from offset {start_idx}/{total_points}...")

        for i in tqdm(range(start_idx, total_points, batch_size), desc=desc):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = list(self.embed_model.embed(texts, batch_size=batch_size))

            points = []
            for chunk_obj, emb in zip(batch, embeddings):
                pt_id = chunk_id_to_uuid(chunk_obj.chunk_id)
                points.append(
                    models.PointStruct(
                        id=pt_id,
                        vector=emb.tolist(),
                        payload=chunk_obj.to_dict(),
                    )
                )

            self.client.upsert(collection_name=collection_name, points=points)

        return total_points

    def search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        query_vector = list(self.embed_model.embed([query]))[0].tolist()

        qdrant_filter = None
        if filters:
            must_clauses = []
            for k, v in filters.items():
                if v is not None:
                    must_clauses.append(
                        models.FieldCondition(
                            key=k,
                            match=models.MatchValue(value=v),
                        )
                    )
            if must_clauses:
                qdrant_filter = models.Filter(must=must_clauses)

        search_result = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        results = []
        for pt in search_result.points:
            results.append({
                "chunk_id": pt.payload.get("chunk_id"),
                "score": float(pt.score),
                "metadata": pt.payload,
            })

        return results

    def close(self) -> None:
        self.client.close()
