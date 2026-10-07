"""FAISS and Qdrant adapters with the same document payload and exclusion rules."""
import os
import uuid
from .data import validate_documents


class VectorRetriever:
    def __init__(self, documents, encoder, backend="faiss", representation="text", image_root="."):
        import numpy as np
        from pathlib import Path
        self.documents = validate_documents(documents)
        self.encoder, self.backend = encoder, backend
        if representation == "text":
            vectors = encoder.texts([d["caption"] for d in documents])
        elif representation == "image" and hasattr(encoder, "images"):
            vectors = encoder.images([Path(image_root) / d["image_path"] for d in documents])
        else:
            raise ValueError("Image retrieval requires CLIP and an image_path for each document")
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[0] != len(documents) or not np.isfinite(vectors).all():
            raise ValueError("Invalid vector matrix")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("Zero vector cannot be indexed")
        vectors = vectors / norms
        self.dimension = vectors.shape[1]
        if backend == "faiss":
            import faiss
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(vectors)
        elif backend == "qdrant":
            from qdrant_client import QdrantClient, models
            url = os.environ.get("QDRANT_URL")
            self.client = (QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"), timeout=30)
                           if url else QdrantClient(":memory:"))
            # Unique collection: existing collections are never deleted or recreated.
            self.collection = "georag_" + uuid.uuid4().hex
            self.client.create_collection(self.collection, vectors_config=models.VectorParams(
                size=self.dimension, distance=models.Distance.COSINE))
            self.client.upsert(self.collection, points=[models.PointStruct(
                id=i, vector=v.tolist(), payload=d) for i, (d, v) in enumerate(zip(documents, vectors))], wait=True)
        else:
            raise ValueError("Choose faiss or qdrant")

    def search(self, query, top_k=3, exclude_ids=()):
        import numpy as np
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        q = np.asarray(self.encoder.texts([query]), dtype="float32")
        norm = np.linalg.norm(q)
        if q.shape != (1, self.dimension) or not np.isfinite(q).all() or norm == 0:
            raise ValueError("Invalid query vector")
        q = q / norm
        excluded = set(exclude_ids)
        # Request extra hits for exclusions without filling beyond the corpus.
        limit = min(len(self.documents), top_k + len(excluded))
        if self.backend == "faiss":
            scores, ids = self.index.search(q, limit)
            hits = [{"document": self.documents[int(i)], "score": float(s)}
                    for i, s in zip(ids[0], scores[0]) if i >= 0]
        else:
            hits = [{"document": h.payload, "score": float(h.score)} for h in
                    self.client.query_points(collection_name=self.collection, query=q[0].tolist(), limit=limit).points]
        return [h for h in hits if h["document"]["id"] not in excluded][:top_k]

    def close(self):
        if self.backend == "qdrant":
            self.client.close()
