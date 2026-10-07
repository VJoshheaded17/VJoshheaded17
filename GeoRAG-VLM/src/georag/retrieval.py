"""Offline lexical baseline and explicit geo/time reranking."""
import math
import re
from collections import Counter
from datetime import date
from .data import validate_documents, validate_metadata


def tokens(text):
    return re.findall(r"\b\w+\b", text.lower())


class LexicalRetriever:
    """TF-IDF cosine baseline; this is not a neural embedding model."""
    def __init__(self, documents):
        self.documents = validate_documents(documents)
        counts = [Counter(tokens(d["caption"])) for d in documents]
        df = Counter(w for c in counts for w in c)
        self.idf = {w: math.log((1 + len(counts)) / (1 + n)) + 1 for w, n in df.items()}
        self.vectors = [self._vector(c) for c in counts]

    def _vector(self, counts):
        vector = {w: n * self.idf[w] for w, n in counts.items() if w in self.idf}
        norm = math.sqrt(sum(v * v for v in vector.values()))
        return {w: v / norm for w, v in vector.items()} if norm else {}

    def search(self, query, top_k=3, exclude_ids=()):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        q = self._vector(Counter(tokens(query)))
        excluded = set(exclude_ids)
        hits = [{"document": d, "score": sum(q.get(w, 0) * v for w, v in vec.items())}
                for d, vec in zip(self.documents, self.vectors) if d["id"] not in excluded]
        return sorted(hits, key=lambda h: (-h["score"], h["document"]["id"]))[:top_k]


def distance_km(lat1, lon1, lat2, lon2):
    a, b, c, d = map(math.radians, (lat1, lon1, lat2, lon2))
    x = math.sin((c-a)/2)**2 + math.cos(a)*math.cos(c)*math.sin((d-b)/2)**2
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1, max(0, x))))


def rerank(hits, query_metadata, text_weight=0.7, geo_weight=0.2, time_weight=0.1,
           geo_scale_km=100, time_scale_days=365):
    """Rerank candidates. Missing metadata contributes zero; no location is guessed."""
    weights = (text_weight, geo_weight, time_weight)
    if any(w < 0 for w in weights) or sum(weights) <= 0 or min(geo_scale_km, time_scale_days) <= 0:
        raise ValueError("Nonnegative weights and positive scales are required")
    validate_metadata(query_metadata)
    result = []
    for hit in hits:
        d = hit["document"]
        validate_metadata(d)
        geo = temporal = 0.0
        if all(x.get("latitude") is not None and x.get("longitude") is not None for x in (d, query_metadata)):
            km = distance_km(float(d["latitude"]), float(d["longitude"]),
                             float(query_metadata["latitude"]), float(query_metadata["longitude"]))
            geo = math.exp(-km / geo_scale_km)
        if d.get("date") and query_metadata.get("date"):
            days = abs((date.fromisoformat(d["date"]) - date.fromisoformat(query_metadata["date"])).days)
            temporal = math.exp(-days / time_scale_days)
        text = max(0, min(1, hit["score"]))
        score = (text_weight*text + geo_weight*geo + time_weight*temporal) / sum(weights)
        result.append({**hit, "score": score, "components": {"text": text, "geo": geo, "time": temporal}})
    return sorted(result, key=lambda h: (-h["score"], h["document"]["id"]))
