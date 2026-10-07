"""Explicit scene metadata and portable JSONL records."""
import json
import re
from datetime import date
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_documents(documents):
    if not documents:
        raise ValueError("The corpus must contain at least one document")
    seen = set()
    for doc in documents:
        if not isinstance(doc.get("id"), str) or not doc["id"] or doc["id"] in seen:
            raise ValueError("Each document needs a unique nonempty string id")
        seen.add(doc["id"])
        if not isinstance(doc.get("caption"), str) or not doc["caption"].strip():
            raise ValueError("Each document needs a nonempty caption")
        validate_metadata(doc)
    return documents


def validate_metadata(record):
    lat, lon = record.get("latitude"), record.get("longitude")
    if (lat is None) != (lon is None):
        raise ValueError("Latitude and longitude must be supplied together")
    if lat is not None and not (-90 <= float(lat) <= 90 and -180 <= float(lon) <= 180):
        raise ValueError("Invalid geographic coordinates")
    if record.get("date"):
        date.fromisoformat(record["date"])
    if record.get("resolution_m") is not None and float(record["resolution_m"]) <= 0:
        raise ValueError("Resolution must be positive")


def filename_metadata(filename):
    """Suggest date/index only. Do not infer signed coordinates or sensor."""
    name = Path(filename).name
    match = re.search(r"(\d{4}-\d{2}-\d{2})Z?", name)
    acquired = match.group(1) if match else None
    if acquired:
        date.fromisoformat(acquired)
    focus = next((x.upper() for x in ("ndvi", "fmr", "ndbi", "ndwi") if x in name.lower()), "unknown")
    return {"date": acquired, "representation": focus, "location": None,
            "latitude": None, "longitude": None, "sensor": None, "resolution_m": None,
            "metadata_status": "filename_hint_unverified"}
