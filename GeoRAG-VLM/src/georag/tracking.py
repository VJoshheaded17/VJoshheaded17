"""Append-only SQLite records with model settings and corpus fingerprint."""
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def fingerprint(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def log_run(path, kind, config, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, created_at TEXT, kind TEXT, config TEXT, records TEXT)")
        cursor = conn.execute("INSERT INTO runs(created_at, kind, config, records) VALUES (?, ?, ?, ?)",
                              (datetime.now(timezone.utc).isoformat(), kind,
                               json.dumps(config, sort_keys=True), json.dumps(records, ensure_ascii=False)))
        return cursor.lastrowid
