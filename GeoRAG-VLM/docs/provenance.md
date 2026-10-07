# Notebook migration

The four uploaded notebooks are the source of this release. Refactored modules and the offline fixture baseline are newly added implementation, not a claim that all features were present in the historical experiments.

| Original notebook | Evidence preserved | Maintained replacement |
|---|---|---|
| `3_10_Ved (1).ipynb` | SQLite logging, Kosmos prompts, ten CLIP caption pairs, FAISS caption search | `caption.py`, `prompts.py`, `tracking.py`, `embeddings.py`, `vector_store.py` |
| `RAG_with_deepseek_Ved (2).ipynb` | MiniLM/Qdrant retrieval and DeepSeek calls | `vector_store.py`, `rag.py`, walkthrough 03 |
| `Copy_of_RAG_with_deepseek_Ved (4).ipynb` | Repeated RAG/no-RAG iterations | Consolidated `paired_comparison`, walkthrough 04 |
| `RAG_vs_No_RAG (1).ipynb` | Two caption documents and saved BLEU output | `legacy_captions.jsonl`, `legacy_observation.json`, controlled evaluation CLI |

Removed from maintained code: duplicate definitions, notebook globals, hard-coded `/content` paths, missing `ground_truth`, destructive collection recreation, raw API response printing, token-download/removal workarounds, and unused dependencies. Payload fields consistently use `caption`; original cells sometimes expected nonexistent `response` fields.

Filename parsing now recognizes dates adjacent to `Z` or `openEO-`. It only proposes unverified date/index hints. It does not invent a sensor, location, or coordinate sign. Index interpretation does not assume a color palette.

The original FAISS experiment computed image vectors but indexed text vectors. Optional image indexing in v1 is a new adapter, not a historical result. Original notebook comments refer to a collaborator's vectorization approach; source attribution is retained in the archive. This release assigns no sole-credit claim or blanket license over collaborators' work.

The archives preserve sanitized source and markdown. All outputs, execution counts, attachments, and nonessential metadata were discarded to prevent publishing secrets or embedded data. Reset-collection cells are disabled. These archives remain historically stateful and are not supported executable pipelines.
