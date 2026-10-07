# Research roadmap

## V1: reproducible project foundation

Implemented: reusable caption/prompt modules, lexical and neural retrieval adapters, optional CLIP image indexing, conservative metadata handling, paired RAG inputs, overlap/retrieval metrics, manual claim summaries, tracking, sanitized provenance, tests and offline examples.

Pending validation: full model inference with the actual imagery, neural retrieval against a substantial corpus, new API credentials and authenticated runs, environment lockfile, independent factual annotations.

## V2: honest benchmark

Start with a small independently annotated set before growing toward 100–500 scenes. Build region/event splits, relevance labels and human reference descriptions. Compare identical-question RAG/no-RAG pairs and matched image-only/metadata VLM prompts. Report factual errors and unverifiable claims alongside text overlap.

## V3: remote-sensing retrieval

Compare generic CLIP, MiniLM, and RemoteCLIP on the same relevance benchmark. Spatial/time reranking currently exists as an isolated helper; expose it in CLI experiments and report its weights and candidate pool before claiming gains. SatCLIP or Clay integrations would require separate adapters and evaluation; they are not interchangeable drop-in text encoders.

## V4: temporal change analysis

Acquire co-registered imagery at multiple dates, control cloud cover and seasonality, create verified change labels, and test vegetation, water or built-up change. Do not infer a temporal change from unrelated captions. Implement a visual pair/change module and compare against a non-RAG baseline before describing the system as change detection.
