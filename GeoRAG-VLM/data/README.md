# Data and provenance

`sample/toy_corpus.jsonl` contains six handwritten synthetic descriptions, not satellite-derived observations. `sample/toy_queries.jsonl` provides four simple relevance fixtures. Results on them validate plumbing only.

`sample/legacy_captions.jsonl` contains 12 literal caption records extracted without executing notebook code: ten paired image-name/caption records from the CLIP notebook and two Qdrant documents from the comparison notebook. Captions can contain hallucinations, incorrect geography, or unsupported object claims. They must not serve as ground truth. Source notebook and zero-based cell index are recorded. No original pixels are bundled.

Copy `scene_manifest.example.jsonl` to an ignored private path and replace its image path and metadata. Each scene needs a unique string `id`, `image_path`, and optional `location`, ISO `date`, paired signed `latitude`/`longitude`, `sensor`, `representation`, and positive `resolution_m`. Keep unknown values null. Do not infer coordinates from unsigned filenames.

A retrieval corpus uses a unique string `id`, nonempty `caption`, `caption_source`, and `verified` flag, plus optional metadata/image path. References belong in a separate query file with `id`, `question`, `reference`, optional `target`, relevance IDs, and exclusions. Never mark a generated caption verified unless a human has checked it against independent evidence.

Keep raw/private data under ignored `data/raw/` or `data/private/`. Record acquisition provenance, band resolution, rendering/resampling, cloud masks, crop bounds, index definitions and legends. Confirm redistribution rights before adding imagery to the public repository. Do not treat resampled image size as native spatial resolution.
