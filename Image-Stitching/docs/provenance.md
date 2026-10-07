# Original submission and portfolio refactor

The uploaded `stitching.py` exactly matches the file inside `submission_vedjoshi.zip`. The uploaded `task2.json` also matches its ZIP entry. No original work has been overwritten.

`archive/` contains the original implementation and scaffold files: `stitching.py`, `task1.py`, `task2.py`, `utils.py`, `pack_submission.sh`, classroom `README.md`, and the saved `task2.json`. `results/original/` contains the actual Task 1 and Task 2 PNGs from the submitted ZIP.

The original Task 1 function aligns image 2 into image 1 and averages overlaps. Its result has visible translucency/ghosting; this is not demonstrated foreground elimination. The original Task 2 function estimates A→B transforms but sometimes applies them as B→A; it also assumes direct overlap with a fixed reference and substitutes identity transforms for unsuccessful matches. These behaviors can produce incorrect alignment and overlap.

The handout requests a binary N×N overlap array. The original saved JSON is fractional and asymmetric. It is preserved under `archive/task2.json` as a historical output, not relabeled as a correct binary result.

The maintained version was refactored for this GitHub portfolio: correct transform convention, graph-based composition, failed-match rejection, independent support masks, inclusive bounds, vectorized blending, binary overlap and deterministic seeded runners. Root-level helper scripts are intentionally different from archived classroom scaffolding; this is not a drop-in graded submission claim.

`docs/asset_manifest.json` records the original ZIP path, repository location, size and SHA-256 of all included PNGs. Input pixels and original result pixels are unchanged. New corrected PNGs, when generated, are placed under `results/corrected/` and described in a separate validation record.

The original BSD license and copyright notice were copied unchanged. Third-party imagery is attributed to the supplied course bundle without inventing a separate image license. No grade, accuracy score, or robustness benchmark was provided or inferred.

## Image transport

The connected GitHub binary-blob operation failed, including for a small text probe. The published package therefore stores byte-exact PNG payloads as base64 plus SHA-256 in `assets/`, with `prepare_assets.py` restoring the documented PNG paths after cloning. The three README SVG previews contain the original PNG bytes as embedded raster images; they are transport containers, not new image generation. Original and corrected PNG outputs remain separately identified.
