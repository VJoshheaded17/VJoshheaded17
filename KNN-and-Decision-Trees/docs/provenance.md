# Coursework source and portfolio maintenance

Source files supplied by Ved Joshi:

- [`archive/knn.py`](../archive/knn.py): original KNN implementation and experiments.
- [`archive/decision.py`](../archive/decision.py): original entropy/Gini trees and experiments.
- [`archive/HW1.tex`](../archive/HW1.tex): unfilled COMPSCI 589 Fall 2026 assignment
  template, containing course instructions and questions. It is not a completed
  report. The two example figure PDFs it references were not supplied, so the
  template is retained as source context rather than a compiled report.

These copies preserve the exact supplied bytes, including line endings.
[`source_manifest.json`](source_manifest.json) records their SHA-256 hashes.
The maintained code and newly generated benchmark plots/results are separate
from the archived coursework. No license is invented for the course template;
its instructor-authored instructions are distinguished from the user's code.

## Corrections

The original KNN normalizes training and test data independently within the
repeated experiments; its initial one-off split normalizes only training data.
The maintained version fits min/max values on the training partition and
applies those values to every query, with safe denominators for constant
features. It supports explicit self-exclusion by row index, multiclass voting,
deterministic ties, batched distance calculations and import-safe modules.

The original entropy tree emits the literal label `"majority"` for empty
branches, and the Gini tree also emits it when attributes are exhausted. The
maintained tree emits an actual training majority class. Each node stores its
local majority so categories absent from training do not cause prediction
errors. Training uses only values observed in the current node, avoiding
fabricated empty branches. Entropy and Gini gains use the same recursive builder.

The source's 85% purity heuristic is commented out and unfinished. It is
implemented here as **subsequent portfolio maintenance**, and its results are
reported as an additional variant rather than a feature of the original code.
The maintained code also supports a depth limit; that option is not varied in
the published benchmarks.

## Evaluation changes

- Seeded, stratified 80/20 outer splits replace the unseeded splits.
- Every k value and normalized/raw KNN variant shares an outer split; every
  tree variant shares an outer split. This makes comparisons paired.
- KNN's k selection uses an inner validation partition of outer training data.
  The outer test sweep remains exploratory and is not used for selection.
- Resubstitution and query-excluded training accuracy are recorded separately.
  The query-excluded diagnostic shares the fitted training scaler and is not
  presented as an independent cross-validation estimate.
- Recorded metrics, per-run measurements and split hashes replace console-only
  output. Dataset sources and dependency versions are recorded.

The course datasets and original saved plots/report were not supplied. The new
runs use bundled scikit-learn WDBC and official UCI Car Evaluation. Consequently
these results do not reconstruct the original homework submission. Algorithms
and impurities are implemented here; scikit-learn is used only for dataset
loading and splitting, not its estimators or preprocessing implementations.

The 15 tests cover normalization reuse, constant columns, voting/ties,
duplicate-neighbor exclusion, categorical gains, majority fallback, XOR,
stopping rules, dataset loaders and deterministic paired experiments.
