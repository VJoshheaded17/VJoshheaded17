# Source and maintenance history

This is a portfolio adaptation of Ved Joshi's submitted `proj1.zip` coursework.
The unchanged supplied `proj1/code/nnScript.py` is retained as
[`archive/nnScript_original.py`](../archive/nnScript_original.py).
The source uses an assignment scaffold; its comments and structure are preserved.
The maintained version documents the subsequent fixes rather than presenting
the entire scaffold as independently authored.

[`source_manifest.json`](source_manifest.json) records SHA-256 hashes and sizes
for the original ZIP and its three files. Only the original Python script is
published from that archive. The source video `demo.mp4.mp4` (62,192,176 bytes)
and `params.pickle` (2,437,155 bytes) are not included in this GitHub project.
The pickle has not been loaded, its contents and metrics have not been verified,
and it is not used for the published demonstration. New checkpoints use arrays
in NPZ format and are loaded with `allow_pickle=False`.

## Corrections in the maintained implementation

- Normalize MNIST pixels by 255; use the bundled demo's documented scale of 16.
- Keep feature selection fitted exclusively on the training partition, and
  persist the mask and pixel scale alongside model weights.
- Evaluate binary cross entropy from logits using `logaddexp`, avoiding
  `log(0)` when sigmoid outputs saturate. Use stable `expit` for sigmoid.
- Add the non-bias L2 derivatives to both weight gradients. The supplied script
  adds regularization to the loss but omits these derivatives; its main program
  sets lambda to zero, so that discrepancy becomes active when lambda is changed.
- Use seeded initialization and dataset partitions, vectorized preprocessing,
  explicit label/shape checks and a training CLI that does not execute on import.
- Predict using logit argmax to avoid saturation ties, while preserving the
  original independent-sigmoid output architecture.
- Preserve optimizer status, accepted-iteration loss, split provenance, test
  predictions and confusion counts instead of reporting an accuracy alone.

The forward pass, analytic backpropagation, bias convention, SciPy conjugate
gradient approach and classification problem follow the original coursework.
No autodifferentiation or neural-network training library is used. SciPy supplies
the optimizer; scikit-learn supplies the optional demo data only.

## Verification scope

The published metrics are from the maintained version on bundled 8×8 digits.
They are not a reconstruction of the original MNIST submission or video.
The original script and checkpoint were not run. The MNIST loader was tested
with a small MAT fixture matching the expected course format, not full MNIST.
The 12 checks cover numerical gradients, bias regularization, extreme logits,
deterministic training/splits, partition separation, shape validation and
checkpoint inference. Subsequent portfolio maintenance is distinguishable from
the archived coursework submission.
