# Benchmark data

## WDBC

The default KNN run uses the bundled
[scikit-learn Wisconsin Diagnostic Breast Cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html):
569 rows, 30 numeric features, with labels 0 = malignant and 1 = benign.
This avoids a dataset download at run time. The user's course `wdbc.csv` was not
attached, so exact equivalence of its row order/label coding has not been verified.

To use your course file instead, pass `--wdbc datasets/wdbc.csv` or another path.
The supported format has no header, 30 numeric feature columns, then a 0/1 label.
For custom files the code does not infer medical label meanings from the numbers.

## Car Evaluation

`car.data` and `car.names` are unmodified files from the official UCI download.
They include 1,728 rows, six categorical inputs and four classes. The default
tree run uses this checked-in data, so it also runs without a dataset fetch.

**Attribution:** Bohanec, M. (1988). *Car Evaluation* [Dataset]. UCI Machine
Learning Repository. https://doi.org/10.24432/C5JP48.

**License:** [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Dataset page: https://archive.ics.uci.edu/dataset/19/car+evaluation.
[`source.json`](source.json) records the download URL, license, sizes and hashes.
The distribution includes the original `car.names` documentation; the current
UCI dataset page supplies the attribution/license information above.

For a course CSV, pass `--car datasets/car.csv`. It must have six categorical
feature columns followed by a label. A header is recognized when its final
column is `class`, `label`, `quality` or `target`. Labels `unacceptable`,
`acceptable`, `good` and `very_good` are mapped to `unacc`, `acc`, `good` and
`vgood`; canonical UCI labels also work. Feature values remain unchanged.
The original course `car.csv` was not attached.
