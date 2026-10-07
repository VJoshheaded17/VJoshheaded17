# Dataset setup

The default `digits` demo uses scikit-learn's bundled 1,797 handwritten digit
images (8×8 pixels, values 0–16). No dataset download is performed by the demo.
This dataset is different from MNIST. See the
[official dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

For the original coursework dataset, place your authorized copy of
`mnist_all.mat` here and run `train.py --dataset mnist`. The loader expects
`train0` through `train9` and `test0` through `test9`, each a two-dimensional
matrix with one flattened image per row and pixels in 0–255. All matrices must
have the same number of columns. Standard 28×28 MNIST has 784 input columns.
Each training class must contain more than 1,000 images because 1,000 per class
are reserved for validation; the supplied test partition is preserved.

The supplied ZIP did not include `mnist_all.mat`; this repository does not
redistribute that course data or claim a verified MNIST accuracy.
