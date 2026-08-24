# Gaussian Mixture Models for Generative Classification (MNIST)

A from-scratch implementation of Gaussian generative classifiers on MNIST, built up in stages: single Gaussian → full-covariance Gaussian → class-conditional Gaussian Mixture Models trained with EM. Done as a course project for CS771 (Machine Learning) at IIT Kanpur.

## What's in here

The core idea is generative classification: instead of learning a decision boundary directly, model `p(x | y = c)` for each class and use Bayes' rule to classify. The notebook builds this up step by step:

- **Single standard Gaussian** — just the mean, covariance fixed to identity. Barely usable, samples are noisy blobs.
- **Full-covariance Gaussian** — learning the actual covariance matrix per class massively improves sample quality, since it captures how pixels co-vary (e.g. an upright vs. inverted 7 — the covariance is what lets a single Gaussian tell them apart without a mixture).
- **Gaussian Mixture Model (K components per class)** — fit with EM, since the mixture assignment (which "style" of digit a sample belongs to) is a latent variable. K-means++ is used for initialization to avoid degenerate EM runs.
- **Classification** — a GMM is fit per class (0-9) on MNIST, then test points are classified via Bayes' rule using the learned class-conditional densities.
- **Missing-feature inference** — the generative model's other superpower: given a test image with a block of pixels censored out, it can still classify using the observed pixels, and reconstruct the missing region using the Gaussian conditional distribution.

## Some results

- Single full-covariance Gaussian per class (no mixture): **~85.7%** test accuracy on MNIST — respectable for something this simple, though nowhere near a CNN.
- With ~21% of the central pixels censored, classification accuracy drops to **~78.6%**, and the model can still reconstruct a plausible version of the missing region using `Σ_missing,observed Σ_observed^-1 (x_observed - μ_observed)`.
- Accuracy vs. number of mixture components K was swept over `[1, 2, 3, 5, 10, 15, 20]` to see where the mixture starts overfitting / running out of data per component.

## Implementation notes

- EM is implemented manually: E-step computes responsibilities `γ_nk`, M-step updates `π_k`, `μ_k`, `Σ_k` — no `sklearn.mixture` used.
- Covariance matrices get a small `εI` regularization term added each M-step, otherwise they go singular pretty fast at 784 dimensions with limited samples per component.
- Matrix inversions in the classification/reconstruction step use `pinv` instead of `inv` since class covariances are often ill-conditioned or singular in this regime.
- K-means++ initialization (`doKMPPInitGreedy`) is used to seed the mixture means before running EM — random initialization tended to collapse components onto each other.

## Running it

1. Download MNIST from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data) and extract into a folder named `mnist/` (needs `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`).
2. Make sure `mnist/`, the `cs771/` helper module (contains `plotData.py` and `utils.py`), and the notebook all sit in the same parent directory.
3. Open `corrected_minor_2.ipynb` and run top to bottom.

## Files

- `corrected_minor_2.ipynb` — the whole implementation and experiments
- `CS771_Minor_Assignment_2.pdf` — write-up with the EM derivation (E-step, M-step updates for π/μ/Σ) and the math behind the classification and reconstruction steps

## Team

Group project for CS771, IIT Kanpur — Khushal Nikam, Prafull Joshi, Sarthak Dumbre, Vishal Junjare.

MNIST loader code adapted from [Hojjat Khodabakhsh's Kaggle notebook](https://www.kaggle.com/code/hojjatk/read-mnist-dataset).
