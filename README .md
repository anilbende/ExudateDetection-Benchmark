# Retinal Exudate Detection: Lightweight CNNs from Scratch vs. Handcrafted-Feature ML

Reproduction code for:

> **Lightweight Deep Learning Trained from Scratch Versus Handcrafted-Feature
> Machine Learning for Retinal Exudate Detection: A Resource-Constrained
> Comparative Benchmarking Study**

[![CI](https://github.com/ORG/retinal-exudate-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/ORG/retinal-exudate-benchmark/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## What this study asks

Most published deep learning for diabetic retinopathy assumes two things:
plenty of annotated images, and ImageNet-pretrained weights to fine-tune. Many
research groups have neither — a single-hospital archive of a few hundred
images, and a machine that cannot reach the internet to download checkpoints.

This study asks what happens in that setting. It compares two lightweight CNNs
(**MobileNetV3-Small** and **-Large**, trained from *random* initialisation)
against three classical classifiers on **handcrafted features** (HOG, LBP,
colour histograms), on the 169-image HEI-MED dataset.

**The finding:** classical machine learning won, clearly. Both CNNs failed to
exceed chance-level discrimination and collapsed to predicting the positive
class for every test image, while logistic regression reached 88.5% accuracy on
the held-out split. Cross-validation over all 169 images then showed the three
classical classifiers were statistically indistinguishable from one another —
so the headline "logistic regression is best" result does not survive contact
with a second data partition. Both results are reported.

This repository reproduces all of it: every table, every figure, every
confidence interval and every significance test.

---

## Reproduction status

The statistical machinery in this repository has been verified to reproduce the
published values **exactly**. These are asserted by the test suite, so a change
that breaks any of them fails CI.

| Published value | Source | Status |
|---|---|---|
| Feature dimensionality 3,042 (LR: 3,043 parameters) | Methods | 2,916 + 26 + 96 + 4 ✅ |
| HOG spatial grid 9 × 9 blocks | Methods | ✅ |
| Split 118 / 25 / 26 at 32.2% / 32.0% / 30.8% positive | Methods | ✅ exact |
| Wilson intervals — all 8 in Table 1 | Table 1 | ✅ exact to 3 d.p. |
| Hanley–McNeil AUC intervals — all 5 | Table 1 | ✅ exact to 3 d.p. |
| Hanley–McNeil SE = 0.041, half-width ±0.08 | Methods | ✅ |
| McNemar statistics 0.000 / 0.042 / 0.308 | Table 4 | ✅ all three |
| McNemar p-values 1.000 / 0.839 / 0.581 | Table 4 | ✅ all three |
| Detectable ΔAUC ≈ 0.09 (range 0.075–0.118) | Methods | ✅ 0.089 (0.073–0.115) |
| Chance baseline: Dice 0.144, IoU 0.081, overlap 20.0% | Table 5 | ✅ analytic **and** permutation |

Two facts about the published analysis were recovered by reproducing these
numbers, and are documented rather than left implicit:

1. **Table 4 uses Edwards' continuity correction.** The reported statistic of
   exactly `0.000` for discordant pairs (12, 13) is unobtainable without it —
   the uncorrected value is 0.040, giving p = 0.841 rather than 1.000.
2. **The "16 held-out exudate-positive images" are validation + test.** The test
   set contains only 8 positives; the stratified split places 8 in validation
   and 8 in test. Running stage 01 confirms this exactly.

See [`docs/MANUSCRIPT_MAP.md`](docs/MANUSCRIPT_MAP.md) for the reasoning.

---

## Quick start

### Install

```bash
git clone https://github.com/ORG/retinal-exudate-benchmark.git
cd retinal-exudate-benchmark

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Or with conda: `conda env create -f environment.yml && conda activate exudate-bench`

### Verify the installation without the dataset

```bash
pytest tests/ -v                    # 80 tests, no images required
```

```bash
python scripts/make_synthetic_dataset.py     # 169 procedurally-drawn images
python scripts/run_all.py --skip-cnn         # full pipeline, ~2 minutes
```

The synthetic images exist so that CI — and any reviewer without HEI-MED — can
execute this code. **They have no clinical meaning and must never be reported
as results.**

### Reproduce the manuscript

```bash
python scripts/00_download_data.py --accept-licence
python scripts/run_all.py
```

Roughly 80 minutes, dominated by CNN training. Outputs land in `results/`.

---

## The pipeline

Each stage is a standalone script, runnable independently. `run_all.py` chains
them.

| # | Script | Produces | Runtime |
|---|---|---|---|
| 00 | `00_download_data.py` | HEI-MED in `data/raw/` | network |
| 01 | `01_build_manifest.py` | Manifest + stratified splits | ~5 s |
| 02 | `02_extract_features.py` | 169 × 3,042 feature matrix | ~46 s |
| 03 | `03_train_classical.py` | **Table 1** (classical rows) | ~10 s |
| 04 | `04_train_cnn.py` | **Table 1** (CNN rows) | **~70 min** |
| 05 | `05_cross_validate.py` | **Tables 3 & 4** | ~2 min |
| 06 | `06_explain_shap.py` | **Figures 2–3, Table 5** | ~30 s |
| 07 | `07_decision_curve.py` | **Figure 4** | ~5 s |
| 08 | `08_efficiency.py` | **Table 2** | ~1 min |
| 09 | `09_make_tables_figures.py` | **Figure 1**, merged tables | ~10 s |

```bash
python scripts/run_all.py --skip-cnn      # everything but stage 04
python scripts/run_all.py --from 05       # resume partway
python scripts/run_all.py --dry-run       # print the plan
```

---

## Method summary

### Dataset

169 colour fundus photographs (HEI-MED), expert-segmented by a single
ophthalmologist. An image is **exudate-positive** if its annotation contains
one or more lesion regions: 54 positive, 115 negative, 32.0% prevalence.
Stratified 70:15:15 split at seed 42 → 118 / 25 / 26.

The manifest builder **asserts** these counts and raises on mismatch. A partial
download would otherwise produce a plausible-looking but different dataset, and
every downstream number would be quietly wrong.

### Classical arm — 3,042 handcrafted features

| Family | Parameters | Dimensions |
|---|---|---|
| HOG | 9 orientations, 16×16 px/cell, 2×2 cells/block → 9×9 block grid | 2,916 |
| LBP (uniform) | P = 24, R = 3 → P + 2 bins | 26 |
| RGB colour histogram | 32 bins × 3 channels | 96 |
| Green-channel statistics | mean, SD, p90, p10 | 4 |
| | | **3,042** |

Classifiers: logistic regression (L2, class-balanced, 2,000 iterations),
SVM-RBF (C = 1.0, class-balanced), random forest (300 trees, depth ≤ 8,
class-balanced). Each is a `StandardScaler → classifier` pipeline, so the
scaler refits inside every CV fold — fitting it once on all 169 images would
leak test-fold statistics into preprocessing and optimistically bias Table 3.

All three feature extractors are deterministic closed-form functions. No
pretraining, no downloads, no fitted preprocessing.

### CNN arm — MobileNetV3 from scratch

`weights=None`, deliberately. AdamW (lr 1e−4, weight decay 1e−3), cosine
annealing, gradient clipping at 1.0, class-weighted cross-entropy with 0.1
label smoothing, weighted random sampler, augmentation (horizontal flip, ±15°
rotation, ±20% brightness/contrast), 160×160 input. Small: 20 epochs; Large:
12. Checkpoint selected on best **validation AUC** — not accuracy, which at 32%
prevalence would happily reward majority-class collapse.

Every available small-data countermeasure is applied. That they were
insufficient *is the finding*, and the code is written so a reviewer can
confirm the collapse was not caused by a missing safeguard.

Setting `pretrained: true` runs transfer learning instead. That answers a
different question and will not reproduce the published numbers.

### Statistics

| Quantity | Method | Why |
|---|---|---|
| Accuracy, sensitivity, precision | Wilson score | Stays in [0,1] and retains coverage at n = 26 near the boundaries, where Wald fails |
| AUC | Hanley–McNeil | Closed-form, accounts for class sizes |
| F1-score | Percentile bootstrap, 5,000 resamples, **unstratified** | No closed-form variance; stratified resampling of the CNNs' single-class predictions returns a degenerate zero-width interval |
| Model comparison | McNemar on pooled out-of-fold predictions | The 26-image test set is underpowered; this uses all 169 |
| Secondary AUC comparison | DeLong | Correlated AUCs on the same cases |

The test set is never used for model selection, hyperparameter tuning or
threshold calibration.

### Explainability

SHAP (linear explainer, training partition as background) on the logistic
regression model, in two distinct outputs:

- **Grouped attribution (Figure 2)** — what *kind* of information the model
  uses, per feature family.
- **Spatial saliency (Figure 3)** — HOG SHAP values carry a block index, so
  they project onto a 9×9 grid and upsample to a pixel-level map.

**The scope limit matters and is stated plainly.** LBP and colour features are
global histograms with no spatial coordinates and cannot be localised *at all*.
The saliency map derives from HOG alone (~27.5% of attribution); the remaining
~72.5% is non-localisable by construction, not merely unmeasured.

Validation against expert masks uses Dice, IoU and overlap against a
chance-level baseline. Because the saliency region is a fixed 20% area
fraction, the chance baseline is available in closed form — which is why the
reported chance overlap is exactly 20.0%. An empirical permutation baseline
agrees with the analytic form to three decimals.

Absolute agreement is modest (Dice 0.250), and two structural factors bound it
before model quality enters the picture: 16-pixel block sampling is coarse
relative to individual exudates, and a block-level map scored against a
pixel-level reference cannot reach high Dice even with perfect localisation.
The defensible claim is that saliency is *non-randomly associated* with
expert-identified lesions — not that it is a validated clinical explanation.
`tests/test_explain.py::test_dice_is_bounded_by_area_mismatch` demonstrates the
ceiling directly.

### Efficiency

Latency is the mean of 30 single-image CPU passes after 5 warm-up passes.
Warm-up matters: the first passes absorb lazy BLAS initialisation and allocator
growth, and including them inflates the mean.

**Feature extraction is timed separately from classifier inference**, because
in the classical arm it dominates (~274 ms vs. 0.11 ms for logistic regression)
and — unlike training-time preprocessing — it is incurred at *every*
prediction. Reporting `.predict` alone would be true of the function call and
badly misleading about the deployed system. This is the source of the headline
reversal: the classical arm trains ~63× faster but infers ~35× slower.

---

## Repository layout

```
├── src/exudate_bench/
│   ├── config.py              # Every published hyperparameter, in one place
│   ├── utils.py               # Seeding, logging, provenance-stamped I/O
│   ├── data/
│   │   ├── download.py        # Licence-gated retrieval
│   │   ├── manifest.py        # Pluggable ground-truth readers + validation
│   │   ├── splits.py          # Stratified partitioning, leakage checks
│   │   └── torch_dataset.py   # Loaders driven by the same manifest
│   ├── features/handcrafted.py
│   ├── models/
│   │   ├── classical.py       # Pipelines, training, cross-validation
│   │   └── cnn.py             # MobileNetV3 (torch imported lazily)
│   ├── evaluation/
│   │   ├── metrics.py         # Wilson, Hanley–McNeil, bootstrap
│   │   ├── statistics.py      # McNemar, DeLong, power calculations
│   │   └── decision_curve.py  # Net benefit
│   ├── explain/
│   │   ├── shap_analysis.py   # Attribution + spatial projection
│   │   └── mask_validation.py # Dice/IoU/overlap vs. chance
│   ├── efficiency/benchmark.py
│   └── viz/figures.py
├── scripts/                   # 00–09 pipeline stages + run_all
├── tests/                     # 80 tests, incl. manuscript regressions
├── docs/
│   ├── DATA.md                # Dataset, licence, annotation formats
│   ├── MANUSCRIPT_MAP.md      # Every table/figure → the code that made it
│   └── REPRODUCIBILITY.md     # Determinism, runtimes, known sensitivities
├── configs/default.yaml
└── .github/workflows/ci.yml
```

`config.py` is the single source of truth: every hyperparameter appears exactly
once, so a reviewer can check each reported value against one file instead of
searching the codebase.

---

## Configuration

```bash
python scripts/run_all.py --config configs/default.yaml
```

`configs/default.yaml` is byte-equivalent to the built-in defaults — it exists
as an editable record of what can be changed. Only listed keys are overridden.
**An unrecognised key raises an error**, so a typo cannot silently leave a
hyperparameter at its default.

---

## Testing

```bash
pytest tests/ -v                                              # 80 tests
pytest tests/test_manuscript_values.py -v                     # published values
pytest tests/ --cov=exudate_bench --cov-report=term-missing
```

| File | Covers |
|---|---|
| `test_manuscript_values.py` | Every published interval, statistic and p-value |
| `test_features.py` | Dimensionality, geometry, determinism, normalisation |
| `test_splits.py` | Partition sizes, stratification, leakage, order-invariance |
| `test_evaluation.py` | Metrics, degenerate classifiers, DCA, pipeline hygiene |
| `test_explain.py` | HOG→spatial projection correctness, agreement measures |

The projection tests are the subtlest: they plant attribution in a known HOG
block and assert it surfaces at the corresponding image location. A wrong
reshape ordering would produce entirely plausible-looking maps pointing at the
wrong part of the image — a failure no visual inspection would catch.

---

## Data availability and licence

**This repository contains no fundus images.** HEI-MED is released for
non-commercial research use by the recipient; onward redistribution is not
permitted. A CI job fails the build if image files are ever committed.

Obtain it from <https://github.com/lgiancaUTH/HEI-MED> and cite:

> Giancardo L, Meriaudeau F, Karnowski TP, Li Y, Garg S, Tobin KW Jr, Chaum E.
> Exudate-based diabetic macular edema detection in fundus images using
> publicly available datasets. *Med Image Anal.* 2012;16(1):216–226.
> doi:10.1016/j.media.2011.07.004

Code is MIT-licensed. See [`LICENSE`](LICENSE) for the scope boundary.

---

## Limitations

Carried over from the manuscript, because code that reproduces a result should
also reproduce its caveats:

1. 169 images from a single historical telemedicine programme; sample size was
   fixed by the dataset, not by design. Metrics are small-sample estimates, not
   precise population parameters.
2. The binary label is a simplified proxy for referable maculopathy, not a
   formal ICDR or DME severity grade.
3. CNN results characterise the from-scratch regime only — not MobileNetV3's
   ceiling under standard transfer learning.
4. Imaging technology dates from ~2010; performance on contemporary cameras may
   differ.
5. Cross-validation covers the classical arm only; retraining each CNN five
   times was outside the compute budget.
6. No external validation on independent datasets.
7. Explainability covers the classical arm and, within it, the HOG component
   only. No CNN-side Grad-CAM was performed.
8. Agreement with expert masks is computational, not independent clinical
   review.
9. Pooled out-of-fold predictions are not fully independent, so those intervals
   may be narrower than their nominal width.

**This is research code, not a medical device.** It must not be used for
clinical diagnosis, screening or patient management.

---

## Citation

See [`CITATION.cff`](CITATION.cff), or use GitHub's "Cite this repository".
Please cite both this software and the HEI-MED source publication.

## Acknowledgements

The authors thank the Hamilton Eye Institute, University of Tennessee Health
Science Center, Oak Ridge National Laboratory and l'Université de Bourgogne for
the collection, curation and public release of the HEI-MED dataset.
