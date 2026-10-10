
# ExudateDetection-Benchmark

Reproducibility implementation for a resource-constrained benchmark of
retinal exudate detection using MobileNetV3 models trained from scratch
and handcrafted-feature classical machine learning classifiers.

## Research scope

The intended experiments include:

- HEI-MED image-level classification.
- MobileNetV3-Small and MobileNetV3-Large trained from scratch.
- Logistic Regression, RBF-SVM and Random Forest.
- HOG, LBP, RGB histogram and green-channel statistical features.
- Five-fold stratified cross-validation for classical models.
- Confidence intervals and paired prediction analysis.
- Logistic Regression SHAP analysis and a random spatial-block baseline.
- Exploratory decision-curve analysis.
- Runtime benchmarking and reproducibility reporting.

## Current implementation status

The repository is being prepared for reproducible execution.

The presence of configuration files and dependencies does not establish
that all experiments have been reproduced. Each pipeline module must be
implemented, tested and checked against the manuscript.

Do not report successful reproduction until the relevant experiments have
run and the generated outputs have been compared with the manuscript.

## Repository structure

Expected structure:

    .
    ├── main.py
    ├── requirements.txt
    ├── requirements-dev.txt
    ├── README.md
    ├── configs/
    │   └── experiment_spec.yaml
    ├── src/
    │   ├── data/
    │   ├── features/
    │   ├── training/
    │   ├── evaluation/
    │   └── reporting/
    ├── tests/
    └── results/
        └── reproduced/

The `src` modules and tests must be added as implementation proceeds.

## Environment

Initial target environment: Python 3.11 or 3.12.

Use a clean virtual environment.

### Windows

    py -3.11 -m venv .venv
    .venv\Scripts\activate
    python -m pip install --upgrade pip

### Linux or macOS

    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip

Install PyTorch and torchvision using the official instructions for the
intended CPU or CUDA environment:

https://pytorch.org/get-started/locally/

Then install the remaining dependencies from the repository:

    python -m pip install -r requirements.txt

Install development tools when running tests:

    python -m pip install -r requirements-dev.txt

Record the actual Python, operating system, PyTorch, torchvision,
scikit-learn, scikit-image, NumPy and SHAP versions used in a successful run.

## Dataset access and labels

Obtain HEI-MED from its authorised source and comply with its licence.

Do not commit the original images to this repository unless the applicable
licence explicitly permits redistribution.

The configured label manifest is:

    data/HEI-MED/labels.csv

It must contain the columns:

    image_path,label

The label column must use:

    0 = negative
    1 = positive

The labels must come from a documented source of ground truth. Do not
infer labels from filenames.

The expected dataset contains 169 images: 54 positive and 115 negative.
The data-loading implementation must verify these counts and report any
missing, duplicate, unreadable or inconsistently labelled images.

Patient identifiers are not available in the current study specification.
The split is therefore image-level, and patient-level independence cannot
be claimed as verified.

## Experimental protocol

The intended primary split is stratified:

- Training: 118 images
- Validation: 25 images
- Test: 26 images

The random seed is 42. Data augmentation must be applied only to training
images. Preprocessing parameters fitted from data must be fitted using
training data only.

The CNNs are trained from scratch. The classification probability
threshold is fixed at 0.5; it must not be tuned on the held-out test set.

Classical-model cross-validation uses five stratified folds. Any scaler
and classifier must be fitted within each training fold.

Do not silently change the input size, feature definitions, split,
training parameters, threshold, statistical procedures or evaluation scope.
Any deviation from the manuscript must be documented.

## Configuration

The experiment configuration is:

    configs/experiment_spec.yaml

Some parameters are intentionally unresolved because they must be
verified from original experiment records or source code.

Do not guess missing training parameters or statistical-test variants.
Resolve them before running the affected stages.

## Running the pipeline

The intended command interface is:

    python main.py --stage prepare
    python main.py --stage classical
    python main.py --stage cnn
    python main.py --stage cross-validation
    python main.py --stage statistics
    python main.py --stage explainability
    python main.py --stage performance
    python main.py --stage report
    python main.py --stage all

Optional arguments include:

    --config configs/experiment_spec.yaml
    --data-dir data/HEI-MED
    --results-dir results/reproduced
    --device auto

These commands require the corresponding modules to exist and implement
the documented module interface. A missing module or unresolved required
parameter should be treated as an implementation blocker, not as a
successful experiment.

## Expected outputs

The completed implementation should save:

- Dataset validation report and split manifest.
- Model configurations and checkpoints.
- Validation-selected CNN checkpoints.
- Test labels, predicted probabilities and thresholded predictions.
- Classical-model cross-validation predictions.
- Metrics and confidence intervals.
- McNemar and bootstrap analysis outputs.
- SHAP values and chance-baseline results.
- Decision-curve and runtime outputs.
- Generated tables, figures and a run manifest.
- Software versions and configuration copies.

All reported values must be calculated from actual experiment outputs.
Never hard-code manuscript values into generated result tables.

## Reproducibility and reporting

Each run should record its configuration, software environment, random
seed, device, timestamps, input manifest and output artefacts.

A completed script is not proof of scientific reproduction. Compare
generated results with the manuscript and document discrepancies,
limitations and deviations.

The small held-out test set, image-level splitting, unavailable patient
identifiers, absence of external validation and lack of CNN cross-validation
must remain explicit limitations unless additional verified evidence
supports a change.

## Data availability

The dataset is not redistributed in this repository. Users must obtain
it from an authorised source under the applicable licence.

Add the verified repository and archival DOI to the manuscript only after
confirming that the code and archive are accessible. Do not publish a
placeholder or invented DOI.

## Ethics and intended use

This benchmark is a research implementation, not a clinically validated
diagnostic product. Results must not be presented as evidence of clinical
utility without appropriate validation.

## License

Add the repository's actual software licence here. Do not assume a licence
for code or data that the repository owner has not authorised.
