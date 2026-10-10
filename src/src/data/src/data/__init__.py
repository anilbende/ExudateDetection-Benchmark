
"""
Data preparation package for ExudateDetection-Benchmark.

This package contains the modules responsible for preparing the
HEI-MED dataset for reproducible retinal exudate detection experiments.

Modules
-------
dataset_loader
    Loads the dataset manifest and validates image files.

label_mapping
    Validates and maps documented ground-truth labels.

split_data
    Generates the configured train/validation/test split and records
    the resulting image assignments.

preprocessing
    Applies image resizing, tensor conversion and configured
    training-only data augmentation.

Implementation requirements
---------------------------
Dataset preparation must follow the central experiment configuration.
Labels must come from documented ground truth, and validation/test
data must not receive training-only augmentation.

Patient-level independence must not be claimed when patient identifiers
are unavailable.

This initializer documents the package; it does not implement or
validate the behaviour of its submodules.
"""
