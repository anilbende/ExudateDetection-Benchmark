
"""
ExudateDetection-Benchmark.

Research code for reproducible retinal exudate detection experiments
using the HEI-MED dataset, MobileNetV3 models trained from scratch,
and handcrafted-feature classical machine learning classifiers.

Subpackages
-----------
data
    Dataset loading, label validation, splitting and preprocessing.
features
    Handcrafted feature extraction.
training
    Classical-model and CNN training.
evaluation
    Metrics, statistical analysis, explainability and benchmarking.
reporting
    Generation and comparison of experiment outputs.

Scientific reproducibility
---------------------------
Experimental settings must be obtained from the central configuration
and verified against the documented study protocol. Evaluation results
must be computed from actual predictions, not hard-coded manuscript
values.

The package initializer does not itself implement or validate the
experimental pipeline.
"""

__version__ = "0.1.0"
