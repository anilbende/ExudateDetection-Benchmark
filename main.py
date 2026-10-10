
#!/usr/bin/env python3
"""
main.py
=======

Master execution entry point for ExudateDetection-Benchmark.

Research scope:
    1. HEI-MED dataset validation and preprocessing
    2. Primary stratified train/validation/test experiment
    3. MobileNetV3-Small and MobileNetV3-Large trained from scratch
    4. Logistic Regression, RBF-SVM and Random Forest
    5. Five-fold stratified cross-validation for classical models
    6. Confidence intervals and McNemar analysis
    7. Logistic Regression SHAP and random spatial-block baseline
    8. Exploratory decision-curve analysis
    9. Computational benchmarking
   10. Manuscript table, figure and reproducibility reporting

Configuration:
    configs/experiment_spec.yaml

Module contract:
    Each pipeline module must expose:

        def run(context: dict) -> dict:
            ...

    Return a dictionary containing:
        {
            "status": "COMPLETED",
            "artifacts": ["relative/or/absolute/path"],
            "warnings": []
        }

    Every declared artifact must exist before the module can be
    considered successful.

IMPORTANT:
    This entry point coordinates the pipeline. It does not replace
    the actual implementations of the individual research modules.
    Successful execution alone does not establish scientific
    reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ============================================================
# 1. Repository paths
# ============================================================

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "experiment_spec.yaml"


# ============================================================
# 2. Pipeline registry
#
# All modules must implement run(context).
# The reporting modules are deliberately separate from the
# model-training and evaluation modules.
# ============================================================

STAGES: dict[str, list[str]] = {
    "prepare": [
        "src.data.dataset_loader",
        "src.data.label_mapping",
        "src.data.split_data",
        "src.data.preprocessing",
    ],
    "classical": [
        "src.features.handcrafted_features",
        "src.training.train_classical",
    ],
    "cnn": [
        "src.training.train_cnn",
    ],
    "cross-validation": [
        "src.evaluation.cross_validation",
    ],
    "statistics": [
        "src.evaluation.metrics",
        "src.evaluation.confidence_intervals",
        "src.evaluation.mcnemar_test",
    ],
    "explainability": [
        "src.evaluation.shap_analysis",
        "src.evaluation.chance_baseline",
    ],
    "performance": [
        "src.evaluation.decision_curve",
        "src.evaluation.benchmark_latency",
    ],
    "report": [
        "src.reporting.generate_tables",
        "src.reporting.generate_figures",
        "src.reporting.compare_results",
    ],
}

STAGE_ORDER = list(STAGES.keys())

VALID_STATUSES = {
    "COMPLETED",
    "FAILED",
    "NOT EXECUTED",
    "COMPLETED_WITH_WARNINGS",
}


# ============================================================
# 3. General utilities
# ============================================================

def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON record, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8",
    )


def create_directories(results_dir: Path) -> None:
    """Create output directories without deleting previous results."""

    folders = [
        "logs",
        "primary",
        "cross_validation",
        "statistics",
        "explainability",
        "performance",
        "tables",
        "figures",
        "reproducibility",
        "models",
    ]

    for folder in folders:
        (results_dir / folder).mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# 4. Configuration loading and validation
# ============================================================

def load_config(config_path: Path) -> dict[str, Any]:
    """Load the central YAML configuration."""

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found:\n{config_path}\n\n"
            "Create configs/experiment_spec.yaml before running "
            "the research pipeline."
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "The experiment configuration must be a YAML mapping."
        )

    required_sections = [
        "project",
        "dataset",
        "primary_split",
        "image",
        "decision",
        "cnn",
        "features",
        "classical_models",
        "evaluation",
        "cross_validation",
        "outputs",
        "reproducibility",
    ]

    missing_sections = [
        section
        for section in required_sections
        if section not in config
    ]

    if missing_sections:
        raise ValueError(
            "Missing configuration sections: "
            + ", ".join(missing_sections)
        )

    return config


def get_nested(
    config: dict[str, Any],
    dotted_path: str,
) -> Any:
    """Retrieve a nested configuration value using dot notation."""

    current: Any = config

    for key in dotted_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]

    return current


def validate_scientific_invariants(
    config: dict[str, Any],
) -> list[str]:
    """
    Validate fixed study-level invariants.

    These checks validate the configuration only. Actual dataset
    counts and predictions must be validated by their own modules.
    """

    errors: list[str] = []

    expected_total = config["dataset"]["expected_total_images"]
    positive = config["dataset"]["expected_positive_images"]
    negative = config["dataset"]["expected_negative_images"]

    if positive + negative != expected_total:
        errors.append(
            "Expected positive and negative counts do not sum "
            "to the expected dataset size."
        )

    split = config["primary_split"]
    split_counts = split["expected_counts"]

    if sum(split_counts.values()) != expected_total:
        errors.append(
            "Expected train/validation/test counts do not sum "
            "to the expected dataset size."
        )

    proportions = split["proportions"]

    if not np.isclose(sum(proportions.values()), 1.0):
        errors.append("Split proportions must sum to 1.0.")

    if split_counts != {
        "train": 118,
        "validation": 25,
        "test": 26,
    }:
        errors.append(
            "Configured split counts differ from the submitted "
            "118/25/26 experimental specification."
        )

    if config["project"]["seed"] != 42:
        errors.append(
            "The submitted study specifies random seed 42."
        )

    if config["decision"]["threshold"] != 0.5:
        errors.append(
            "The author-confirmed fixed classification threshold is 0.5."
        )

    image = config["image"]

    if image["height"] != 160 or image["width"] != 160:
        errors.append(
            "The submitted study specifies 160 x 160 image inputs."
        )

    features = config["features"]

    feature_components = [
        features["hog"]["expected_features"],
        features["lbp"]["expected_features"],
        features["rgb_histogram"]["expected_features"],
        features["green_channel_statistics"]["expected_features"],
    ]

    if sum(feature_components) != features["expected_total_features"]:
        errors.append(
            "Feature dimensions do not sum to the configured total."
        )

    if features["expected_total_features"] != 3042:
        errors.append(
            "The submitted study specifies 3,042 handcrafted features."
        )

    if config["cross_validation"]["folds"] != 5:
        errors.append(
            "The submitted study specifies five-fold classical-model CV."
        )

    if config["outputs"]["never_hardcode_reported_results"] is not True:
        errors.append(
            "The configuration must prohibit hard-coding reported results."
        )

    return errors


# ============================================================
# 5. Stage-specific unresolved-setting checks
# ============================================================

# These values must be recovered from original experiment records,
# verified source code, or documented as deviations.
#
# A stage is blocked only when its required settings are unresolved.
# This avoids blocking dataset validation because, for example, a
# DCA threshold step is not yet known.

STAGE_REQUIRED_SETTINGS: dict[str, list[str]] = {
    "prepare": [],
    "classical": [],
    "cnn": [
        "cnn.batch_size",
        "cnn.scheduler_parameters.T_max",
        "cnn.scheduler_parameters.eta_min",
        "cnn.scheduler_parameters.step_frequency",
        "cnn.early_stopping.patience",
        "cnn.early_stopping.min_delta",
    ],
    "cross-validation": [],
    "statistics": [
        "mcnemar.test_variant",
    ],
    "explainability": [
        "chance_baseline.shap_block_selection_rule",
    ],
    "performance": [
        "decision_curve_analysis.threshold_probabilities.step",
    ],
    "report": [],
}


def find_unresolved_settings(
    config: dict[str, Any],
    stage: str,
) -> list[str]:
    """Find null settings that block the requested stage."""

    unresolved = []

    for dotted_path in STAGE_REQUIRED_SETTINGS.get(stage, []):
        if get_nested(config, dotted_path) is None:
            unresolved.append(dotted_path)

    return unresolved


def validate_stage_settings(
    config: dict[str, Any],
    stage: str,
) -> None:
    """Stop a stage when essential scientific parameters are unknown."""

    unresolved = find_unresolved_settings(config, stage)

    if unresolved:
        formatted = "\n".join(
            f"  - {item}" for item in unresolved
        )

        raise ValueError(
            f"Stage '{stage}' cannot run as specified because "
            "the following scientific settings are unresolved:\n"
            f"{formatted}\n\n"
            "Recover these values from the original experiment "
            "records or document an explicit methodological deviation. "
            "Do not silently guess values."
        )


# ============================================================
# 6. Random seeds and device selection
# ============================================================

def set_random_seeds(seed: int) -> None:
    """Set Python, NumPy and PyTorch random seeds."""

    random.seed(seed)
    np.random.seed(seed)

    # Set before CUDA initialisation where supported.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(
                True,
                warn_only=True,
            )
        except (AttributeError, TypeError):
            pass

    except ImportError:
        # Dataset-only and classical stages can still be available.
        pass


def resolve_device(requested: str) -> str:
    """Select CPU or CUDA and fail clearly if CUDA was required."""

    if requested == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:
        if requested == "cuda":
            raise RuntimeError(
                "CUDA was requested, but PyTorch is not installed."
            )
        return "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but it is unavailable."
            )
        return "cuda"

    return "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 7. Runtime and software metadata
# ============================================================

def collect_software_metadata() -> dict[str, Any]:
    """Record runtime and relevant dependency versions."""

    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }

    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("sklearn", "scikit-learn"),
        ("skimage", "scikit-image"),
        ("yaml", "PyYAML"),
        ("shap", "shap"),
        ("matplotlib", "matplotlib"),
    ]

    for import_name, display_name in packages:
        try:
            module = importlib.import_module(import_name)
            metadata[display_name] = str(
                getattr(module, "__version__", "unknown")
            )
        except ImportError:
            metadata[display_name] = "NOT INSTALLED"

    try:
        import torch

        metadata["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            metadata["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            metadata["gpu_name"] = None

    except ImportError:
        metadata["cuda_available"] = False
        metadata["gpu_name"] = None

    return metadata


# ============================================================
# 8. Consistent module contract and output validation
# ============================================================

def validate_module_result(
    module_name: str,
    result: Any,
    results_dir: Path,
) -> dict[str, Any]:
    """
    Validate the result returned by a research module.

    Modules must return a dictionary with:
        status: COMPLETED or COMPLETED_WITH_WARNINGS
        artifacts: list of paths, if any
        warnings: list of warning strings, if any
    """

    if not isinstance(result, dict):
        raise TypeError(
            f"{module_name}.run(context) must return a dictionary."
        )

    status = result.get("status")

    if status not in {"COMPLETED", "COMPLETED_WITH_WARNINGS"}:
        raise ValueError(
            f"{module_name} returned invalid status: {status!r}"
        )

    artifacts = result.get("artifacts", [])

    if not isinstance(artifacts, list):
        raise TypeError(
            f"{module_name} must return artifacts as a list."
        )

    verified_artifacts = []

    for item in artifacts:
        path = Path(item)

        if not path.is_absolute():
            path = ROOT / path

        if not path.exists():
            raise FileNotFoundError(
                f"{module_name} declared an artifact that does not "
                f"exist: {path}"
            )

        verified_artifacts.append({
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
        })

    warnings = result.get("warnings", [])

    if not isinstance(warnings, list):
        raise TypeError(
            f"{module_name} must return warnings as a list."
        )

    return {
        "status": status,
        "artifacts": verified_artifacts,
        "warnings": warnings,
        "summary": result.get("summary", {}),
    }


def execute_module(
    module_name: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Import and execute a required module."""

    record: dict[str, Any] = {
        "module": module_name,
        "status": "NOT EXECUTED",
        "started_utc": utc_now(),
    }

    print(f"\nExecuting module: {module_name}")

    try:
        module = importlib.import_module(module_name)

        run_function = getattr(module, "run", None)

        if not callable(run_function):
            raise AttributeError(
                f"{module_name} must expose run(context)."
            )

        result = run_function(context)

        validated = validate_module_result(
            module_name=module_name,
            result=result,
            results_dir=context["results_dir"],
        )

        record.update(validated)

    except Exception as exc:
        record["status"] = "FAILED"
        record["error"] = str(exc)
        record["traceback"] = traceback.format_exc()
        raise

    finally:
        record["finished_utc"] = utc_now()

    return record


# ============================================================
# 9. Stage execution and explicit failure tracking
# ============================================================

def execute_stage(
    stage: str,
    context: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    """Execute a stage and record every module status."""

    print("\n" + "=" * 68)
    print(f"STAGE: {stage.upper()}")
    print("=" * 68)

    stage_record = {
        "stage": stage,
        "status": "STARTED",
        "started_utc": utc_now(),
        "modules": [],
    }

    manifest["stages"].append(stage_record)

    validate_stage_settings(context["config"], stage)

    try:
        for module_name in STAGES[stage]:
            try:
                module_record = execute_module(module_name, context)
                stage_record["modules"].append(module_record)

            except Exception as exc:
                stage_record["modules"].append({
                    "module": module_name,
                    "status": "FAILED",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "finished_utc": utc_now(),
                })

                # Mark remaining modules as not executed.
                remaining = STAGES[stage][
                    STAGES[stage].index(module_name) + 1:
                ]

                for pending in remaining:
                    stage_record["modules"].append({
                        "module": pending,
                        "status": "NOT EXECUTED",
                        "reason": "An earlier module in this stage failed.",
                    })

                raise

        stage_record["status"] = (
            "COMPLETED_WITH_WARNINGS"
            if any(
                module.get("status") == "COMPLETED_WITH_WARNINGS"
                for module in stage_record["modules"]
            )
            else "COMPLETED"
        )

    except Exception as exc:
        stage_record["status"] = "FAILED"
        stage_record["error"] = str(exc)
        raise

    finally:
        stage_record["finished_utc"] = utc_now()


def mark_remaining_stages_not_executed(
    manifest: dict[str, Any],
    requested_stages: list[str],
    completed_stages: set[str],
    failed_stage: str | None,
) -> None:
    """Record stages skipped because an earlier stage failed."""

    for stage in requested_stages:
        if stage in completed_stages or stage == failed_stage:
            continue

        if any(item["stage"] == stage for item in manifest["stages"]):
            continue

        manifest["stages"].append({
            "stage": stage,
            "status": "NOT EXECUTED",
            "reason": (
                f"Pipeline stopped after stage '{failed_stage}' failed."
                if failed_stage
                else "Stage was not reached."
            ),
            "modules": [
                {
                    "module": module_name,
                    "status": "NOT EXECUTED",
                }
                for module_name in STAGES[stage]
            ],
        })


# ============================================================
# 10. Command-line interface
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible HEI-MED retinal exudate research pipeline."
        )
    )

    parser.add_argument(
        "--stage",
        required=True,
        choices=["all", *STAGE_ORDER],
        help="Run the complete pipeline or one experiment stage.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to experiment_spec.yaml.",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "HEI-MED",
        help="Directory containing the HEI-MED dataset.",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "reproduced",
        help="Directory for newly generated results.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device.",
    )

    return parser.parse_args()


# ============================================================
# 11. Main
# ============================================================

def main() -> int:
    args = parse_arguments()

    config_path = args.config.resolve()
    data_dir = args.data_dir.resolve()
    results_dir = args.results_dir.resolve()

    create_directories(results_dir)

    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    manifest_path = (
        results_dir / "logs" / f"run_{run_id}.json"
    )

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "STARTED",
        "started_utc": utc_now(),
        "requested_stage": args.stage,
        "config_path": str(config_path),
        "data_dir": str(data_dir),
        "results_dir": str(results_dir),
        "stages": [],
        "scientific_reproduction_verified": False,
    }

    requested_stages = (
        STAGE_ORDER.copy()
        if args.stage == "all"
        else [args.stage]
    )

    completed_stages: set[str] = set()
    failed_stage: str | None = None

    try:
        # Configuration and scientific preflight
        config = load_config(config_path)

        invariant_errors = validate_scientific_invariants(config)

        if invariant_errors:
            raise ValueError(
                "Scientific configuration validation failed:\n- "
                + "\n- ".join(invariant_errors)
            )

        # Record the configuration digest for traceability.
        manifest["config_sha256"] = sha256_file(config_path)
        manifest["seed"] = int(config["project"]["seed"])

        set_random_seeds(manifest["seed"])

        device = resolve_device(args.device)

        manifest["device"] = device
        manifest["software_versions"] = collect_software_metadata()

        context = {
            "config": config,
            "config_path": config_path,
            "root_dir": ROOT,
            "data_dir": data_dir,
            "results_dir": results_dir,
            "device": device,
            "seed": manifest["seed"],
            "threshold": config["decision"]["threshold"],
            "run_id": run_id,
        }

        print("HEI-MED research pipeline")
        print(f"Configuration: {config_path}")
        print(f"Data directory: {data_dir}")
        print(f"Results directory: {results_dir}")
        print(f"Device: {device}")
        print(f"Threshold: {context['threshold']}")

        # Execute stages in the declared order.
        for stage in requested_stages:
            failed_stage = stage

            execute_stage(
                stage=stage,
                context=context,
                manifest=manifest,
            )

            completed_stages.add(stage)
            failed_stage = None

        manifest["status"] = "COMPLETED"

        print("\nAll requested stages completed successfully.")

        print(
            "Reminder: this status records software execution only. "
            "It does not prove that the manuscript results have been "
            "reproduced or scientifically validated."
        )

    except Exception as exc:
        manifest["status"] = "FAILED"
        manifest["error"] = str(exc)
        manifest["traceback"] = traceback.format_exc()

        print(f"\nPIPELINE FAILED: {exc}", file=sys.stderr)

        if failed_stage is not None:
            print(
                f"Failed stage: {failed_stage}",
                file=sys.stderr,
            )

        print(
            "Inspect the run manifest for the failure details.",
            file=sys.stderr,
        )

    finally:
        mark_remaining_stages_not_executed(
            manifest=manifest,
            requested_stages=requested_stages,
            completed_stages=completed_stages,
            failed_stage=failed_stage,
        )

        manifest["finished_utc"] = utc_now()

        # A completed software run must not automatically be
        # represented as verified scientific reproduction.
        manifest["scientific_reproduction_verified"] = False

        write_json(manifest_path, manifest)

        print(f"\nRun manifest: {manifest_path}")

    return 0 if manifest["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
