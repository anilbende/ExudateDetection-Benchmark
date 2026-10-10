
"""
Explicit ground-truth label mapping for ExudateDetection-Benchmark.

Negative class: 0
Positive class: 1

Labels must be mapped from documented dataset annotations.
This module must never infer labels from image filenames.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _get_nested(mapping: dict, *keys: str, default=None):
    """Read a nested configuration value safely."""
    value = mapping

    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]

    return value


def _normalise_label(value: Any) -> str:
    """Convert a label to a normalized string for explicit lookup."""
    if pd.isna(value):
        raise ValueError("A missing ground-truth label was encountered.")

    return str(value).strip().lower()


def run(context: dict[str, Any]) -> dict[str, Any]:
    """
    Map documented ground-truth labels to numeric binary classes.

    Expected configuration:
        dataset.manifest_columns.label
        dataset.label_mapping.negative
        dataset.label_mapping.positive

    The input manifest is expected to be the validated manifest generated
    by dataset_loader.py.
    """
    config = context.get("config", {})
    repository_root = Path(
        context.get("repository_root", Path.cwd())
    ).resolve()

    output_dir = Path(
        context.get("output_dir", repository_root / "results" / "primary")
    )
    if not output_dir.is_absolute():
        output_dir = (repository_root / output_dir).resolve()

    input_manifest = output_dir / "validated_manifest.csv"

    if not input_manifest.is_file():
        raise FileNotFoundError(
            f"Validated manifest not found: {input_manifest}. "
            "Run dataset_loader.py successfully first."
        )

    manifest = pd.read_csv(input_manifest)

    label_column = _get_nested(
        config,
        "dataset",
        "manifest_columns",
        "label",
        default="label",
    )

    if label_column not in manifest.columns:
        raise ValueError(
            f"Configured label column '{label_column}' "
            "is missing from the validated manifest."
        )

    mapping_config = _get_nested(
        config,
        "dataset",
        "label_mapping",
        default=None,
    )

    if not isinstance(mapping_config, dict):
        raise ValueError(
            "Configure dataset.label_mapping explicitly in the YAML file."
        )

    negative_value = mapping_config.get("negative")
    positive_value = mapping_config.get("positive")

    if negative_value is None or positive_value is None:
        raise ValueError(
            "Both negative and positive label mappings must be configured."
        )

    if negative_value == positive_value:
        raise ValueError(
            "Negative and positive labels must have different values."
        )

    # Numeric class IDs used by the experiments.
    configured_mapping = {
        _normalise_label(negative_value): 0,
        _normalise_label(positive_value): 1,
    }

    observed_labels = manifest[label_column].map(_normalise_label)
    observed_unique = set(observed_labels.unique())
    allowed_labels = set(configured_mapping)

    unexpected_labels = observed_unique - allowed_labels

    if unexpected_labels:
        raise ValueError(
            "The manifest contains labels that are not covered by the "
            f"configured mapping: {sorted(unexpected_labels)}. "
            "Check the official dataset annotations and configuration."
        )

    manifest["label_original"] = manifest[label_column]
    manifest["label"] = observed_labels.map(configured_mapping).astype("int64")

    expected_total = _get_nested(
        config, "dataset", "expected_total_images", default=169
    )
    expected_positive = _get_nested(
        config, "dataset", "expected_positive_images", default=54
    )
    expected_negative = _get_nested(
        config, "dataset", "expected_negative_images", default=115
    )

    counts = manifest["label"].value_counts().to_dict()
    actual_negative = int(counts.get(0, 0))
    actual_positive = int(counts.get(1, 0))

    problems = []

    if len(manifest) != int(expected_total):
        problems.append(
            f"Expected {expected_total} rows; found {len(manifest)}."
        )

    if actual_negative != int(expected_negative):
        problems.append(
            f"Expected {expected_negative} negative images; "
            f"found {actual_negative}."
        )

    if actual_positive != int(expected_positive):
        problems.append(
            f"Expected {expected_positive} positive images; "
            f"found {actual_positive}."
        )

    report = {
        "status": "passed" if not problems else "failed",
        "total_images": int(len(manifest)),
        "negative_label_id": 0,
        "positive_label_id": 1,
        "negative_count": actual_negative,
        "positive_count": actual_positive,
        "configured_expected_total": int(expected_total),
        "configured_expected_negative": int(expected_negative),
        "configured_expected_positive": int(expected_positive),
        "label_mapping": configured_mapping,
        "patient_level_independence_verified": False,
        "problems": problems,
    }

    report_path = output_dir / "label_mapping_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    if problems:
        raise ValueError(
            "Label validation failed: " + " ".join(problems)
        )

    mapped_path = output_dir / "mapped_manifest.csv"
    manifest.to_csv(mapped_path, index=False)

    return {
        "status": "passed",
        "artifacts": [
            str(mapped_path),
            str(report_path),
        ],
        "summary": report,
        "warnings": [
            "Confirm the configured source labels against the official "
            "HEI-MED annotations before reporting these class counts."
        ],
    }
