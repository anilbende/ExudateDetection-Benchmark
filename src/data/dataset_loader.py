
"""
Dataset loading and validation for ExudateDetection-Benchmark.

Responsibilities
----------------
1. Read the configured HEI-MED manifest.
2. Validate required columns and manifest entries.
3. Resolve image paths and check that image files are readable.
4. Save a validated copy of the manifest and a validation report.

Ground-truth label mapping is handled by label_mapping.py.
This module does not infer labels from filenames or modify source images.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError


def _get_nested(mapping: dict, *keys: str, default=None):
    """Return a nested configuration value without raising KeyError."""
    value = mapping

    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]

    return value


def _resolve_image_path(
    image_path: str,
    dataset_root: Path,
    repository_root: Path,
) -> Path:
    """Resolve a manifest image path using documented locations."""
    relative_path = Path(image_path)

    if relative_path.is_absolute():
        return relative_path.resolve()

    candidates = [
        (dataset_root / relative_path).resolve(),
        (repository_root / relative_path).resolve(),
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Return the primary expected location for a useful error message.
    return candidates[0]


def _is_inside(path: Path, parent: Path) -> bool:
    """Check whether path is contained within parent."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _check_image(path: Path) -> tuple[bool, str]:
    """Check that a file can be opened and verified as an image."""
    if not path.is_file():
        return False, "file_not_found"

    try:
        with Image.open(path) as image:
            image.verify()

        return True, "readable"

    except (OSError, ValueError, UnidentifiedImageError):
        return False, "unreadable_or_invalid_image"


def run(context: dict[str, Any]) -> dict[str, Any]:
    """
    Load and validate the dataset manifest.

    Expected context keys:
        config: parsed experiment configuration
        repository_root: repository root directory (optional)
        output_dir: directory for generated artifacts (optional)

    Returns:
        A dictionary containing stage status, artifact paths,
        validation counts, and warnings.
    """
    config = context.get("config", {})

    if not isinstance(config, dict):
        raise TypeError("context['config'] must be a dictionary.")

    repository_root = Path(
        context.get("repository_root", Path.cwd())
    ).resolve()

    dataset_root_value = _get_nested(
        config, "dataset", "data_root", default="data/HEI-MED"
    )
    dataset_root = Path(dataset_root_value)

    if not dataset_root.is_absolute():
        dataset_root = (repository_root / dataset_root).resolve()

    manifest_value = _get_nested(
        config,
        "dataset",
        "manifest_file",
        default="data/HEI-MED/labels.csv",
    )
    manifest_path = Path(manifest_value)

    if not manifest_path.is_absolute():
        # Prefer the path as written in the configuration when it exists.
        repo_manifest = (repository_root / manifest_path).resolve()
        root_manifest = (dataset_root / manifest_path).resolve()

        if repo_manifest.is_file():
            manifest_path = repo_manifest
        elif root_manifest.is_file():
            manifest_path = root_manifest
        else:
            manifest_path = repo_manifest

    columns = _get_nested(
        config,
        "dataset",
        "manifest_columns",
        default={"image_path": "image_path", "label": "label"},
    )

    image_column = columns.get("image_path", "image_path")
    label_column = columns.get("label", "label")

    output_dir = Path(
        context.get("output_dir", repository_root / "results" / "primary")
    )
    if not output_dir.is_absolute():
        output_dir = (repository_root / output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. "
            "Check dataset.manifest_file in the experiment configuration."
        )

    manifest = pd.read_csv(manifest_path)

    required_columns = {image_column, label_column}
    missing_columns = required_columns - set(manifest.columns)

    if missing_columns:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing_columns)}. "
            f"Columns found: {list(manifest.columns)}"
        )

    if manifest.empty:
        raise ValueError("The dataset manifest contains no rows.")

    if manifest[image_column].isna().any():
        raise ValueError("The manifest contains missing image paths.")

    if manifest[label_column].isna().any():
        raise ValueError(
            "The manifest contains missing labels. "
            "Resolve the documented ground-truth labels before continuing."
        )

    image_values = manifest[image_column].astype(str).str.strip()

    if image_values.eq("").any():
        raise ValueError("The manifest contains blank image paths.")

    if image_values.duplicated().any():
        duplicates = image_values[image_values.duplicated()].tolist()
        raise ValueError(
            "Duplicate image paths found in the manifest: "
            f"{duplicates[:10]}"
        )

    validated = manifest.copy()
    resolved_paths = []
    image_statuses = []
    invalid_images = []

    for row_index, image_value in enumerate(image_values):
        resolved_path = _resolve_image_path(
            image_value,
            dataset_root,
            repository_root,
        )

        # Prevent a relative path from escaping the repository or dataset
        # directories. Absolute paths are allowed for local configurations.
        original_path = Path(image_value)
        if not original_path.is_absolute():
            allowed = (
                _is_inside(resolved_path, dataset_root)
                or _is_inside(resolved_path, repository_root)
            )
            if not allowed:
                raise ValueError(
                    f"Image path escapes the configured data locations: "
                    f"{image_value}"
                )

        is_valid, status = _check_image(resolved_path)

        resolved_paths.append(str(resolved_path))
        image_statuses.append(status)

        if not is_valid:
            invalid_images.append(
                {
                    "row": int(row_index),
                    "image_path": image_value,
                    "resolved_path": str(resolved_path),
                    "status": status,
                }
            )

    validated["resolved_image_path"] = resolved_paths
    validated["image_validation_status"] = image_statuses

    expected_total = _get_nested(
        config, "dataset", "expected_total_images", default=None
    )

    if expected_total is not None and len(validated) != int(expected_total):
        raise ValueError(
            f"Expected {expected_total} manifest rows, "
            f"but found {len(validated)}."
        )

    if invalid_images:
        report = {
            "status": "failed",
            "manifest": str(manifest_path),
            "total_manifest_rows": int(len(validated)),
            "invalid_image_count": len(invalid_images),
            "invalid_images": invalid_images,
        }

        report_path = output_dir / "dataset_validation.json"
        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        raise ValueError(
            f"{len(invalid_images)} image file(s) failed validation. "
            f"See {report_path}"
        )

    validated_manifest_path = output_dir / "validated_manifest.csv"
    validated.to_csv(validated_manifest_path, index=False)

    report = {
        "status": "passed",
        "manifest": str(manifest_path),
        "dataset_root": str(dataset_root),
        "total_manifest_rows": int(len(validated)),
        "readable_images": int(len(validated)),
        "invalid_image_count": 0,
        "image_path_column": image_column,
        "label_column": label_column,
        "label_mapping_performed": False,
        "patient_level_independence_verified": False,
    }

    report_path = output_dir / "dataset_validation.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return {
        "status": "passed",
        "artifacts": [
            str(validated_manifest_path),
            str(report_path),
        ],
        "summary": report,
        "warnings": [
            "Ground-truth labels have not yet been mapped to numeric classes.",
            "Patient-level independence has not been verified.",
        ],
    }
