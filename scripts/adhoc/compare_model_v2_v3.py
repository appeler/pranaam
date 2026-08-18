"""Run a paired audit of the released v2 and candidate v3 English models.

The script reconstructs Pranaam's deterministic SEPRI calibration and
evaluation partitions, calibrates v2 on the same calibration partition used by
v3, and compares both models on identical household-head rows. It writes only
aggregate results; raw names and row-level predictions are never serialized.

Run from the repository root:

```
uv run --group train python scripts/adhoc/compare_model_v2_v3.py \
  --v2-model PATH --v2-vocabulary PATH --v3-dir PATH --output PATH
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import string
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pranaam.model import ModelConfig, NameClassifier, NameTokenizer, load_classifier
from pranaam.model_v3 import (
    ByteTokenizer,
    ModelArtifactMetadata,
    load_byte_classifier,
)
from training.train_v3 import (
    _sigmoid,
    evaluate_logits,
    filter_names_within_byte_limit,
    fit_platt_scaler,
    load_land_names,
    prepare_splits,
)

Metric = Callable[[np.ndarray, np.ndarray, np.ndarray], float]
_DELETE_ASCII_PUNCTUATION = str.maketrans("", "", string.punctuation)


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for an audit input."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _v2_logits(
    model: NameClassifier,
    tokenizer: NameTokenizer,
    frame: pd.DataFrame,
    batch_size: int,
) -> np.ndarray:
    """Return the binary log-odds represented by v2's two output logits."""
    parts: list[np.ndarray] = []
    names = frame["name"].astype(str).tolist()
    with torch.inference_mode():
        for start in range(0, len(names), batch_size):
            logits = model(tokenizer.encode(names[start : start + batch_size]))
            parts.append((logits[:, 1] - logits[:, 0]).numpy())
    return np.concatenate(parts).astype(np.float64, copy=False)


def _weighted_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    confidence: float = 0.8,
) -> dict[str, float]:
    """Compute weighted binary and calibration metrics."""
    predictions = probabilities >= 0.5
    positives = labels == 1
    tp = float(weights[predictions & positives].sum())
    fp = float(weights[predictions & ~positives].sum())
    fn = float(weights[~predictions & positives].sum())
    tn = float(weights[~predictions & ~positives].sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    bounded = np.clip(probabilities, 1e-15, 1 - 1e-15)
    loss = -(labels * np.log(bounded) + (1 - labels) * np.log(1 - bounded))
    retained = (probabilities <= 1 - confidence) | (probabilities >= confidence)
    total_weight = float(weights.sum())
    retained_weight = float(weights[retained].sum())
    return {
        "accuracy": (tp + tn) / total_weight,
        "precision_muslim": precision,
        "recall_muslim": recall,
        "f1_muslim": f1,
        "brier": float(np.average((probabilities - labels) ** 2, weights=weights)),
        "log_loss": float(np.average(loss, weights=weights)),
        "coverage": retained_weight / total_weight,
        "retained_accuracy": float(
            np.average(
                predictions[retained] == positives[retained],
                weights=weights[retained],
            )
        )
        if retained_weight
        else math.nan,
    }


def _bootstrap_improvements(
    frame: pd.DataFrame,
    v2: np.ndarray,
    v3: np.ndarray,
    *,
    draws: int,
    seed: int,
    confidence: float,
) -> dict[str, dict[str, float]]:
    """Return name-cluster bootstrap intervals for paired v3 improvements."""
    labels = frame["label"].to_numpy(dtype=np.float64)
    group_codes, unique_names = pd.factorize(frame["name"], sort=False)
    group_rows = [
        np.flatnonzero(group_codes == group) for group in range(len(unique_names))
    ]
    rng = np.random.default_rng(seed)
    metrics = (
        "accuracy",
        "precision_muslim",
        "recall_muslim",
        "f1_muslim",
        "brier",
        "log_loss",
    )
    full_weights = np.ones(len(frame), dtype=np.float64)
    full_v2 = _weighted_metrics(v2, labels, full_weights, confidence)
    full_v3 = _weighted_metrics(v3, labels, full_weights, confidence)
    values = {metric: np.empty(draws, dtype=np.float64) for metric in metrics}
    for draw in range(draws):
        sampled_groups = rng.integers(0, len(group_rows), size=len(group_rows))
        rows = np.concatenate([group_rows[group] for group in sampled_groups])
        weights = np.ones(len(rows), dtype=np.float64)
        old = _weighted_metrics(v2[rows], labels[rows], weights, confidence)
        new = _weighted_metrics(v3[rows], labels[rows], weights, confidence)
        for metric in metrics:
            if metric in {"brier", "log_loss"}:
                values[metric][draw] = old[metric] - new[metric]
            else:
                values[metric][draw] = new[metric] - old[metric]

    return {
        metric: {
            "point_improvement": (
                full_v2[metric] - full_v3[metric]
                if metric in {"brier", "log_loss"}
                else full_v3[metric] - full_v2[metric]
            ),
            "bootstrap_mean": float(np.mean(samples)),
            "ci_95_low": float(np.quantile(samples, 0.025)),
            "ci_95_high": float(np.quantile(samples, 0.975)),
        }
        for metric, samples in values.items()
    }


def _subgroup_results(
    frame: pd.DataFrame,
    v2: np.ndarray,
    v3: np.ndarray,
    column: str,
    *,
    confidence: float,
    minimum_rows: int = 50,
) -> dict[str, Any]:
    """Summarize both models within sufficiently large audit subgroups."""
    labels = frame["label"].to_numpy(dtype=np.float64)
    result: dict[str, Any] = {}
    for value in sorted(frame[column].dropna().unique(), key=str):
        selected = frame[column].eq(value).to_numpy()
        rows = int(selected.sum())
        if rows < minimum_rows:
            continue
        weights = np.ones(rows, dtype=np.float64)
        old = _weighted_metrics(v2[selected], labels[selected], weights, confidence)
        new = _weighted_metrics(v3[selected], labels[selected], weights, confidence)
        result[str(value)] = {
            "rows": rows,
            "muslim": int(labels[selected].sum()),
            "v2_calibrated": old,
            "v3": new,
            "accuracy_change": new["accuracy"] - old["accuracy"],
            "recall_change": new["recall_muslim"] - old["recall_muslim"],
            "brier_improvement": old["brier"] - new["brier"],
        }
    return result


def _survey_weight(document: dict[str, Any]) -> float:
    """Return the English split mode recorded in v3 training metadata."""
    training = document.get("training")
    if not isinstance(training, dict):
        raise ValueError("v3 metadata must include training settings")
    weight = float(training.get("survey_weight", 0.0))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("v3 survey_weight must be finite and nonnegative")
    return weight


def _validate_split_counts(
    document: dict[str, Any],
    splits: dict[str, pd.DataFrame],
) -> None:
    """Fail if reconstructed audit partitions differ from artifact metadata."""
    try:
        expected = document["training"]["splits"]
        for split_name in ("calibration", "test"):
            frame = splits[split_name]
            expected_split = expected[split_name]
            if len(frame) != int(expected_split["rows"]):
                raise ValueError(f"{split_name} row count does not match v3 metadata")
            if int(frame["label"].sum()) != int(expected_split["muslim"]):
                raise ValueError(f"{split_name} label count does not match v3 metadata")
    except (KeyError, TypeError) as exc:
        raise ValueError("v3 metadata must include training split counts") from exc


def _prepare_comparison_splits(
    document: dict[str, Any],
    metadata: ModelArtifactMetadata,
    land_dir: Path,
    reds_dir: Path,
    survey_weight: float,
) -> tuple[dict[str, pd.DataFrame], ByteTokenizer]:
    """Reconstruct the exact splits supported by the candidate model."""
    tokenizer = ByteTokenizer(metadata.architecture.max_bytes)
    splits = prepare_splits(
        land_dir,
        reds_dir,
        "eng",
        survey_weight=survey_weight,
    )
    supported_splits = {
        name: filter_names_within_byte_limit(frame, tokenizer)
        for name, frame in splits.items()
    }
    _validate_split_counts(document, supported_splits)
    return supported_splits, tokenizer


def compare(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the paired audit and return its aggregate report."""
    metadata_path = args.v3_dir / "metadata.json"
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("v3 metadata must be a JSON object")
    metadata = ModelArtifactMetadata.from_file(metadata_path)
    if metadata.model_version != "3.0" or metadata.language != "eng":
        raise ValueError("Expected English model v3 metadata")
    survey_weight = _survey_weight(document)
    splits, v3_tokenizer = _prepare_comparison_splits(
        document,
        metadata,
        args.land_dir,
        args.reds_dir,
        survey_weight,
    )
    calibration = splits["calibration"].reset_index(drop=True)
    test = splits["test"].reset_index(drop=True)

    v2_config = ModelConfig()
    v2_model = load_classifier(args.v2_model, v2_config)
    v2_tokenizer = NameTokenizer.from_file(args.v2_vocabulary, v2_config)
    v2_calibration_logits = _v2_logits(
        v2_model,
        v2_tokenizer,
        calibration,
        args.batch_size,
    )
    v2_slope, v2_intercept = fit_platt_scaler(
        v2_calibration_logits,
        calibration["label"].to_numpy(dtype=np.float64),
    )
    v2_test_logits = _v2_logits(v2_model, v2_tokenizer, test, args.batch_size)
    v2_raw = _sigmoid(v2_test_logits)
    v2_calibrated = _sigmoid(v2_slope * v2_test_logits + v2_intercept)

    v3_model = load_byte_classifier(
        args.v3_dir / "model.safetensors",
        metadata.architecture,
    )
    v3_logits = evaluate_logits(
        v3_model,
        v3_tokenizer,
        test,
        args.batch_size,
        torch.device("cpu"),
    )
    v3 = _sigmoid(
        metadata.calibration.slope * v3_logits + metadata.calibration.intercept
    )

    names = test["name"].astype(str)
    frequencies = names.map(names.value_counts())
    normalized_v2 = names.str.lower().str.translate(_DELETE_ASCII_PUNCTUATION)
    vocabulary = set(args.v2_vocabulary.read_text(encoding="utf-8").splitlines()[2:])
    word_lists = normalized_v2.str.split()
    oov_counts = word_lists.map(
        lambda words: sum(word not in vocabulary for word in words)
    )
    word_counts = word_lists.str.len()
    land_names = set(load_land_names(args.land_dir, "eng")["name"])

    test["word_count"] = word_counts.clip(upper=4).map(
        {1: "1", 2: "2", 3: "3", 4: "4+"}
    )
    test["v2_oov"] = np.where(
        oov_counts.eq(0),
        "none",
        np.where(oov_counts.eq(word_counts), "all", "some"),
    )
    test["land_overlap"] = np.where(names.isin(land_names), "yes", "no")
    test["name_frequency"] = np.where(frequencies.eq(1), "one row", "multiple rows")

    labels = test["label"].to_numpy(dtype=np.float64)
    row_weights = np.ones(len(test), dtype=np.float64)
    unique_name_weights = 1 / frequencies.to_numpy(dtype=np.float64)
    conflicting = test.groupby("name", sort=False)["label"].nunique().gt(1)
    conflicting_names = set(conflicting[conflicting].index)

    if args.audit_frame is not None:
        group_ids, _ = pd.factorize(test["name"], sort=False)
        deidentified = pd.DataFrame(
            {
                "name_group": group_ids,
                "label": labels.astype(int),
                "word_count": test["word_count"],
                "v2_oov": test["v2_oov"],
                "land_overlap": test["land_overlap"],
                "name_frequency": test["name_frequency"],
            }
        )
        args.audit_frame.parent.mkdir(parents=True, exist_ok=True)
        deidentified.to_csv(args.audit_frame, index=False)

    report: dict[str, Any] = {
        "estimand": {
            "unit": "directly labeled SEPRI household-head row",
            "population": (
                "stable name-hash buckets 80-99 in the available SEPRI files"
                if survey_weight > 0
                else "all directly labeled household heads in the available SEPRI files"
            ),
            "comparison": "paired predictions on identical rows",
            "threshold": 0.5,
            "abstention_confidence": metadata.confidence_threshold,
            "uncertainty": (
                "percentile bootstrap resampling normalized-name clusters; "
                "95% intervals; paired within each draw"
            ),
            "test_selection_caveat": (
                "held out from fitting and calibration, but previously inspected "
                "during architecture selection"
            ),
        },
        "data_integrity": {
            "rows": len(test),
            "unique_normalized_names": int(names.nunique()),
            "duplicate_rows": int(len(test) - names.nunique()),
            "muslim_rows": int(labels.sum()),
            "missing_names": int(names.isna().sum()),
            "missing_labels": int(test["label"].isna().sum()),
            "conflicting_normalized_names": len(conflicting_names),
            "rows_with_conflicting_name_labels": int(
                names.isin(conflicting_names).sum()
            ),
            "land_overlap_rows": int(test["land_overlap"].eq("yes").sum()),
        },
        "calibration": {
            "v2_platt_slope": v2_slope,
            "v2_platt_intercept": v2_intercept,
            "v2_calibration_rows": len(calibration),
            "v3_platt_slope": metadata.calibration.slope,
            "v3_platt_intercept": metadata.calibration.intercept,
            "v3_calibration_rows": metadata.calibration.rows,
        },
        "row_weighted": {
            "v2_as_released": _weighted_metrics(
                v2_raw, labels, row_weights, metadata.confidence_threshold
            ),
            "v2_recalibrated": _weighted_metrics(
                v2_calibrated, labels, row_weights, metadata.confidence_threshold
            ),
            "v3": _weighted_metrics(
                v3, labels, row_weights, metadata.confidence_threshold
            ),
        },
        "unique_name_weighted": {
            "v2_recalibrated": _weighted_metrics(
                v2_calibrated,
                labels,
                unique_name_weights,
                metadata.confidence_threshold,
            ),
            "v3": _weighted_metrics(
                v3,
                labels,
                unique_name_weights,
                metadata.confidence_threshold,
            ),
        },
        "paired_cluster_bootstrap_v3_vs_recalibrated_v2": _bootstrap_improvements(
            test,
            v2_calibrated,
            v3,
            draws=args.bootstrap_draws,
            seed=args.seed,
            confidence=metadata.confidence_threshold,
        ),
        "subgroups": {
            column: _subgroup_results(
                test,
                v2_calibrated,
                v3,
                column,
                confidence=metadata.confidence_threshold,
            )
            for column in ("word_count", "v2_oov", "land_overlap", "name_frequency")
        },
        "artifacts": {
            "v2_hugging_face_revision": args.v2_revision,
            "v2_model_sha256": _sha256(args.v2_model),
            "v2_vocabulary_sha256": _sha256(args.v2_vocabulary),
            "v3_hugging_face_revision": args.v3_revision,
            "v3_model_sha256": _sha256(args.v3_dir / "model.safetensors"),
            "v3_metadata_sha256": _sha256(args.v3_dir / "metadata.json"),
        },
        "bootstrap": {"draws": args.bootstrap_draws, "seed": args.seed},
    }
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-model", type=Path, required=True)
    parser.add_argument("--v2-vocabulary", type=Path, required=True)
    parser.add_argument("--v3-dir", type=Path, required=True)
    parser.add_argument(
        "--v2-revision",
        default="6b92b6a5a5d8abe69f0cd10429dd48e079c00e5f",
    )
    parser.add_argument(
        "--v3-revision",
        default="96f9cd59cdfe91a8f73f4c72677ec6eeec891a1f",
    )
    parser.add_argument("--land-dir", type=Path, default=Path("../land"))
    parser.add_argument("--reds-dir", type=Path, default=Path("../reds"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-frame", type=Path)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def main() -> None:
    """Write the aggregate comparison report."""
    args = parse_args()
    report = compare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
