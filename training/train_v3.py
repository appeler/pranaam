"""Train, calibrate, and evaluate Pranaam's byte-level v3 models."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

from pranaam.model_v3 import (
    ByteModelConfig,
    ByteNameClassifier,
    ByteTokenizer,
    normalize_name,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

Language = Literal["eng", "hin"]
logger = logging.getLogger(__name__)


def stable_bucket(value: str, buckets: int = 100) -> int:
    """Map a normalized name to a stable split bucket."""
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def _collapse_name_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.dropna(subset=["name", "label"]).copy()
    frame["name"] = frame["name"].astype(str).map(normalize_name)
    frame = frame[frame["name"].ne("")]
    consistency = frame.groupby("name", sort=False)["label"].nunique()
    valid_names = consistency[consistency.eq(1)].index
    return frame[frame["name"].isin(valid_names)].drop_duplicates("name")


def load_land_names(land_dir: Path, language: Language) -> pd.DataFrame:
    """Load nonconflicting caste-derived binary labels from Bihar land data."""
    logger.info("loading land names for language=%s", language)
    names = pd.read_csv(
        land_dir / "data/ryot_hindi_caste.csv.gz",
        usecols=["name_of_ryot", "caste"],
    )
    crosswalk = pd.read_stata(
        land_dir / "data/caste_codes/caste_code_landrecords.dta",
        columns=["caste_landrecords", "muslim"],
        convert_categoricals=False,
    )
    crosswalk = crosswalk[crosswalk["muslim"].isin([0, 1])].copy()
    consistency = crosswalk.groupby("caste_landrecords")["muslim"].nunique()
    valid_castes = consistency[consistency.eq(1)].index
    caste_labels = (
        crosswalk[crosswalk["caste_landrecords"].isin(valid_castes)]
        .drop_duplicates("caste_landrecords")
        .set_index("caste_landrecords")["muslim"]
    )
    names["label"] = names["caste"].map(caste_labels)
    names = names.dropna(subset=["name_of_ryot", "label"])

    if language == "hin":
        selected = names.rename(columns={"name_of_ryot": "name"})[["name", "label"]]
    else:
        translations = pd.read_parquet(
            land_dir / "data/hindi_names_religion_translated.parquet",
            columns=["name", "eng_name"],
        ).dropna(subset=["name", "eng_name"])
        translation_consistency = translations.groupby("name", sort=False)[
            "eng_name"
        ].nunique()
        valid_translations = translation_consistency[
            translation_consistency.eq(1)
        ].index
        translations = translations[
            translations["name"].isin(valid_translations)
        ].drop_duplicates("name")
        selected = names.merge(
            translations,
            left_on="name_of_ryot",
            right_on="name",
            how="inner",
            validate="many_to_one",
        )[["eng_name", "label"]].rename(columns={"eng_name": "name"})

    selected["label"] = selected["label"].astype(np.float32)
    result = _collapse_name_labels(selected).reset_index(drop=True)
    logger.info("loaded %d consistent land names", len(result))
    return result


def load_sepri_heads(reds_dir: Path) -> pd.DataFrame:
    """Load household-head names with directly recorded head religion."""
    logger.info("loading SEPRI household heads")
    paths = [
        reds_dir / "data/Sepri1/HH/SECTION01.dta",
        reds_dir / "data/Sepri2/HH/SECTION01.dta",
    ]
    frames = [
        pd.read_stata(
            path,
            columns=["q1_4", "q1_9"],
            convert_categoricals=True,
        )
        for path in paths
    ]
    heads = pd.concat(frames, ignore_index=True).dropna(subset=["q1_4", "q1_9"])
    heads = heads.rename(columns={"q1_4": "name"})
    heads["name"] = heads["name"].astype(str).map(normalize_name)
    heads = heads[heads["name"].ne("")].copy()
    heads["label"] = heads["q1_9"].eq("Muslim").astype(np.float32)
    result = heads[["name", "label"]].reset_index(drop=True)
    logger.info("loaded %d directly labeled SEPRI heads", len(result))
    return result


def prepare_splits(
    land_dir: Path,
    reds_dir: Path,
    language: Language,
    survey_weight: float = 8.0,
) -> dict[str, pd.DataFrame]:
    """Build name-grouped training, validation, calibration, and test splits."""
    if not math.isfinite(survey_weight) or survey_weight < 0:
        raise ValueError("survey_weight must be finite and nonnegative")
    land = load_land_names(land_dir, language)
    land["bucket"] = land["name"].map(stable_bucket)
    land["source"] = "land"

    if language == "hin":
        return {
            "train": land[land["bucket"].lt(70)],
            "validation": land[land["bucket"].between(70, 79)],
            "calibration": land[land["bucket"].between(80, 89)],
            "test": land[land["bucket"].ge(90)],
        }

    survey = load_sepri_heads(reds_dir)
    survey["source"] = "sepri"
    survey_names = set(survey["name"])
    land = land[~land["name"].isin(survey_names)].copy()
    survey["bucket"] = survey["name"].map(stable_bucket)
    land["weight"] = 1.0
    if survey_weight == 0:
        return {
            "train": land[land["bucket"].lt(80)],
            "validation": land[land["bucket"].between(80, 89)],
            "calibration": land[land["bucket"].ge(90)],
            "test": survey,
        }
    survey["weight"] = survey_weight
    return {
        "train": pd.concat(
            [land, survey[survey["bucket"].lt(50)]],
            ignore_index=True,
        ),
        "validation": survey[survey["bucket"].between(50, 64)],
        "calibration": survey[survey["bucket"].between(65, 79)],
        "test": survey[survey["bucket"].ge(80)],
    }


def limit_training_rows(
    frame: pd.DataFrame,
    max_rows: int,
    *,
    seed: int,
) -> pd.DataFrame:
    """Deterministically cap training rows while retaining every data stratum."""
    if max_rows <= 0:
        raise ValueError("max_train_rows must be positive")
    if max_rows >= len(frame):
        return frame.copy()

    strata = ["label"]
    if "source" in frame:
        strata.insert(0, "source")
    groups = [group for _, group in frame.groupby(strata, sort=True, observed=True)]
    if max_rows < len(groups):
        raise ValueError(
            f"max_train_rows must be at least {len(groups)} to retain every "
            "source/label stratum"
        )

    allocations = np.ones(len(groups), dtype=int)
    capacities = np.array([len(group) - 1 for group in groups], dtype=int)
    remaining = max_rows - len(groups)
    if remaining:
        quotas = remaining * capacities / capacities.sum()
        extras = np.floor(quotas).astype(int)
        allocations += extras
        leftover = remaining - int(extras.sum())
        order = np.argsort(-(quotas - extras), kind="stable")
        allocations[order[:leftover]] += 1

    sampled: list[pd.DataFrame] = []
    grouped_allocations = zip(groups, allocations, strict=True)
    for group_number, (group, count) in enumerate(grouped_allocations):
        generator = np.random.default_rng(seed + group_number)
        selected = np.sort(generator.choice(len(group), size=count, replace=False))
        sampled.append(group.iloc[selected])
    return pd.concat(sampled, ignore_index=True)


def _batches(
    frame: pd.DataFrame,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> Iterator[tuple[list[str], np.ndarray, np.ndarray]]:
    names = frame["name"].to_numpy(dtype=object)
    labels = frame["label"].to_numpy(dtype=np.float32)
    weights = (
        frame["weight"].to_numpy(dtype=np.float32)
        if "weight" in frame
        else np.ones(len(frame), dtype=np.float32)
    )
    indices = np.arange(len(frame))
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        yield names[batch].tolist(), labels[batch], weights[batch]


def _device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_logits(
    model: ByteNameClassifier,
    tokenizer: ByteTokenizer,
    frame: pd.DataFrame,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return one raw logit per row without retaining encoded names."""
    model.eval()
    parts: list[np.ndarray] = []
    with torch.inference_mode():
        for names, _, _ in _batches(frame, batch_size, shuffle=False, seed=0):
            token_ids = tokenizer.encode(names).to(device)
            parts.append(model(token_ids).cpu().numpy())
    return np.concatenate(parts).astype(np.float64, copy=False)


def train_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: ByteModelConfig,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[ByteNameClassifier, list[dict[str, float]]]:
    """Fit a v3 classifier and retain the epoch with lowest validation loss."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    model = ByteNameClassifier(config).to(device)
    tokenizer = ByteTokenizer(config.max_bytes)
    sample_weights = (
        train["weight"] if "weight" in train else pd.Series(1.0, index=train.index)
    )
    positives = float(sample_weights[train["label"].eq(1)].sum())
    negatives = float(sample_weights[train["label"].eq(0)].sum())
    if positives == 0 or negatives == 0:
        raise ValueError("training data must contain positive and negative labels")
    pos_weight = torch.tensor(
        negatives / positives,
        dtype=torch.float32,
        device=device,
    )
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
        reduction="none",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_weight = 0.0
        for names, labels, weights in _batches(
            train,
            batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            token_ids = tokenizer.encode(names).to(device)
            targets = torch.from_numpy(labels).to(device)
            batch_weights = torch.from_numpy(weights).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids)
            loss = (loss_function(logits, targets) * batch_weights).sum() / (
                batch_weights.sum()
            )
            loss.backward()
            optimizer.step()
            batch_weight_total = float(batch_weights.sum().detach().cpu())
            total_loss += float(loss.detach().cpu()) * batch_weight_total
            total_weight += batch_weight_total

        validation_logits = evaluate_logits(
            model,
            tokenizer,
            validation,
            batch_size,
            device,
        )
        validation_targets = validation["label"].to_numpy(dtype=np.float64)
        validation_loss = binary_log_loss(
            _sigmoid(validation_logits), validation_targets
        )
        epoch_result = {
            "epoch": float(epoch + 1),
            "train_weighted_loss": total_loss / total_weight,
            "validation_log_loss": validation_loss,
        }
        history.append(epoch_result)
        logger.info("epoch_result=%s", json.dumps(epoch_result))
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a model state")
    model.load_state_dict(best_state)
    model.to("cpu").eval()
    return model, history


def fit_platt_scaler(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit a positive-slope logistic calibrator on held-out predictions."""
    inputs = torch.tensor(logits, dtype=torch.float64)
    targets = torch.tensor(labels, dtype=torch.float64)
    log_slope = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    intercept = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_slope, intercept],
        max_iter=500,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        calibrated_logits = log_slope.exp() * inputs + intercept
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            calibrated_logits, targets
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_slope.detach().exp()), float(intercept.detach())


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1 + exponent)
    return result


def binary_log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Return mean binary logarithmic loss with finite clipping."""
    bounded = np.clip(probabilities, 1e-15, 1 - 1e-15)
    return float(
        -np.mean(labels * np.log(bounded) + (1 - labels) * np.log(1 - bounded))
    )


def binary_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    confidence_threshold: float,
) -> dict[str, float | int | None]:
    """Compute discrimination, calibration, and abstention metrics."""
    predictions = probabilities >= 0.5
    positives = labels == 1
    tp = int(np.sum(predictions & positives))
    fp = int(np.sum(predictions & ~positives))
    fn = int(np.sum(~predictions & positives))
    tn = int(np.sum(~predictions & ~positives))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    abstained = (probabilities > 1 - confidence_threshold) & (
        probabilities < confidence_threshold
    )
    retained = ~abstained
    retained_accuracy = (
        float(np.mean(predictions[retained] == positives[retained]))
        if np.any(retained)
        else None
    )
    bins = np.minimum((probabilities * 10).astype(int), 9)
    ece = 0.0
    for index in range(10):
        selected = bins == index
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(
                float(np.mean(probabilities[selected]))
                - float(np.mean(labels[selected]))
            )

    return {
        "rows": len(labels),
        "accuracy": (tp + tn) / len(labels),
        "precision_muslim": precision,
        "recall_muslim": recall,
        "f1_muslim": f1,
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "log_loss": binary_log_loss(probabilities, labels),
        "ece_10_bin": ece,
        "coverage": float(np.mean(retained)),
        "retained_accuracy": retained_accuracy,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def confidence_threshold(value: str) -> float:
    """Parse an abstention threshold supported by inference metadata."""
    threshold = float(value)
    if not 0.5 < threshold < 1:
        raise argparse.ArgumentTypeError("confidence must be between 0.5 and 1")
    return threshold


def survey_weight(value: str) -> float:
    """Parse a nonnegative SEPRI training weight."""
    weight = float(value)
    if not math.isfinite(weight) or weight < 0:
        raise argparse.ArgumentTypeError("survey weight must be finite and nonnegative")
    return weight


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_training(args: argparse.Namespace) -> None:
    """Execute the complete grouped-split training and evaluation workflow."""
    language: Language = args.language
    output_dir = args.output_dir / language
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = prepare_splits(
        args.land_dir,
        args.reds_dir,
        language,
        survey_weight=args.survey_weight,
    )
    if args.max_train_rows is not None:
        splits["train"] = limit_training_rows(
            splits["train"],
            args.max_train_rows,
            seed=args.seed,
        )
    counts = {
        name: {
            "rows": len(frame),
            "muslim": int(frame["label"].sum()),
        }
        for name, frame in splits.items()
    }
    logger.info("split_counts=%s", json.dumps(counts))

    config = ByteModelConfig(
        max_bytes=args.max_bytes,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
    )
    device = _device(args.device)
    logger.info("device=%s", device)
    model, history = train_model(
        splits["train"],
        splits["validation"],
        config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
    )
    model.to(device)
    tokenizer = ByteTokenizer(config.max_bytes)
    calibration_logits = evaluate_logits(
        model,
        tokenizer,
        splits["calibration"],
        args.batch_size,
        device,
    )
    calibration_labels = splits["calibration"]["label"].to_numpy(dtype=np.float64)
    slope, intercept = fit_platt_scaler(calibration_logits, calibration_labels)
    test_logits = evaluate_logits(
        model,
        tokenizer,
        splits["test"],
        args.batch_size,
        device,
    )
    test_labels = splits["test"]["label"].to_numpy(dtype=np.float64)
    raw_metrics = binary_metrics(_sigmoid(test_logits), test_labels, args.confidence)
    calibrated = _sigmoid(slope * test_logits + intercept)
    calibrated_metrics = binary_metrics(calibrated, test_labels, args.confidence)

    model.to("cpu")
    save_file(model.state_dict(), output_dir / "model.safetensors")
    calibration_source = (
        "SEPRI household heads, stable name-hash calibration split"
        if language == "eng" and args.survey_weight > 0
        else "Bihar land caste labels, stable name-hash calibration split"
    )
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "model_version": "3.0",
        "language": language,
        "supported_scripts": ["LATIN" if language == "eng" else "DEVANAGARI"],
        "architecture": asdict(config),
        "normalization": "Unicode NFKC, casefold, collapse whitespace",
        "calibration": {
            "slope": slope,
            "intercept": intercept,
            "source": calibration_source,
            "rows": len(splits["calibration"]),
        },
        "abstention": {"confidence_threshold": args.confidence},
        "training": {
            "seed": args.seed,
            "source": (
                "Bihar land caste-derived labels plus directly labeled "
                "SEPRI household heads"
                if language == "eng" and args.survey_weight > 0
                else "Bihar land names with caste-derived binary labels"
            ),
            "splits": counts,
            **(
                {"survey_weight": args.survey_weight}
                if language == "eng" and args.survey_weight > 0
                else {}
            ),
        },
        "evaluation": {
            "source": (
                "SEPRI household heads"
                if language == "eng"
                else "Bihar land caste labels, stable name-hash test split"
            ),
            "raw": raw_metrics,
            "calibrated": calibrated_metrics,
        },
    }
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(
        output_dir / "training-report.json",
        {"history": history, "metadata": metadata},
    )
    logger.info("calibrated_test=%s", json.dumps(calibrated_metrics))


def parse_args() -> argparse.Namespace:
    """Parse command-line options for a reproducible training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=["eng", "hin"], required=True)
    parser.add_argument("--land-dir", type=Path, default=Path("../land"))
    parser.add_argument("--reds-dir", type=Path, default=Path("../reds"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-bytes", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--confidence", type=confidence_threshold, default=0.8)
    parser.add_argument("--survey-weight", type=survey_weight, default=8.0)
    parser.add_argument("--max-train-rows", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    run_training(parse_args())
