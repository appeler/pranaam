#!/usr/bin/env python3
"""Convert the released Keras name models to parity-tested safetensors files.

Run from the repository root with the historical conversion dependencies:

``uv run --with-editable . --with tensorflow==2.21.* --with tf-keras==2.21.* scripts/adhoc/convert_keras_to_pytorch.py SOURCE OUTPUT``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import tf_keras as keras
import torch
from safetensors.torch import save_file

from pranaam.model import ModelConfig, NameClassifier, NameTokenizer

LANGUAGES = ("eng", "hin")
EDGE_CASE_SAMPLES = {
    "eng": [
        "Shah Rukh Khan",
        "Amitabh Bachchan",
        "MOHAMMED ALI",
        "O'Connor-Smith",
        "unknown-token-xyzzy",
        " ".join(["singh"] * 51),
    ],
    "hin": [
        "शाहरुख खान",
        "अमिताभ बच्चन",
        "सलमान खान",
        "अक्षय-कुमार",
        "अज्ञातनाम",
        " ".join(["सिंह"] * 51),
    ],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_archive(archive_path: Path, destination: Path) -> Path:
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(destination, filter="data")
    candidates = [path for path in destination.iterdir() if path.is_dir()]
    if len(candidates) != 1:
        raise ValueError("Model archive must contain exactly one top-level directory")
    return candidates[0]


def _load_source(path: Path, temporary_dir: Path) -> Path:
    if path.is_dir():
        return path
    if not path.is_file():
        raise FileNotFoundError(path)
    return _extract_archive(path, temporary_dir)


def _copy_weights(source: Any, target: NameClassifier) -> None:
    network = source.layers[-1]
    weights = network.get_weights()
    if len(weights) != 3:
        raise ValueError(f"Expected three learned arrays, found {len(weights)}")
    embedding, classifier_kernel, classifier_bias = weights

    expected = {
        "embedding": tuple(target.embedding.weight.shape),
        "classifier_kernel": tuple(reversed(target.classifier.weight.shape)),
        "classifier_bias": tuple(target.classifier.bias.shape),
    }
    actual = {
        "embedding": tuple(embedding.shape),
        "classifier_kernel": tuple(classifier_kernel.shape),
        "classifier_bias": tuple(classifier_bias.shape),
    }
    if actual != expected:
        raise ValueError(f"Unexpected learned-array shapes: {actual}; expected {expected}")

    with torch.no_grad():
        target.embedding.weight.copy_(torch.from_numpy(embedding))
        target.classifier.weight.copy_(torch.from_numpy(classifier_kernel.T.copy()))
        target.classifier.bias.copy_(torch.from_numpy(classifier_bias))
    target.eval()


def _parity_samples(language: str, vocabulary: list[str]) -> list[str]:
    usable_tokens = vocabulary[2:]
    stride = max(1, len(usable_tokens) // 512)
    selected = usable_tokens[::stride][:512]
    pairs = [
        f"{token} {selected[(index + 1) % len(selected)]}"
        for index, token in enumerate(selected)
    ]
    return [*EDGE_CASE_SAMPLES[language], *selected, *pairs]


def _convert_language(
    language: str,
    source_dir: Path,
    output_dir: Path,
    tolerance: float,
) -> dict[str, Any]:
    source_path = source_dir / f"{language}_model.keras"
    source_model = keras.models.load_model(source_path)
    vectorizer = source_model.layers[0]
    vocabulary = vectorizer.get_vocabulary()

    config = ModelConfig()
    tokenizer = NameTokenizer(vocabulary, config)
    converted = NameClassifier(config)
    _copy_weights(source_model, converted)

    samples = _parity_samples(language, vocabulary)
    keras_probabilities = np.asarray(
        source_model.predict(np.asarray(samples, dtype=str), verbose=0)
    )
    torch_probabilities = converted.predict_proba(tokenizer.encode(samples)).numpy()
    max_absolute_error = float(
        np.max(np.abs(keras_probabilities - torch_probabilities))
    )
    if max_absolute_error > tolerance:
        raise RuntimeError(
            f"{language} parity failed: {max_absolute_error:.3g} > {tolerance:.3g}"
        )

    language_dir = output_dir / language
    language_dir.mkdir(parents=True, exist_ok=True)
    weights_path = language_dir / "model.safetensors"
    vocabulary_path = language_dir / "vocabulary.txt"
    save_file(
        {key: value.detach().contiguous() for key, value in converted.state_dict().items()},
        weights_path,
    )
    vocabulary_path.write_text("\n".join(vocabulary) + "\n", encoding="utf-8")

    return {
        "language": language,
        "samples": len(samples),
        "max_absolute_error": max_absolute_error,
        "source_model_sha256": _sha256(source_path),
        "model_sha256": _sha256(weights_path),
        "vocabulary_sha256": _sha256(vocabulary_path),
    }


def convert(
    source: Path,
    output_dir: Path,
    report_path: Path,
    tolerance: float,
) -> dict[str, Any]:
    """Convert both models and write a machine-readable parity record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pranaam-keras-source-") as temporary:
        source_dir = _load_source(source, Path(temporary))
        results = [
            _convert_language(language, source_dir, output_dir, tolerance)
            for language in LANGUAGES
        ]

    config = ModelConfig()
    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architecture": "embedding_mean_classifier",
                "classes": ["not-muslim", "muslim"],
                "preprocessing": "keras_lower_and_strip_ascii_punctuation",
                **vars(config),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = {
        "source": source.name,
        "source_archive_sha256": _sha256(source) if source.is_file() else None,
        "tolerance": tolerance,
        "config_sha256": _sha256(config_path),
        "models": results,
    }
    serialized_report = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (output_dir / "conversion.json").write_text(
        serialized_report, encoding="utf-8"
    )
    report_path.write_text(serialized_report, encoding="utf-8")
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("scripts/adhoc/keras_to_pytorch_parity.json"),
    )
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the conversion from command-line arguments."""
    args = _parse_args(argv)
    report = convert(args.source, args.output_dir, args.report, args.tolerance)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
