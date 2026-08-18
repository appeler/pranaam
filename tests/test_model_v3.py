"""Tests for the byte-level v3 name model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from safetensors.torch import save_file

from pranaam.model_v3 import (
    CURRENT_METADATA_SCHEMA_VERSION,
    ByteModelConfig,
    ByteNameClassifier,
    ByteTokenizer,
    CalibrationConfig,
    LabelSource,
    ModelArtifactMetadata,
    ReferencePopulation,
    load_byte_classifier,
    normalize_name,
    supports_name_script,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_normalize_name_handles_unicode_case_and_spacing() -> None:
    assert normalize_name("  SHAH\u3000Rukh  Khān ") == "shah rukh khān"


@pytest.mark.parametrize(
    ("name", "scripts", "expected"),
    [
        ("Shah Rukh Khān", ("LATIN",), True),
        ("शाहरुख खान", ("DEVANAGARI",), True),
        ("Shah खान", ("LATIN",), False),
        ("محمد خان", ("LATIN",), False),
        ("123 -", ("LATIN",), False),
    ],
)
def test_supported_script_requires_letters_from_one_expected_script(
    name: str, scripts: tuple[str, ...], expected: bool
) -> None:
    assert supports_name_script(name, scripts) is expected


def test_tokenizer_preserves_order_and_encodes_every_utf8_byte() -> None:
    tokenizer = ByteTokenizer(max_bytes=12)

    encoded = tokenizer.encode(["ab", "ba", "ख"])

    assert encoded.shape == (3, 12)
    assert encoded[0].tolist() != encoded[1].tolist()
    assert encoded[0, :4].tolist() == [1, ord("a") + 3, ord("b") + 3, 2]
    assert encoded[2, 0].item() == 1
    assert encoded[2, 4].item() == 2


def test_tokenizer_truncates_content_without_dropping_end_boundary() -> None:
    tokenizer = ByteTokenizer(max_bytes=5)
    encoded = tokenizer.encode(["abcdef"])

    assert encoded.tolist() == [[1, ord("a") + 3, ord("b") + 3, ord("c") + 3, 2]]
    assert tokenizer.max_content_bytes == 3
    assert tokenizer.normalized_utf8_length("éé") == 4
    assert tokenizer.truncates("éé")


def test_tokenizer_rejects_impossible_sequence_length() -> None:
    with pytest.raises(ValueError, match="boundaries"):
        ByteTokenizer(max_bytes=2)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"max_bytes": 2}, "max_bytes"),
        ({"embedding_dim": 0}, "dimensions"),
        ({"hidden_dim": 0}, "dimensions"),
        ({"output_dim": 0}, "dimensions"),
        ({"dropout": -0.1}, "dropout"),
        ({"dropout": 1.0}, "dropout"),
    ],
)
def test_model_config_rejects_invalid_architecture(
    settings: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ByteModelConfig(**settings)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"slope": 0}, "slope"),
        ({"slope": float("inf")}, "slope"),
        ({"intercept": float("nan")}, "intercept"),
        ({"rows": 0}, "rows"),
    ],
)
def test_calibration_config_rejects_invalid_parameters(
    settings: dict[str, object], message: str
) -> None:
    values = {
        "slope": 1.0,
        "intercept": 0.0,
        "population": ReferencePopulation.SEPRI_HOUSEHOLD_HEADS,
        "rows": 1,
    }
    values.update(settings)
    with pytest.raises(ValueError, match=message):
        CalibrationConfig(**values)  # type: ignore[arg-type]


def test_classifier_returns_valid_binary_probabilities() -> None:
    config = ByteModelConfig(
        max_bytes=16,
        embedding_dim=4,
        hidden_dim=6,
        output_dim=8,
        dropout=0,
    )
    model = ByteNameClassifier(config)
    token_ids = ByteTokenizer(config.max_bytes).encode(["shah", "खान"])

    probabilities = model.predict_proba(token_ids)

    assert probabilities.shape == (2, 2)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))


def test_load_byte_classifier_restores_safe_state(tmp_path: Path) -> None:
    config = ByteModelConfig(
        max_bytes=16,
        embedding_dim=4,
        hidden_dim=6,
        output_dim=8,
        dropout=0,
    )
    source = ByteNameClassifier(config)
    path = tmp_path / "model.safetensors"
    save_file(source.state_dict(), path)

    loaded = load_byte_classifier(path, config)

    assert loaded.training is False
    for name, tensor in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor)


def test_metadata_loads_validated_inference_contract(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    path.write_text(
        """{
  "schema_version": 2,
  "model_version": "3.0",
  "language": "eng",
  "supported_scripts": ["LATIN"],
  "architecture": {
    "max_bytes": 64,
    "embedding_dim": 8,
    "hidden_dim": 12,
    "output_dim": 16,
    "dropout": 0.1
  },
  "calibration": {
    "slope": 0.8,
    "intercept": -0.2,
    "rows": 100
  },
  "provenance": {
    "reference_population": "SEPRI household heads",
    "label_source": "Bihar land caste/community labels and SEPRI household religion",
    "calibration_population": "SEPRI household heads",
    "training_seed": 7,
    "normalization": "Unicode NFKC, casefold, collapse whitespace"
  },
  "abstention": {"confidence_threshold": 0.8}
}
""",
        encoding="utf-8",
    )

    metadata = ModelArtifactMetadata.from_file(path)

    assert metadata.schema_version == CURRENT_METADATA_SCHEMA_VERSION
    assert metadata.language == "eng"
    assert metadata.architecture.max_bytes == 64
    assert metadata.calibration.slope == 0.8
    assert metadata.calibration.population is ReferencePopulation.SEPRI_HOUSEHOLD_HEADS
    assert metadata.provenance.label_source is LabelSource.LAND_CASTE_AND_SEPRI_RELIGION


@pytest.mark.parametrize("threshold", [0.5, 1.0])
def test_metadata_rejects_invalid_abstention_threshold(
    tmp_path: Path, threshold: float
) -> None:
    path = tmp_path / "metadata.json"
    path.write_text(
        f"""{{
  "schema_version": 2,
  "model_version": "3.0",
  "language": "eng",
  "supported_scripts": ["LATIN"],
  "architecture": {{}},
  "calibration": {{"slope": 1, "intercept": 0, "rows": 1}},
  "provenance": {{
    "reference_population": "SEPRI household heads",
    "label_source": "Bihar land caste/community labels and SEPRI household religion",
    "calibration_population": "SEPRI household heads",
    "training_seed": 7,
    "normalization": "Unicode NFKC, casefold, collapse whitespace"
  }},
  "abstention": {{"confidence_threshold": {threshold}}}
}}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="confidence_threshold"):
        ModelArtifactMetadata.from_file(path)


def test_metadata_adapts_only_the_shipped_schema_one_contract(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    path.write_text(
        """{
  "schema_version": 1,
  "model_version": "3.0",
  "language": "eng",
  "supported_scripts": ["LATIN"],
  "normalization": "Unicode NFKC, casefold, collapse whitespace",
  "architecture": {},
  "calibration": {
    "slope": 1,
    "intercept": 0,
    "source": "SEPRI household heads, stable name-hash calibration split",
    "rows": 10
  },
  "training": {"seed": 7},
  "abstention": {"confidence_threshold": 0.8}
}
""",
        encoding="utf-8",
    )

    metadata = ModelArtifactMetadata.from_file(path)

    assert metadata.schema_version == 1
    assert metadata.provenance.reference_population is (
        ReferencePopulation.SEPRI_HOUSEHOLD_HEADS
    )
    assert metadata.calibration.population is (
        ReferencePopulation.SEPRI_HOUSEHOLD_HEADS
    )


def test_metadata_rejects_reused_schema_one_version(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    path.write_text(
        """{
  "schema_version": 1,
  "model_version": "3.0",
  "language": "eng",
  "supported_scripts": ["LATIN"],
  "normalization": "Unicode NFKC, casefold, collapse whitespace",
  "architecture": {},
  "calibration": {"slope": 1, "intercept": 0, "rows": 10},
  "provenance": {},
  "training": {"seed": 7},
  "abstention": {"confidence_threshold": 0.8}
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported schema 1"):
        ModelArtifactMetadata.from_file(path)


@pytest.mark.parametrize("schema_version", [True, 1.0, 3])
def test_metadata_rejects_unsupported_schema_versions(
    tmp_path: Path, schema_version: object
) -> None:
    path = tmp_path / "metadata.json"
    path.write_text(
        f'{{"schema_version": {str(schema_version).lower()}}}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Unsupported model metadata schema"):
        ModelArtifactMetadata.from_file(path)
