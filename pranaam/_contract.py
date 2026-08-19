"""Inference-contract metadata for score-form results.

Implements the common-column block of the appeler inference result contract,
version 1.1 (https://github.com/appeler/appellation). Pranaam returns
score-form results: one calibrated probability for a stated quantity, with
no label and no per-category probability columns. For a binary target a
single score carries the whole distribution, and the cutoff that would turn
it into a label belongs to the caller's decision problem, not to this
package.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

CONTRACT_VERSION = "1.1"
ESTIMATE_TYPE = "name-pattern estimate"
RESULT_FORM = "score"

ABSTENTION_REASONS = frozenset(
    {
        "missing-name",
        "no-letters",
        "unsupported-script",
        "unsupported-context",
        "input-truncated",
        "out-of-vocabulary",
        "out-of-dictionary",
        "uncertain-score",
        "insufficient-evidence",
    }
)


@dataclass(frozen=True)
class ResultProvenance:
    """Constant provenance facts for one result frame."""

    target: str
    input_scope: str
    model_id: str
    model_version: str
    model_revision: str
    reference_population: str
    calibration_status: str
    calibration_reference: str
    uncertainty_method: str | None = None
    uncertainty_level: float | None = None


def preserve_reserved_columns(data: pd.DataFrame, reserved: list[str]) -> pd.DataFrame:
    """Rename input columns that collide with estimator output.

    A colliding input column is preserved under ``input_<name>``, with a
    numeric suffix when that name is also taken.

    Args:
        data: Input frame, not mutated.
        reserved: Column names the estimator is about to append.

    Returns:
        A copy of ``data`` with colliding columns renamed.
    """
    renames: dict[str, str] = {}
    taken = set(data.columns)
    for name in reserved:
        if name not in taken:
            continue
        candidate = f"input_{name}"
        suffix = 1
        while candidate in taken:
            candidate = f"input_{name}_{suffix}"
            suffix += 1
        renames[name] = candidate
        taken.add(candidate)
    return data.rename(columns=renames) if renames else data.copy()


def metadata_columns(
    provenance: ResultProvenance,
    *,
    scored: list[bool],
    script_supported: list[bool | None],
    abstention_reasons: list[str | None],
) -> dict[str, object]:
    """Build the contract's common columns for one result frame.

    Args:
        provenance: Constant provenance facts for this result.
        scored: Whether each row produced a usable score.
        script_supported: Whether each row's input script is supported.
        abstention_reasons: Reason per abstaining row, ``None`` elsewhere.

    Returns:
        Column name to values, in contract order.

    Raises:
        ValueError: If the invariants between the three row vectors fail or a
            reason is outside the shared vocabulary.
    """
    rows = len(scored)
    if len(script_supported) != rows or len(abstention_reasons) != rows:
        raise ValueError("scored, script_supported, and reasons must align")
    abstained = [reason is not None for reason in abstention_reasons]
    for row in range(rows):
        if not scored[row] and not abstained[row]:
            raise ValueError("an unscored row must abstain")
        if script_supported[row] is False and scored[row]:
            raise ValueError("an unsupported-script row cannot be scored")
        reason = abstention_reasons[row]
        if reason is not None and reason not in ABSTENTION_REASONS:
            raise ValueError(f"unknown abstention reason: {reason}")

    def constant(value: object) -> object:
        return pd.array([value] * rows, dtype="string")

    return {
        "inference_contract_version": constant(CONTRACT_VERSION),
        "estimate_type": constant(ESTIMATE_TYPE),
        "result_form": constant(RESULT_FORM),
        "target": constant(provenance.target),
        "input_scope": constant(provenance.input_scope),
        "scored": pd.array(scored, dtype="boolean"),
        "script_supported": pd.array(script_supported, dtype="boolean"),
        "abstained": pd.array(abstained, dtype="boolean"),
        "abstention_reason": pd.array(abstention_reasons, dtype="string"),
        "model_id": constant(provenance.model_id),
        "model_version": constant(provenance.model_version),
        "model_revision": constant(provenance.model_revision),
        "reference_population": constant(provenance.reference_population),
        "calibration_status": constant(provenance.calibration_status),
        "calibration_reference": constant(provenance.calibration_reference),
        "uncertainty_method": constant(provenance.uncertainty_method),
        "uncertainty_level": pd.array(
            [provenance.uncertainty_level] * rows, dtype="Float64"
        ),
    }
