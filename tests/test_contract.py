"""Contract 1.1 score-form columns, uncertainty, and prior shifting."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from pranaam._contract import (
    ResultProvenance,
    metadata_columns,
    preserve_reserved_columns,
)
from pranaam.naam import Naam, _shift_prior
from tests.test_naam import metadata, mock_language_model

PROVENANCE = ResultProvenance(
    target="muslim-name-pattern",
    input_scope="full-name",
    model_id="test",
    model_version="3.0",
    model_revision="0" * 40,
    reference_population="test population",
    calibration_status="platt-scaled",
    calibration_reference="test population",
)


class TestContractColumns:
    def test_columns_are_complete_and_typed(self):
        columns = metadata_columns(
            PROVENANCE,
            scored=[True, False],
            script_supported=[True, None],
            abstention_reasons=[None, "missing-name"],
        )
        assert columns["inference_contract_version"][0] == "1.1"
        assert columns["result_form"][0] == "score"
        assert list(columns["abstained"]) == [False, True]
        assert columns["scored"].dtype.name == "boolean"
        assert pd.isna(columns["uncertainty_method"][0])

    def test_unscored_row_must_abstain(self):
        with pytest.raises(ValueError, match="unscored row must abstain"):
            metadata_columns(
                PROVENANCE,
                scored=[False],
                script_supported=[True],
                abstention_reasons=[None],
            )

    def test_unsupported_script_row_cannot_be_scored(self):
        with pytest.raises(ValueError, match="cannot be scored"):
            metadata_columns(
                PROVENANCE,
                scored=[True],
                script_supported=[False],
                abstention_reasons=[None],
            )

    def test_reason_outside_vocabulary_rejected(self):
        with pytest.raises(ValueError, match="unknown abstention reason"):
            metadata_columns(
                PROVENANCE,
                scored=[False],
                script_supported=[True],
                abstention_reasons=["made-up"],
            )

    def test_misaligned_vectors_rejected(self):
        with pytest.raises(ValueError, match="must align"):
            metadata_columns(
                PROVENANCE,
                scored=[True, True],
                script_supported=[True],
                abstention_reasons=[None],
            )

    def test_reserved_input_columns_are_preserved(self):
        data = pd.DataFrame({"name": ["x"], "scored": [1], "input_scored": [2]})
        result = preserve_reserved_columns(data, ["scored"])
        assert list(result.columns) == ["name", "input_scored_1", "input_scored"]
        assert data.columns.tolist() == ["name", "scored", "input_scored"]


class TestFrameInput:
    @patch.object(Naam, "_model_for")
    def test_dataframe_preserves_rows_order_and_index(self, mock_model_for: Mock):
        model = mock_language_model()
        model.predict.return_value = np.array([0.9, 0.2])
        mock_model_for.return_value = model
        data = pd.DataFrame({"nm": ["One", "Two"], "keep": [1, 2]}, index=[7, 9])

        result = Naam.estimate_muslim_name_pattern(data, "nm")

        assert list(result.index) == [7, 9]
        assert result["keep"].tolist() == [1, 2]
        assert result["muslim_score"].tolist() == [0.9, 0.2]
        assert "muslim_score" not in data.columns

    @patch.object(Naam, "_model_for")
    def test_reserved_input_column_is_renamed(self, mock_model_for: Mock):
        model = mock_language_model()
        model.predict.return_value = np.array([0.9])
        mock_model_for.return_value = model
        data = pd.DataFrame({"nm": ["One"], "scored": ["mine"]})

        result = Naam.estimate_muslim_name_pattern(data, "nm")

        assert result["input_scored"].tolist() == ["mine"]
        assert bool(result["scored"].iloc[0])

    def test_missing_or_duplicate_column_raises(self):
        with pytest.raises(ValueError, match="name_column is required"):
            Naam.estimate_muslim_name_pattern(pd.DataFrame({"nm": ["x"]}))
        with pytest.raises(ValueError, match="exactly once"):
            Naam.estimate_muslim_name_pattern(pd.DataFrame({"a": [1]}), "nm")

    def test_unsupported_input_type_raises(self):
        with pytest.raises(TypeError, match="must be a DataFrame"):
            Naam.estimate_muslim_name_pattern(42)  # type: ignore[arg-type]


class TestUncertainty:
    @patch.object(Naam, "_model_for")
    def test_monte_carlo_columns_summarize_dropout_samples(self, mock_model_for: Mock):
        model = mock_language_model()
        model.predict.return_value = np.array([0.6])
        model.predict_monte_carlo.return_value = np.array([[0.4], [0.6], [0.8]])
        mock_model_for.return_value = model

        result = Naam.estimate_muslim_name_pattern(
            ["Name"], uncertainty_level=0.9, mc_iterations=3
        )

        row = result.iloc[0]
        assert row["muslim_score"] == 0.6
        assert row["muslim_score_mc_mean"] == pytest.approx(0.6)
        assert row["muslim_score_mc_lower"] < row["muslim_score_mc_upper"]
        assert row["uncertainty_method"] == "monte-carlo-dropout"
        assert row["uncertainty_level"] == 0.9

    @patch.object(Naam, "_model_for")
    def test_point_estimates_carry_no_interval_columns(self, mock_model_for: Mock):
        model = mock_language_model()
        model.predict.return_value = np.array([0.6])
        mock_model_for.return_value = model

        result = Naam.estimate_muslim_name_pattern(["Name"])

        assert "muslim_score_mc_mean" not in result.columns
        assert pd.isna(result.iloc[0]["uncertainty_method"])

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"uncertainty_level": 0}, "uncertainty_level"),
            ({"uncertainty_level": 1}, "uncertainty_level"),
            ({"uncertainty_level": 0.9, "mc_iterations": 1}, "mc_iterations"),
            ({"prior": 0}, "prior"),
            ({"prior": 1.5}, "prior"),
        ],
    )
    def test_out_of_domain_options_raise(self, kwargs: dict, message: str):
        with pytest.raises(ValueError, match=message):
            Naam.estimate_muslim_name_pattern(["Name"], **kwargs)

    def test_dropout_layers_are_restored_after_sampling(self):
        from pranaam.model_v3 import ByteModelConfig, ByteNameClassifier, ByteTokenizer
        from pranaam.naam import _LanguageModel

        classifier = ByteNameClassifier(ByteModelConfig())
        classifier.eval()
        model = _LanguageModel(
            classifier=classifier,
            tokenizer=ByteTokenizer(ByteModelConfig().max_bytes),
            metadata=metadata(),
        )

        samples = model.predict_monte_carlo(["Test Name"], 4)

        assert samples.shape == (4, 1)
        assert all(not module.training for module in classifier.modules())


class TestPriorShift:
    def test_shift_matches_the_odds_ratio_definition(self):
        shifted = _shift_prior(np.array([0.5]), reference_prior=0.1, target_prior=0.5)
        # Reference odds 1 for a 0.5 score; target/reference prior odds ratio is 9.
        assert shifted[0] == pytest.approx(9 / 10)

    def test_identity_when_priors_agree(self):
        scores = np.array([0.1, 0.5, 0.9])
        shifted = _shift_prior(scores, reference_prior=0.3, target_prior=0.3)
        assert shifted == pytest.approx(scores)

    @patch.object(Naam, "_model_for")
    def test_prior_columns_record_both_base_rates(self, mock_model_for: Mock):
        model = mock_language_model()
        model.predict.return_value = np.array([0.5])
        mock_model_for.return_value = model

        result = Naam.estimate_muslim_name_pattern(["Name"], prior=0.5)

        row = result.iloc[0]
        assert row["reference_prior"] == pytest.approx(0.1)
        assert row["target_prior"] == pytest.approx(0.5)
        assert row["muslim_score"] == pytest.approx(0.9)

    @patch.object(Naam, "_model_for")
    def test_prior_without_a_reference_base_rate_raises(self, mock_model_for: Mock):
        model = mock_language_model(metadata())
        object.__setattr__(model.metadata, "reference_prior", None)
        model.predict.return_value = np.array([0.5])
        mock_model_for.return_value = model

        with pytest.raises(ValueError, match="no reference base rate"):
            Naam.estimate_muslim_name_pattern(["Name"], prior=0.5)


class TestReferencePriorParsing:
    @pytest.mark.parametrize(
        "evaluation",
        [
            None,
            {"calibrated": None},
            {"calibrated": {}},
            {"calibrated": {"true_positive": 1, "false_negative": 1, "rows": 0}},
            {"calibrated": {"true_positive": 0, "false_negative": 0, "rows": 10}},
            {"calibrated": {"true_positive": 9, "false_negative": 9, "rows": 10}},
            {"calibrated": {"true_positive": "x", "false_negative": 1, "rows": 10}},
        ],
    )
    def test_unusable_confusion_matrices_yield_no_prior(self, evaluation: object):
        from pranaam.model_v3 import _reference_prior

        document = {} if evaluation is None else {"evaluation": evaluation}
        assert _reference_prior(document) is None

    def test_usable_confusion_matrix_yields_the_positive_share(self):
        from pranaam.model_v3 import _reference_prior

        document = {
            "evaluation": {
                "calibrated": {
                    "true_positive": 40,
                    "false_negative": 10,
                    "rows": 500,
                }
            }
        }
        assert _reference_prior(document) == pytest.approx(0.1)


class TestModelOutputValidation:
    @pytest.mark.parametrize(
        ("returned", "message"),
        [
            (np.array([0.5, 0.5]), "shape"),
            (np.array([np.nan]), "non-finite"),
            (np.array([1.5]), "between zero and one"),
        ],
    )
    @patch.object(Naam, "_model_for")
    def test_bad_model_output_is_reported(
        self, mock_model_for: Mock, returned: np.ndarray, message: str
    ):
        model = mock_language_model()
        model.predict.return_value = returned
        mock_model_for.return_value = model

        with pytest.raises(RuntimeError, match=message):
            Naam.estimate_muslim_name_pattern(["Name"])

    @patch.object(Naam, "_model_for")
    def test_model_loading_failure_is_reported(self, mock_model_for: Mock):
        mock_model_for.side_effect = OSError("hub unreachable")

        with pytest.raises(RuntimeError, match="Prediction failed"):
            Naam.estimate_muslim_name_pattern(["Name"])
