"""Interactive web interface for Pranaam name-pattern estimates."""

import re
from typing import Literal

import pandas as pd

import pranaam
import streamlit as st


def download_file(df: pd.DataFrame) -> None:
    """Offer prediction results through Streamlit's native download control."""
    st.download_button(
        label="Download results as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="pranaam-results.csv",
        mime="text/csv",
    )


def parse_names(value: str) -> list[str]:
    """Parse comma-separated, newline-separated, or mixed manual input."""
    return [name.strip() for name in re.split(r"[,\n]+", value) if name.strip()]


def predict_dataframe(
    df: pd.DataFrame, name_column: str, lang: Literal["eng", "hin"]
) -> pd.DataFrame:
    """Add estimates without changing row order or multiplying duplicates."""
    result = df.copy()
    valid_rows = result[name_column].notna()
    names = result.loc[valid_rows, name_column]
    invalid_rows = [
        index for index, value in names.items() if not isinstance(value, str)
    ]
    if invalid_rows:
        raise TypeError(
            f"Column {name_column!r} contains a non-text value at row "
            f"{invalid_rows[0]!r}"
        )
    if names.empty:
        raise ValueError(f"Column {name_column!r} has no names to predict")

    predictions = pranaam.estimate_muslim_name_pattern(names.tolist(), lang=lang)
    dtypes = {
        "name_pattern_estimate": "string",
        "muslim_score": "Float64",
        "abstained": "boolean",
        "abstention_reason": "string",
        "script_supported": "boolean",
        "normalized_utf8_bytes": "Int64",
        "input_truncated": "boolean",
        "reference_population": "string",
        "label_source": "string",
        "calibration_population": "string",
        "model_language": "string",
        "model_version": "string",
        "model_revision": "string",
        "model_max_name_bytes": "Int64",
    }
    for column, dtype in dtypes.items():
        result[column] = pd.Series(pd.NA, index=result.index, dtype=dtype)
        result.loc[valid_rows, column] = predictions[column].to_numpy()
    return result


def render_summary(result: pd.DataFrame) -> None:
    """Render aggregate counts without presenting estimates as identities."""
    muslim_count = result["name_pattern_estimate"].eq("muslim-associated").sum()
    non_muslim_count = result["name_pattern_estimate"].eq("not-muslim-associated").sum()
    abstained_count = result["abstained"].fillna(False).sum()
    st.write(
        f"**Model summary**: {muslim_count} Muslim-associated, "
        f"{non_muslim_count} non-Muslim-associated, and "
        f"{abstained_count} abstained name-pattern estimates"
    )


def app() -> None:
    """Render the Pranaam Streamlit interface."""
    st.title("🔮 Pranaam: Muslim name-pattern estimates")

    with st.sidebar:
        st.header("About")
        st.write(
            "The English model is calibrated against SEPRI household-head "
            "labels. The Hindi model is calibrated against a held-out split "
            "of Bihar land-record labels."
        )
        st.write("See the model card for split-specific evaluation results.")
        st.write("[GitHub Repository](https://github.com/appeler/pranaam)")
        st.write("[Documentation](https://appeler.github.io/pranaam/)")
        st.write("[Model card](https://huggingface.co/gojiberries/pranaam)")

    st.write(
        """
    This app estimates whether a name follows patterns labeled **Muslim** or
    **not-Muslim** in the selected model's reference data. Results identify
    that reference population and the source of its labels.
    """
    )
    st.warning(
        "Religion is sensitive personal information. These are uncertain "
        "name-pattern estimates, not a person's self-identified religion. "
        "Do not use them to make decisions about individuals."
    )

    input_method = st.radio(
        "Choose input method:", ["Enter names manually", "Upload CSV file"]
    )

    if input_method == "Enter names manually":
        st.subheader("Enter Names")

        lang = st.selectbox(
            "Select language:",
            ["eng", "hin"],
            format_func=lambda x: "English" if x == "eng" else "Hindi",
        )

        if lang == "eng":
            example = "Shah Rukh Khan, Amitabh Bachchan, Salman Khan"
            names_input = st.text_area(
                "Enter names (one per line or comma-separated):",
                placeholder=example,
                height=100,
            )
        else:
            example = "शाहरुख खान, अमिताभ बच्चन"
            names_input = st.text_area(
                "Enter names in Hindi (one per line or comma-separated):",
                placeholder=example,
                height=100,
            )

        if st.button("Estimate name patterns"):
            if names_input.strip():
                names = parse_names(names_input)

                with st.spinner("Estimating name patterns..."):
                    try:
                        result = pranaam.estimate_muslim_name_pattern(names, lang=lang)

                        st.subheader("Results")
                        st.dataframe(result, use_container_width=True)

                        render_summary(result)

                        download_file(result)

                    except Exception as e:
                        st.error(f"Error estimating name patterns: {e!s}")
            else:
                st.warning("Please enter at least one name.")

    else:
        # CSV Upload
        st.subheader("Upload CSV File")

        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("**Data loaded successfully!**")
                st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

                with st.expander("Preview data"):
                    st.dataframe(df.head(), use_container_width=True)

                name_col = st.selectbox("Select column containing names:", df.columns)
                lang = st.selectbox(
                    "Select language:",
                    ["eng", "hin"],
                    format_func=lambda x: "English" if x == "eng" else "Hindi",
                )

                if st.button("Estimate name patterns for all names"):
                    with st.spinner("Processing names..."):
                        try:
                            result_df = predict_dataframe(df, name_col, lang)

                            st.subheader("Results")
                            st.dataframe(result_df, use_container_width=True)

                            render_summary(result_df)

                            download_file(result_df)

                        except Exception as e:
                            st.error(f"Error processing file: {e!s}")

            except Exception as e:
                st.error(f"Error loading file: {e!s}")
        else:
            st.info("Please upload a CSV file to continue.")

    st.markdown("---")
    st.markdown(
        """
    **Note**: This tool is for aggregate research and education. Its estimates
    are based on statistical patterns and must not be treated as facts about
    individuals or used for consequential decisions.
    """
    )


if __name__ == "__main__":
    app()
