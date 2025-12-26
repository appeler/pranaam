"""Main prediction module for religion classification."""

from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd
import tensorflow as tf

from .base import Base
from .logging import get_logger

logger = get_logger()


def is_english(text: str) -> bool:
    """Check if text contains only ASCII characters (English).

    Args:
        text: Input text to check

    Returns:
        bool: True if text is ASCII-only (English), False otherwise
    """
    try:
        text.encode(encoding="utf-8").decode("ascii")
        return True
    except UnicodeDecodeError:
        return False


class Naam(Base):
    """Main class for religion prediction from names."""

    MODELFN: str = "model"
    weights_loaded: bool = False
    model: tf.keras.Model | None = None
    model_path: Path | None = None
    classes: Final[list[str]] = ["not-muslim", "muslim"]
    cur_lang: str = "eng"
    model_name: Final[str] = "eng_and_hindi_models_v2"

    @classmethod
    def pred_rel(
        cls,
        names: str | list[str] | pd.Series,
        lang: Literal["eng", "hin"] = "eng",
        latest: bool = False,
    ) -> pd.DataFrame:
        """Predict religion based on name(s).

        Args:
            names: Single name string, list of names, or pandas Series of names
            lang: Language of input ('eng' for English, 'hin' for Hindi)
            latest: Whether to download latest model version

        Returns:
            pd.DataFrame: DataFrame with columns: name, pred_label, pred_prob_muslim

        Raises:
            ValueError: If invalid language specified
            RuntimeError: If model loading fails or TensorFlow not available
        """
        if lang not in ["eng", "hin"]:
            raise ValueError(f"Unsupported language: {lang}. Use 'eng' or 'hin'")

        match names:
            case str():
                name_list = [names]
            case pd.Series():
                name_list = names.tolist()
            case _:
                name_list = list(names)

        if not name_list:
            raise ValueError("Input names list cannot be empty")

        for i, name in enumerate(name_list):
            if not name or not name.strip():
                raise ValueError(
                    f"Name at index {i} is empty or contains only whitespace"
                )

        if not cls.weights_loaded or cls.cur_lang != lang:
            cls._load_model(lang, latest)

        try:
            if cls.model is None:
                raise RuntimeError("Model not loaded properly")

            results = cls.model.predict(name_list, verbose=0)
            predictions = tf.argmax(results, axis=1)
            probabilities = tf.nn.softmax(results)

            labels = [cls.classes[pred] for pred in predictions.numpy()]
            muslim_probs = [
                float(np.around(prob[1] * 100)) for prob in probabilities.numpy()
            ]

            return pd.DataFrame(
                {
                    "name": name_list,
                    "pred_label": labels,
                    "pred_prob_muslim": muslim_probs,
                }
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e

    @classmethod
    def _load_model(cls, lang: Literal["eng", "hin"], latest: bool = False) -> None:
        """Load the appropriate model for the specified language.

        Args:
            lang: Language code ('eng' or 'hin')
            latest: Whether to download latest model version

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            cls.model_path = cls.load_model_data(cls.model_name, latest)
            if cls.model_path is None:
                raise RuntimeError("Failed to load model data")

            model_filename = f"{lang}_model.keras"
            model_full_path = cls.model_path / cls.model_name / model_filename

            logger.info(f"Loading {lang} model from {model_full_path}")

            # Load Keras 3 compatible model using tf-keras
            cls.model = cls._load_model_with_compatibility(str(model_full_path), lang)
            cls.weights_loaded = True
            cls.cur_lang = lang

        except Exception as e:
            logger.error(f"Failed to load {lang} model: {e}")
            raise RuntimeError(f"Failed to load {lang} model: {e}") from e

    @classmethod
    def _load_model_with_compatibility(
        cls, model_path: str, lang: Literal["eng", "hin"]
    ) -> tf.keras.Model:
        """Load Keras 3 compatible model using tf-keras for compatibility.

        Args:
            model_path: Path to .keras model file
            lang: Language code ('eng' or 'hin')

        Returns:
            tf.keras.Model: Loaded and configured model

        Raises:
            RuntimeError: If model loading fails
        """
        # Fix Windows encoding for model assets with Unicode content
        import sys

        if sys.platform == "win32":
            import os

            os.environ.setdefault("PYTHONIOENCODING", "utf-8")

        try:
            # Use tf-keras for loading the migrated Keras 3 models
            import tf_keras as keras

            logger.info(f"Loading {lang} model with tf-keras compatibility layer")
            return keras.models.load_model(model_path)
        except ImportError:
            logger.info(
                f"tf-keras not available, trying standard Keras for {lang} model"
            )
            try:
                return tf.keras.models.load_model(model_path)
            except Exception as e:
                raise RuntimeError(
                    f"Standard Keras loading failed for {lang} model: {e}"
                ) from e
        except Exception as e:
            logger.error(f"tf-keras loading failed for {lang} model: {e}")
            raise RuntimeError(f"Model loading failed for {lang} model: {e}") from e
