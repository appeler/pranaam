"""Main prediction module for religion classification."""

from typing import List, Optional, Union

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
        True if text is ASCII-only (English), False otherwise
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
    model: Optional[tf.keras.Model] = None
    model_path: Optional[str] = None
    classes: List[str] = ["not-muslim", "muslim"]
    cur_lang: str = "eng"
    model_name: str = "eng_and_hindi_models_v1"

    @classmethod
    def pred_rel(
        cls, names: Union[str, List[str]], lang: str = "eng", latest: bool = False
    ) -> pd.DataFrame:
        """Predict religion based on name(s).

        Args:
            names: Single name string or list of names
            lang: Language of input ('eng' for English, 'hin' for Hindi)
            latest: Whether to download latest model version

        Returns:
            DataFrame with columns: name, pred_label, pred_prob_muslim

        Raises:
            ValueError: If invalid language specified
            RuntimeError: If model loading fails
        """
        if lang not in ["eng", "hin"]:
            raise ValueError(f"Unsupported language: {lang}. Use 'eng' or 'hin'")

        # Convert single string to list for consistent processing
        if isinstance(names, str):
            name_list = [names]
        else:
            name_list = list(names)

        if not name_list:
            raise ValueError("Input names list cannot be empty")

        # Load model if not loaded or language changed
        if not cls.weights_loaded or cls.cur_lang != lang:
            cls._load_model(lang, latest)

        # Make predictions
        try:
            if cls.model is None:
                raise RuntimeError("Model not loaded properly")

            results = cls.model.predict(name_list, verbose=0)
            predictions = tf.argmax(results, axis=1)
            probabilities = tf.nn.softmax(results)

            # Extract results
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
    def _load_model(cls, lang: str, latest: bool = False) -> None:
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

            model_subpath = "eng_model" if lang == "eng" else "hin_model"
            model_full_path = f"{cls.model_path}/{cls.model_name}/{model_subpath}"

            logger.info(f"Loading {lang} model from {model_full_path}")

            # Try different loading methods based on TensorFlow version
            cls.model = cls._load_model_with_compatibility(model_full_path, lang)
            cls.weights_loaded = True
            cls.cur_lang = lang

        except Exception as e:
            logger.error(f"Failed to load {lang} model: {e}")
            raise RuntimeError(f"Failed to load {lang} model: {e}") from e

    @classmethod
    def _load_model_with_compatibility(
        cls, model_path: str, lang: str
    ) -> tf.keras.Model:
        """Load model with simple error handling.

        Args:
            model_path: Path to model directory
            lang: Language code for error messages

        Returns:
            Loaded TensorFlow model

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            if (
                "keras 3" in str(e).lower()
                or "file format not supported" in str(e).lower()
            ):
                raise RuntimeError(
                    f"Model format compatibility issue. "
                    f"Try: pip install 'pranaam[tensorflow-compat]'"
                ) from e
            else:
                raise RuntimeError(f"Model loading failed: {e}") from e
