from .logging import get_logger
from .base import Base

import numpy as np
import pandas as pd
import tensorflow as tf

logger = get_logger()


def isEnglish(s):
    try:
        s.encode(encoding="utf-8").decode("ascii")
    except UnicodeDecodeError:
        return False
    else:
        return True


class Naam(Base):
    MODELFN = "model"
    weights_loaded = False
    model = None
    classes = ["not-muslim", "muslim"]
    cur_lang = "eng"

    @classmethod
    def pred_rel(cls, input, lang="eng", latest=False):
        """
        Predict religion based on name
        Args:
            input (str): Name in Hindi Or English text
        Returns:
            output (str): religion name
        """

        if not cls.weights_loaded or cls.cur_lang != lang:
            cls.model_path = cls.load_model_data(latest)
            if lang == "eng":
                cls.model = tf.keras.models.load_model(f"{cls.model_path}/saved_model/eng_model")
            else:
                cls.model = tf.keras.models.load_model(f"{cls.model_path}/saved_model/hin_model")
            cls.weights_loaded = True
            cls.cur_lang = lang

        results = cls.model.predict(input)
        res_args = tf.argmax(results, 1)
        probs = tf.nn.softmax(results)

        labels = []
        muslim_probs = []
        for i in range(0, len(input)):
            labels.append(cls.classes[res_args[i]])
            muslim_probs.append(np.around(probs[i][1] * 100))
        return pd.DataFrame(data={"name": input, "pred_label": labels, "pred_prob_muslim": muslim_probs})
