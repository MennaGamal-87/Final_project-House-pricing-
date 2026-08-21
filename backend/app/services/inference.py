import json
import logging

import joblib
import numpy as np
import pandas as pd

from app.core.config import get_settings

logger = logging.getLogger("app.inference")
settings = get_settings()


class ModelService:
    """Holds the loaded model pipeline in memory. Load once at startup, reuse per request."""

    def __init__(self) -> None:
        self.model = None
        self.target: str = "raw"  # "log" means the model was trained on np.log1p(price)
        # Bounds for safely clipping log-space predictions before expm1 - mirrors the
        # notebook's safe_expm1() helper so a model that extrapolates in log-space
        # (mainly linear models) can never return an astronomical, meaningless price.
        self.log_pred_min: float | None = None
        self.log_pred_max: float | None = None

    def load(self) -> None:
        logger.info("Loading model from %s", settings.resolved_model_path)
        self.model = joblib.load(settings.resolved_model_path)

        try:
            with open(settings.resolved_metadata_path) as f:
                metadata = json.load(f)
            self.target = metadata.get("target", "raw")
            self.log_pred_min = metadata.get("log_pred_min")
            self.log_pred_max = metadata.get("log_pred_max")
            logger.info("Model metadata: %s", metadata)
        except FileNotFoundError:
            logger.warning("No model_metadata.json found - assuming raw (non-log) target.")

    def predict(self, X: pd.DataFrame) -> float:
        if self.model is None:
            raise RuntimeError("Model has not been loaded yet.")
        raw_output = self.model.predict(X)[0]

        if self.target == "log":
            if self.log_pred_min is not None and self.log_pred_max is not None:
                raw_output = float(np.clip(raw_output, self.log_pred_min, self.log_pred_max))
            price = float(np.expm1(raw_output))
        else:
            price = float(raw_output)

        return max(price, 0.0)


model_service = ModelService()
