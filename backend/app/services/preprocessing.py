import json

import pandas as pd

from app.core.config import get_settings
from app.schemas.prediction import PredictionRequest

settings = get_settings()

# Column order/names must exactly match what the notebook trained on:
#   numeric_features   = ["carpet_area_sqft", "floor_num", "bathroom", "balcony"]
#   categorical_features = ["location_grouped", "Furnishing", "Transaction", "Ownership", "facing"]
NUMERIC_FEATURES = ["carpet_area_sqft", "floor_num", "bathroom", "balcony"]
CATEGORICAL_FEATURES = ["location_grouped", "Furnishing", "Transaction", "Ownership", "facing"]


def _load_allowed_locations() -> set[str]:
    try:
        with open(settings.resolved_locations_path) as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()


ALLOWED_LOCATIONS = _load_allowed_locations()


def request_to_dataframe(payload: PredictionRequest) -> pd.DataFrame:
    """Build the exact one-row DataFrame the trained Pipeline expects.

    Because the exported model is a full scikit-learn Pipeline (imputer + scaler +
    one-hot encoder bundled in), we do NOT need to manually encode anything here -
    the pipeline does it. We only need to (a) use identical column names, and
    (b) map any location the model has never seen to "other", exactly like training.
    """
    location_grouped = payload.location if payload.location in ALLOWED_LOCATIONS else "other"

    row = {
        "carpet_area_sqft": payload.carpet_area_sqft,
        "floor_num": payload.floor_num,
        "bathroom": payload.bathroom,
        "balcony": payload.balcony,
        "location_grouped": location_grouped,
        "Furnishing": payload.furnishing,
        "Transaction": payload.transaction,
        "Ownership": payload.ownership,
        "facing": payload.facing,
    }

    return pd.DataFrame([row], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)
