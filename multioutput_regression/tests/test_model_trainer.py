import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

from pipeline.p1_model_trainer import XGBoostTrainer


def test_train_model(mock_dX, mock_dY):  # Pass the fixtures as arguments
    """Tests if the model is trained and returns an XGBoost model."""
    params = {
        "objective": "reg:squarederror",  # Basic parameter
        "n_estimators": 50,                # Small number for faster testing
        "seed": 42                        # For reproducibility
    }
    model_trainer = XGBoostTrainer(params)
    model = model_trainer.train_model(mock_dX, mock_dY)

    assert isinstance(model, MultiOutputRegressor), "Model should be a MultiOutputRegressor instance"
    assert isinstance(model.estimator, xgb.XGBRegressor), "Base estimator should be XGBRegressor"
