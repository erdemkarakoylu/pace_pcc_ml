from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from loguru import logger
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
import xgboost as xgb


class XGBoostTrainer:
    """Train a multi-output XGBoost model (one regressor per target)."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.model: MultiOutputRegressor | None = None
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None

    def _build(self) -> MultiOutputRegressor:
        base = xgb.XGBRegressor(tree_method="hist", n_jobs=-1, **self.params)
        return MultiOutputRegressor(base)

    def train_model(self, X_train, y_train, val_size: float = 0.2, seed: int = 42) -> MultiOutputRegressor:
        """Fit with a quick internal holdout for sanity (no eval_set to avoid 2D y issues)."""
        Xtr, Xva, ytr, yva = train_test_split(X_train, y_train, test_size=val_size, random_state=seed)
        est = self._build()
        logger.info("Fitting (quick val) on %d rows, %d features → %d targets", Xtr.shape[0], Xtr.shape[1], ytr.shape[1])
        est.fit(Xtr, ytr)
        rmse = float(np.sqrt(mean_squared_error(yva, est.predict(Xva))))
        logger.info("Quick validation RMSE: %.4f", rmse)
        self.model = est
        self.X_train_ = np.asarray(X_train)
        self.y_train_ = np.asarray(y_train)
        return est

    def fit_full(self, X_train, y_train) -> MultiOutputRegressor:
        """Fit on the full provided training set (used after HPO)."""
        est = self._build()
        logger.info(f"Fitting on FULL training set: {X_train.shape[0]} rows × {X_train.shape[1]} features")
        est.fit(X_train, y_train)
        self.model = est
        self.X_train_ = np.asarray(X_train)
        self.y_train_ = np.asarray(y_train)
        return est

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        path.parent.mkdir(parents=True, exist_ok=True
        )
        joblib.dump(self.model, path)
        logger.success("Saved model → %s", path)

    def predict(self, X_new) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        return self.model.predict(X_new)

    def predict_with_uncertainty(self, X_new, num_bootstrap_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Call train_model()/fit_full() first.")
        n_targets = self.y_train_.shape[1]
        preds = np.zeros((num_bootstrap_samples, X_new.shape[0], n_targets))
        for i in range(num_bootstrap_samples):
            Xb, yb = resample(self.X_train_, self.y_train_, random_state=1000 + i)
            est = self._build()
            est.fit(Xb, yb)
            preds[i] = est.predict(X_new)
        return preds.mean(axis=0), preds.std(axis=0)
