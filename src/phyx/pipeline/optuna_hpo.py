import numpy as np
import optuna
from loguru import logger
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from .model_trainer import XGBoostTrainer


def objective(trial: optuna.Trial, X, Y, kfolds:int=3) -> float:
    """Optuna objective with KFold CV (matches your notebook behavior)."""
    params = dict(
        objective="reg:squarederror",
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        n_estimators=trial.suggest_int("n_estimators", 50, 500),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        gamma=trial.suggest_float("gamma", 1e-8, 1.0, log=True),
    )

    cv = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    rmses = []
    for tr_idx, va_idx in cv.split(X):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        Ytr, Yva = Y.iloc[tr_idx], Y.iloc[va_idx]
        trainer = XGBoostTrainer(params=params)
        model = trainer.train_model(Xtr, Ytr, val_size=0.2, seed=42)  # quick internal val too
        Yhat = model.predict(Xva)
        rmses.append(np.sqrt(mean_squared_error(Yva, Yhat)))

    score = float(np.mean(rmses))
    logger.debug("Trial {} ({}-fold) → RMSE={:.4f}", trial.number, kfolds, score, params)

    return score
