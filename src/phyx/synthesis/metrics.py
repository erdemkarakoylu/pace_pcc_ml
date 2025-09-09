from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import json
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)


def _safe_std(arr: np.ndarray, ddof: int = 1) -> float:
    """Sample std by default; returns NaN if undefined or zero."""
    arr = np.asarray(arr)
    if arr.size <= ddof:
        return float("nan")
    s = float(np.std(arr, ddof=ddof))
    return s if s > 0 else float("nan")


class ModelEvaluator:
    """
    Metrics-only evaluator for multi-output regression.

    Features:
      - Per-target metrics: MSE, RMSE, MAE, MAPE, R², MAE/std(y_true)
      - Aggregate metrics across targets (means), incl. MAE/std aggregate
      - Save metrics to JSON/CSV
    """

    # ---------------------
    # Aggregate metrics
    # ---------------------
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Back-compat aggregate metrics across targets (mean of per-target metrics).

        Returns:
            mse, r2, mae, rmse  (aggregate/mean across targets)
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

        if y_true.ndim == 1:
            y_true = y_true[:, None]
            y_pred = y_pred[:, None]

        mses, maes, r2s = [], [], []
        for t in range(y_true.shape[1]):
            yt, yp = y_true[:, t], y_pred[:, t]
            mses.append(mean_squared_error(yt, yp))
            maes.append(mean_absolute_error(yt, yp))
            r2s.append(r2_score(yt, yp))

        mse = float(np.mean(mses))
        mae = float(np.mean(maes))
        rmse = float(np.sqrt(mse))
        r2 = float(np.mean(r2s))
        return mse, r2, mae, rmse

    # ---------------------
    # Per-target metrics
    # ---------------------
    def evaluate_per_target(self, dY_true: pd.DataFrame, dY_pred: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Per-target metrics in a legacy-friendly shape:
            {target: {"mse","rmse","mae","mape","r2","mae_over_std"}}
        """
        if list(dY_true.columns) != list(dY_pred.columns):
            raise ValueError("Target columns do not match between true and pred.")

        scores: Dict[str, Dict[str, float]] = {}
        for col in dY_true.columns:
            yt = dY_true[col].to_numpy()
            yp = dY_pred[col].to_numpy()

            mse = mean_squared_error(yt, yp)
            rmse = float(np.sqrt(mse))
            mae = mean_absolute_error(yt, yp)
            try:
                mape = float(mean_absolute_percentage_error(yt, yp))
                if not np.isfinite(mape):
                    mape = float("nan")
            except Exception:
                mape = float("nan")
            r2 = r2_score(yt, yp)
            std_true = _safe_std(yt, ddof=1)  # sample std
            mae_over_std = float(mae / std_true) if np.isfinite(std_true) else float("nan")

            scores[col] = {
                "mse": float(mse),
                "rmse": rmse,
                "mae": float(mae),
                "mape": mape,
                "r2": float(r2),
                "mae_over_std": mae_over_std,
            }
        return scores

    def aggregate_from_per_target(self, per_target: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate metrics by mean over targets (ignoring NaNs where relevant).
        Includes 'mae_over_std' aggregate.
        """
        def mean_key(k: str) -> float:
            vals = [v.get(k, float("nan")) for v in per_target.values()]
            return float(np.nanmean(vals)) if len(vals) else float("nan")

        mse = mean_key("mse")
        mae = mean_key("mae")
        rmse = float(np.sqrt(mse)) if np.isfinite(mse) else float("nan")
        r2 = mean_key("r2")
        mae_over_std = mean_key("mae_over_std")

        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mae_over_std": mae_over_std}

    # ---------------------
    # Persistence
    # ---------------------
    @staticmethod
    def save_metrics(aggregate: Dict[str, float], per_target: Dict[str, Dict[str, float]], outdir: Path) -> None:
        outdir = Path(outdir).resolve()
        outdir.mkdir(parents=True, exist_ok=True)

        (outdir / "metrics.json").write_text(json.dumps(aggregate, indent=2))
        pd.DataFrame(per_target).T.to_csv(outdir / "metrics_per_target.csv", index=True)

        logger.info("Saved → {}", outdir / "metrics.json")
        logger.info("Saved → {}", outdir / "metrics_per_target.csv")

    # ---------------------
    # One-call (metrics only)
    # ---------------------
    def evaluate_and_save(
        self,
        Y_true: pd.DataFrame,
        Y_pred: pd.DataFrame,
        outdir: Path,
    ) -> Dict[str, Any]:
        """
        Compute per-target + aggregate metrics and write to disk.
        Returns: {"aggregate": {...}, "per_target": {...}}
        """
        per_target = self.evaluate_per_target(Y_true, Y_pred)
        aggregate = self.aggregate_from_per_target(per_target)
        self.save_metrics(aggregate, per_target, outdir)
        return {"aggregate": aggregate, "per_target": per_target}
