import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error


class ModelEvaluator:
    """Simple evaluator mirroring your p3 module."""

    def evaluate(self, y_true, y_pred):
        """Return MSE, R2, MAE, RMSE."""
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mae = median_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        return mse, r2, mae, rmse
