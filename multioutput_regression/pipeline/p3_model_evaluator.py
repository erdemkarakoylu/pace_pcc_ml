from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
import numpy as np

class ModelEvaluator:
    def evaluate(self, y_true, y_pred):
        """Evaluates model performance with multiple metrics."""
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mae = median_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return mse, r2, mae, rmse
