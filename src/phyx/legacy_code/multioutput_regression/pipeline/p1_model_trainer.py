import mlflow
#import mlflow.xgboost
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import numpy as np
from sklearn.utils import resample

class XGBoostTrainer:
    def __init__(self, params):
        self.params = params        

    def train_model(self, X_train, y_train):
        """Trains the XGBoost model, starting an MLflow run only if none is active."""
        if mlflow.active_run() is None:
            run_context = mlflow.start_run()
        else:
            # If already inside a run, don't start a new one
            run_context = None

        try:
            model = MultiOutputRegressor(xgb.XGBRegressor(**self.params))
            model.fit(X_train, y_train)
        finally:
            if run_context is not None:
                mlflow.end_run()
        return model

    def predict_with_uncertainty(self, model, X_new, num_bootstrap_samples=100):
        """
        Predicts target values with uncertainty estimation using bootstrapping.
        """

        all_predictions = np.zeros((num_bootstrap_samples, X_new.shape[0], len(self.params["objective"])))
        for i in range(num_bootstrap_samples):
            X_resampled, y_resampled = resample(self.X_train, self.y_train)
            model.fit(X_resampled, y_resampled)
            all_predictions[i] = model.predict(X_new)
        
        # Calculate statistics across bootstrap samples
        mean_predictions = np.mean(all_predictions, axis=0)
        std_dev_predictions = np.std(all_predictions, axis=0)

        return mean_predictions, std_dev_predictions
