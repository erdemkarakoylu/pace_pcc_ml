import mlflow
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
#from xgboost import XGBRegressor
from pipeline.p1_model_trainer import XGBoostTrainer

def objective(trial, X, Y):
    params= dict(
        objective='reg:squarederror',
        learning_rate=trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        n_estimators=trial.suggest_int('n_estimators', 50, 500),
        subsample=trial.suggest_uniform('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        gamma=trial.suggest_float('gamma', 1e-8, 1.0, log=True)
    )
    model_trainer = XGBoostTrainer(params=params)

    # Log hyperparameters:
    with mlflow.start_run(nested=True):
        cv = KFold(n_splits=3, shuffle=True,  random_state=42)
        cv_scores = []
        for train_idx, valid_idx in cv.split(X):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            Y_train, Y_valid = Y.iloc[train_idx], Y.iloc[valid_idx]
            model = model_trainer.train_model(X_train, Y_train)
            Y_pred = model.predict(X_valid)
            cv_scores.append(np.sqrt(mean_squared_error(Y_valid, Y_pred)))
        avg_rmse = np.mean(cv_scores)
        mlflow.log_metric('rmse', avg_rmse)
        mlflow.log_params(params)
    return avg_rmse
