import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger as logging
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils import resample


class XGBoostTrainer:
    def __init__(self, params):
        self.params = params

    def train_model(self, X_train, y_train):
        """Trains the XGBoost model."""
        with mlflow.start_run():
            model = MultiOutputRegressor(xgb.XGBRegressor(**self.params))
            model.fit(X_train, y_train)
            return model
        
    def predict_with_uncertainty(
            self, model, X_new, num_bootstrap_samples=100, return_all_predictions=False):
        """
        Predicts target values with uncertainty estimation using bootstrapping.
        """

        all_predictions = np.zeros(
            (
                num_bootstrap_samples, X_new.shape[0], 
                len(self.params["objective"])
                )
            )
        for i in range(num_bootstrap_samples):
            X_resampled, y_resampled = resample(self.X_train, self.y_train)
            model.fit(X_resampled, y_resampled)
            all_predictions[i] = model.predict(X_new)
        
        if return_all_predictions:
            return all_predictions
        # Calculate statistics across bootstrap samples
        mean_predictions = np.mean(all_predictions, axis=0)
        std_dev_predictions = np.std(all_predictions, axis=0)
        return mean_predictions, std_dev_predictions
    

# Configure logging
# Configure loguru's output (adjust as needed)
logging.add(sys.stderr, format="{time} - {level} - {message}", level="INFO")


def load_and_preprocess_data(data_file, output_vars):
    """
    Loads Rrs data from a CSV file and preprocesses it.

    Parameters
    ----------
    data_file : str
        Path to the CSV file containing Rrs data.
    output_vars : list of str
        Names of the output variables (columns) in the data.

    Returns
    -------
    X_train, X_test, y_train, y_test : pandas DataFrames
        Training and testing sets for features (X) and target variables (y).
    """
    data = pd.read_csv(data_file) 

    X = data.drop(output_vars, axis=1)
    y = data[output_vars]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def create_hierarchical_pipelines(y):
    """
    Creates hierarchical pipelines for each output variable, including the XGBoost models.

    Parameters
    ----------
    y : pandas DataFrame
        DataFrame containing the target variables.

    Returns
    -------
    pipelines : dict
        A dictionary where keys are output variable names and values are corresponding pipelines.
    """
    pipelines = {}
    for output_var in y.columns:
        first_level_model = MultiOutputRegressor(xgb.XGBRegressor()) 
        second_level_model = MultiOutputRegressor(xgb.XGBRegressor())

        pipe = Pipeline(steps=[
            ('first_level_model', first_level_model),
            ('second_level_model', second_level_model)
        ])
        pipelines[output_var] = pipe
    return pipelines


def train_hierarchical_pipelines(pipelines, X_train, y_train, params):
    """
    Trains the hierarchical pipelines for each output variable using the XGBoostTrainer.

    Parameters
    ----------
    pipelines : dict
        A dictionary where keys are output variable names and values are corresponding pipelines.
    X_train : pandas DataFrame
        Training set for features.
    y_train : pandas DataFrame
        Training set for target variables.
    params : dict
        Parameters for the XGBoost model.
    """
    trainer = XGBoostTrainer(params)

    for output_var in y_train.columns:
        pipe = pipelines[output_var]

        # Train the first-level model
        y_train_mean = y_train.mean(axis=1)
        first_level_model = trainer.train_model(X_train, y_train_mean.to_frame())

        # Set the trained first-level model in the pipeline
        pipe.set_params(first_level_model=first_level_model)

        # Now fit the entire pipeline (this will also fit the second-level model)
        y_train_residual = y_train[output_var] - first_level_model.predict(X_train)
        pipe.fit(X_train, y_train_residual.to_frame())
        

def save_pipelines(pipelines, save_path='.'): 
    """
    Saves the trained pipelines to disk using joblib.

    Parameters
    ----------
    pipelines : dict
        A dictionary where keys are output variable names and values are corresponding pipelines.
    save_path : str or pathlib.Path, default='.'
        The directory path where the pipelines should be saved. Defaults to the current working directory.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    for output_var, pipeline in pipelines.items():
        file_path = save_path / f'pipeline_{output_var}.joblib'
        joblib.dump(pipeline, file_path)
        logging.info(f"Saved pipeline for {output_var} to {file_path}")


def load_hierarchical_pipeline_and_predict(output_var, X_test, with_uncertainty=False, num_bootstrap_samples=100):
    """
    Loads a saved hierarchical pipeline and makes predictions on new data, optionally with uncertainty estimation.

    Parameters
    ----------
    output_var : str
        Name of the output variable for which to load the pipeline.
    X_test : pandas DataFrame
        Testing set for features.
    with_uncertainty : bool, default=False
        Whether to perform uncertainty estimation using bootstrapping
    num_bootstrap_samples : int, default=100
        Number of bootstrap samples for uncertainty estimation (only used if `with_uncertainty` is True)

    Returns
    -------
    If `with_uncertainty` is False:
        y_pred : numpy array
            Predicted values for the specified output variable.
    If `with_uncertainty` is True:
        mean_predictions, std_dev_predictions : numpy arrays
            Mean and standard deviation of predictions across bootstrap samples
    """
    loaded_pipeline = joblib.load(f'pipeline_{output_var}.joblib')
    
    if with_uncertainty:
        trainer = XGBoostTrainer({}) 
        first_level_pred, first_level_std = trainer.predict_with_uncertainty(
            loaded_pipeline.named_steps['first_level_model'], X_test, num_bootstrap_samples
        )
        second_level_pred, second_level_std = trainer.predict_with_uncertainty(
            loaded_pipeline.named_steps['second_level_model'], X_test, num_bootstrap_samples
        )

        mean_predictions = first_level_pred + second_level_pred
        std_dev_predictions = np.sqrt(first_level_std**2 + second_level_std**2) 
        return mean_predictions, std_dev_predictions
    else:
        y_pred = loaded_pipeline.predict(X_test)
        return y_pred

def evaluate(y_test, y_pred, output_var, with_uncertainty=False):
    """
    Evaluates the model's performance using various metrics.

    Parameters
    ----------
    y_test : pandas Series or DataFrame
        Testing set for target variables
    y_pred : numpy array or tuple of numpy arrays
        Predicted values. If `with_uncertainty` is True, it should be a tuple (mean_predictions, std_dev_predictions)
    output_var : str
        Name of the output variable being evaluated
    with_uncertainty: bool, default=False
        Whether uncertainty estimation was performed

    Logs
    -------
    Metrics (R^2, MAE, STD) for the specified output variable
    """
    from sklearn.metrics import mean_absolute_error, r2_score

    if with_uncertainty:
        mean_pred, std_pred = y_pred
        r2 = r2_score(y_test[output_var], mean_pred)
        mae = mean_absolute_error(y_test[output_var], mean_pred)
        std = np.mean(std_pred)
        logging.info(f"Metrics for {output_var}: R^2={r2:.3f}, MAE={mae:.3f}, STD={std:.3f}")
    else:
        r2 = r2_score(y_test[output_var], y_pred)
        mae = mean_absolute_error(y_test[output_var], y_pred)
        logging.info(f"Metrics for {output_var}: R^2={r2:.3f}, MAE={mae:.3f}")

# Main execution
if __name__ == "__main__":
    data_file = "your_rrs_data.csv" 
    output_vars = ['output_var1', 'output_var2', ...] 

    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_file, output_vars)
    pipelines = create_hierarchical_pipelines(y_train) 

    # Define XGBoost parameters
    params = {
        'objective': 'reg:squarederror', 
        'n_estimators': 100,
        'learning_rate': 0.1,
        # Add other parameters as needed
    }

    train_hierarchical_pipelines