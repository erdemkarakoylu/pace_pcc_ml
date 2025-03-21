from p0_data_loader import DataLoader
from p1_model_trainer import XGBoostTrainer
from p2_nested_cv import NestedCV
from p3_model_evaluator import ModelEvaluator
from sklearn.model_selection import train_test_split
from pathlib import Path
from logger_utils import logger


if __name__ == "__main__":
    # Data Loading and Preparation

    project_path = Path(__file__).resolve().parent
    data_path = project_path.parent / 'data'
    logger.info("Loading data.")

    data_loader = DataLoader(data_path)  # Update with your data path
    dX, dY = data_loader.load_data()

    # Splitting Data
    dX_train, dX_test, dY_train, dY_test = train_test_split(dX, dY, test_size=0.2, random_state=42)

    # Example parameters for the XGBoost model (for the initial model, will be tuned)
    params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.1,
        "max_depth": 3,
        "n_estimators": 100,
    }

    # Class Instantiations
    model_trainer = XGBoostTrainer(params)  # Instantiate the model trainer
    nested_cv = NestedCV(params)           # Instantiate the nested cross-validator
    model_evaluator = ModelEvaluator()     # Instantiate the model evaluator

    # Perform nested cross-validation and get the best hyperparameters
    best_params = nested_cv.perform_nested_cv(dX_train, dY_train, model_trainer)

    # Update model parameters based on the findings of the nested cross-validation
    model_trainer.params.update(best_params)

    # Train the final model using the tuned hyperparameters
    final_model = model_trainer.train_model(dX_train, dY_train)

    # Make predictions on new data (X_test) along with uncertainty estimates
    mean_predictions, std_dev_predictions = model_trainer.predict_with_uncertainty(final_model, dX_test)

    # Evaluate the model's performance
    mse, r2 = model_evaluator.evaluate(dY_test, mean_predictions)
    print(f"MSE: {mse}, R^2: {r2}")

    # Log evaluation results to MLflow (if desired)
    # ...
