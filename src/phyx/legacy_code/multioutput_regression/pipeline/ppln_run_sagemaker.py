import boto3
import sagemaker
import os
from sagemaker.remote_function import remote
from sagemaker.session import Session
import mlflow
import optuna
import sys
from pathlib import Path
import pandas as pd
import warnings

 # Import  custom pipeline modules
from pipeline.p0_data_loader import DataLoader
from pipeline.p1_model_trainer import XGBoostTrainer
from pipeline.p2_optuna_hpo import objective
from pipeline.p3_model_evaluator import ModelEvaluator
from sklearn.model_selection import train_test_split
from loguru import logger


# Initialize SageMaker session
session = Session()
role = sagemaker.get_execution_role()

# Define S3 bucket where data is stored
s3_bucket = "your-bucket-name"
s3_prefix = "craig_pfc_2023/step_2_cleaned"

# Configure the remote function environment
@remote(
    instance_type="ml.m5.4xlarge",  # Choose appropriate instance type
    instance_count=1,
    framework="sklearn",
    framework_version="1.2-1",
    base_job_name="optuna-hpo-job",
    keep_alive_period_in_seconds=3600,  # Keep instance alive for 1 hour
    role=role
)
def run_hpo_job():
    
    # Set up S3 client
    s3 = boto3.client('s3')
    
    # Download data files from S3
    s3_bucket = "buket_name"
    s3_prefix = "craig_pfc_2023/step_2_cleane"
    local_data_path = Path.home() / 'data/craig_pfc_2023/step_2_cleaned'
    
    for file_name in ['df_rrs.pqt', 'df_phy.pqt', 'df_env.pqt']:
        s3.download_file(
            s3_bucket, 
            f"{s3_prefix}/{file_name}", 
            f"{local_data_path}/{file_name}"
        )
    
    # Set up project path
    project_path = Path.cwd()
    sys.path.append(project_path.as_posix())
    
    # Filter warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Load data
    data_path = Path.home() / 'data/craig_pfc_2023/step_2_cleaned'
    assert data_path.exists()
    
    loader = DataLoader(
        data_path=data_path, rrs_file='df_rrs.pqt', 
        phy_file='df_phy.pqt', env_file='df_env.pqt')
    dX, dX_env, dY = loader.load_data()
    
    # Select subset of environmental features
    dX_env_sub = dX_env[['lat', 'temp']]
    dX = pd.concat((dX, dX_env_sub), axis=1)
    
    # CHANGE: No subsampling - use all data
    logger.info(f"\nUsing full dataset: Features shape = {dX.shape}, Targets shape = {dY.shape}")
    
    # Train/Test Split
    dX_train, dX_test, dY_train, dY_test = train_test_split(
        dX, dY, test_size=0.2, random_state=42)
    logger.info(f"\nTrain/Test split completed --> Train shape: {dX_train.shape}, Test shape: {dX_test.shape}")
    
    # Initial model training
    initial_params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.1,
        "max_depth": 3,
        "n_estimators": 100,
    }
    model_trainer = XGBoostTrainer(initial_params)
    model = model_trainer.train_model(dX_train, dY_train)
    logger.info("Initial model trained with basic hyperparameters.")
    
    # Run predictions on the test set
    preds = model.predict(dX_test)
    evaluator = ModelEvaluator()
    mse, r2, mae, rmse = evaluator.evaluate(dY_test, preds)
    
    logger.info("Initial Evaluation Results:")
    logger.info(f"MSE: {mse:.3f}, R2: {r2:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    
    # CHANGE: Run full HPO with more trials
    with mlflow.start_run():
        study = optuna.create_study(direction="minimize")
        # CHANGE: Increased number of trials from 5 to 50
        study.optimize(lambda trial: objective(trial, dX_train, dY_train), n_trials=50)
        best_params = study.best_trial.params
        print("Best hyperparameters:", best_params)
        
        # Train final model with best parameters
        best_model_trainer = XGBoostTrainer(best_params)
        best_model = best_model_trainer.train_model(dX_train, dY_train)
        
        # Evaluate final model
        final_preds = best_model.predict(dX_test)
        final_mse, final_r2, final_mae, final_rmse = evaluator.evaluate(dY_test, final_preds)
        
        logger.info("Final Model Evaluation Results:")
        logger.info(f"MSE: {final_mse:.3f}, R2: {final_r2:.3f}, MAE: {final_mae:.3f}, RMSE: {final_rmse:.3f}")
        
        # Save best model to S3
        import joblib
        model_path = "best_model.joblib"
        joblib.dump(best_model, model_path)
        s3.upload_file(model_path, s3_bucket, f"{s3_prefix}/best_model.joblib")
    
    logger.info("=== HPO Pipeline Completed ===")
    
    # Return the best parameters and evaluation metrics
    return {
        "best_params": best_params,
        "metrics": {
            "mse": final_mse,
            "r2": final_r2,
            "mae": final_mae,
            "rmse": final_rmse
        }
    }

# Run the job
results = run_hpo_job()
logger.info("HPO job completed!")
print("Best parameters:", results["best_params"])
print("Final metrics:", results["metrics"])