from loguru import logger
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
import numpy as np
import shap
import matplotlib.pyplot as pp

def model_eval (dY_true, dY_pred):
    scores_dict = dict()
    for col in dY_true.columns:  # Iterate over each output column
        mse = mean_squared_error(
            dY_true[col], dY_pred[col])  
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(
            dY_true[col], dY_pred[col])
        r2 = r2_score(
            dY_true[col], dY_pred[col])
        mae_to_dev_ratio = mae / dY_true[col].std()
        
        logger.info(f"\nMetrics for {col}:")
        logger.info(f"  MSE: {mse:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R-squared: {r2:.4f}")
        logger.info(f"  MAE/StDev_true {mae_to_dev_ratio:.3f}")
        scores_dict[col] = dict(mse=mse, rmse=rmse, mae=mae, r2=r2, mae_2_true_std_ratio=mae_to_dev_ratio)
    return scores_dict

def compute_shapley_values(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values

def get_main_features(shap_vals,  X):
    # Get the displayed feature names
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    feature_order = np.argsort(mean_abs_shap)[::-1] #Sort by importance, descending
    displayed_feature_names = X.columns[feature_order].tolist()
    return displayed_feature_names

def plot_shap(
        shap_values, X, target_name:str, input_types: str='rrs', 
        max_display=-1, xlabel=None, title=None):
    if max_display < 0:
        shap.summary_plot(
            shap_values, X, plot_type='layered_violin', color='coolwarm', show=False)
    else:
        shap.summary_plot(
            shap_values, X, plot_type='layered_violin', color='coolwarm', 
            show=False, max_display=max_display)
    if xlabel:
        pp.xlabel(xlabel)
    if title:
        pp.title(title)
    pp.savefig(f'shap_{input_types}_{target_name}.png', bbox_inches='tight', dpi=300)
