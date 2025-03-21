import matplotlib.pyplot as pp
import numpy as np

def plot_pca_variance(pca, max_components=None):
    """
    Plots the cumulative explained variance ratio as the number of PCA components increases.

    Parameters
    ----------
    pca : sklearn.decomposition.PCA object
        Fitted PCA object.
    """
    pp.figure()
    pp.plot(np.cumsum(pca.explained_variance_ratio_))
    pp.xlabel('Number of Components')
    pp.ylabel('Cumulative Explained Variance')
    pp.title('PCA Variance Explained')
    pp.show()

def plot_rfe_ranking(selector, X_train, output_var):
    """
    Plots the ranking of features selected by RFE for a specific output variable.

    Parameters
    ----------
    selector : sklearn.feature_selection.RFE object
        Fitted RFE object for the specified output variable.
    X_train : pandas DataFrame
        Training set for features.
    output_var : str
        Name of the output variable.
    """
    pp.figure()
    pp.barh(X_train.columns[selector.support_], selector.ranking_[selector.support_])
    pp.xlabel('Ranking')
    pp.ylabel('Features')
    pp.title(f'RFE Feature Ranking for {output_var}')
    pp.show()

def plot_actual_vs_predicted(y_test, y_pred, output_var):
    """
    Plots a scatter plot comparing actual and predicted values for a specific output variable.

    Parameters
    ----------
    y_test : pandas Series
        Testing set for the specified output variable.
    y_pred : numpy array
        Predicted values for the specified output variable.
    output_var : str
        Name of the output variable.
    """
    pp.figure()
    pp.scatter(y_test, y_pred)
    pp.xlabel('Actual Values')
    pp.ylabel('Predicted Values')
    pp.title(f'Actual vs. Predicted for {output_var}')
    pp.show()

def plot_residuals(y_test, y_pred, output_var):
    """
    Plots residuals (predicted - actual) against predicted values for a specific output variable.

    Parameters
    ----------
    y_test : pandas Series
        Testing set for the specified output variable.
    y_pred : numpy array
        Predicted values for the specified output variable.
    output_var : str
        Name of the output variable.
    """
    residuals = y_pred - y_test
    pp.figure()
    pp.scatter(y_pred, residuals)
    pp.xlabel('Predicted Values')
    pp.ylabel('Residuals')
    pp.title(f'Residuals vs. Predicted for {output_var}')
    pp.axhline(y=0, color='r', linestyle='--')
    pp.show()

