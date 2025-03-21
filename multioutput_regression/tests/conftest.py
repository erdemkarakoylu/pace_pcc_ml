# conftest.py

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor

from pipeline.logger_utils import setup_logger
from pipeline.p0_data_loader import DataLoader

setup_logger()

@pytest.fixture
def data_dir():
    """Fixture to get or create the test data directory."""
    data_dir = Path("./tests/test_data")
    data_dir.mkdir(parents=True, exist_ok=True)  # Create if doesn't exist
    return data_dir

# Parameters for data generation (adjust as needed)
@pytest.fixture
def n_samples():
    return 1000  
@pytest.fixture
def n_features():
    return 500  
@pytest.fixture
def n_outputs():
    return 7  
@pytest.fixture
def corr_strength():
    return 0.7
@pytest.fixture
def noise_level():
    return 0.05

@pytest.fixture
def wavelengths():
    return np.linspace(400, 800, n_features)

@pytest.fixture
def base_signals(wavelengths, n_outputs):
    return np.exp(-(wavelengths[:, np.newaxis] - np.random.rand(n_outputs) * 80)**2 / 30)

@pytest.fixture
def signals(base_signals, n_features, corr_strength):
    interaction_matrix = np.random.rand(n_outputs, n_outputs) * 0.5
    base_signals = base_signals @ interaction_matrix  

    corr_matrix = corr_strength * np.ones((n_features, n_features)) + (1 - corr_strength) * np.eye(n_features)
    return base_signals.dot(corr_matrix)

# Synthetic Data Generation Fixtures
@pytest.fixture
def dX_synthetic(n_samples, n_features, signals, corr_strength, noise_level):
    # Create a lower-dimensional latent space (e.g., representing principal components)
    n_latent_features = 5  
    latent_features = np.random.randn(n_samples, n_latent_features)

    # Simulate highly correlated features from the latent space
    feature_loadings = np.random.randn(n_features, n_latent_features) 
    features = latent_features @ feature_loadings.T + np.random.randn(n_samples, n_features) * 0.1  

    # Add noise for extra realism
    features += np.random.randn(n_samples, n_features) * noise_level

    # Return a DataFrame with appropriate column names
    return pd.DataFrame(features, columns=[f"w{i+1}" for i in range(n_features)])


@pytest.fixture
def dY_synthetic(n_samples, n_features, n_outputs, dX_synthetic, noise_level):
    # Coefficients relating latent features to outputs
    latent_coefficients = np.random.randn(n_latent_features, n_outputs)

    # Gather statistics (since you're using them in the original code)
    dX_stats = dX_synthetic.describe()
    dY_stats = []
    for i in range(n_outputs):
        y_mean = 0.5  # Placeholder mean - adjust based on your expected output range
        y_std = 1.0  # Placeholder std dev - adjust based on your expected output range
        dY_stats.append({'mean': y_mean, 'std': y_std})

    # Create synthetic targets
    synthetic_targets = []
    for i in range(n_outputs):
        y = dX_synthetic.values @ latent_coefficients[:, i] + np.random.randn(n_samples) * dY_stats[i]['std'] + dY_stats[i]['mean']
        synthetic_targets.append(y)

    # Add noise for extra realism
    synthetic_targets = np.array(synthetic_targets) + np.random.randn(n_outputs, n_samples) * noise_level

    # Return a DataFrame with appropriate column names
    return pd.DataFrame(synthetic_targets.T, columns=[f"ph{i+1}" for i in range(n_outputs)])


@pytest.fixture
def mock_dX():
    return pd.DataFrame(np.random.rand(10, 5), columns=['w1', 'w2', 'w3', 'w4', 'w5'])

@pytest.fixture
def mock_dY():
    return pd.DataFrame(np.random.rand(10, 3), columns=['ph1', 'ph2', 'ph3'])

@pytest.fixture
def mock_data_loader(mocker, mock_dX, mock_dY):
    mock = mocker.Mock(spec=DataLoader)
    mock.load_data.return_value = mock_dX, mock_dY
    return mock


@pytest.fixture  
def mock_xgb_regressor(mocker):
    mock = mocker.patch('xgboost.XGBRegressor', autospec=True)
    mock.fit.return_value = None  # Directly mock the fit method on the class
    mock.predict.return_value = np.random.rand(10, 3)  
    return mock


@pytest.fixture
def mock_trainer(mock_xgb_regressor):
    """
    This fixture now returns a MultiOutputRegressor wrapping the mocked XGBRegressor.
    """
    return MultiOutputRegressor(mock_xgb_regressor)