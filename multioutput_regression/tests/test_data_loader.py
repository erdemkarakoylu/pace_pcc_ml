import pandas as pd
import numpy as np

from pipeline.p0_data_loader import DataLoader


def test_load_data(data_dir):
    """Tests if data is loaded and preprocessed correctly."""
    test_data_path = data_dir.resolve()
    data_loader = DataLoader(test_data_path,
                             rrs_file='dx_synthetic.pqt',
                             phy_file='dy_synthetic.pqt')  # Assuming your test data is in this directory
    dX, dY = data_loader.load_data()
    
    assert dX is not None and dY is not None, "DataFrames should not be None"
    assert isinstance(dX, pd.DataFrame), "dX should be a DataFrame"        
    assert not dX.empty, "dX should not be empty"
    assert not dY.empty, "dY should not be empty"
    assert isinstance(dY, pd.DataFrame), "dY should be a DataFrame"
    assert dX.shape[1] > 0, "dX should have at least one feature column"
    assert dY.shape[1] > 0, "dY should have at least one target column"
    # Add further tests for specific preprocessing steps
    assert np.all(dX.dtypes == np.floating), "dX should contain float values"
    assert np.all(dY.dtypes == np.floating), "dX should contain float values"
