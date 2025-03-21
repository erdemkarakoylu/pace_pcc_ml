from pathlib import Path

import pandas as pd
from loguru import logger


class DataLoader:
    def __init__(
            self, data_path, rrs_file='df_rrs.pqt',
            phy_file='df_phyto.pqt', env_file=None):
        data_path = Path(data_path).resolve()
        preprocessed_data_path = data_path #/ '02_extracted/pcc_sims/subset'
        self.data_path = preprocessed_data_path
        self.X_name = rrs_file
        self.Y_name = phy_file
        self.X_env_name = env_file
        logger.debug(f"Data directory set to {self.data_path.as_posix()}")
        logger.debug(f'Rrs file used: {self.X_name}')
        logger.debug(f'Phytoplankton file use {self.Y_name}')
        if env_file is not None:
            logger.debug(f"Env file used: {self.X_env_name}")
   
    def load_data(self):
        """Loads and preprocesses data."""
        try:
            dX = pd.read_parquet(self.data_path / self.X_name, engine='pyarrow')
            dY = pd.read_parquet(self.data_path / self.Y_name, engine='pyarrow')
            
            # Data preprocessing (handle missing values, type conversion, etc.)
            dX = dX.astype(float) 
            dY = dY.astype(float) 
            if self.X_env_name is not None:
                dX_env = pd.read_parquet(
                    self.data_path / self.X_env_name, engine='pyarrow')
                dX_env = dX_env.astype(float)
                return dX, dX_env, dY  
            return dX, dY

        except FileNotFoundError:
            logger.error(f"Error: Files not found in directory {self.data_path}")
            return None, None  # Return None on error
        except pd.errors.EmptyDataError:
            logger.error(f"Error: Empty data files in directory {self.data_path}")
            return None, None
        except Exception as e:  # Catch other potential exceptions
            logger.error(f"Error loading data: {e}")
            return None, None 