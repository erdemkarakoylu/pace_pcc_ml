from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from loguru import logger

class DataLoader:
    """Load X (Rrs), optional env, and Y targets exactly as stored on disk."""

    def __init__(
        self,
        data_path: Path | str,
        rrs_file: str = "df_rrs.pqt",
        phy_file: str = "df_phyto.pqt",
        env_file: Optional[str] = None,
        env_features = ['lat', 'temp']
    ):
        self.data_path = Path(data_path).resolve()
        self.rrs_file = rrs_file
        self.phy_file = phy_file
        self.env_file = env_file
        self.env_features = env_features

    def load_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
        dX = pd.read_parquet(self.data_path / self.rrs_file).astype(float)
        dY = pd.read_parquet(self.data_path / self.phy_file).astype(float)
        dX_env = None
        if self.env_file:
            dX_env = pd.read_parquet(self.data_path / self.env_file).astype(float)[self.env_features]
        logger.info("Loaded X=%s, Y=%s%s", dX.shape, dY.shape, f", X_env={dX_env.shape}" if dX_env is not None else "")
        return dX, dX_env, dY
