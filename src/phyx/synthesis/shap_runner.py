from pathlib import Path
from typing import Union, Sequence, Dict, Any

import joblib
import pandas as pd
from loguru import logger

from .shap_tools import SHAPSynthesizer


def run_shap_and_plots(
    model_or_path: Union[Path, str, object],
    X_test: pd.DataFrame,
    target_names: Sequence[str],
    outdir: Path,
    *,
    input_types: str = "rrs",
    shap_max_display: int = -1,
    save_rankings: bool = True,
) -> Dict[str, Any]:
    """
    Compute SHAP values and save per-target summary plots (+ rankings).

    Parameters
    ----------
    model_or_path
        A fitted multi-output model (e.g., MultiOutputRegressor(XGBRegressor)) OR
        a filesystem path to a serialized joblib model.
    X_test
        Feature matrix as a pandas DataFrame (column names are used in plots/rankings).
    target_names
        Sequence of target names (order must match model outputs).
    outdir
        Directory where plots and CSVs will be written.
    input_types
        Tag used in output filenames (e.g., 'rrs', 'rrs+env').
    shap_max_display
        Max number of features to show in SHAP summary (<=0 means 'show all').
    save_rankings
        If True, writes per-target and overall mean|SHAP| rankings to CSV.

    Returns
    -------
    Dict[str, Any]
        {
          "shap_paths": [list of .png plot paths],
          "per_target_rank": {target: path_to_csv},
          "overall_rank_csv": path_to_csv_or_empty_string
        }
    """
    # Load model if a path is provided
    if isinstance(model_or_path, (str, Path)):
        model_path = Path(model_or_path)
        model = joblib.load(model_path)
        logger.info("Loaded model from: {}", model_path)
    else:
        model = model_or_path

    synth = SHAPSynthesizer()
    return synth.explain_all_targets(
        model=model,
        X_test=X_test,
        target_names=target_names,
        outdir=outdir,
        input_types=input_types,
        shap_max_display=shap_max_display,
        save_rankings=save_rankings,
    )
