from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from loguru import logger


def run_shap_and_plots(model_path: Path, X_df, target_names: Sequence[str], outdir: Path, nsamples: int = 5000) -> Path:
    """Compute SHAP values on a sample of X_df and save summary plots per target."""
    outdir.mkdir(parents=True, exist_ok=True)
    feats = np.array(X_df.columns)
    X = X_df.to_numpy()
    n = X.shape[0]
    if n == 0:
        raise ValueError("Empty test set; cannot compute SHAP.")
    rng = np.random.default_rng(0)
    idx = np.arange(n) if n <= nsamples else rng.choice(n, nsamples, replace=False)
    Xs = X[idx]

    model = joblib.load(model_path)
    shap_values = []
    for i, est in enumerate(model.estimators_):
        explainer = shap.TreeExplainer(est)
        sv = explainer.shap_values(Xs)
        shap_values.append(sv)

        # summary plot for this target
        plt.figure()
        shap.summary_plot(sv, Xs, feature_names=feats, show=False)
        fig_path = outdir / f"shap_summary_target_{target_names[i] if i < len(target_names) else i}.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        logger.info("Saved SHAP summary → %s", fig_path)

    npz_path = outdir / "shap_values.npz"
    np.savez(npz_path, shap_values=shap_values, feature_names=feats, row_index=idx)
    logger.success("Saved SHAP values → %s", npz_path)
    return npz_path
