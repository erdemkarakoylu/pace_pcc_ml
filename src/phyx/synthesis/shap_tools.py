from pathlib import Path
from typing import List, Dict, Any, Sequence, Optional

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from loguru import logger


class SHAPSynthesizer:
    """
    SHAP utilities for multi-output regression:
      - compute SHAP values per target
      - save per-target SHAP summary plots (legacy look)
      - rank features by mean|SHAP| (per-target and overall)
    """

    # ------------------
    # Core computation
    # ------------------

    @staticmethod
    def compute_shap_for_multioutput(model, X: pd.DataFrame) -> List[np.ndarray]:
        """
        Compute SHAP values for each target of a MultiOutputRegressor (or a single regressor).
        Returns list of arrays, one per target, each of shape (n_samples, n_features).
        """
        vals: List[np.ndarray] = []
        if hasattr(model, "estimators_"):  # MultiOutputRegressor
            for est in model.estimators_:
                explainer = shap.TreeExplainer(est)
                sv = explainer.shap_values(X.to_numpy())
                vals.append(np.asarray(sv))
        else:
            explainer = shap.TreeExplainer(model)
            vals = [np.asarray(explainer.shap_values(X.to_numpy()))]
        return vals


    # ------------------
    # Rankings
    # ------------------

    @staticmethod
    def feature_ranking(sv: np.ndarray, feature_names: Sequence[str]) -> pd.DataFrame:
        """
        Produce a ranking DataFrame given SHAP values for ONE target.
        """
        sv = np.asarray(sv)
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        return pd.DataFrame(
            {"feature": np.array(feature_names)[order], "mean_abs_shap": mean_abs[order]}
        ).reset_index(drop=True)

    @staticmethod
    def overall_ranking(sv_list: List[np.ndarray], feature_names: Sequence[str]) -> pd.DataFrame:
        """
        Aggregate rankings across targets by averaging mean|SHAP|.
        """
        per = [np.abs(sv).mean(axis=0) for sv in sv_list]  # list of (f,) arrays
        overall = np.mean(np.vstack(per), axis=0)
        order = np.argsort(overall)[::-1]
        return pd.DataFrame(
            {"feature": np.array(feature_names)[order], "mean_abs_shap": overall[order]}
        ).reset_index(drop=True)

    # ------------------
    # Plots
    # ------------------
    @staticmethod
    def save_summary_plot(
        sv: np.ndarray,
        X: pd.DataFrame,
        target_name: str,
        outdir: Path,
        input_types: str = "rrs",
        max_display: int = -1,
        title: Optional[str] = None,
    ) -> Path:
        """
        Save a legacy-style summary plot:
        filename: shap_{input_types}_{target_name}.png
        """
        outdir = Path(outdir).resolve()
        outdir.mkdir(parents=True, exist_ok=True)

        # SHAP plot (layered violin, coolwarm), saved headlessly
        if max_display is not None and max_display > 0:
            shap.summary_plot(sv, X, plot_type="layered_violin", color="coolwarm", show=False, max_display=max_display)
        else:
            shap.summary_plot(sv, X, plot_type="layered_violin", color="coolwarm", show=False)

        if title:
            import matplotlib.pyplot as _plt
            _plt.title(title)

        path = outdir / f"shap_{input_types}_{target_name}.png"
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()
        logger.info("Saved SHAP summary → {}", path)
        return path

    # ------------------
    # Driver
    # ------------------
    def explain_all_targets(
        self,
        model,
        X_test: pd.DataFrame,
        target_names: Sequence[str],
        outdir: Path,
        input_types: str = "rrs",
        shap_max_display: int = -1,
        save_rankings: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute per-target SHAP values, save summary plots for each target,
        and optionally save per-target & overall feature rankings.

        Returns:
            {
              "shap_paths": [str, ...],
              "per_target_rank": {target: "path/to/csv"},
              "overall_rank_csv": "path/to/csv"
            }
        """
        sv_list = self.compute_shap_for_multioutput(model, X_test)

        shap_paths: List[Path] = []
        per_rank_paths: Dict[str, str] = {}
        for i, sv in enumerate(sv_list):
            tgt = str(target_names[i]) if i < len(target_names) else f"t{i}"
            # plot
            pth = self.save_summary_plot(
                sv, X_test, target_name=tgt, outdir=outdir,
                input_types=input_types, max_display=shap_max_display,
                title=f"SHAP summary — {tgt}"
            )
            shap_paths.append(pth)

            # per-target ranking
            if save_rankings:
                rank_df = self.feature_ranking(sv, list(X_test.columns))
                rp = Path(outdir) / f"shap_feature_ranking_{tgt}.csv"
                rank_df.to_csv(rp, index=False)
                per_rank_paths[tgt] = str(rp)
                logger.info("Saved feature ranking (target={}) → {}", tgt, rp)

        overall_rank_path = ""
        if save_rankings and sv_list:
            overall_df = self.overall_ranking(sv_list, list(X_test.columns))
            overall_rank_path = str(Path(outdir) / "shap_feature_ranking_overall.csv")
            overall_df.to_csv(overall_rank_path, index=False)
            logger.info("Saved overall feature ranking → {}", overall_rank_path)

        return {
            "shap_paths": [str(p) for p in shap_paths],
            "per_target_rank": per_rank_paths,
            "overall_rank_csv": overall_rank_path,
        }
