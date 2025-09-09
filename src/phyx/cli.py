from pathlib import Path
import json, sys
import typer, optuna
from loguru import logger
from sklearn.model_selection import train_test_split

from .pipeline.data_loader import DataLoader
from .pipeline.model_trainer import XGBoostTrainer
from .pipeline.model_evaluator import ModelEvaluator
from .pipeline.optuna_hpo import objective
from .synthesis.shap_runner import run_shap_and_plots

app = typer.Typer(add_completion=False)

def _configure_logging(verbose=False):
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO", colorize=True, enqueue=True)

@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")):
    _configure_logging(verbose)

@app.command()
def run_all(
    data_path: Path = typer.Option(..., help="Folder with df_rrs.pqt / df_phyto_types.pqt"),
    outdir: Path = typer.Option(Path("artifacts/pace"), help="Output directory"),
    n_trials: int = typer.Option(50, help="Optuna trials"),
    cv_folds: int  = typer.Option(3, help="CV folds per Optuna trial (1 = single holdout)"),
    test_size: float = typer.Option(0.2, help="Holdout fraction"),
    seed: int = typer.Option(42, help="Random seed"),
    fit_all: bool = typer.Option(False, help="After evaluation, fit a final production model on train+test"),
):
    """HPO → train on training split → evaluate on test split → SHAP plots.
       Optionally fit a final 'production' model on all data.
    """
    # 1) Load pre-clipped dataset
    dl = DataLoader(data_path)
    X_df, Xenv_df, Y_df = dl.load_data()
    if Xenv_df is not None:
        X_df = X_df.join(Xenv_df)

    # fixed split
    Xtr, Xte, Ytr, Yte = train_test_split(X_df, Y_df, test_size=test_size, random_state=seed)
    outdir.mkdir(parents=True, exist_ok=True)

    # 2) HPO
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, Xtr, Ytr), n_trials=n_trials, kfolds=cv_folds, show_progress_bar=False)
    best_params = study.best_trial.params | {"objective": "reg:squarederror"}
    (outdir / "best_params.json").write_text(json.dumps(best_params, indent=2))
    logger.success("Best params saved → {}", outdir / "best_params.json")

    # 3) Train FULL on training split with best params
    trainer = XGBoostTrainer(params=best_params)
    trainer.fit_full(Xtr.to_numpy(), Ytr.to_numpy())
    model_path = outdir / "model.pkl"
    trainer.save(model_path)

    # 4) Evaluate on test split
    evaluator = ModelEvaluator()
    yhat = trainer.predict(Xte.to_numpy())
    mse, r2, mae, rmse = evaluator.evaluate(Yte.to_numpy(), yhat)
    metrics = {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.success("Metrics → {}", metrics)

    # 5) SHAP on test split
    run_shap_and_plots(model_path, Xte, list(Y_df.columns), outdir)

    # 6) Optional: fit production model on ALL data (no metrics)
    if fit_all:
        trainer.fit_full(X_df.to_numpy(), Y_df.to_numpy())
        trainer.save(outdir / "model_production.pkl")
        logger.success("Saved production model trained on ALL data → {}", outdir / "model_production.pkl")
