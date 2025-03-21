import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold


class NestedCV:
    """Performs nested cross-validation and logs results to MLflow."""

    def __init__(self, params, outer_folds=5, inner_folds=3):
        """Initializes the NestedCV class.

        Args:
            params (dict): Dictionary of hyperparameters for the XGBoost model.
            outer_folds (int, optional): Number of outer folds. Defaults to 5.
            inner_folds (int, optional): Number of inner folds. Defaults to 3.
        """
        self.params = params
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds

    def perform_nested_cv(self, X, y, model_trainer):
        """Performs nested cross-validation.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.DataFrame): Target outputs.
            model_trainer (ModelTrainer): Object responsible for training the model.

        Returns:
            list: List of MSE scores for each outer fold.
        """
        outer_cv = KFold(n_splits=self.outer_folds, shuffle=True, random_state=42)
        outer_scores = []

        for train_index, test_index in outer_cv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            with mlflow.start_run(nested=True):  # Start nested MLflow run
                # Inner loop (hyperparameter tuning)
                inner_cv = KFold(n_splits=self.inner_folds, shuffle=True, random_state=42)
                clf = GridSearchCV(estimator=model_trainer, param_grid=self.params, cv=inner_cv, scoring='neg_mean_squared_error')
                clf.fit(X_train, y_train)

                # Evaluate on the outer test fold
                y_pred = clf.predict(X_test)
                outer_scores.append(mean_squared_error(y_test, y_pred))
                mlflow.log_metric("mse", outer_scores[-1])

                # Log best parameters to MLflow
                mlflow.log_params(clf.best_params_)

        return outer_scores
