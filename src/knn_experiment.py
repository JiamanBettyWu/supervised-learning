import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path

current_dir = Path(__file__).parent


from evaluation_utils import ModelEvaluator, create_train_test_split

RANDOM_SEED = 5
CV_FOLDS = 2
N_JOBS = -1


class KNNExperiment:
    """
    K-Nearest Neighbors Experiment
    """

    def __init__(self, dataset_name, X, y, is_multiclass, class_labels, output_dir):
        """
        Initialize experiment

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        feature_names : list
            List of feature names
        class_labels : list
            List of class labels
        output_dir : str
            Directory to save results
        """
        self.dataset_name = dataset_name
        self.X = X
        self.y = y
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.is_multiclass = is_multiclass
        self.class_labels = class_labels

        # Set scoring metric
        if is_multiclass:
            self.scoring = "f1_macro"
            self.scoring_display = "Macro-F1"
        else:
            self.scoring = "f1"
            self.scoring_display = "F1-Score"

        # Initialize evaluator
        self.evaluator = ModelEvaluator(output_dir, random_state=RANDOM_SEED)

        self.X_train, self.X_test, self.y_train, self.y_test = create_train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )

        print(f"\n{'='*80}")
        print(f"KNN EXPERIMENT: {dataset_name}")
        print(f"{'='*80}")
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        print(f"Features: {X.shape[1]}")
        print(f"Scoring metric: {self.scoring_display}")

    def run_baseline(self):
        """
        Run baseline KNN experiment with default parameters
        """
        print(f"\n{'-'*30}")
        print("RUNNING BASELINE KNN EXPERIMENT")
        print(f"{'-'*30}")

        # Create pipeline
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]
        )

        # Cross-validate
        cv_results = cross_validate(
            pipeline,
            self.X_train,
            self.y_train,
            cv=CV_FOLDS,
            scoring=self.scoring,
            n_jobs=N_JOBS,
            return_train_score=False,
        )

        mean_score = np.mean(cv_results["test_score"])
        print(f"\nBaseline {self.scoring_display} (CV mean): {mean_score:.4f}")

        # Fit on full training data
        pipeline.fit(self.X_train, self.y_train)

        # Evaluate on test set
        y_pred = pipeline.predict(self.X_test)
        y_pred_proba = pipeline.predict_proba(self.X_test)

        metrics = self.evaluator.evaluate_classification(
            self.y_test,
            y_pred,
            y_pred_proba,
            class_labels=self.class_labels,
            title=f"{self.dataset_name} - KNN Baseline",
        )

        # Runtime
        runtime = self.evaluator.measure_runtime(
            pipeline, self.X_train, self.y_train, self.X_test
        )

        return {"model": pipeline, "metrics": metrics, "runtime": runtime}

    def tune_hyperparameters(self):
        """
        Tune KNN hyperparameters via GridSearchCV
        """
        print(f"\n{'-'*30}")
        print("TUNING KNN HYPERPARAMETERS")
        print(f"{'-'*30}")

        # Create pipeline
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]
        )

        # Parameter grid
        param_grid = {
            "knn__n_neighbors": [3, 5, 7],  # [3, 5, 7, 9, 15, 20] speed up
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["manhattan", "minkowski"],
            "knn__leaf_size": [30, 50],  # [20, 30, 40, 50] speed up
        }
        print(f"\nParameter grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")

        # Grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=CV_FOLDS,
            scoring=self.scoring,
            n_jobs=N_JOBS,
            verbose=1,
            return_train_score=True,
        )

        print(f"\nRunning GridSearchCV with {CV_FOLDS}-fold CV...")
        grid_search.fit(self.X_train, self.y_train)

        print(f"\nGrid search complete!")
        print(f"\nBest parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"\nBest CV {self.scoring_display}: {grid_search.best_score_:.4f}")

        # Train best model
        best_pipeline = grid_search.best_estimator_

        # Evaluate on test set
        y_pred = best_pipeline.predict(self.X_test)
        y_pred_proba = best_pipeline.predict_proba(self.X_test)

        metrics = self.evaluator.evaluate_classification(
            self.y_test,
            y_pred,
            y_pred_proba,
            class_labels=self.class_labels,
            title=f"{self.dataset_name} - KNN Tuned",
        )

        print(f"\nTunned KNN k-value: {best_pipeline.named_steps['knn'].n_neighbors}")

        # Runtime
        runtime = self.evaluator.measure_runtime(
            best_pipeline, self.X_train, self.y_train, self.X_test
        )

        # Save grid search results
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv(
            self.output_dir / "tables" / "knn_grid_search_results.csv", index=False
        )

        return {
            "model": best_pipeline,
            "metrics": metrics,
            "runtime": runtime,
            "grid_search": grid_search,
        }

    def generate_learning_curve(self, model):
        """
        Generate learning curve for the KNN model
        """
        print(f"\n{'-'*30}")
        print("GENERATING LEARNING CURVE")
        print(f"{'-'*30}")

        self.evaluator.generate_learning_curve(
            model,
            self.X_train,
            self.y_train,
            title=f"{self.dataset_name} - KNN Learning Curve",
            scoring=self.scoring,
            cv=CV_FOLDS,
        )

    def generate_complexity_curves(self):
        """
        Generate model complexity curves for key parameters
        """
        print(f"\n{'-'*80}")
        print("GENERATING COMPLEXITY CURVES")
        print(f"{'-'*80}")

        # Base model for complexity curves
        base_model = Pipeline(
            [("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]
        )

        # number of neighbors
        self.evaluator.generate_complexity_curve(
            base_model,
            self.X_train,
            self.y_train,
            param_name="knn__n_neighbors",
            param_range=[
                2,
                3,
                5,
                10,
                15,
                20,
                30,
                50,
                70,
            ],  # [3, 5, 7, 10, 15, 20, 25, 30]
            cv=CV_FOLDS,
            scoring=self.scoring,
            title=f"{self.dataset_name} - KNN Number of Neighbors",
        )

        print("\nInterpretation:")
        print("  - k=1: Maximum variance (overfitting), memorizes training data")
        print("  - Large k: Maximum bias (underfitting), oversmooths decision boundary")
        print("  - Optimal k: Balances bias-variance tradeoff")

    def run_complete_experiment(self):
        """
        Run complete KNN experiment: baseline, tuning, learning curve, complexity curves
        """
        results = {}

        # Baseline
        baseline_results = self.run_baseline()
        results["baseline"] = baseline_results

        # Hyperparameter tuning
        tuned_results = self.tune_hyperparameters()
        results["tuned"] = tuned_results

        # Learning curve
        self.generate_learning_curve(tuned_results["model"])

        # Complexity curves
        self.generate_complexity_curves()

        # summary
        print(f"\n{'='*80}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*80}")

        print(f"\nBaseline Performance:")
        for metric, value in baseline_results["metrics"].items():
            print(f"  {metric}: {value:.4f}")

        print(f"\nTuned Performance:")
        for metric, value in tuned_results["metrics"].items():
            print(f"  {metric}: {value:.4f}")

        print(f"\nRuntime Comparison:")
        print(
            f"  Baseline: {results['baseline']['runtime']['fit_time_mean']:.4f}s fit, "
            f"{results['baseline']['runtime']['predict_time_mean']:.4f}s predict"
        )
        print(
            f"  Tuned: {results['tuned']['runtime']['fit_time_mean']:.4f}s fit, "
            f"{results['tuned']['runtime']['predict_time_mean']:.4f}s predict"
        )

        summary = {
            "Dataset": self.dataset_name,
            "Algorithm": "kNN",
            "Baseline_Score": results["baseline"]["metrics"].get(
                self.scoring, results["baseline"]["metrics"].get("accuracy")
            ),
            "Tuned_Score": results["tuned"]["metrics"].get(
                self.scoring, results["tuned"]["metrics"].get("accuracy")
            ),
            "Best_Params": str(results["tuned"]["grid_search"].best_params_),
            "Best_k": results["tuned"]["model"].named_steps["knn"].n_neighbors,
            "Best_weights": results["tuned"]["model"].named_steps["knn"].weights,
            "Best_metric": results["tuned"]["model"].named_steps["knn"].metric,
        }

        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(self.output_dir / "tables" / "knn_summary.csv", index=False)

        print(f"\n All outputs saved to: {self.output_dir}")

        return results
