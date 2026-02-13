import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path

current_dir = Path(__file__).parent

from evaluation_utils import ModelEvaluator, create_train_test_split

RANDOM_SEED = 42
CV_FOLDS = 2
N_JOBS = -1


class SVMExperiment:
    """
    Support Vector Machine Experiment
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

        print("\n" + "=" * 80)
        print(f"SVM EXPERIMENT: {dataset_name}")
        print(f"{'='*80}")
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        print(f"Features: {X.shape[1]}")
        print(f"Scoring metric: {self.scoring_display}")

    def run_baseline(self):
        """
        Run baseline SVM experiment with default parameters
        """

        print(f"\n{'-'*80}")
        print(f"Running baseline SVM experiment")
        print(f"{'-'*80}")

        # Create pipeline with scaling and SVM
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("svm", SVC(random_state=RANDOM_SEED, probability=True))]
        )

        # cross-validate baseline model
        cv_results = cross_validate(
            pipeline,
            self.X_train,
            self.y_train,
            cv=CV_FOLDS,
            scoring=self.scoring,
            n_jobs=N_JOBS,
            return_train_score=True,
        )

        # Fit on full training data and evaluate on test set
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
            title=f"{self.dataset_name} - SVM Baseline",
        )

        # Runtime
        runtime = self.evaluator.measure_runtime(
            pipeline, self.X_train, self.y_train, self.X_test
        )

        return {"model": pipeline, "metrics": metrics, "runtime": runtime}

    def tune_hyperparameters(self):
        """
        Tune SVM hyperparameters using GridSearchCV
        """
        print(f"\n{'-'*80}")
        print("Tuning SVM hyperparameters...")
        print(f"{'-'*80}")

        # Create pipeline
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("svm", SVC(random_state=RANDOM_SEED, probability=True))]
        )

        # Define hyperparameter grid
        param_grid = [
            # Linear kernel - for high-dim sparse data
            {"svm__kernel": ["linear"], "svm__C": [0.01, 0.1, 1, 10, 100]},
            # RBF kernel - for non-linear patterns
            {
                "svm__kernel": ["rbf"],
                "svm__C": [0.01, 0.1, 1, 10, 100],
                "svm__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
            },
            # Polynomial kernel (optional, often slower)
            {
                "svm__kernel": ["poly"],
                "svm__C": [0.1, 1, 10],  # Reduced range
                "svm__gamma": ["scale", "auto"],
                "svm__degree": [2, 3],  # Usually 2 or 3 is enough
            },
        ]

        print(f"\nParameter grid:")
        for i, params in enumerate(param_grid, 1):
            print(f"  Kernel config {i}:")
            for param, values in params.items():
                print(f"    {param}: {values}")

        # Run GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=CV_FOLDS,
            scoring=self.scoring,
            n_jobs=N_JOBS,
            return_train_score=True,
            verbose=2,
        )

        print(f"\nRunning GridSearchCV with {CV_FOLDS}-fold cross-validation...")
        grid_search.fit(self.X_train, self.y_train)
        print(f"\nGrid search complete!")

        print(f"\nBest kernel: {grid_search.best_params_['svm__kernel']}")
        print(f"Best C: {grid_search.best_params_['svm__C']}")
        if "svm__gamma" in grid_search.best_params_:
            print(f"Best gamma: {grid_search.best_params_['svm__gamma']}")

        best_pipeline = grid_search.best_estimator_

        # Evaluate best model on test set
        y_pred = best_pipeline.predict(self.X_test)
        y_pred_proba = best_pipeline.predict_proba(self.X_test)

        metrics = self.evaluator.evaluate_classification(
            self.y_test,
            y_pred,
            y_pred_proba,
            class_labels=self.class_labels,
            title=f"{self.dataset_name} - SVM Tuned",
        )

        # Runtime
        runtime = self.evaluator.measure_runtime(
            best_pipeline, self.X_train, self.y_train, self.X_test
        )

        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv(
            self.output_dir / "tables" / f"svm_grid_search_results.csv", index=False
        )

        return {
            "model": best_pipeline,
            "metrics": metrics,
            "runtime": runtime,
            "grid_search": grid_search,
        }

    def generate_learning_curves(self, model):
        """
        Generate learning curve for the SVM model
        """
        print(f"\n{'-'*30}")
        print("GENERATING LEARNING CURVE")
        print(f"{'-'*30}")

        self.evaluator.generate_learning_curve(
            model,
            self.X_train,
            self.y_train,
            title=f"{self.dataset_name} - SVM Learning Curve",
            scoring=self.scoring,
            cv=CV_FOLDS,
        )

    def generate_complexity_curves(self):
        """
        Generate complexity curve for the SVM model
        """
        print(f"\n{'-'*30}")
        print("GENERATING COMPLEXITY CURVE")
        print(f"{'-'*30}")

        # Kernel 1: linear
        # use base model for complexity curve
        base_linear_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="linear", random_state=RANDOM_SEED, probability=True)),
            ]
        )

        # Vary C parameter for complexity curve
        self.evaluator.generate_complexity_curve(
            base_linear_model,
            self.X_train,
            self.y_train,
            title=f"{self.dataset_name} - SVM (Linear Kernel) Complexity Curve",
            param_name="svm__C",
            param_range=[0.01, 0.1, 1, 10, 100],
        )

        # Kernel 2: RBF
        base_rbf_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", C=1.0, random_state=RANDOM_SEED, probability=True)),
            ]
        )

        # vary gamma parameter for complexity curve
        self.evaluator.generate_complexity_curve(
            base_rbf_model,
            self.X_train,
            self.y_train,
            param_name="svm__gamma",
            param_range=[0.0001, 0.001, 0.01, 0.1, 1, 10],
            title=f"{self.dataset_name} - SVM RBF Kernel (gamma)",
        )

        base_rbf_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", gamma='scale', random_state=RANDOM_SEED, probability=True)),
            ]
        ) # fix gamma to scale for C complexity curve

        # vary C parameter for complexity curve
        self.evaluator.generate_complexity_curve(
            base_rbf_model,
            self.X_train,
            self.y_train,
            param_name="svm__C",
            param_range=[0.01, 0.1, 1, 10, 100],
            title=f"{self.dataset_name} - SVM RBF Kernel (C)",
        )

    def run_complete_experiment(self):
        """
        Run complete SVM experiment: baseline, tuning, learning curve, complexity curves
        """
        results = {}

        # Baseline
        baseline_results = self.run_baseline()
        results["baseline"] = baseline_results

        # Hyperparameter tuning
        tuned_results = self.tune_hyperparameters()
        results["tuned"] = tuned_results

        # Learning curve
        self.generate_learning_curves(tuned_results["model"])

        # Complexity curves
        self.generate_complexity_curves()

        # Summary
        print(f"\n{'='*80}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*80}")

        print(f"\nBaseline Performance:")
        for metric, value in results["baseline"]["metrics"].items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")

        print(f"\nTuned Performance:")
        for metric, value in results["tuned"]["metrics"].items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")

        print(f"\nRuntime:")
        print(
            f"  Baseline: {results['baseline']['runtime']['fit_time_mean']:.4f}s fit, "
            f"{results['baseline']['runtime']['predict_time_mean']:.4f}s predict"
        )
        print(
            f"  Tuned: {results['tuned']['runtime']['fit_time_mean']:.4f}s fit, "
            f"{results['tuned']['runtime']['predict_time_mean']:.4f}s predict"
        )

        # Save summary
        summary = {
            "Dataset": self.dataset_name,
            "Algorithm": "SVM",
            "Baseline_Score": results["baseline"]["metrics"].get(
                self.scoring, results["baseline"]["metrics"].get("accuracy")
            ),
            "Tuned_Score": results["tuned"]["metrics"].get(
                self.scoring, results["tuned"]["metrics"].get("accuracy")
            ),
            "Best_Params": str(results["tuned"]["grid_search"].best_params_),
        }

        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(self.output_dir / "tables" / "svm_summary.csv", index=False)

        print(f"\n All outputs saved to: {self.output_dir}")

        return results
