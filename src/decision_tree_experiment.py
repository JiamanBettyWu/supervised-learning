import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path

current_dir = Path(__file__).parent
# if str(current_dir) not in sys.path:
#     sys.path.insert(0, str(current_dir))

from evaluation_utils import ModelEvaluator, create_train_test_split

RANDOM_SEED = 5
CV_FOLDS = 5
N_JOBS = -1


class DecisionTreeExperiment:
    """
    Complete Decision Tree experiment with:
    - Baseline (default parameters)
    - Hyperparameter tuning via grid search
    - Learning curves
    - Complexity curves (pruning analysis)
    - Runtime profiling
    - Final evaluation
    """

    def __init__(
        self, dataset_name, X, y, output_dir, is_multiclass=False, class_labels=None
    ):
        """
        Initialize experiment

        Parameters:
        -----------
        dataset_name : str
            Name of dataset (for output files)
        X : array-like
            Features
        y : array-like
            Labels
        output_dir : str or Path
            Directory for outputs
        is_multiclass : bool
            Whether this is multiclass classification
        class_labels : list, optional
            Class labels for confusion matrix
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

        # train test split 30% test
        self.X_train, self.X_test, self.y_train, self.y_test = create_train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )

        print(f"\n{'='*80}")
        print(f"DECISION TREE EXPERIMENT: {dataset_name}")
        print(f"{'='*80}")
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        print(f"Features: {X.shape[1]}")
        print(f"Scoring metric: {self.scoring_display}")

    def run_baseline(self):
        """
        Train baseline Decision Tree with default parameters
        """
        print(f"\n{'-'*80}")
        print("BASELINE MODEL (Default Parameters)")
        print(f"{'-'*80}")

        # Baseline model
        dt_baseline = DecisionTreeClassifier(
            random_state=RANDOM_SEED,
        )

        # Train
        dt_baseline.fit(self.X_train, self.y_train)

        # Evaluate on test set
        y_pred = dt_baseline.predict(self.X_test)
        y_pred_proba = dt_baseline.predict_proba(self.X_test)

        metrics = self.evaluator.evaluate_classification(
            self.y_test,
            y_pred,
            y_pred_proba,
            class_labels=self.class_labels,
            title=f"{self.dataset_name} - DT Baseline",
        )

        # Runtime
        runtime = self.evaluator.measure_runtime(
            dt_baseline, self.X_train, self.y_train, self.X_test
        )

        print(f"\nBaseline tree depth: {dt_baseline.get_depth()}")
        print(f"Baseline tree leaves: {dt_baseline.get_n_leaves()}")

        return {"model": dt_baseline, "metrics": metrics, "runtime": runtime}

    def tune_hyperparameters(self):
        """
        Tune Decision Tree hyperparameters via GridSearchCV
        Focus on pruning parameters
        """
        print(f"\n{'-'*80}")
        print("HYPERPARAMETER TUNING")
        print(f"{'-'*80}")

        # Parameter grid - focus on pruning/regularization
        param_grid = {
            "max_depth": [3, 5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10, 20, 50],
            "min_samples_leaf": [1, 2, 4, 8, 16],
            "ccp_alpha": [0.0, 0.001, 0.005, 0.01, 0.02, 0.05],  # Post-pruning
        }

        print(f"\nParameter grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")

        # Grid search
        dt = DecisionTreeClassifier(random_state=RANDOM_SEED)

        grid_search = GridSearchCV(
            dt,
            param_grid,
            cv=CV_FOLDS,
            scoring=self.scoring,
            n_jobs=N_JOBS,
            verbose=1,
            return_train_score=True,
        )

        print(f"\nRunning GridSearchCV with {CV_FOLDS}-fold CV...")
        grid_search.fit(self.X_train, self.y_train)

        print(f"\n Grid search complete!")
        print(f"\nBest parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"\nBest CV {self.scoring_display}: {grid_search.best_score_:.4f}")

        # Train best model
        best_dt = grid_search.best_estimator_

        # Evaluate on test set
        y_pred = best_dt.predict(self.X_test)
        y_pred_proba = best_dt.predict_proba(self.X_test)

        metrics = self.evaluator.evaluate_classification(
            self.y_test,
            y_pred,
            y_pred_proba,
            class_labels=self.class_labels,
            title=f"{self.dataset_name} - DT Tuned",
        )

        # Runtime
        runtime = self.evaluator.measure_runtime(
            best_dt, self.X_train, self.y_train, self.X_test
        )

        print(f"\nTuned tree depth: {best_dt.get_depth()}")
        print(f"Tuned tree leaves: {best_dt.get_n_leaves()}")

        # Save grid search results
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv(
            self.output_dir / "tables" / "dt_grid_search_results.csv", index=False
        )

        return {
            "model": best_dt,
            "metrics": metrics,
            "runtime": runtime,
            "grid_search": grid_search,
        }

    def generate_learning_curves(self, model):
        """
        Generate learning curves for the model
        """
        print(f"\n{'-'*80}")
        print("GENERATING LEARNING CURVES")
        print(f"{'-'*80}")

        self.evaluator.generate_learning_curve(
            model,
            self.X_train,
            self.y_train,
            cv=CV_FOLDS,
            scoring=self.scoring,
            title=f"{self.dataset_name} - Decision Tree",
        )

    def generate_complexity_curves(self):
        """
        Generate model complexity curves for key parameters
        """
        print(f"\n{'-'*80}")
        print("GENERATING COMPLEXITY CURVES")
        print(f"{'-'*80}")

        # Base model for complexity curves
        base_model = DecisionTreeClassifier(random_state=RANDOM_SEED)

        # 1. Max depth curve
        self.evaluator.generate_complexity_curve(
            base_model,
            self.X_train,
            self.y_train,
            param_name="max_depth",
            param_range=[1, 2, 3, 5, 7, 10, 15, 20, 25, 30],
            cv=CV_FOLDS,
            scoring=self.scoring,
            title=f"{self.dataset_name} - DT Max Depth",
        )

        # 2. Min samples split curve
        self.evaluator.generate_complexity_curve(
            base_model,
            self.X_train,
            self.y_train,
            param_name="min_samples_split",
            param_range=[2, 5, 10, 20, 50, 100, 200],
            cv=CV_FOLDS,
            scoring=self.scoring,
            title=f"{self.dataset_name} - DT Min Samples Split",
        )

        # 3. CCP alpha (pruning) curve
        self.evaluator.generate_complexity_curve(
            base_model,
            self.X_train,
            self.y_train,
            param_name="ccp_alpha",
            param_range=[0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
            cv=CV_FOLDS,
            scoring=self.scoring,
            title=f"{self.dataset_name} - DT CCP Alpha (Pruning)",
        )

    def run_complete_experiment(self):
        """
        Run complete Decision Tree experiment
        """
        results = {}

        # 1. Baseline
        results["baseline"] = self.run_baseline()

        # 2. Hyperparameter tuning
        results["tuned"] = self.tune_hyperparameters()

        # 3. Learning curves (use tuned model)
        self.generate_learning_curves(results["tuned"]["model"])

        # 4. Complexity curves
        self.generate_complexity_curves()

        # 5. Summary
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
            "Algorithm": "Decision Tree",
            "Baseline_Score": results["baseline"]["metrics"].get(
                self.scoring, results["baseline"]["metrics"].get("accuracy")
            ),
            "Tuned_Score": results["tuned"]["metrics"].get(
                self.scoring, results["tuned"]["metrics"].get("accuracy")
            ),
            "Best_Params": str(results["tuned"]["grid_search"].best_params_),
            "Tree_Depth": results["tuned"]["model"].get_depth(),
            "Tree_Leaves": results["tuned"]["model"].get_n_leaves(),
        }

        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(self.output_dir / "tables" / "dt_summary.csv", index=False)

        print(f"\n All outputs saved to: {self.output_dir}")

        return results


def main():
    """
    Example usage - update with your actual data loading
    """
    print("Decision Tree Experiment Template")
    print("=" * 80)
    print("\nTo use this script:")
    print("1. Load your data (X, y)")
    print("2. Initialize DecisionTreeExperiment with your data")
    print("3. Call run_complete_experiment()")
    print("\nExample:")
    print("""
    # Load data
    X, y = load_your_data()
    
    # For Adult Income (binary)
    experiment = DecisionTreeExperiment(
        dataset_name='Adult Income',
        X=X, y=y,
        output_dir='results/adult_income/decision_tree',
        is_multiclass=False,
        class_labels=['<=50K', '>50K']
    )
    
    # For Wine Quality (multiclass)
    experiment = DecisionTreeExperiment(
        dataset_name='Wine Quality',
        X=X, y=y,
        output_dir='results/wine_quality/decision_tree',
        is_multiclass=True,
        class_labels=['Quality 1', 'Quality 2', ..., 'Quality 8']
    )
    
    # Run experiment
    results = experiment.run_complete_experiment()
    """)


if __name__ == "__main__":
    main()
