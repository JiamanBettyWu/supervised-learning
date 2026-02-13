import numpy as np
import pandas as pd
import time
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


class ModelEvaluator:
    """
    Comprehensive evaluation for supervised learning models
    Generates learning curves, complexity curves, and runtime profiles
    """

    def __init__(self, output_dir, random_state=5):
        """
        Initialize evaluator

        Parameters:
        -----------
        output_dir : str or Path
            Directory to save evaluation outputs
        random_state : int
            Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)

        self.random_state = random_state
        self.results = {}

    def generate_learning_curve(
        self,
        estimator,
        X,
        y,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        title="Learning Curve",
    ):
        """
        Generate learning curve showing training and validation performance vs dataset size

        This diagnoses bias (underfitting) vs variance (overfitting):
        - High bias: Both curves converge to poor performance → need more capacity
        - High variance: Large gap between curves → need more data or regularization

        Parameters:
        -----------
        estimator : sklearn estimator
            Model to evaluate (must be unfitted)
        X : array-like
            Training features
        y : array-like
            Training labels
        cv : int
            Number of cross-validation folds
        scoring : str
            Metric to use ('accuracy', 'f1', 'f1_macro', etc.)
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        train_sizes : array-like
            Proportions of training set to use
        title : str
            Plot title

        Returns:
        --------
        dict : Dictionary with train_sizes, train_scores, val_scores
        """
        print(f"\nGenerating learning curve: {title}")
        print(f"  Metric: {scoring}")
        print(f"  CV folds: {cv}")

        # Generate learning curve
        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator,
            X,
            y,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=self.random_state,
            shuffle=True,
        )

        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Create figure
        plt.figure(figsize=(10, 6))

        # Plot training scores
        plt.plot(
            train_sizes_abs,
            train_mean,
            "o-",
            color="#3498db",
            label="Training score",
            linewidth=2,
            markersize=8,
        )
        plt.fill_between(
            train_sizes_abs,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.2,
            color="#3498db",
        )

        # Plot validation scores
        plt.plot(
            train_sizes_abs,
            val_mean,
            "o-",
            color="#e74c3c",
            label="Validation score",
            linewidth=2,
            markersize=8,
        )
        plt.fill_between(
            train_sizes_abs,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.2,
            color="#e74c3c",
        )

        plt.xlabel("Training Set Size", fontsize=12, fontweight="bold")
        plt.ylabel(f"{scoring.upper()} Score", fontsize=12, fontweight="bold")
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(loc="best", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save figure
        filename = (
            title.lower().replace(" ", "_").replace(":", "") + "_learning_curve.png"
        )
        plt.savefig(
            self.output_dir / "figures" / filename, dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"   Saved: {filename}")

        # Diagnose bias/variance
        final_gap = train_mean[-1] - val_mean[-1]
        final_val_score = val_mean[-1]

        print(f"\n  Bias/Variance Diagnosis:")
        print(f"    Final training score: {train_mean[-1]:.4f}")
        print(f"    Final validation score: {val_mean[-1]:.4f}")
        print(f"    Train-val gap: {final_gap:.4f}")

        if final_val_score < 0.7:
            print(f"     High BIAS (underfitting): Model too simple")
        elif final_gap > 0.1:
            print(f"     High VARIANCE (overfitting): Train-val gap large")
        else:
            print(f"     Good balance between bias and variance")

        # Save data
        results = {
            "train_sizes": train_sizes_abs,
            "train_scores_mean": train_mean,
            "train_scores_std": train_std,
            "val_scores_mean": val_mean,
            "val_scores_std": val_std,
        }

        df = pd.DataFrame(results)
        csv_filename = (
            title.lower().replace(" ", "_").replace(":", "") + "_learning_curve.csv"
        )
        df.to_csv(self.output_dir / "tables" / csv_filename, index=False)

        return results

    def generate_complexity_curve(
        self,
        estimator,
        X,
        y,
        param_name,
        param_range,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        title="Complexity Curve",
    ):
        """
        Generate model complexity curve showing performance vs hyperparameter

        This shows how model complexity affects bias-variance tradeoff:
        - Increasing complexity typically reduces bias but increases variance
        - Optimal complexity balances the two

        Parameters:
        -----------
        estimator : sklearn estimator
            Model to evaluate (base estimator, unfitted)
        X : array-like
            Training features
        y : array-like
            Training labels
        param_name : str
            Name of parameter to vary (e.g., 'max_depth', 'C', 'n_neighbors')
        param_range : array-like
            Values of parameter to test
        cv : int
            Number of cross-validation folds
        scoring : str
            Metric to use
        n_jobs : int
            Number of parallel jobs
        title : str
            Plot title

        Returns:
        --------
        dict : Dictionary with param_range, train_scores, val_scores
        """
        print(f"\nGenerating complexity curve: {title}")
        print(f"  Parameter: {param_name}")
        print(f"  Range: {param_range}")
        print(f"  Metric: {scoring}")

        # Generate validation curve
        train_scores, val_scores = validation_curve(
            estimator,
            X,
            y,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
        )

        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Create figure
        plt.figure(figsize=(10, 6))

        # Plot training scores
        plt.plot(
            param_range,
            train_mean,
            "o-",
            color="#3498db",
            label="Training score",
            linewidth=2,
            markersize=8,
        )
        plt.fill_between(
            param_range,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.2,
            color="#3498db",
        )

        # Plot validation scores
        plt.plot(
            param_range,
            val_mean,
            "o-",
            color="#e74c3c",
            label="Validation score",
            linewidth=2,
            markersize=8,
        )
        plt.fill_between(
            param_range,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.2,
            color="#e74c3c",
        )

        plt.xlabel(param_name, fontsize=12, fontweight="bold")
        plt.ylabel(f"{scoring.upper()} Score", fontsize=12, fontweight="bold")
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(loc="best", fontsize=11)
        plt.grid(True, alpha=0.3)

        # Handle log scale for certain parameters
        if param_name in ["C", "alpha", "gamma", "ccp_alpha"]:
            plt.xscale("log")

        plt.tight_layout()

        # Save figure
        filename = (
            title.lower().replace(" ", "_").replace(":", "") + "_complexity_curve.png"
        )
        plt.savefig(
            self.output_dir / "figures" / filename, dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"Saved: {filename}")

        # Find best parameter value
        best_idx = np.argmax(val_mean)
        best_param = param_range[best_idx]
        best_score = val_mean[best_idx]

        print(f"\n  Best {param_name}: {best_param}")
        print(f"  Best validation score: {best_score:.4f}")

        # Save data
        results = {
            "param_value": param_range,
            "train_scores_mean": train_mean,
            "train_scores_std": train_std,
            "val_scores_mean": val_mean,
            "val_scores_std": val_std,
        }

        df = pd.DataFrame(results)
        csv_filename = (
            title.lower().replace(" ", "_").replace(":", "") + "_complexity_curve.csv"
        )
        df.to_csv(self.output_dir / "tables" / csv_filename, index=False)

        return results

    def measure_runtime(self, estimator, X_train, y_train, X_test, n_repeats=5):
        """
        Measure training and prediction time with multiple repeats for stability

        Parameters:
        -----------
        estimator : sklearn estimator
            Fitted model to evaluate
        X_train : array-like
            Training features (for fit timing)
        y_train : array-like
            Training labels
        X_test : array-like
            Test features (for predict timing)
        n_repeats : int
            Number of times to repeat timing (for averaging)

        Returns:
        --------
        dict : Dictionary with mean and std of fit and predict times
        """
        print(f"\n  Measuring runtime ({n_repeats} repeats)...")

        fit_times = []
        predict_times = []

        for i in range(n_repeats):
            # Clone estimator to get fresh instance
            from sklearn.base import clone

            est = clone(estimator)

            # Time fitting
            start = time.time()
            est.fit(X_train, y_train)
            fit_time = time.time() - start
            fit_times.append(fit_time)

            # Time prediction
            start = time.time()
            _ = est.predict(X_test)
            predict_time = time.time() - start
            predict_times.append(predict_time)

        results = {
            "fit_time_mean": np.mean(fit_times),
            "fit_time_std": np.std(fit_times),
            "predict_time_mean": np.mean(predict_times),
            "predict_time_std": np.std(predict_times),
        }

        print(
            f"    Fit time: {results['fit_time_mean']:.4f} ± {results['fit_time_std']:.4f} sec"
        )
        print(
            f"    Predict time: {results['predict_time_mean']:.4f} ± {results['predict_time_std']:.4f} sec"
        )

        return results

    def evaluate_classification(
        self,
        y_true,
        y_pred,
        y_pred_proba=None,
        class_labels=None,
        title="Model Evaluation",
    ):
        """
        Comprehensive classification evaluation with metrics and confusion matrix

        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_pred_proba : array-like, optional
            Predicted probabilities (for AUC metrics)
        class_labels : list, optional
            Class labels for confusion matrix
        title : str
            Title for outputs

        Returns:
        --------
        dict : Dictionary with all metrics
        """
        print(f"\n{title}")
        print("=" * 60)

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Handle binary vs multiclass
        n_classes = len(np.unique(y_true))

        if n_classes == 2:
            # Binary classification
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)

            # AUC metrics if probabilities provided
            if y_pred_proba is not None:
                if y_pred_proba.ndim == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                roc_auc = roc_auc_score(y_true, y_pred_proba)
                pr_auc = average_precision_score(y_true, y_pred_proba)
            else:
                roc_auc = None
                pr_auc = None

            metrics = {
                "accuracy": accuracy,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }

            print(f"Accuracy:  {accuracy:.4f}")
            print(f"F1-score:  {f1:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            if roc_auc:
                print(f"ROC-AUC:   {roc_auc:.4f}")
                print(f"PR-AUC:    {pr_auc:.4f}")

        else:
            # Multiclass classification
            f1_macro = f1_score(y_true, y_pred, average="macro")
            f1_weighted = f1_score(y_true, y_pred, average="weighted")
            precision_macro = precision_score(y_true, y_pred, average="macro")
            recall_macro = recall_score(y_true, y_pred, average="macro")

            metrics = {
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
            }

            print(f"Accuracy:      {accuracy:.4f}")
            print(f"Macro-F1:      {f1_macro:.4f}")
            print(f"Weighted-F1:   {f1_weighted:.4f}")
            print(f"Macro-Prec:    {precision_macro:.4f}")
            print(f"Macro-Recall:  {recall_macro:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
        )
        plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
        plt.ylabel("True Label", fontsize=12, fontweight="bold")
        plt.title(f"{title} - Confusion Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()

        filename = title.lower().replace(" ", "_") + "_confusion_matrix.png"
        plt.savefig(
            self.output_dir / "figures" / filename, dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"\n✓ Saved confusion matrix: {filename}")

        # Per-class report
        if n_classes > 2:
            print("\nPer-class Performance:")
            report = classification_report(y_true, y_pred, target_names=class_labels)
            print(report)

            # Save report
            report_filename = (
                title.lower().replace(" ", "_") + "_classification_report.txt"
            )
            with open(self.output_dir / "tables" / report_filename, "w") as f:
                f.write(report)

        return metrics

    def create_runtime_comparison_table(
        self, results_dict, output_name="runtime_comparison"
    ):
        """
        Create comparison table of runtime across models

        Parameters:
        -----------
        results_dict : dict
            Dictionary mapping model names to runtime results
        output_name : str
            Output filename prefix

        Returns:
        --------
        DataFrame : Comparison table
        """
        data = []
        for model_name, times in results_dict.items():
            data.append(
                {
                    "Model": model_name,
                    "Fit Time (s)": f"{times['fit_time_mean']:.4f} ± {times['fit_time_std']:.4f}",
                    "Predict Time (s)": f"{times['predict_time_mean']:.4f} ± {times['predict_time_std']:.4f}",
                }
            )

        df = pd.DataFrame(data)

        # Save as CSV
        csv_path = self.output_dir / "tables" / f"{output_name}.csv"
        df.to_csv(csv_path, index=False)

        # Also save as LaTeX
        latex_path = self.output_dir / "tables" / f"{output_name}.tex"
        latex_table = df.to_latex(
            index=False,
            caption=f"Runtime Comparison",
            label=f"tab:{output_name}",
            escape=False,
        )
        with open(latex_path, "w") as f:
            f.write(latex_table)

        print(f"\n✓ Saved runtime comparison: {output_name}.csv and .tex")

        return df


# Utility function for stratified train/test split
def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Create stratified train/test split

    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Labels
    test_size : float
        Proportion for test set
    random_state : int
        Random seed

    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test)
    """

    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
