import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings
from sklearn.preprocessing import LabelEncoder
from decision_tree_experiment import DecisionTreeExperiment

sys.path.insert(0, str(Path(__file__).parent))


warnings.filterwarnings("ignore")


RANDOM_SEED = 5
np.random.seed(RANDOM_SEED)


class MasterExperimentRunner:
    """
    Orchestrates all experiments for the assignment
    """

    def __init__(self, base_output_dir="results"):
        """
        Initialize runner

        Parameters:
        -----------
        base_output_dir : str
            Base directory for all results
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)

        self.results = {"adult_income": {}, "wine_quality": {}}

    def load_adult_income_data(self, csv_path="data/raw/adult.csv"):
        """
        Load and preprocess Adult Income dataset
        """
        print("\n" + "=" * 80)
        print("LOADING ADULT INCOME DATASET")
        print("=" * 80)

        df = pd.read_csv(csv_path)

        # Strip whitespace
        str_cols = df.select_dtypes(include=["object"]).columns
        for col in str_cols:
            df[col] = df[col].str.strip()

        # Separate features and target
        # Assuming target is 'class' or 'income'
        if "class" in df.columns:
            target_col = "class"
        elif "income" in df.columns:
            target_col = "income"
        else:
            raise ValueError("Cannot find target column (expected 'class' or 'income')")

        y = df[target_col]
        X = df.drop(columns=[target_col])

        # Remove fnlwgt if present
        if "fnlwgt" in X.columns:
            print("\nRemoving 'fnlwgt' (census weight, not predictive)")
            X = X.drop(columns=["fnlwgt"])

        # Encode target

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_labels = le.classes_

        # One-hot encode categorical features
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

        print(f"\nOriginal features: {X.shape[1]}")
        print(f"  Numeric: {len(X.select_dtypes(include=[np.number]).columns)}")
        print(f"  Categorical: {len(categorical_cols)}")

        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        feature_names = X_encoded.columns.tolist()

        print(f"\nAfter one-hot encoding: {X_encoded.shape[1]} features")
        print(f"Class distribution:")
        for label, count in zip(*np.unique(y_encoded, return_counts=True)):
            print(f"  {class_labels[label]}: {count} ({count/len(y_encoded)*100:.1f}%)")

        return X_encoded.values, y_encoded, feature_names, class_labels

    def load_wine_quality_data(self, csv_path="data/raw/wine.csv"):
        """
        Load and preprocess Wine Quality dataset
        """
        print("\n" + "=" * 80)
        print("LOADING WINE QUALITY DATASET")
        print("=" * 80)

        df = pd.read_csv(csv_path)

        # Separate features and target
        target_col = "quality"
        y = df[target_col].values
        X = df.drop(columns=[target_col])

        # Encode wine type if present
        if "type" in X.columns:
            print("\nEncoding wine type (red=0, white=1)")

            le = LabelEncoder()
            X["type"] = le.fit_transform(X["type"])

        feature_names = X.columns.tolist()

        # Quality classes
        class_labels = [f"Quality {q}" for q in sorted(np.unique(y))]

        print(f"\nFeatures: {X.shape[1]} (all numeric)")
        print(f"Classes: {len(np.unique(y))}")
        print(f"Class distribution:")
        for quality, count in zip(*np.unique(y, return_counts=True)):
            print(f"  Quality {quality}: {count} ({count/len(y)*100:.1f}%)")

        return X.values, y, feature_names, class_labels

    def run_adult_income_experiments(self, X, y, feature_names, class_labels):
        """
        Run all experiments for Adult Income dataset
        """
        print("\n" + "=" * 80)
        print("ADULT INCOME - ALL EXPERIMENTS")
        print("=" * 80)

        output_dir = self.base_output_dir / "adult_income"

        # Decision Trees
        print("\n" + "-" * 80)
        print("1. DECISION TREES")
        print("-" * 80)

        from decision_tree_experiment import DecisionTreeExperiment

        # dt_exp = DecisionTreeExperiment(
        #     dataset_name='Adult Income',
        #     X=X, y=y,
        #     output_dir=output_dir / 'decision_tree',
        #     is_multiclass=False,
        #     class_labels=class_labels
        # )
        # self.results['adult_income']['decision_tree'] = dt_exp.run_complete_experiment()

        # TODO: Add remaining algorithms
        # - kNN
        # - SVM
        # - Neural Networks (sklearn)
        # - Neural Networks (PyTorch)

        # KNN
        # print("\n" + "-"*80)
        # print("2. KNN")
        # print("-"*80)
        # from knn_experiment import KNNExperiment
        # knn_exp = KNNExperiment(
        #     dataset_name='Adult Income',
        #     X=X, y=y,
        #     output_dir=output_dir / 'knn',
        #     is_multiclass=False,
        #     class_labels=class_labels
        # )
        # self.results['adult_income']['knn'] = knn_exp.run_complete_experiment()

        # SVM
        print("\n" + "-" * 80)
        print("3. SVM")
        print("-" * 80)
        from svm_experiment import SVMExperiment

        svm_exp = SVMExperiment(
            dataset_name="Adult Income",
            X=X,
            y=y,
            output_dir=output_dir / "svm",
            is_multiclass=False,
            class_labels=class_labels,
        )
        self.results["adult_income"]["svm"] = svm_exp.run_complete_experiment()

        print("\n✓ Adult Income experiments complete!")

    def run_wine_quality_experiments(self, X, y, feature_names, class_labels):
        """
        Run all experiments for Wine Quality dataset
        """
        print("\n" + "=" * 30)
        print("WINE QUALITY - ALL EXPERIMENTS")
        print("=" * 30)

        output_dir = self.base_output_dir / "wine_quality"

        # # Decision Trees
        # print("\n" + "-"*30)
        # print("1. DECISION TREES")
        # print("-"*30)

        # from decision_tree_experiment import DecisionTreeExperiment

        # dt_exp = DecisionTreeExperiment(
        #     dataset_name='Wine Quality',
        #     X=X, y=y,
        #     output_dir=output_dir / 'decision_tree',
        #     is_multiclass=True,
        #     class_labels=class_labels
        # )
        # self.results['wine_quality']['decision_tree'] = dt_exp.run_complete_experiment()

        # TODO: Add remaining algorithms

        # # KNN
        # print("\n" + "-"*80)
        # print("2. KNN")
        # print("-"*80)
        # from knn_experiment import KNNExperiment
        # knn_exp = KNNExperiment(
        #     dataset_name='Wine Quality',
        #     X=X, y=y,
        #     output_dir=output_dir / 'knn',
        #     is_multiclass=True,
        #     class_labels=class_labels
        # )
        # self.results['wine_quality']['knn'] = knn_exp.run_complete_experiment()

        # SVM
        print("\n" + "-" * 80)
        print("3. SVM")
        print("-" * 80)
        from svm_experiment import SVMExperiment

        svm_exp = SVMExperiment(
            dataset_name="Wine Quality",
            X=X,
            y=y,
            output_dir=output_dir / "svm",
            is_multiclass=True,
            class_labels=class_labels,
        )
        self.results["wine_quality"]["svm"] = svm_exp.run_complete_experiment()

        print("\nWine Quality experiments complete!")

    def generate_cross_model_comparison(self):
        """
        Generate comparison tables and plots across all models
        """
        print("\n" + "=" * 80)
        print("CROSS-MODEL COMPARISON")
        print("=" * 80)

        # TODO: Implement
        # - Performance comparison table
        # - Runtime comparison table
        # - Bias-variance analysis
        # - Sensitivity to hyperparameters

        pass

    def run_all_experiments(self):
        """
        Run complete experimental pipeline for both datasets
        """
        print("\n" + "*" * 30)
        print("EXPERIMENT RUNNER")
        print("*" * 30)
        print(f"Output directory: {self.base_output_dir}")

        # RUN Adult
        X_adult, y_adult, features_adult, labels_adult = self.load_adult_income_data()
        self.run_adult_income_experiments(
            X_adult, y_adult, features_adult, labels_adult
        )

        # # RUN Wine
        X_wine, y_wine, features_wine, labels_wine = self.load_wine_quality_data()
        self.run_wine_quality_experiments(X_wine, y_wine, features_wine, labels_wine)

        # Cross-model comparison
        self.generate_cross_model_comparison()

        print("\n" + "*" * 30)
        print("ALL EXPERIMENTS COMPLETE!")
        print("*" * 30)

        return self.results


def main():
    """
    Main execution
    """
    # Initialize runner
    runner = MasterExperimentRunner(base_output_dir="results")

    # Run all experiments
    results = runner.run_all_experiments()

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Review generated figures in results/*/figures/
2. Check performance metrics in results/*/tables/
3. Analyze learning curves for bias/variance
4. Compare complexity curves across models
5. Review runtime comparisons
6. Write report sections based on results
7. Test hypotheses: do results support or contradict predictions?
    """)


if __name__ == "__main__":
    main()
