"""
Exploratory Data Analysis for Adult Income Dataset
CS7641 Spring 2026 - Supervised Learning Assignment

This script performs comprehensive EDA to inform hypothesis generation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)

# Set style for publication-quality plots
# plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9


class AdultIncomeEDA:
    """Comprehensive EDA for Adult Income dataset"""

    def __init__(self, data_path, output_dir="results/adult_income/eda"):
        """
        Initialize EDA analyzer

        Parameters:
        -----------
        data_path : str
            Path to the adult income CSV file
        output_dir : str
            Directory to save figures and summary
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "figures").mkdir(exist_ok=True)

        self.df = None
        self.summary = {}

    def load_data(self):
        """Load and perform initial inspection"""
        self.df = pd.read_csv(self.data_path)

        str_columns = self.df.select_dtypes(include=["object"]).columns
        for col in str_columns:
            self.df[col] = self.df[col].str.strip()

        print(f"\nDataset shape: {self.df.shape}")
        print(f"Rows: {self.df.shape[0]:,}")
        print(f"Columns: {self.df.shape[1]}")

        # Data types
        print("\nData types:")
        print(self.df.dtypes)

        return self

    def basic_statistics(self):
        """Compute basic statistical summaries"""

        print("BASIC STATISTICS")

        missing = self.df.isnull().sum()
        missing_pct = 100 * missing / len(self.df)
        missing_df = pd.DataFrame(
            {"Missing_Count": missing, "Missing_Percentage": missing_pct}
        )
        missing_df = missing_df[missing_df["Missing_Count"] > 0].sort_values(
            "Missing_Count", ascending=False
        )

        print("\nMissing Values:")
        if len(missing_df) > 0:
            print(missing_df)
        else:
            print("No missing values detected!")

        self.summary["missing_values"] = missing_df.to_dict()

        # Check for '?' as missing value indicator (common in Adult dataset)
        question_marks = {}
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                count = (self.df[col] == "?").sum()
                if count > 0:
                    question_marks[col] = count

        if question_marks:
            print("\n'?' values found (potential missing indicators):")
            for col, count in question_marks.items():
                pct = 100 * count / len(self.df)
                print(f"  {col}: {count:,} ({pct:.2f}%)")

        self.summary["question_mark_values"] = question_marks

        # Continuous variables summary
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print(f"\nContinuous features ({len(numeric_cols)}):")
        print(self.df[numeric_cols].describe())

        # Categorical variables
        categorical_cols = self.df.select_dtypes(include=["object"]).columns
        print(f"\nCategorical features ({len(categorical_cols)}):")
        for col in categorical_cols:
            n_unique = self.df[col].nunique()
            print(f"  {col}: {n_unique} unique values")

        self.summary["n_numeric_features"] = len(numeric_cols)
        self.summary["n_categorical_features"] = len(categorical_cols)

        return self

    def target_analysis(self, target_col="class"):
        """Analyze target variable distribution"""
        print("\n" + "=" * 80)
        print("TARGET VARIABLE ANALYSIS")
        print("=" * 80)

        # Target distribution
        target_counts = self.df[target_col].value_counts()
        target_props = self.df[target_col].value_counts(normalize=True)

        print(f"\nTarget column: '{target_col}'")
        print("\nClass distribution:")
        for cls, count in target_counts.items():
            pct = target_props[cls] * 100
            print(f"  {cls}: {count:,} ({pct:.2f}%)")

        # Calculate imbalance ratio
        class_counts = target_counts.values
        imbalance_ratio = max(class_counts) / min(class_counts)
        print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")

        # Store in summary
        self.summary["target_column"] = target_col
        self.summary["class_distribution"] = target_counts.to_dict()
        self.summary["imbalance_ratio"] = imbalance_ratio

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Count plot
        target_counts.plot(kind="bar", ax=axes[0], color=["#3498db", "#e74c3c"])
        axes[0].set_title("Target Class Distribution (Counts)", fontweight="bold")
        axes[0].set_xlabel("Income Class")
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis="x", rotation=45)
        for i, v in enumerate(target_counts.values):
            axes[0].text(i, v + 500, f"{v:,}", ha="center", va="bottom")

        # Proportion plot
        target_props.plot(kind="bar", ax=axes[1], color=["#3498db", "#e74c3c"])
        axes[1].set_title("Target Class Distribution (Proportions)", fontweight="bold")
        axes[1].set_xlabel("Income Class")
        axes[1].set_ylabel("Proportion")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].set_ylim([0, 1])
        for i, v in enumerate(target_props.values):
            axes[1].text(i, v + 0.02, f"{v:.2%}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "figures" / "target_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"\nSaved: target_distribution.png")

        return self

    def feature_distributions(self):
        """Analyze feature distributions"""
        print("\n" + "=" * 80)
        print("FEATURE DISTRIBUTIONS")
        print("=" * 80)

        # Numeric features
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if "fnlwgt" in numeric_cols:
            # fnlwgt is typically not used as a feature
            print(
                "\nNote: 'fnlwgt' detected - this is a census weight, typically not used as feature"
            )

        # Plot numeric distributions
        n_numeric = len(numeric_cols)
        if n_numeric > 0:
            n_cols = 2
            n_rows = (n_numeric + 1) // 2

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
            axes = axes.flatten() if n_numeric > 1 else [axes]

            for idx, col in enumerate(numeric_cols):
                ax = axes[idx]
                self.df[col].hist(bins=50, ax=ax, edgecolor="black", alpha=0.7)
                ax.set_title(f"{col} Distribution", fontweight="bold")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")

                # Add statistics
                mean_val = self.df[col].mean()
                median_val = self.df[col].median()
                ax.axvline(
                    mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.1f}"
                )
                ax.axvline(
                    median_val,
                    color="green",
                    linestyle="--",
                    label=f"Median: {median_val:.1f}",
                )
                ax.legend()

            # Hide extra subplots
            for idx in range(n_numeric, len(axes)):
                axes[idx].axis("off")

            plt.tight_layout()
            plt.savefig(
                self.output_dir / "figures" / "numeric_distributions.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print(f"\nSaved: numeric_distributions.png")

        # Categorical features (top categories for high-cardinality features)
        categorical_cols = self.df.select_dtypes(include=["object"]).columns.tolist()

        # Remove target if present
        if "income" in categorical_cols:
            categorical_cols.remove("income")

        print(f"\nCategorical feature value counts:")
        for col in categorical_cols[:5]:  # Show first 5
            print(f"\n{col}:")
            print(self.df[col].value_counts().head(10))

        # Plot categorical distributions for select features
        selected_cats = [
            col for col in categorical_cols if self.df[col].nunique() <= 15
        ][:6]

        if selected_cats:
            n_cols = 2
            n_rows = (len(selected_cats) + 1) // 2

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
            axes = axes.flatten() if len(selected_cats) > 1 else [axes]

            for idx, col in enumerate(selected_cats):
                ax = axes[idx]
                value_counts = self.df[col].value_counts().head(10)
                value_counts.plot(kind="barh", ax=ax)
                ax.set_title(f"{col} Distribution (Top 10)", fontweight="bold")
                ax.set_xlabel("Count")
                ax.set_ylabel(col)

            # Hide extra subplots
            for idx in range(len(selected_cats), len(axes)):
                axes[idx].axis("off")

            plt.tight_layout()
            plt.savefig(
                self.output_dir / "figures" / "categorical_distributions.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print(f"\nSaved: categorical_distributions.png")

        return self

    def correlation_analysis(self, target_col="class"):
        """Analyze correlations between features and with target"""
        print("\n" + "=" * 80)
        print("CORRELATION ANALYSIS")
        print("=" * 80)

        # Numeric correlations
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) > 1:
            # Correlation matrix
            corr_matrix = self.df[numeric_cols].corr()

            print("\nNumeric feature correlations:")
            print(corr_matrix)

            # Plot correlation heatmap
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8},
            )
            plt.title(
                "Numeric Feature Correlation Matrix", fontweight="bold", fontsize=14
            )
            plt.tight_layout()
            plt.savefig(
                self.output_dir / "figures" / "correlation_matrix.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print(f"\nSaved: correlation_matrix.png")

        # Analyze relationship between features and target
        # For numeric features
        if len(numeric_cols) > 0:
            # Encode target for correlation
            target_encoded = (self.df[target_col] == ">50K").astype(int)

            print("\nCorrelation with target (point-biserial):")
            for col in numeric_cols:
                corr = self.df[col].corr(target_encoded)
                print(f"  {col}: {corr:.3f}")

        return self

    def feature_target_relationships(self, target_col="class"):
        """Analyze relationships between key features and target"""
        print("\n" + "=" * 80)
        print("FEATURE-TARGET RELATIONSHIPS")
        print("=" * 80)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # Select key numeric features
        key_numeric = numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols

        if key_numeric:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()

            for idx, col in enumerate(key_numeric):
                ax = axes[idx]

                # Box plot by target class
                self.df.boxplot(column=col, by=target_col, ax=ax)
                ax.set_title(f"{col} by Income Class", fontweight="bold")
                ax.set_xlabel("Income Class")
                ax.set_ylabel(col)
                plt.sca(ax)
                plt.xticks(rotation=45)

            plt.suptitle("")  # Remove default title
            plt.tight_layout()
            plt.savefig(
                self.output_dir / "figures" / "feature_target_numeric.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print(f"\nSaved: feature_target_numeric.png")

        # Categorical features vs target
        categorical_cols = self.df.select_dtypes(include=["object"]).columns.tolist()
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)

        selected_cats = [
            col for col in categorical_cols if self.df[col].nunique() <= 10
        ][:4]

        if selected_cats:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()

            for idx, col in enumerate(selected_cats):
                ax = axes[idx]

                # Stacked bar chart
                ct = pd.crosstab(self.df[col], self.df[target_col], normalize="index")
                ct.plot(kind="bar", stacked=True, ax=ax, color=["#3498db", "#e74c3c"])
                ax.set_title(f"{col} vs Income (Normalized)", fontweight="bold")
                ax.set_xlabel(col)
                ax.set_ylabel("Proportion")
                ax.legend(title="Income", bbox_to_anchor=(1.05, 1))
                plt.sca(ax)
                plt.xticks(rotation=45, ha="right")

            plt.tight_layout()
            plt.savefig(
                self.output_dir / "figures" / "feature_target_categorical.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print(f"\nSaved: feature_target_categorical.png")

        return self

    def data_quality_checks(self):
        """Perform data quality and leakage checks"""
        print("\n" + "=" * 80)
        print("DATA QUALITY & LEAKAGE CHECKS")
        print("=" * 80)

        # Duplicates
        n_duplicates = self.df.duplicated().sum()
        print(f"\nDuplicate rows: {n_duplicates:,}")

        if n_duplicates > 0:
            dup_pct = 100 * n_duplicates / len(self.df)
            print(f"  ({dup_pct:.2f}% of dataset)")

        self.summary["n_duplicates"] = n_duplicates

        # Check for constant features
        constant_features = []
        for col in self.df.columns:
            if self.df[col].nunique() == 1:
                constant_features.append(col)

        if constant_features:
            print(f"\nConstant features (zero variance): {constant_features}")
        else:
            print("\nNo constant features detected.")

        self.summary["constant_features"] = constant_features

        # Check for high cardinality features (potential ID columns)
        high_cardinality = []
        n_rows = len(self.df)
        for col in self.df.columns:
            n_unique = self.df[col].nunique()
            if n_unique > 0.9 * n_rows:  # >90% unique values
                high_cardinality.append(
                    {
                        "column": col,
                        "unique_values": n_unique,
                        "uniqueness_ratio": n_unique / n_rows,
                    }
                )

        if high_cardinality:
            print("\nHigh cardinality features (potential IDs/leakage):")
            for item in high_cardinality:
                print(
                    f"  {item['column']}: {item['unique_values']:,} unique "
                    f"({item['uniqueness_ratio']:.2%})"
                )
        else:
            print("\nNo high cardinality features detected.")

        self.summary["high_cardinality_features"] = high_cardinality

        return self

    def dimensionality_analysis(self):
        """Analyze dimensionality after one-hot encoding"""
        print("\n" + "=" * 80)
        print("DIMENSIONALITY ANALYSIS")
        print("=" * 80)

        categorical_cols = self.df.select_dtypes(include=["object"]).columns.tolist()
        if "income" in categorical_cols:
            categorical_cols.remove("income")

        # Estimate post-encoding dimensionality
        numeric_dim = len(self.df.select_dtypes(include=[np.number]).columns)
        categorical_dim = sum(self.df[col].nunique() for col in categorical_cols)

        # One-hot encoding creates (n_categories - 1) per feature, or n_categories if drop_first=False
        estimated_dim_drop_first = numeric_dim + categorical_dim - len(categorical_cols)
        estimated_dim_no_drop = numeric_dim + categorical_dim

        print(f"\nOriginal features: {len(self.df.columns) - 1}")  # -1 for target
        print(f"  Numeric: {numeric_dim}")
        print(f"  Categorical: {len(categorical_cols)}")

        print(f"\nEstimated dimensions after one-hot encoding:")
        print(f"  With drop_first=True: {estimated_dim_drop_first}")
        print(f"  With drop_first=False: {estimated_dim_no_drop}")

        print(
            f"\nDimensionality increase: {estimated_dim_drop_first / len(self.df.columns):.1f}x"
        )

        # Implications
        print("\nImplications for algorithms:")
        print("  - kNN: High-dimensional space → curse of dimensionality")
        print("  - Decision Trees: May benefit from feature interactions")
        print("  - SVM: Feature scaling critical; kernel choice important")
        print("  - Neural Networks: Larger input layer needed")

        self.summary["dimensionality"] = {
            "original_features": len(self.df.columns) - 1,
            "numeric_features": numeric_dim,
            "categorical_features": len(categorical_cols),
            "estimated_encoded_dim_drop_first": estimated_dim_drop_first,
            "estimated_encoded_dim_no_drop": estimated_dim_no_drop,
        }

        return self

    def generate_summary_report(self):
        """Generate and save comprehensive EDA summary"""
        print("\n" + "=" * 80)
        print("GENERATING SUMMARY REPORT")
        print("=" * 80)

        summary_text = f"""
ADULT INCOME DATASET - EXPLORATORY DATA ANALYSIS SUMMARY
================================================================

DATASET OVERVIEW
-----------------
Total Samples: {len(self.df):,}
Total Features: {len(self.df.columns) - 1}
  - Numeric: {self.summary.get('n_numeric_features', 'N/A')}
  - Categorical: {self.summary.get('n_categorical_features', 'N/A')}

TARGET VARIABLE
----------------
Column: {self.summary.get('target_column', 'N/A')}
Classes: {list(self.summary.get('class_distribution', {}).keys())}
Distribution: {self.summary.get('class_distribution', {})}
Imbalance Ratio: {self.summary.get('imbalance_ratio', 'N/A'):.2f}:1

DATA QUALITY
-------------
Missing Values: {sum(self.summary.get('question_mark_values', {}).values())} '?' markers found
Duplicate Rows: {self.summary.get('n_duplicates', 'N/A'):,}
Constant Features: {len(self.summary.get('constant_features', []))}

DIMENSIONALITY
---------------
Original Features: {self.summary['dimensionality']['original_features']}
After One-Hot (drop_first=True): {self.summary['dimensionality']['estimated_encoded_dim_drop_first']}

KEY OBSERVATIONS FOR HYPOTHESIS GENERATION
-------------------------------------------
1. CLASS IMBALANCE: ~{self.summary.get('imbalance_ratio', 0):.1f}:1 ratio suggests:
   - Accuracy alone is insufficient (baseline: ~{max(self.summary.get('class_distribution', {}).values()) / sum(self.summary.get('class_distribution', {}).values()) * 100:.1f}%)
   - F1 and PR-AUC are critical metrics
   - Threshold tuning may be necessary

2. FEATURE TYPES: Mix of categorical (one-hot encoded) and continuous
   - High dimensionality after encoding → curse of dimensionality for kNN
   - Binary/sparse features → linear separability possible
   - Feature scaling critical for distance-based and SVM

3. DATA CHARACTERISTICS:
   - Primarily binary/categorical features after encoding
   - Some continuous features (age, hours-per-week, etc.)
   - Suggests potential for linear decision boundaries

HYPOTHESIS IMPLICATIONS
------------------------
These observations suggest hypotheses about:
- SVM performance (margin-based, handles high-dim)
- kNN challenges (curse of dimensionality)
- Decision tree behavior (handles mixed types well)
- Neural network architecture requirements

================================================================
Generated: {pd.Timestamp.now()}
"""

        # Save summary
        summary_path = self.output_dir / "eda_summary.txt"
        with open(summary_path, "w") as f:
            f.write(summary_text)

        print(f"\nSaved: eda_summary.txt")
        print(summary_text)

        return self


def main():
    """Main execution function"""

    data_path = "data/raw/adult.csv"

    # Initialize and run EDA
    eda = AdultIncomeEDA(data_path)

    # Execute analysis pipeline
    (
        eda.load_data()
        .basic_statistics()
        .target_analysis()
        .feature_distributions()
        .correlation_analysis()
        .feature_target_relationships()
        .data_quality_checks()
        .dimensionality_analysis()
        .generate_summary_report()
    )

    print("\n" + "=" * 80)
    print("EDA COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {eda.output_dir}")
    print("\nGenerated files:")
    print("  - figures/target_distribution.png")
    print("  - figures/numeric_distributions.png")
    print("  - figures/categorical_distributions.png")
    print("  - figures/correlation_matrix.png")
    print("  - figures/feature_target_numeric.png")
    print("  - figures/feature_target_categorical.png")
    print("  - eda_summary.txt")
    print("\nNext steps:")
    print("  1. Review all figures and summary")
    print("  2. Formulate hypotheses based on observations")
    print("  3. Begin preprocessing and model development")


if __name__ == "__main__":
    main()
