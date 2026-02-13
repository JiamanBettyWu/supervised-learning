"""
Exploratory Data Analysis for Wine Quality Dataset
CS7641 Spring 2026 - Supervised Learning Assignment

This script performs comprehensive EDA for the multiclass wine quality task.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
RANDOM_SEED = 2
np.random.seed(RANDOM_SEED)

# Set style for publication-quality plots
sns.set_style("darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9


class WineQualityEDA:
    """Comprehensive EDA for Wine Quality dataset"""

    def __init__(self, data_path, output_dir="results/wine_quality/eda"):
        """
        Initialize EDA analyzer

        Parameters:
        -----------
        data_path : str
            Path to the wine quality CSV file
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
        print("=" * 80)
        print("LOADING WINE QUALITY DATASET")
        print("=" * 80)

        # Load data
        self.df = pd.read_csv(self.data_path)

        print(f"\nDataset shape: {self.df.shape}")
        print(f"Rows: {self.df.shape[0]:,}")
        print(f"Columns: {self.df.shape[1]}")

        # Display first few rows
        print("\nFirst 5 rows:")
        print(self.df.head())

        # Display last few rows (check for red/white if combined)
        print("\nLast 5 rows:")
        print(self.df.tail())

        # Data types
        print("\nData types:")
        print(self.df.dtypes)

        # Check if this is combined red+white dataset
        if "type" in self.df.columns:
            print("\nWine types:")
            print(self.df["type"].value_counts())
            self.summary["combined_dataset"] = True
        else:
            self.summary["combined_dataset"] = False

        return self

    def basic_statistics(self):
        """Compute basic statistical summaries"""
        print("\n" + "=" * 80)
        print("BASIC STATISTICS")
        print("=" * 80)

        # Missing values
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
            print("No missing values detected! ✓")

        self.summary["missing_values"] = missing_df.to_dict()
        self.summary["has_missing"] = len(missing_df) > 0

        # All features should be numeric except possibly 'type'
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=["object"]).columns.tolist()

        print(f"\nNumeric features: {len(numeric_cols)}")
        print(f"Categorical features: {len(categorical_cols)}")

        # Statistical summary
        print("\nDescriptive statistics:")
        print(self.df[numeric_cols].describe())

        self.summary["n_numeric_features"] = len(numeric_cols)
        self.summary["n_categorical_features"] = len(categorical_cols)

        return self

    def target_analysis(self, target_col="quality"):
        """Analyze target variable distribution"""
        print("\n" + "=" * 80)
        print("TARGET VARIABLE ANALYSIS (MULTICLASS)")
        print("=" * 80)

        # Target distribution
        target_counts = self.df[target_col].value_counts().sort_index()
        target_props = self.df[target_col].value_counts(normalize=True).sort_index()

        print(f"\nTarget column: '{target_col}'")
        print(f"Task type: Multiclass classification")
        print(f"Number of classes: {self.df[target_col].nunique()}")

        print("\nClass distribution:")
        for cls in sorted(self.df[target_col].unique()):
            count = target_counts[cls]
            pct = target_props[cls] * 100
            print(f"  Quality {cls}: {count:,} ({pct:.2f}%)")

        # Calculate class balance metrics
        class_counts = target_counts.values
        imbalance_ratio = max(class_counts) / min(class_counts)
        print(f"\nMaximum imbalance ratio: {imbalance_ratio:.2f}:1")
        print(f"  (Most common / Least common)")

        # Store in summary
        self.summary["target_column"] = target_col
        self.summary["n_classes"] = self.df[target_col].nunique()
        self.summary["class_distribution"] = target_counts.to_dict()
        self.summary["imbalance_ratio"] = imbalance_ratio

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Count plot
        target_counts.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="black")
        axes[0].set_title("Wine Quality Distribution (Counts)", fontweight="bold")
        axes[0].set_xlabel("Quality Rating")
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis="x", rotation=0)
        for i, (idx, v) in enumerate(target_counts.items()):
            axes[0].text(i, v + 50, f"{v:,}", ha="center", va="bottom")

        # Proportion plot
        target_props.plot(kind="bar", ax=axes[1], color="coral", edgecolor="black")
        axes[1].set_title("Wine Quality Distribution (Proportions)", fontweight="bold")
        axes[1].set_xlabel("Quality Rating")
        axes[1].set_ylabel("Proportion")
        axes[1].tick_params(axis="x", rotation=0)
        axes[1].set_ylim([0, max(target_props.values) * 1.2])
        for i, (idx, v) in enumerate(target_props.items()):
            axes[1].text(i, v + 0.01, f"{v:.2%}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "figures" / "target_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"\nSaved: target_distribution.png")

        # If combined dataset, analyze by type
        if "type" in self.df.columns:
            print("\nQuality distribution by wine type:")
            type_quality = pd.crosstab(self.df["type"], self.df[target_col])
            print(type_quality)

            # Visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            type_quality.T.plot(kind="bar", ax=ax, width=0.8)
            ax.set_title("Wine Quality Distribution by Type", fontweight="bold")
            ax.set_xlabel("Quality Rating")
            ax.set_ylabel("Count")
            ax.legend(title="Wine Type")
            ax.tick_params(axis="x", rotation=0)
            plt.tight_layout()
            plt.savefig(
                self.output_dir / "figures" / "quality_by_type.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print(f"Saved: quality_by_type.png")

        return self

    def feature_distributions(self):
        """Analyze feature distributions"""
        print("\n" + "=" * 80)
        print("FEATURE DISTRIBUTIONS")
        print("=" * 80)

        # Get numeric columns (exclude target)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if "quality" in numeric_cols:
            numeric_cols.remove("quality")

        print(f"\nAnalyzing {len(numeric_cols)} numeric features")

        # Plot distributions
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if len(numeric_cols) > 1 else [axes]

        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            self.df[col].hist(
                bins=40, ax=ax, edgecolor="black", alpha=0.7, color="teal"
            )
            ax.set_title(f"{col}", fontweight="bold")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")

            # Add statistics
            mean_val = self.df[col].mean()
            median_val = self.df[col].median()
            std_val = self.df[col].std()

            ax.axvline(
                mean_val,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"μ={mean_val:.2f}",
            )
            ax.axvline(
                median_val,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Med={median_val:.2f}",
            )
            ax.legend(fontsize=8)

            # Check for skewness
            skew = self.df[col].skew()
            if abs(skew) > 1:
                ax.text(
                    0.02,
                    0.98,
                    f"Skew: {skew:.2f}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

        # Hide extra subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "figures" / "feature_distributions.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"\nSaved: feature_distributions.png")

        # Analyze skewness
        print("\nFeature skewness:")
        for col in numeric_cols:
            skew = self.df[col].skew()
            print(
                f"  {col:25s}: {skew:6.2f} {'(heavily skewed)' if abs(skew) > 1 else ''}"
            )

        return self

    def correlation_analysis(self, target_col="quality"):
        """Analyze correlations between features and with target"""
        print("\n" + "=" * 80)
        print("CORRELATION ANALYSIS")
        print("=" * 80)

        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # Correlation matrix
        corr_matrix = self.df[numeric_cols].corr()

        print("\nTop correlations with target (quality):")
        target_corr = (
            corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
        )
        for feature, corr in target_corr.items():
            print(f"  {feature:25s}: {corr:6.3f}")

        # Find highly correlated feature pairs (potential multicollinearity)
        print("\nHighly correlated feature pairs (|r| > 0.7):")
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7 and col1 != target_col and col2 != target_col:
                    high_corr_pairs.append((col1, col2, corr_val))
                    print(f"  {col1} <-> {col2}: {corr_val:.3f}")

        if not high_corr_pairs:
            print("  None found.")

        self.summary["high_correlations"] = high_corr_pairs

        # Plot correlation heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            annot_kws={"fontsize": 8},
        )
        plt.title("Feature Correlation Matrix", fontweight="bold", fontsize=14)
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "figures" / "correlation_matrix.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"\nSaved: correlation_matrix.png")

        return self

    def feature_target_relationships(self, target_col="quality"):
        """Analyze relationships between features and target"""
        print("\n" + "=" * 80)
        print("FEATURE-TARGET RELATIONSHIPS")
        print("=" * 80)

        # Get numeric columns (exclude target)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove("quality")

        # Select top correlated features
        corr_with_target = (
            self.df[numeric_cols + [target_col]].corr()[target_col].drop(target_col)
        )
        top_features = corr_with_target.abs().nlargest(6).index.tolist()

        print(f"\nAnalyzing top {len(top_features)} features correlated with quality:")
        for feat in top_features:
            print(f"  {feat}: {corr_with_target[feat]:.3f}")

        # Box plots for top features
        n_cols = 3
        n_rows = (len(top_features) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if len(top_features) > 1 else [axes]

        for idx, col in enumerate(top_features):
            ax = axes[idx]

            # Box plot by quality rating
            self.df.boxplot(column=col, by=target_col, ax=ax)
            ax.set_title(f"{col} by Quality Rating", fontweight="bold")
            ax.set_xlabel("Quality Rating")
            ax.set_ylabel(col)
            plt.sca(ax)
            plt.xticks(rotation=0)

        # Hide extra subplots
        for idx in range(len(top_features), len(axes)):
            axes[idx].axis("off")

        plt.suptitle("")  # Remove default title
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "figures" / "feature_target_relationships.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"\nSaved: feature_target_relationships.png")

        # Violin plots for better distribution visualization
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if len(top_features) > 1 else [axes]

        for idx, col in enumerate(top_features[: len(axes)]):
            ax = axes[idx]
            sns.violinplot(data=self.df, x=target_col, y=col, ax=ax, palette="Set2")
            ax.set_title(f"{col} Distribution by Quality", fontweight="bold")
            ax.set_xlabel("Quality Rating")
            ax.set_ylabel(col)

        # Hide extra subplots
        for idx in range(len(top_features), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "figures" / "feature_target_violin.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"Saved: feature_target_violin.png")

        return self

    def wine_type_analysis(self):
        """Analyze differences between red and white wines if applicable"""
        if "type" not in self.df.columns:
            print("\nNo wine type column found - skipping type analysis")
            return self

        print("\n" + "=" * 80)
        print("WINE TYPE ANALYSIS")
        print("=" * 80)

        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if "quality" in numeric_cols:
            numeric_cols.remove("quality")

        # Compare features between types
        print("\nFeature comparison between Red and White wines:")
        comparison = self.df.groupby("type")[numeric_cols].mean()
        print(comparison.T)

        # Statistical tests for difference
        from scipy import stats

        print("\nSignificant differences (t-test p < 0.05):")
        for col in numeric_cols:
            red_vals = self.df[self.df["type"] == "red"][col]
            white_vals = self.df[self.df["type"] == "white"][col]
            t_stat, p_val = stats.ttest_ind(red_vals, white_vals)
            if p_val < 0.05:
                print(
                    f"  {col:25s}: p={p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'}"
                )

        # Visualize differences
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        # Select features with largest differences
        mean_diff = (
            self.df.groupby("type")[numeric_cols]
            .mean()
            .diff()
            .iloc[-1]
            .abs()
            .nlargest(6)
        )

        for idx, col in enumerate(mean_diff.index):
            ax = axes[idx]
            self.df.boxplot(column=col, by="type", ax=ax)
            ax.set_title(f"{col}", fontweight="bold")
            ax.set_xlabel("Wine Type")
            ax.set_ylabel(col)

        plt.suptitle(
            "Feature Differences: Red vs White Wine", fontweight="bold", fontsize=14
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "figures" / "red_vs_white_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"\nSaved: red_vs_white_comparison.png")

        return self

    def data_quality_checks(self):
        """Perform data quality checks"""
        print("\n" + "=" * 80)
        print("DATA QUALITY CHECKS")
        print("=" * 80)

        # Duplicates
        n_duplicates = self.df.duplicated().sum()
        print(f"\nDuplicate rows: {n_duplicates:,}")

        if n_duplicates > 0:
            dup_pct = 100 * n_duplicates / len(self.df)
            print(f"  ({dup_pct:.2f}% of dataset)")
            print(
                "  Note: Duplicates in wine quality may be legitimate (multiple wines with same properties)"
            )

        self.summary["n_duplicates"] = n_duplicates

        # Outliers (using IQR method)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if "quality" in numeric_cols:
            numeric_cols.remove("quality")

        print("\nOutlier detection (IQR method):")
        outlier_summary = {}
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = (
                (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            ).sum()
            outlier_pct = 100 * outliers / len(self.df)

            if outliers > 0:
                print(f"  {col:25s}: {outliers:4d} outliers ({outlier_pct:.1f}%)")
                outlier_summary[col] = outliers

        self.summary["outliers"] = outlier_summary

        # Check for constant features
        constant_features = []
        for col in self.df.columns:
            if self.df[col].nunique() == 1:
                constant_features.append(col)

        if constant_features:
            print(f"\nConstant features: {constant_features}")
        else:
            print("\nNo constant features ✓")

        self.summary["constant_features"] = constant_features

        return self

    def multiclass_considerations(self, target_col="quality"):
        """Analyze multiclass-specific considerations"""
        print("\n" + "=" * 80)
        print("MULTICLASS CLASSIFICATION CONSIDERATIONS")
        print("=" * 80)

        n_classes = self.df[target_col].nunique()
        print(f"\nNumber of classes: {n_classes}")

        # Class balance
        class_dist = self.df[target_col].value_counts(normalize=True).sort_index()
        print("\nClass balance:")
        for cls, prop in class_dist.items():
            print(f"  Quality {cls}: {prop:.1%}")

        # Ordinal nature
        print("\nOrdinal structure:")
        print("  Wine quality is inherently ORDINAL (3 < 4 < 5 < ... < 9)")
        print("  However, treating as nominal multiclass for this assignment")
        print("  → Misclassifying 3 as 9 weighted same as 3 as 4")

        # Metric implications
        print("\nMetric implications:")
        print("  - Macro-F1: Treats all classes equally (good for imbalanced)")
        print(
            "  - Accuracy: Overall correctness (baseline = {:.1f}%)".format(
                class_dist.max() * 100
            )
        )
        print("  - Confusion matrix: Essential for understanding per-class errors")
        print(
            "  - Per-class precision/recall: Identify which qualities are hard to predict"
        )

        # Adjacent class analysis
        print("\nAdjacent class consideration:")
        print("  Some algorithms may naturally learn ordinal structure")
        print("  Could evaluate 'off-by-one' accuracy as secondary metric")

        return self

    def generate_summary_report(self):
        """Generate and save comprehensive EDA summary"""
        print("\n" + "=" * 80)
        print("GENERATING SUMMARY REPORT")
        print("=" * 80)

        summary_text = f"""
WINE QUALITY DATASET - EXPLORATORY DATA ANALYSIS SUMMARY
================================================================

DATASET OVERVIEW
-----------------
Total Samples: {len(self.df):,}
Total Features: {len(self.df.columns) - 1}
Combined Dataset (Red + White): {self.summary.get('combined_dataset', False)}
Missing Values: {self.summary.get('has_missing', False)}

TARGET VARIABLE
----------------
Column: {self.summary.get('target_column', 'N/A')}
Task Type: Multiclass Classification
Number of Classes: {self.summary.get('n_classes', 'N/A')}
Class Distribution: {self.summary.get('class_distribution', {})}
Imbalance Ratio: {self.summary.get('imbalance_ratio', 'N/A'):.2f}:1

FEATURE CHARACTERISTICS
------------------------
All Numeric Features: {self.summary.get('n_numeric_features', 'N/A')}
Categorical: {self.summary.get('n_categorical_features', 'N/A')} (wine type if combined)

DATA QUALITY
-------------
Missing Values: {'None ✓' if not self.summary.get('has_missing', False) else 'Present'}
Duplicate Rows: {self.summary.get('n_duplicates', 'N/A'):,}
Outliers Detected: {len(self.summary.get('outliers', {}))} features with outliers

KEY OBSERVATIONS FOR HYPOTHESIS GENERATION
-------------------------------------------
1. MULTICLASS NATURE:
   - {self.summary.get('n_classes', 'N/A')} quality classes (ordinal but treated as nominal)
   - Class imbalance present → Macro-F1 critical
   - Confusion matrix essential for per-class analysis
   
2. FEATURE CHARACTERISTICS:
   - All continuous/numeric features (chemical properties)
   - Some features highly correlated with quality
   - Feature scaling likely beneficial for most algorithms
   - No missing values → simpler preprocessing

3. CORRELATION STRUCTURE:
   - Several features show moderate correlation with quality
   - Some multicollinearity present between features
   - Suggests potential for feature importance analysis

4. WINE TYPE (if applicable):
   - Red and white wines have significantly different chemical profiles
   - Wine type may be important predictive feature
   - Could analyze separately or include as feature

HYPOTHESIS IMPLICATIONS
------------------------
These observations suggest hypotheses about:
- SVM with RBF kernel may capture non-linear relationships
- Decision trees can naturally handle feature interactions
- Neural networks may benefit from multiple hidden layers
- kNN may work well given continuous features (with scaling)
- Ensemble methods might excel due to feature correlations

MULTICLASS METRICS REQUIREMENTS
---------------------------------
- Primary: Macro-F1 (equal weight to all classes)
- Primary: Accuracy (overall correctness)
- Required: Confusion matrix analysis
- Required: Per-class precision/recall discussion

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

    # File path - UPDATE THIS to your actual data path
    data_path = "data/raw/wine.csv"

    # Check if file exists
    if not Path(data_path).exists():
        print(f"ERROR: Data file not found at {data_path}")
        print(
            "Please update the data_path variable to point to your Wine Quality dataset"
        )
        return

    # Initialize and run EDA
    eda = WineQualityEDA(data_path)

    # Execute analysis pipeline
    (
        eda.load_data()
        .basic_statistics()
        .target_analysis()
        .feature_distributions()
        .correlation_analysis()
        .feature_target_relationships()
        .wine_type_analysis()
        .data_quality_checks()
        .multiclass_considerations()
        .generate_summary_report()
    )

    print("\n" + "=" * 80)
    print("EDA COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {eda.output_dir}")
    print("\nGenerated files:")
    print("  - figures/target_distribution.png")
    print("  - figures/quality_by_type.png (if applicable)")
    print("  - figures/feature_distributions.png")
    print("  - figures/correlation_matrix.png")
    print("  - figures/feature_target_relationships.png")
    print("  - figures/feature_target_violin.png")
    print("  - figures/red_vs_white_comparison.png (if applicable)")
    print("  - eda_summary.txt")
    print("\nNext steps:")
    print("  1. Review all figures and summary")
    print("  2. Formulate hypotheses based on observations")
    print("  3. Begin preprocessing and model development")


if __name__ == "__main__":
    main()
