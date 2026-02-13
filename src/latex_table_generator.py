"""
LaTeX Table Generator for EDA Statistics
CS7641 Spring 2026 - Supervised Learning Assignment

This module converts EDA summary statistics into publication-ready LaTeX tables.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class LaTeXTableGenerator:
    """Generate LaTeX tables from EDA statistics"""

    def __init__(self, output_dir="results/latex_tables"):
        """
        Initialize LaTeX table generator

        Parameters:
        -----------
        output_dir : str
            Directory to save LaTeX table files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tables = {}

    def generate_dataset_overview_table(self, df, target_col, dataset_name):
        """
        Generate dataset overview table

        Parameters:
        -----------
        df : DataFrame
            The dataset
        target_col : str
            Name of target column
        dataset_name : str
            Name for the table caption
        """
        # Gather statistics
        n_samples = len(df)
        n_features = len(df.columns) - 1  # Exclude target
        n_numeric = len(df.select_dtypes(include=[np.number]).columns)
        n_categorical = len(df.select_dtypes(include=["object"]).columns)
        if target_col in df.select_dtypes(include=["object"]).columns:
            n_categorical -= 1

        missing_total = df.isnull().sum().sum()
        duplicates = df.duplicated().sum()

        # Create DataFrame
        overview_df = pd.DataFrame(
            {
                "Property": [
                    "Total Samples",
                    "Total Features",
                    "Numeric Features",
                    "Categorical Features",
                    "Missing Values",
                    "Duplicate Rows",
                ],
                "Value": [
                    f"{n_samples:,}",
                    f"{n_features}",
                    f"{n_numeric}",
                    f"{n_categorical}",
                    f"{missing_total:,}",
                    f"{duplicates:,}",
                ],
            }
        )

        # Convert to LaTeX
        latex_table = overview_df.to_latex(
            index=False,
            caption=f"{dataset_name} Dataset Overview",
            label=f'tab:{dataset_name.lower().replace(" ", "_")}_overview',
            position="h",
            column_format="lr",
            escape=False,
        )

        # Save
        filename = f'{dataset_name.lower().replace(" ", "_")}_overview.tex'
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            f.write(latex_table)

        print(f"✓ Generated: {filename}")
        self.tables["overview"] = latex_table
        return latex_table

    def generate_class_distribution_table(self, df, target_col, dataset_name):
        """
        Generate class distribution table

        Parameters:
        -----------
        df : DataFrame
            The dataset
        target_col : str
            Name of target column
        dataset_name : str
            Name for the table caption
        """
        # Get class distribution
        class_counts = df[target_col].value_counts().sort_index()
        class_props = df[target_col].value_counts(normalize=True).sort_index()

        # Create DataFrame
        dist_df = pd.DataFrame(
            {
                "Class": class_counts.index.astype(str),
                "Count": class_counts.values,
                "Percentage": [f"{p*100:.2f}\\%" for p in class_props.values],
            }
        )

        # Add total row
        total_row = pd.DataFrame(
            {
                "Class": ["Total"],
                "Count": [class_counts.sum()],
                "Percentage": ["100.00\\%"],
            }
        )
        dist_df = pd.concat([dist_df, total_row], ignore_index=True)

        # Convert to LaTeX with booktabs style
        latex_table = dist_df.to_latex(
            index=False,
            caption=f"{dataset_name} Class Distribution",
            label=f'tab:{dataset_name.lower().replace(" ", "_")}_class_dist',
            position="h",
            column_format="lrr",
            escape=False,
        )

        # Enhance with booktabs
        latex_table = latex_table.replace("\\toprule", "\\toprule")
        latex_table = latex_table.replace("Total &", "\\midrule\nTotal &")

        # Save
        filename = f'{dataset_name.lower().replace(" ", "_")}_class_distribution.tex'
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            f.write(latex_table)

        print(f"✓ Generated: {filename}")
        self.tables["class_dist"] = latex_table
        return latex_table

    def generate_feature_statistics_table(
        self, df, target_col, dataset_name, max_features=10
    ):
        """
        Generate descriptive statistics table for numeric features

        Parameters:
        -----------
        df : DataFrame
            The dataset
        target_col : str
            Name of target column
        dataset_name : str
            Name for the table caption
        max_features : int
            Maximum number of features to include
        """
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        # Limit to max_features
        numeric_cols = numeric_cols[:max_features]

        # Calculate statistics
        stats_df = df[numeric_cols].describe().T
        stats_df = stats_df[["mean", "std", "min", "max"]]
        stats_df["Feature"] = stats_df.index
        stats_df = stats_df[["Feature", "mean", "std", "min", "max"]]

        # Format numbers
        for col in ["mean", "std", "min", "max"]:
            stats_df[col] = stats_df[col].apply(lambda x: f"{x:.2f}")

        # Rename columns
        stats_df.columns = ["Feature", "Mean", "Std Dev", "Min", "Max"]

        # Convert to LaTeX
        latex_table = stats_df.to_latex(
            index=False,
            caption=f"{dataset_name} Numeric Feature Statistics",
            label=f'tab:{dataset_name.lower().replace(" ", "_")}_feature_stats',
            position="h",
            column_format="lrrrr",
            escape=False,
        )

        # Save
        filename = f'{dataset_name.lower().replace(" ", "_")}_feature_statistics.tex'
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            f.write(latex_table)

        print(f"✓ Generated: {filename}")
        self.tables["feature_stats"] = latex_table
        return latex_table

    def generate_correlation_table(self, df, target_col, dataset_name, top_n=10):
        """
        Generate correlation with target table

        Parameters:
        -----------
        df : DataFrame
            The dataset
        target_col : str
            Name of target column
        dataset_name : str
            Name for the table caption
        top_n : int
            Number of top correlated features to show
        """
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Calculate correlations
        if target_col in numeric_cols:
            # Numeric target (like wine quality)
            correlations = df[numeric_cols].corr()[target_col].drop(target_col)
        else:
            # Categorical target (like adult income) - encode it
            target_encoded = pd.get_dummies(df[target_col], drop_first=True).iloc[:, 0]
            correlations = df[numeric_cols].corrwith(target_encoded)

        # Get top correlations by absolute value
        correlations = correlations.abs().sort_values(ascending=False).head(top_n)

        # Create DataFrame
        corr_df = pd.DataFrame(
            {
                "Feature": correlations.index,
                "Absolute Correlation": [f"{c:.3f}" for c in correlations.values],
            }
        )

        # Convert to LaTeX
        latex_table = corr_df.to_latex(
            index=False,
            caption=f"{dataset_name} Top {top_n} Features by Correlation with Target",
            label=f'tab:{dataset_name.lower().replace(" ", "_")}_correlations',
            position="h",
            column_format="lr",
            escape=False,
        )

        # Save
        filename = f'{dataset_name.lower().replace(" ", "_")}_correlations.tex'
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            f.write(latex_table)

        print(f"✓ Generated: {filename}")
        self.tables["correlations"] = latex_table
        return latex_table

    def generate_categorical_summary_table(self, df, dataset_name, max_categories=5):
        """
        Generate summary table for categorical features

        Parameters:
        -----------
        df : DataFrame
            The dataset
        dataset_name : str
            Name for the table caption
        max_categories : int
            Maximum number of categorical features to include
        """
        # Get categorical columns
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        # Remove target if present
        if "income" in categorical_cols:
            categorical_cols.remove("income")
        if "quality" in categorical_cols:
            categorical_cols.remove("quality")

        if len(categorical_cols) == 0:
            print("⚠ No categorical features to summarize")
            return None

        # Limit to max_categories
        categorical_cols = categorical_cols[:max_categories]

        # Create summary
        summary_data = []
        for col in categorical_cols:
            n_unique = df[col].nunique()
            most_common = df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A"
            most_common_pct = (df[col] == most_common).mean() * 100

            summary_data.append(
                {
                    "Feature": col,
                    "Unique Values": n_unique,
                    "Most Common": most_common,
                    "Frequency": f"{most_common_pct:.1f}\\%",
                }
            )

        summary_df = pd.DataFrame(summary_data)

        # Convert to LaTeX
        latex_table = summary_df.to_latex(
            index=False,
            caption=f"{dataset_name} Categorical Feature Summary",
            label=f'tab:{dataset_name.lower().replace(" ", "_")}_categorical',
            position="h",
            column_format="lrlr",
            escape=False,
        )

        # Save
        filename = f'{dataset_name.lower().replace(" ", "_")}_categorical_summary.tex'
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            f.write(latex_table)

        print(f"✓ Generated: {filename}")
        self.tables["categorical"] = latex_table
        return latex_table

    def generate_all_tables(self, df, target_col, dataset_name):
        """
        Generate all standard tables for a dataset

        Parameters:
        -----------
        df : DataFrame
            The dataset
        target_col : str
            Name of target column
        dataset_name : str
            Name for tables (e.g., 'Adult Income', 'Wine Quality')
        """
        print(f"\n{'='*60}")
        print(f"Generating LaTeX Tables for {dataset_name}")
        print(f"{'='*60}\n")

        # Generate all tables
        self.generate_dataset_overview_table(df, target_col, dataset_name)
        self.generate_class_distribution_table(df, target_col, dataset_name)
        self.generate_feature_statistics_table(df, target_col, dataset_name)
        self.generate_correlation_table(df, target_col, dataset_name)
        self.generate_categorical_summary_table(df, dataset_name)

        print(f"\n✓ All tables saved to: {self.output_dir}")
        return self.tables

    def create_master_latex_file(self, dataset_name):
        """
        Create a master .tex file that includes all tables

        Parameters:
        -----------
        dataset_name : str
            Name of the dataset
        """
        prefix = dataset_name.lower().replace(" ", "_")

        master_content = f"""% LaTeX Tables for {dataset_name} Dataset
% CS7641 Spring 2026 - Supervised Learning Assignment
% Auto-generated by LaTeXTableGenerator

% To use these tables in your report, either:
% 1. Copy individual table code from the .tex files, OR
% 2. Use \\input{{path/to/table.tex}} in your main document

% Required packages (add to your preamble):
% \\usepackage{{booktabs}}
% \\usepackage{{caption}}

\\section{{{dataset_name} Dataset}}

\\subsection{{Dataset Overview}}
\\input{{{prefix}_overview.tex}}

\\subsection{{Class Distribution}}
\\input{{{prefix}_class_distribution.tex}}

\\subsection{{Feature Statistics}}
\\input{{{prefix}_feature_statistics.tex}}

\\subsection{{Feature Correlations}}
\\input{{{prefix}_correlations.tex}}

% If categorical features exist:
% \\subsection{{Categorical Features}}
% \\input{{{prefix}_categorical_summary.tex}}
"""

        # Save master file
        master_path = self.output_dir / f"{prefix}_tables_master.tex"
        with open(master_path, "w") as f:
            f.write(master_content)

        print(f"\n✓ Master LaTeX file: {master_path.name}")
        print(f"  Include in your report with: \\input{{{master_path}}}")


def generate_latex_tables_from_csv(csv_path, target_col, dataset_name, output_dir=None):
    """
    Convenience function to generate all tables from a CSV file

    Parameters:
    -----------
    csv_path : str
        Path to CSV file
    target_col : str
        Name of target column
    dataset_name : str
        Name for the dataset (e.g., 'Adult Income')
    output_dir : str, optional
        Output directory for tables

    Returns:
    --------
    LaTeXTableGenerator
        Generator object with all tables
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Strip whitespace from string columns
    str_columns = df.select_dtypes(include=["object"]).columns
    for col in str_columns:
        df[col] = df[col].str.strip()

    # Create generator
    if output_dir is None:
        output_dir = f'results/{dataset_name.lower().replace(" ", "_")}/latex_tables'

    generator = LaTeXTableGenerator(output_dir)

    # Generate all tables
    generator.generate_all_tables(df, target_col, dataset_name)
    generator.create_master_latex_file(dataset_name)

    return generator


# Example usage functions
def generate_adult_income_tables(csv_path="data/raw/adult_income.csv"):
    """Generate all LaTeX tables for Adult Income dataset"""
    return generate_latex_tables_from_csv(
        csv_path=csv_path, target_col="class", dataset_name="Adult Income"
    )


def generate_wine_quality_tables(csv_path="data/raw/wine_quality.csv"):
    """Generate all LaTeX tables for Wine Quality dataset"""
    return generate_latex_tables_from_csv(
        csv_path=csv_path, target_col="quality", dataset_name="Wine Quality"
    )


if __name__ == "__main__":
    """
    Run this script to generate LaTeX tables for both datasets
    """
    import sys

    print("=" * 60)
    print("LaTeX Table Generator for CS7641 Assignment")
    print("=" * 60)

    # Try Adult Income
    adult_path = "data/raw/adult.csv"
    if Path(adult_path).exists():
        print(f"\n✓ Found Adult Income dataset")
        try:
            generate_adult_income_tables(adult_path)
        except Exception as e:
            print(f"❌ Error generating Adult Income tables: {e}")
    else:
        print(f"\n⚠ Adult Income dataset not found at: {adult_path}")

    # Try Wine Quality
    wine_path = "data/raw/wine.csv"
    if Path(wine_path).exists():
        print(f"\n✓ Found Wine Quality dataset")
        try:
            generate_wine_quality_tables(wine_path)
        except Exception as e:
            print(f"❌ Error generating Wine Quality tables: {e}")
    else:
        print(f"\n⚠ Wine Quality dataset not found at: {wine_path}")

    print("\n" + "=" * 60)
    print("LaTeX Table Generation Complete")
    print("=" * 60)
    print("\nTo use in your Overleaf report:")
    print("1. Upload all .tex files from results/*/latex_tables/")
    print("2. Add to preamble: \\usepackage{booktabs}")
    print("3. Use \\input{table_name.tex} where needed")
    print("   OR copy table code directly from .tex files")
