"""
Enhanced EDA Runner with LaTeX Table Generation
CS7641 Spring 2026 - Supervised Learning Assignment

Runs comprehensive EDA and generates publication-ready LaTeX tables.
"""

import sys
from pathlib import Path


def check_file_exists(filepath, dataset_name):
    """Check if data file exists"""
    if not Path(filepath).exists():
        print(f"❌ ERROR: {dataset_name} data file not found at: {filepath}")
        print(f"   Please update the path or place the file at the specified location.")
        return False
    print(f"✓ Found {dataset_name} data file")
    return True


def run_adult_eda_with_tables(data_path):
    """Run Adult Income EDA and generate LaTeX tables"""
    print("\n" + "=" * 80)
    print("RUNNING ADULT INCOME EDA + LATEX TABLES")
    print("=" * 80 + "\n")

    # Import EDA
    from adult_eda import AdultIncomeEDA

    # Run EDA
    eda = AdultIncomeEDA(data_path)
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

    print(f"\n✓ Adult Income EDA complete! Results in: {eda.output_dir}")

    # Generate LaTeX tables
    print("\n" + "-" * 60)
    print("Generating LaTeX Tables...")
    print("-" * 60)

    from latex_table_generator import generate_adult_income_tables

    try:
        generator = generate_adult_income_tables(data_path)
        print(f"\n✓ LaTeX tables generated!")
    except Exception as e:
        print(f"❌ Error generating LaTeX tables: {e}")
        return False

    return True


def run_wine_eda_with_tables(data_path):
    """Run Wine Quality EDA and generate LaTeX tables"""
    print("\n" + "=" * 80)
    print("RUNNING WINE QUALITY EDA + LATEX TABLES")
    print("=" * 80 + "\n")

    # Import EDA
    from wine_eda import WineQualityEDA

    # Run EDA
    eda = WineQualityEDA(data_path)
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

    print(f"\n✓ Wine Quality EDA complete! Results in: {eda.output_dir}")

    # Generate LaTeX tables
    print("\n" + "-" * 60)
    print("Generating LaTeX Tables...")
    print("-" * 60)

    from latex_table_generator import generate_wine_quality_tables

    try:
        generator = generate_wine_quality_tables(data_path)
        print(f"\n✓ LaTeX tables generated!")
    except Exception as e:
        print(f"❌ Error generating LaTeX tables: {e}")
        return False

    return True


def main():
    """Main execution"""
    print("=" * 80)
    print("CS7641 SPRING 2026 - SUPERVISED LEARNING EDA + LATEX TABLES")
    print("=" * 80)

    # Configuration - UPDATE THESE PATHS
    adult_path = "data/raw/adult.csv"
    wine_path = "data/raw/wine.csv"

    # Check which datasets are available
    adult_available = check_file_exists(adult_path, "Adult Income")
    wine_available = check_file_exists(wine_path, "Wine Quality")

    if not adult_available and not wine_available:
        print("\n❌ No data files found. Please update paths in this script.")
        print("\nExpected locations:")
        print(f"  - Adult Income: {adult_path}")
        print(f"  - Wine Quality: {wine_path}")
        sys.exit(1)

    # Run available analyses
    success = True

    if adult_available:
        try:
            run_adult_eda_with_tables(adult_path)
        except Exception as e:
            print(f"\n❌ Error in Adult Income EDA: {e}")
            import traceback

            traceback.print_exc()
            success = False

    if wine_available:
        try:
            run_wine_eda_with_tables(wine_path)
        except Exception as e:
            print(f"\n❌ Error in Wine Quality EDA: {e}")
            import traceback

            traceback.print_exc()
            success = False

    # Final summary
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)

    if success:
        print("\n✓ All requested analyses completed successfully!")

        print("\n📊 Generated outputs:")
        if adult_available:
            print("\n  Adult Income:")
            print("    Figures:")
            print("      - results/adult_income/eda/figures/*.png")
            print("    Summary:")
            print("      - results/adult_income/eda/eda_summary.txt")
            print("    LaTeX Tables:")
            print("      - results/adult_income/latex_tables/*.tex")

        if wine_available:
            print("\n  Wine Quality:")
            print("    Figures:")
            print("      - results/wine_quality/eda/figures/*.png")
            print("    Summary:")
            print("      - results/wine_quality/eda/eda_summary.txt")
            print("    LaTeX Tables:")
            print("      - results/wine_quality/latex_tables/*.tex")

        print("\n" + "-" * 80)
        print("📝 Using LaTeX Tables in Overleaf:")
        print("-" * 80)
        print("1. Upload all .tex files from results/*/latex_tables/ to Overleaf")
        print("2. Add to your preamble: \\usepackage{booktabs}")
        print("3. Include tables with: \\input{table_name.tex}")
        print("   OR copy-paste table code from .tex files directly")

        print("\n💡 Example usage in your report:")
        print("""
\\section{Exploratory Data Analysis}
\\subsection{Adult Income Dataset}
\\input{adult_income_overview.tex}
\\input{adult_income_class_distribution.tex}
        """)

        print("\n📚 Next steps:")
        print("   1. Review all generated figures")
        print("   2. Read the eda_summary.txt files")
        print("   3. Review LaTeX tables and customize if needed")
        print("   4. Formulate hypotheses based on observations")
        print("   5. Begin preprocessing pipeline development")
    else:
        print("\n⚠ Some errors occurred during execution")
        print("   Check error messages above for details")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
