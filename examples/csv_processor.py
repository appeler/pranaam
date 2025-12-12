#!/usr/bin/env python3
"""
CSV Processor - Command Line Utility for Pranaam

A practical command-line tool that reads a CSV file with names,
adds religion predictions, and saves the results.

Usage:
    python csv_processor.py input.csv output.csv --name-column "name" --language eng
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

import pranaam


def create_sample_csv(filename: str = "sample_names.csv") -> None:
    """Create a sample CSV file for testing."""
    sample_data = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "full_name": [
                "Shah Rukh Khan",
                "Priya Sharma",
                "Mohammed Ali",
                "Raj Patel",
                "Fatima Khan",
                "John Smith",
            ],
            "department": ["Engineering", "Marketing", "Finance", "HR", "Sales", "IT"],
            "city": ["Mumbai", "Delhi", "Bangalore", "Chennai", "Pune", "Hyderabad"],
        }
    )
    sample_data.to_csv(filename, index=False)
    print(f"📝 Created sample file: {filename}")


def process_csv(
    input_file: str, output_file: str, name_column: str, language: str
) -> None:
    """Process CSV file and add religion predictions."""

    # Validate input file
    if not Path(input_file).exists():
        print(f"❌ Error: Input file '{input_file}' not found")
        sys.exit(1)

    try:
        # Read CSV
        print(f"📖 Reading {input_file}...")
        df = pd.read_csv(input_file)
        print(f"   Found {len(df)} rows")

        # Validate name column
        if name_column not in df.columns:
            print(f"❌ Error: Column '{name_column}' not found in CSV")
            print(f"   Available columns: {list(df.columns)}")
            sys.exit(1)

        # Check for missing names
        missing_names = df[name_column].isna().sum()
        if missing_names > 0:
            print(f"⚠️  Warning: {missing_names} rows have missing names")
            df = df.dropna(subset=[name_column])
            print(f"   Processing {len(df)} rows with valid names")

        # Get predictions
        print(f"🔮 Getting predictions for {len(df)} names (language: {language})...")
        predictions = pranaam.pred_rel(df[name_column], lang=language)

        # Merge predictions back to original data
        # Rename columns to avoid conflicts
        predictions = predictions.rename(
            columns={
                "name": name_column,
                "pred_label": f"{name_column}_religion",
                "pred_prob_muslim": f"{name_column}_confidence_muslim",
            }
        )

        df_with_predictions = df.merge(predictions, on=name_column, how="left")

        # Save results
        print(f"💾 Saving results to {output_file}...")
        df_with_predictions.to_csv(output_file, index=False)

        # Show summary
        print("\n📊 Processing Summary:")
        print(f"   Input rows: {len(df)}")
        print(f"   Output rows: {len(df_with_predictions)}")

        religion_counts = df_with_predictions[f"{name_column}_religion"].value_counts()
        print(f"   Predictions: {dict(religion_counts)}")

        # Show confidence distribution
        conf_col = f"{name_column}_confidence_muslim"
        high_conf_muslim = (df_with_predictions[conf_col] > 90).sum()
        high_conf_not_muslim = (df_with_predictions[conf_col] < 10).sum()
        print(
            f"   High confidence (>90%): {high_conf_muslim + high_conf_not_muslim} predictions"
        )

        print(f"\n✅ Successfully processed {input_file} → {output_file}")

    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")
        sys.exit(1)


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Add religion predictions to CSV files using pranaam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python csv_processor.py data.csv results.csv --name-column "full_name"
    python csv_processor.py data.csv results.csv --name-column "employee_name" --language hin
    python csv_processor.py --create-sample  # Create sample file for testing
        """,
    )

    parser.add_argument("input_file", nargs="?", help="Input CSV file path")
    parser.add_argument("output_file", nargs="?", help="Output CSV file path")
    parser.add_argument(
        "--name-column",
        default="name",
        help="Name of the column containing names (default: 'name')",
    )
    parser.add_argument(
        "--language",
        choices=["eng", "hin"],
        default="eng",
        help="Language for predictions: eng (English) or hin (Hindi) (default: eng)",
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample CSV file for testing",
    )

    args = parser.parse_args()

    # Handle sample creation
    if args.create_sample:
        create_sample_csv()
        print(
            "You can now test with: python csv_processor.py sample_names.csv results.csv --name-column 'full_name'"
        )
        return

    # Validate required arguments
    if not args.input_file or not args.output_file:
        print("❌ Error: Both input_file and output_file are required")
        print("Use --help for usage information or --create-sample to create test data")
        sys.exit(1)

    # Process the CSV
    process_csv(args.input_file, args.output_file, args.name_column, args.language)


if __name__ == "__main__":
    main()
