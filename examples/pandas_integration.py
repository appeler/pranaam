#!/usr/bin/env python3
"""
Pandas Integration Examples for Pranaam

This script demonstrates how to use pranaam with pandas DataFrames for real-world data processing.
"""

import pandas as pd

import pranaam


def create_sample_data():
    """Create sample data for demonstration."""
    return pd.DataFrame(
        {
            "employee_id": [1001, 1002, 1003, 1004, 1005, 1006],
            "name": [
                "Shah Rukh Khan",
                "Priya Sharma",
                "Mohammed Ali",
                "Raj Patel",
                "Fatima Khan",
                "Amitabh Bachchan",
            ],
            "department": [
                "Engineering",
                "Marketing",
                "Finance",
                "HR",
                "Engineering",
                "Management",
            ],
            "salary": [75000, 65000, 70000, 60000, 80000, 120000],
        }
    )


def basic_dataframe_processing():
    """Show basic DataFrame processing with pranaam."""
    print("📊 Basic DataFrame Processing")
    print("=" * 50)

    # Create sample data
    df = create_sample_data()
    print("Original data:")
    print(df)
    print()

    # Get predictions for the name column
    predictions = pranaam.pred_rel(df["name"], lang="eng")
    print("Predictions:")
    print(predictions)
    print()

    # Merge predictions back to original DataFrame
    # Note: pranaam returns name, pred_label, pred_prob_muslim
    df_with_predictions = df.merge(
        predictions[["name", "pred_label", "pred_prob_muslim"]], on="name", how="left"
    )

    print("Combined data with predictions:")
    print(df_with_predictions)
    print()


def analysis_examples():
    """Show analysis examples using the predictions."""
    print("📈 Analysis Examples")
    print("=" * 50)

    df = create_sample_data()
    predictions = pranaam.pred_rel(df["name"], lang="eng")
    df_combined = df.merge(
        predictions[["name", "pred_label", "pred_prob_muslim"]], on="name"
    )

    # Basic statistics
    print("Religion distribution:")
    print(df_combined["pred_label"].value_counts())
    print()

    # Average salary by predicted religion
    print("Average salary by predicted religion:")
    salary_by_religion = df_combined.groupby("pred_label")["salary"].agg(
        ["mean", "count"]
    )
    print(salary_by_religion)
    print()

    # Department distribution by predicted religion
    print("Department distribution by predicted religion:")
    dept_religion = pd.crosstab(df_combined["department"], df_combined["pred_label"])
    print(dept_religion)
    print()


def confidence_filtering():
    """Show how to work with prediction confidence scores."""
    print("🎯 Confidence-Based Filtering")
    print("=" * 50)

    df = create_sample_data()
    predictions = pranaam.pred_rel(df["name"], lang="eng")
    df_combined = df.merge(
        predictions[["name", "pred_label", "pred_prob_muslim"]], on="name"
    )

    # Show confidence distribution
    print("Confidence distribution:")
    print("Name                | Prediction   | Confidence")
    print("-" * 50)
    for _, row in df_combined.iterrows():
        confidence = max(row["pred_prob_muslim"], 100 - row["pred_prob_muslim"])
        print(f"{row['name']:<18} | {row['pred_label']:<10} | {confidence:>6.1f}%")
    print()

    # Filter high-confidence predictions (>90%)
    high_confidence = df_combined[
        (df_combined["pred_prob_muslim"] > 90) | (df_combined["pred_prob_muslim"] < 10)
    ]
    print("High-confidence predictions (>90%):")
    print(high_confidence[["name", "pred_label", "pred_prob_muslim"]])
    print()


def save_results():
    """Show how to save results to different formats."""
    print("💾 Saving Results")
    print("=" * 50)

    df = create_sample_data()
    predictions = pranaam.pred_rel(df["name"], lang="eng")
    df_combined = df.merge(
        predictions[["name", "pred_label", "pred_prob_muslim"]], on="name"
    )

    # Save to CSV
    output_file = "employee_predictions.csv"
    df_combined.to_csv(output_file, index=False)
    print(f"✅ Results saved to {output_file}")

    # Show what was saved
    print("Saved data preview:")
    print(df_combined.head())

    # Clean up
    import os

    os.remove(output_file)
    print(f"🧹 Cleaned up {output_file}")


if __name__ == "__main__":
    print("🐼 Pranaam + Pandas Integration Examples")
    print("=" * 60)
    print("This script shows how to integrate pranaam with pandas for data analysis.\n")

    basic_dataframe_processing()
    analysis_examples()
    confidence_filtering()
    save_results()

    print("✅ All pandas integration examples completed!")
    print("Next steps: Check out csv_processor.py for command-line data processing.")
