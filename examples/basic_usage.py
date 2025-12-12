#!/usr/bin/env python3
"""
Basic Usage Examples for Pranaam

This script demonstrates the most common usage patterns for the pranaam package.
"""

import pranaam


def single_name_prediction():
    """Predict religion for a single name."""
    print("🔮 Single Name Prediction")
    print("=" * 40)

    # English name
    result = pranaam.pred_rel("Shah Rukh Khan", lang="eng")
    print("English name:")
    print(result)
    print()

    # Hindi name
    result = pranaam.pred_rel("शाहरुख खान", lang="hin")
    print("Hindi name:")
    print(result)
    print()


def multiple_names_prediction():
    """Predict religion for multiple names."""
    print("📝 Multiple Names Prediction")
    print("=" * 40)

    # List of English names
    names = ["Shah Rukh Khan", "Amitabh Bachchan", "Salman Khan", "Akshay Kumar"]

    result = pranaam.pred_rel(names, lang="eng")
    print("Batch prediction results:")
    print(result)
    print()


def mixed_examples():
    """Show predictions for mixed cultural names."""
    print("🌍 Mixed Cultural Names")
    print("=" * 40)

    diverse_names = [
        "Mohammed Ali",
        "Priya Sharma",
        "Fatima Khan",
        "Raj Patel",
        "John Smith",
    ]

    result = pranaam.pred_rel(diverse_names, lang="eng")

    print("Name                | Prediction   | Confidence")
    print("-" * 45)
    for _, row in result.iterrows():
        print(
            f"{row['name']:<18} | {row['pred_label']:<10} | {row['pred_prob_muslim']:>6.1f}%"
        )
    print()


if __name__ == "__main__":
    print("🔥 Pranaam Basic Usage Examples")
    print("=" * 50)
    print(
        "This script shows simple usage patterns for religion prediction from names.\n"
    )

    single_name_prediction()
    multiple_names_prediction()
    mixed_examples()

    print("✅ All examples completed!")
    print("Next steps: Check out pandas_integration.py for data processing examples.")
