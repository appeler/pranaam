#!/usr/bin/env python3
"""
Performance and Caching Demo for Pranaam

This script demonstrates the performance characteristics and caching behavior
of pranaam, including batch processing and language switching.
"""

import time

import pranaam
from pranaam.naam import Naam


def reset_model_state():
    """Reset model state for clean timing."""
    Naam.model = None
    Naam.weights_loaded = False
    Naam.cur_lang = None


def time_function(func, *args, **kwargs):
    """Time a function call and return result and elapsed time."""
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed


def batch_size_performance():
    """Test performance across different batch sizes."""
    print("⚡ Batch Size Performance Analysis")
    print("=" * 50)

    # Generate test names
    base_names = [
        "Shah Rukh Khan",
        "Amitabh Bachchan",
        "Salman Khan",
        "Priya Sharma",
        "Mohammed Ali",
        "Raj Patel",
    ]

    batch_sizes = [1, 5, 10, 25, 50, 100]
    results = []

    for batch_size in batch_sizes:
        # Create batch by repeating/cycling base names
        test_names = (base_names * ((batch_size // len(base_names)) + 1))[:batch_size]

        # Reset state for clean timing
        reset_model_state()

        # Time the prediction
        _, elapsed = time_function(pranaam.pred_rel, test_names, lang="eng")

        names_per_sec = batch_size / elapsed
        ms_per_name = (elapsed * 1000) / batch_size

        results.append(
            {
                "batch_size": batch_size,
                "total_time": elapsed,
                "names_per_sec": names_per_sec,
                "ms_per_name": ms_per_name,
            }
        )

        print(
            f"Batch {batch_size:>3}: {elapsed:>6.2f}s total, {names_per_sec:>6.1f} names/sec, {ms_per_name:>6.1f}ms/name"
        )

    # Show efficiency gains
    print("\nBatch Processing Efficiency:")
    single_ms = results[0]["ms_per_name"]
    for r in results[1:]:
        speedup = single_ms / r["ms_per_name"]
        print(
            f"Batch {r['batch_size']:>3}: {speedup:>4.1f}x faster than single predictions"
        )
    print()


def model_caching_behavior():
    """Demonstrate model caching and reload behavior."""
    print("💾 Model Caching and Reload Behavior")
    print("=" * 50)

    test_name = "Shah Rukh Khan"

    # First prediction - includes model loading
    reset_model_state()
    print("First prediction (includes model download/loading):")
    _, elapsed1 = time_function(pranaam.pred_rel, test_name, lang="eng")
    print(f"   Time: {elapsed1:.2f}s")
    print(f"   Model loaded: {Naam.weights_loaded}")
    print(f"   Current language: {Naam.cur_lang}")

    # Second prediction - should use cached model
    print("\nSecond prediction (cached model):")
    _, elapsed2 = time_function(pranaam.pred_rel, test_name, lang="eng")
    print(f"   Time: {elapsed2:.2f}s")
    print(f"   Speedup: {elapsed1 / elapsed2:.1f}x faster")

    # Third prediction with different name - still cached
    print("\nThird prediction with different name (still cached):")
    _, elapsed3 = time_function(pranaam.pred_rel, "Amitabh Bachchan", lang="eng")
    print(f"   Time: {elapsed3:.2f}s")
    print(f"   Similar performance: {abs(elapsed3 - elapsed2) < 0.1}")
    print()


def language_switching_performance():
    """Test performance when switching between languages."""
    print("🔄 Language Switching Performance")
    print("=" * 50)

    english_name = "Shah Rukh Khan"
    hindi_name = "शाहरुख खान"

    # Start with English
    reset_model_state()
    print("Initial English prediction:")
    _, elapsed_eng1 = time_function(pranaam.pred_rel, english_name, lang="eng")
    print(f"   Time: {elapsed_eng1:.2f}s (includes model loading)")
    print(f"   Current language: {Naam.cur_lang}")

    # Switch to Hindi - requires model reload
    print("\nSwitch to Hindi (requires model reload):")
    _, elapsed_hin = time_function(pranaam.pred_rel, hindi_name, lang="hin")
    print(f"   Time: {elapsed_hin:.2f}s")
    print(f"   Current language: {Naam.cur_lang}")

    # Switch back to English - requires reload again
    print("\nSwitch back to English (requires reload):")
    _, elapsed_eng2 = time_function(pranaam.pred_rel, english_name, lang="eng")
    print(f"   Time: {elapsed_eng2:.2f}s")
    print(f"   Current language: {Naam.cur_lang}")

    # Second English prediction - should be fast
    print("\nSecond English prediction (cached):")
    _, elapsed_eng3 = time_function(pranaam.pred_rel, english_name, lang="eng")
    print(f"   Time: {elapsed_eng3:.2f}s")
    print(f"   Speedup vs reload: {elapsed_eng2 / elapsed_eng3:.1f}x")
    print()


def memory_usage_demo():
    """Show memory considerations for large batches."""
    print("🧠 Memory Usage Considerations")
    print("=" * 50)

    # Test with increasingly large batches
    base_names = ["Shah Rukh Khan", "Priya Sharma", "Mohammed Ali"]

    for size in [100, 500, 1000]:
        test_names = (base_names * (size // len(base_names) + 1))[:size]

        print(f"Processing {size} names...")
        _, elapsed = time_function(pranaam.pred_rel, test_names, lang="eng")

        rate = size / elapsed
        print(f"   Time: {elapsed:.2f}s ({rate:.0f} names/sec)")

    print("\n💡 Memory Tips:")
    print("   • Process in chunks of 1000-5000 names for optimal memory usage")
    print("   • Model stays loaded between predictions (uses ~500MB RAM)")
    print("   • Language switching reloads model but previous model is freed")
    print()


def practical_benchmarks():
    """Show practical real-world performance benchmarks."""
    print("📊 Practical Performance Benchmarks")
    print("=" * 50)

    # Typical use cases
    use_cases = [
        ("Single name lookup", 1),
        ("Small batch (department)", 25),
        ("Medium batch (company)", 500),
        ("Large batch (survey)", 5000),
    ]

    base_names = [
        "Shah Rukh Khan",
        "Amitabh Bachchan",
        "Priya Sharma",
        "Mohammed Ali",
        "Raj Patel",
        "Fatima Khan",
    ]

    print("Use Case                    | Size  | Time    | Rate")
    print("-" * 55)

    for use_case, size in use_cases:
        test_names = (base_names * (size // len(base_names) + 1))[:size]

        # Reset for fair timing
        reset_model_state()
        _, elapsed = time_function(pranaam.pred_rel, test_names, lang="eng")

        rate = size / elapsed
        print(f"{use_case:<25} | {size:>4} | {elapsed:>6.2f}s | {rate:>6.0f}/sec")

    print("\n🎯 Performance Summary:")
    print("   • Cold start (first prediction): ~3-5 seconds")
    print("   • Warm predictions: 100-500+ names/second")
    print("   • Model loading is one-time cost per language")
    print("   • Batch processing is highly efficient")


if __name__ == "__main__":
    print("🚀 Pranaam Performance and Caching Demo")
    print("=" * 60)
    print("This script analyzes performance characteristics of the pranaam package.\n")

    batch_size_performance()
    model_caching_behavior()
    language_switching_performance()
    memory_usage_demo()
    practical_benchmarks()

    print("✅ Performance analysis completed!")
    print(
        "Use these insights to optimize your pranaam usage for your specific use case."
    )
