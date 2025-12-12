# Pranaam Examples

This directory contains practical examples showing how to use the pranaam package for religion prediction from names.

## Overview

| Example | Description | Complexity | Use Case |
|---------|-------------|------------|----------|
| [`basic_usage.py`](basic_usage.py) | Simple usage patterns | Beginner | Learning the API |
| [`pandas_integration.py`](pandas_integration.py) | DataFrame processing | Intermediate | Data analysis |
| [`csv_processor.py`](csv_processor.py) | Command-line utility | Intermediate | Batch processing |
| [`performance_demo.py`](performance_demo.py) | Performance analysis | Advanced | Optimization |

## Quick Start

```bash
# Install pranaam
pip install pranaam

# Run basic examples
python examples/basic_usage.py
python examples/pandas_integration.py  
python examples/performance_demo.py

# Try the CSV processor
python examples/csv_processor.py --create-sample
python examples/csv_processor.py sample_names.csv results.csv --name-column "full_name"
```

## Example Details

### 1. Basic Usage (`basic_usage.py`)
**What it shows:**
- Single name prediction
- Multiple name prediction  
- Working with English and Hindi names
- Understanding output format

**Run it:**
```bash
python examples/basic_usage.py
```

**Expected output:**
```
🔮 Single Name Prediction
English name:
             name pred_label  pred_prob_muslim
0  Shah Rukh Khan     muslim              71.0

📝 Multiple Names Prediction
Batch prediction results:
             name    pred_label  pred_prob_muslim
0   Shah Rukh Khan     muslim              71.0
1  Amitabh Bachchan not-muslim              15.2
```

### 2. Pandas Integration (`pandas_integration.py`)
**What it shows:**
- Loading data from pandas DataFrame
- Merging predictions with existing data
- Data analysis with predictions
- Confidence-based filtering
- Saving results to CSV

**Run it:**
```bash
python examples/pandas_integration.py
```

**Key concepts:**
- Use `df["name"]` as input to `pranaam.pred_rel()`
- Merge results back using `pd.merge()` on name column
- Filter by confidence levels for quality control
- Analyze demographics with prediction results

### 3. CSV Processor (`csv_processor.py`)
**What it shows:**
- Command-line interface for batch processing
- Reading/writing CSV files
- Error handling and validation
- Progress reporting

**Run it:**
```bash
# Create sample data
python examples/csv_processor.py --create-sample

# Process the sample file
python examples/csv_processor.py sample_names.csv results.csv --name-column "full_name"

# For Hindi names
python examples/csv_processor.py data.csv results.csv --name-column "name" --language hin
```

**Features:**
- Handles missing names gracefully
- Validates input files and columns
- Shows processing statistics
- Configurable name column and language

### 4. Performance Demo (`performance_demo.py`)
**What it shows:**
- Batch processing performance
- Model caching behavior  
- Language switching costs
- Memory usage patterns
- Real-world benchmarks

**Run it:**
```bash
python examples/performance_demo.py
```

**Performance insights:**
- First prediction: 3-5 seconds (includes model loading)
- Cached predictions: 100-500+ names/second
- Batch processing is much more efficient than individual calls
- Language switching requires model reload (~2-3 seconds)

## Common Patterns

### Basic Prediction
```python
import pranaam

# Single name
result = pranaam.pred_rel("Shah Rukh Khan", lang="eng")

# Multiple names  
names = ["Shah Rukh Khan", "Priya Sharma"]
result = pranaam.pred_rel(names, lang="eng")
```

### DataFrame Integration
```python
import pandas as pd
import pranaam

# Load your data
df = pd.read_csv("your_data.csv")

# Get predictions
predictions = pranaam.pred_rel(df["name_column"], lang="eng")

# Merge back to original data
df_with_predictions = df.merge(predictions, left_on="name_column", right_on="name")
```

### Confidence Filtering
```python
# Keep only high-confidence predictions
high_conf = df_with_predictions[
    (df_with_predictions["pred_prob_muslim"] > 90) |  # High confidence Muslim
    (df_with_predictions["pred_prob_muslim"] < 10)    # High confidence Not Muslim
]
```

## Performance Tips

1. **Batch Processing**: Always process multiple names at once rather than individual calls
2. **Model Caching**: The model stays loaded between calls - reuse the same process
3. **Language Switching**: Minimize switches between English and Hindi models  
4. **Memory Management**: Process very large datasets in chunks of 1000-5000 names
5. **First Call**: The first prediction takes longer due to model loading

## Supported Languages

- **English (`eng`)**: Names transliterated to English script
- **Hindi (`hin`)**: Names in Devanagari script

## Output Format

All examples return pandas DataFrames with:
- `name`: The input name
- `pred_label`: Predicted religion ("muslim" or "not-muslim") 
- `pred_prob_muslim`: Probability of being Muslim (0-100)

## Error Handling

The examples demonstrate proper error handling for:
- Missing input files
- Invalid column names
- Empty or malformed names
- Network issues during model download
- Memory constraints with large batches

## Next Steps

After running these examples:
1. Integrate pranaam into your existing data pipeline
2. Adapt the CSV processor for your specific file formats
3. Use the performance insights to optimize your batch sizes
4. Implement confidence thresholds appropriate for your use case

For more information, see the [main documentation](https://appeler.github.io/pranaam/).