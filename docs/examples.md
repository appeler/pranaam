# Examples

This page contains practical examples of using pranaam in various scenarios.

## 🚀 Interactive Jupyter Notebooks

For hands-on, executable examples, check out our interactive Jupyter notebooks:

- **[Basic Usage](examples/basic_usage.ipynb)**: Learn the fundamentals with single and batch predictions
- **[Pandas Integration](examples/pandas_integration.ipynb)**: DataFrame processing and data analysis workflows  
- **[CSV Processing](examples/csv_processing.ipynb)**: File processing techniques for real-world datasets
- **[Performance Benchmarks](examples/performance_benchmarks.ipynb)**: Optimization strategies and performance analysis

These notebooks are fully executable and demonstrate best practices for using pranaam in production environments.

## 📝 Code Examples

Below are quick reference examples for common use cases:

## Basic Examples

### Single Name Prediction

```python
import pranaam

# Basic prediction
result = pranaam.pred_rel("Abdul Kalam")
print(result)

# Output:
#        name pred_label  pred_prob_muslim
# 0  Abdul Kalam     muslim              85.5
```

### Multiple Names

```python
import pranaam

# List of names
bollywood_actors = [
    "Shah Rukh Khan",
    "Amitabh Bachchan",
    "Salman Khan",
    "Aamir Khan",
    "Akshay Kumar"
]

results = pranaam.pred_rel(bollywood_actors, lang="eng")
print(results)
```

## Data Analysis Examples

### Working with CSV Files

```python
import pandas as pd
import pranaam

# Read names from CSV
df = pd.read_csv('names.csv')

# Predict religion for name column
predictions = pranaam.pred_rel(df['name'], lang="eng")

# Merge predictions with original data
df_with_predictions = pd.concat([df, predictions[['pred_label', 'pred_prob_muslim']]], axis=1)

# Save results
df_with_predictions.to_csv('results.csv', index=False)
```

### Batch Processing

```python
import pranaam
import pandas as pd
from typing import List

def process_large_dataset(names: List[str], batch_size: int = 1000) -> pd.DataFrame:
    """Process large datasets in batches to manage memory."""
    all_results = []

    for i in range(0, len(names), batch_size):
        batch = names[i:i+batch_size]
        batch_results = pranaam.pred_rel(batch, lang="eng")
        all_results.append(batch_results)
        print(f"Processed batch {i//batch_size + 1}/{(len(names)-1)//batch_size + 1}")

    return pd.concat(all_results, ignore_index=True)

# Usage
large_name_list = ["Name" + str(i) for i in range(10000)]  # Your actual names
results = process_large_dataset(large_name_list)
```

### Statistical Analysis

```python
import pranaam
import pandas as pd
import matplotlib.pyplot as plt

# Sample dataset
politicians = [
    "Narendra Modi", "Rahul Gandhi", "Mamata Banerjee",
    "Arvind Kejriwal", "Yogi Adityanath", "Akhilesh Yadav",
    "Mayawati", "Sharad Pawar", "Uddhav Thackeray"
]

# Get predictions
results = pranaam.pred_rel(politicians)

# Statistical summary
print("Religion Distribution:")
print(results['pred_label'].value_counts())

print("\nAverage Confidence Scores:")
print(results.groupby('pred_label')['pred_prob_muslim'].mean())

# Visualization
plt.figure(figsize=(10, 6))
plt.hist(results['pred_prob_muslim'], bins=20, alpha=0.7)
plt.xlabel('Muslim Probability Score')
plt.ylabel('Count')
plt.title('Distribution of Prediction Confidence Scores')
plt.show()
```

## Hindi Language Examples

### Devanagari Script

```python
import pranaam

# Hindi names in Devanagari script
hindi_names = [
    "शाहरुख खान",     # Shah Rukh Khan
    "अमिताभ बच्चन",    # Amitabh Bachchan
    "आमिर खान",       # Aamir Khan
    "अक्षय कुमार",     # Akshay Kumar
    "सलमान खान"       # Salman Khan
]

results = pranaam.pred_rel(hindi_names, lang="hin")
print(results)
```

### Mixed Script Dataset

```python
import pranaam
import pandas as pd

# Dataset with mixed English and Hindi names
mixed_data = pd.DataFrame({
    'name': ["Shah Rukh Khan", "शाहरुख खान", "Amitabh Bachchan", "अमिताभ बच्चन"],
    'script': ['English', 'Hindi', 'English', 'Hindi']
})

# Process by script type
english_results = pranaam.pred_rel(
    mixed_data[mixed_data['script'] == 'English']['name'],
    lang="eng"
)

hindi_results = pranaam.pred_rel(
    mixed_data[mixed_data['script'] == 'Hindi']['name'],
    lang="hin"
)

# Combine results
all_results = pd.concat([english_results, hindi_results], ignore_index=True)
```

## Advanced Usage

### Custom Confidence Thresholds

```python
import pranaam
import pandas as pd

# Get predictions with custom confidence analysis
names = ["Mohammed Ali", "John Smith", "Ahmad Khan", "David Wilson"]
results = pranaam.pred_rel(names)

# Add confidence categories
def categorize_confidence(prob):
    if prob >= 80:
        return "High"
    elif prob >= 60:
        return "Medium"
    else:
        return "Low"

results['confidence_level'] = results['pred_prob_muslim'].apply(categorize_confidence)

# Filter high-confidence predictions only
high_confidence = results[results['confidence_level'] == 'High']
print("High confidence predictions:")
print(high_confidence)
```

### Error Handling

```python
import pranaam
import pandas as pd

def safe_predict(names, lang="eng"):
    """Predict with error handling."""
    try:
        results = pranaam.pred_rel(names, lang=lang)
        return results
    except ValueError as e:
        print(f"ValueError: {e}")
        return pd.DataFrame()
    except FileNotFoundError as e:
        print(f"Model file error: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error: {e}")
        return pd.DataFrame()

# Usage
problematic_input = ["", None, "Valid Name"]  # Contains invalid entries
results = safe_predict([name for name in problematic_input if name])
```

### Performance Optimization

```python
import pranaam
import time

# Preload models for better performance in production
# First call loads the model into memory
_ = pranaam.pred_rel("Test Name")

# Subsequent calls are faster
large_dataset = ["Name" + str(i) for i in range(1000)]

start_time = time.time()
results = pranaam.pred_rel(large_dataset)
end_time = time.time()

print(f"Processed {len(large_dataset)} names in {end_time - start_time:.2f} seconds")
print(f"Average: {(end_time - start_time)/len(large_dataset)*1000:.2f} ms per name")
```

## Integration Examples

### Flask Web Application

```python
from flask import Flask, request, jsonify
import pranaam

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_religion():
    data = request.json
    names = data.get('names', [])
    lang = data.get('lang', 'eng')

    try:
        results = pranaam.pred_rel(names, lang=lang)
        return jsonify(results.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
```

### Command Line Scripts

```bash
# Using the built-in CLI
predict_religion --input "Shah Rukh Khan" --lang eng

# Batch processing from file
predict_religion --input-file names.txt --output-file results.csv --lang eng
```

## Real-world Use Cases

### Research Applications

```python
import pranaam
import pandas as pd

# Academic research on name-based demographics
def analyze_author_demographics(author_names):
    """Analyze religious demographics of academic authors."""
    results = pranaam.pred_rel(author_names)

    # Calculate statistics
    total_authors = len(results)
    muslim_authors = len(results[results['pred_label'] == 'muslim'])

    demographics = {
        'total_authors': total_authors,
        'muslim_percentage': (muslim_authors / total_authors) * 100,
        'non_muslim_percentage': ((total_authors - muslim_authors) / total_authors) * 100
    }

    return demographics, results

# Example usage
ieee_authors = ["A. Sharma", "M. Ahmed", "R. Singh", "S. Ali"]
stats, detailed_results = analyze_author_demographics(ieee_authors)
```

### Business Intelligence

```python
import pranaam
import pandas as pd

# Customer base analysis
def analyze_customer_base(customer_df):
    """Analyze customer demographics for business insights."""
    # Predict religion from customer names
    predictions = pranaam.pred_rel(customer_df['customer_name'])

    # Merge with customer data
    analysis_df = pd.concat([customer_df, predictions[['pred_label', 'pred_prob_muslim']]], axis=1)

    # Business insights
    demographic_summary = analysis_df.groupby('pred_label').agg({
        'purchase_amount': ['mean', 'sum', 'count'],
        'pred_prob_muslim': 'mean'
    }).round(2)

    return analysis_df, demographic_summary

# Example usage
customer_data = pd.DataFrame({
    'customer_name': ['Ahmed Khan', 'Priya Sharma', 'Mohammad Ali'],
    'purchase_amount': [1500, 2000, 1200]
})

results, summary = analyze_customer_base(customer_data)
```
