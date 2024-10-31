# Quickstart

# VoteM8 Quickstart Guide

VoteM8 is a Python package that provides multiple consensus scoring methods for ranking and analyzing data. This guide will help you get started with the basic usage of VoteM8.

## Basic Usage

### 1. Simple Consensus Scoring

Here's a minimal example showing how to calculate consensus scores for your data:

```python
import pandas as pd
from votem8 import apply_consensus_scoring

# Load your data
data = pd.DataFrame({
    'ID': ['A1', 'A2', 'A3'],
    'Score1': [0.9, 0.8, 0.7],
    'Score2': [0.85, 0.9, 0.75],
    'Score3': [0.95, 0.85, 0.8]
})

# Apply consensus scoring using all available methods
results = apply_consensus_scoring(data)
```

### 2. Customizing the Analysis

You can customize the consensus scoring process with various parameters:

```python
results = apply_consensus_scoring(
    data,
    methods=['ECR', 'RbR', 'TOPSIS'],  # Specify methods to use
    columns=['Score1', 'Score2', 'Score3'],  # Specify columns to consider
    id_column='ID',  # Specify ID column
    normalize=True,  # Normalize scores to [0,1]
    aggregation='best'  # Use 'best' or 'avg' for aggregation
)
```

### 3. Handling Missing Values

VoteM8 provides several strategies for handling missing values:

```python
results = apply_consensus_scoring(
    data,
    nan_strategy='fill_mean',  # Options: 'raise', 'drop', 'fill_mean', 'fill_median', 'interpolate'
)
```

### 4. Using Weights

You can assign different weights to your scoring columns:

```python
weights = {
    'Score1': 0.5,
    'Score2': 0.3,
    'Score3': 0.2
}

results = apply_consensus_scoring(
    data,
    weights=weights
)
```

### 5. Command Line Interface

VoteM8 also provides a CLI for quick analysis:

```bash
votem8 input.csv --methods ECR RbR --columns Score1 Score2 Score3 --normalize --output results.csv
```

## Available Methods

VoteM8 includes the following consensus scoring methods:

- ECR (Exponential Consensus Ranking)
- RbR (Rank by Rank)
- RbV (Rank by Vote)
- Zscore
- TOPSIS
- WASPAS
- VIKOR
- ARAS
- WPM (Weighted Product Method)
- WSM (Weighted Sum Method)
- BinaryPareto
- Pareto

To see all available methods:

```python
from votem8 import get_available_methods

methods = get_available_methods()
print(methods)
```

## Output Format

The output is a pandas DataFrame containing:

- The ID column from your input data
- One column per consensus method showing the normalized consensus scores
- Rows sorted by consensus scores (highest to lowest)

## Next Steps

- Check the full documentation for detailed method descriptions
- Explore advanced features like custom weighting methods
- Learn about integrating VoteM8 into your data analysis pipeline
