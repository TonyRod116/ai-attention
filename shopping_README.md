# Shopping - CS50 AI Project

Write an AI to predict whether online shopping customers will complete a purchase.

## Overview

This project implements a nearest-neighbor classifier to predict purchasing intent based on session data from about 12,000 user sessions. The model measures both sensitivity (True Positive Rate) and specificity (True Negative Rate) to ensure good performance in predicting both buyers and non-buyers.

## Files

- `shopping.py` - Main implementation with three required functions
- `shopping_requirements.txt` - Python dependencies
- `shopping.csv` - Dataset (download from CS50)

## Installation

1. Install required dependencies:
```bash
pip install -r shopping_requirements.txt
```

2. Download the dataset from: https://cdn.cs50.net/ai/2023/x/projects/4/shopping.zip

## Usage

Run the program with the CSV data file:

```bash
python shopping.py shopping.csv
```

Expected output:
```
Correct: 4088
Incorrect: 844
True Positive Rate: 41.02%
True Negative Rate: 90.55%
```

## Implementation Details

### Functions Implemented

1. **`load_data(filename)`**
   - Loads shopping data from CSV file
   - Converts data types according to specification
   - Returns tuple (evidence, labels)

2. **`train_model(evidence, labels)`**
   - Trains a KNeighborsClassifier with k=1
   - Returns fitted model

3. **`evaluate(labels, predictions)`**
   - Calculates sensitivity and specificity
   - Returns tuple (sensitivity, specificity)

### Data Conversion

- **Integers**: Administrative, Informational, ProductRelated, Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend
- **Floats**: Administrative_Duration, Informational_Duration, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay
- **Month**: Jan=0, Feb=1, ..., Dec=11
- **VisitorType**: Returning_Visitor=1, Other=0
- **Weekend**: TRUE=1, FALSE=0

## Testing

Use CS50's testing tools:

```bash
check50 ai50/projects/2024/x/shopping
style50 shopping.py
```

## Submission

Submit using CS50's submission system:

```bash
submit50 ai50/projects/2024/x/shopping
```

Or push to GitHub under branch `ai50/projects/2024/x/shopping`.

## Data Source

Sakar, C.O., Polat, S.O., Katircioglu, M. et al. *Neural Comput & Applic (2018)*
https://link.springer.com/article/10.1007%2Fs00521-018-3523-0
