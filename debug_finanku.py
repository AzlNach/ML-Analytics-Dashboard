import pandas as pd
import numpy as np

print("Loading FinanKu dataset...")
df = pd.read_csv('training_data/FinanKu Data All.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Check for any problematic data types
print("\nChecking for complex data types...")
for i, row in df.iterrows():
    for col, val in row.items():
        if isinstance(val, (list, tuple)):
            print(f"Found list/tuple in Row {i}, Col '{col}': {val} (type: {type(val)})")
        elif isinstance(val, str) and ('[' in val or '{' in val):
            print(f"Found potential complex string in Row {i}, Col '{col}': {val}")
    
    if i > 20:  # Check first 20 rows
        break

print("\nData types:")
print(df.dtypes)

# Check for NaN values
print(f"\nMissing values per column:")
print(df.isnull().sum())

# Sample some data
print(f"\nFirst 3 rows:")
print(df.head(3))
