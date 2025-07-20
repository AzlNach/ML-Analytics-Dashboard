import pandas as pd
import json
import requests

# Simulate the exact frontend data processing
print("Loading FinanKu dataset...")
df = pd.read_csv('training_data/FinanKu Data All.csv')

# Convert to the format the frontend sends (list of dictionaries)
csv_data = df.head(100).to_dict('records')  # Test with first 100 rows

print(f"Converted to {len(csv_data)} records")
print("Sample record:", csv_data[0])

# Clean data like the frontend does
def clean_data_for_api(data):
    cleaned_data = []
    for row_index, row in enumerate(data):
        clean_row = {}
        for key, value in row.items():
            try:
                if value is None or pd.isna(value):
                    clean_row[key] = None
                elif isinstance(value, (int, float)) and (pd.isna(value) or not pd.isfinite(value)):
                    clean_row[key] = None
                elif isinstance(value, list):
                    if len(value) > 0:
                        first_val = value[0]
                        clean_row[key] = first_val if isinstance(first_val, (int, float)) else str(first_val)
                    else:
                        clean_row[key] = None
                elif isinstance(value, dict):
                    clean_row[key] = str(value)
                elif isinstance(value, str):
                    lower_value = value.lower().strip()
                    if value == '' or lower_value in ['nan', 'null', 'undefined']:
                        clean_row[key] = None
                    else:
                        try:
                            num_value = float(value)
                            if not pd.isna(num_value) and pd.isfinite(num_value) and value.strip() == str(num_value):
                                clean_row[key] = num_value
                            else:
                                clean_row[key] = value
                        except:
                            clean_row[key] = value
                else:
                    clean_row[key] = value
            except Exception as e:
                print(f"Error cleaning value for {key} in row {row_index}: {e}")
                clean_row[key] = None
        cleaned_data.append(clean_row)
    return cleaned_data

cleaned_csv_data = clean_data_for_api(csv_data)
print("Sample cleaned record:", cleaned_csv_data[0])

# Test the decision tree endpoint directly
payload = {
    'csv_data': cleaned_csv_data,
    'target_column': 'City',
    'selected_features': ['Age', 'Avg. Annual Income/Month', 'Balance Q1', 'NumOfProducts Q1'],
    'max_depth': 5
}

print("Sending request to decision tree endpoint...")
print(f"Payload keys: {list(payload.keys())}")
print(f"CSV data length: {len(payload['csv_data'])}")
print(f"Target column: {payload['target_column']}")
print(f"Selected features: {payload['selected_features']}")

try:
    response = requests.post(
        'http://localhost:5000/api/decision-tree',
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Response status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("Success! Decision tree result:")
        print(f"Accuracy: {result.get('accuracy')}")
        print(f"Model ID: {result.get('model_id')}")
        print(f"Feature importance: {result.get('feature_importance')}")
    else:
        print("Error response:")
        print(response.text)
        
except Exception as e:
    print(f"Request failed: {e}")
