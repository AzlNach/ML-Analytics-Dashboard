"""
Test script to verify the fix for the FinanKu Data unhashable type error.
This simulates the exact problem and tests the solution.
"""
import json

def test_complex_data_handling():
    # Simulate problematic data that might come from frontend
    test_data = [
        {
            'Customer ID': 15565701,
            'City': 'Jakarta',
            'Age': 29,
            'Balance Q1': [123.45],  # This is a list - the problem!
            'Balance Q2': {'value': 0.0},  # This is a dict - also problematic!
            'BadField': '[1, 2, 3]',  # String representation of list
            'NullField': None,
            'NormalField': 100
        },
        {
            'Customer ID': 15565702,
            'City': 'Bandung', 
            'Age': 35,
            'Balance Q1': [456.78, 999.99],  # List with multiple values
            'Balance Q2': {'value': 100.0},
            'BadField': '{"key": "value"}',  # String representation of dict
            'NullField': float('nan'),
            'NormalField': 200
        }
    ]
    
    print("Original problematic data:")
    print(json.dumps(test_data, indent=2, default=str))
    
    # Apply the cleaning logic from our fix
    cleaned_data = []
    for i, row in enumerate(test_data):
        cleaned_row = {}
        for key, value in row.items():
            try:
                if value is None:
                    cleaned_row[key] = None
                elif isinstance(value, (list, tuple)):
                    # Handle list/array values
                    if len(value) > 0:
                        first_val = value[0]
                        if isinstance(first_val, (int, float)):
                            cleaned_row[key] = float(first_val)
                        else:
                            cleaned_row[key] = str(first_val)
                    else:
                        cleaned_row[key] = None
                    print(f"Converted list at row {i}, column '{key}': {value} -> {cleaned_row[key]}")
                elif isinstance(value, dict):
                    # Convert dict to string
                    cleaned_row[key] = str(value)
                    print(f"Converted dict at row {i}, column '{key}': {value} -> {cleaned_row[key]}")
                elif isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                    # Handle string representation of complex types
                    try:
                        import ast
                        parsed = ast.literal_eval(value)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            cleaned_row[key] = parsed[0] if isinstance(parsed[0], (int, float)) else str(parsed[0])
                        else:
                            cleaned_row[key] = value
                        print(f"Converted complex string at row {i}, column '{key}': {value} -> {cleaned_row[key]}")
                    except:
                        cleaned_row[key] = value
                else:
                    cleaned_row[key] = value
            except Exception as e:
                print(f"Error processing value at row {i}, column '{key}': {e}")
                cleaned_row[key] = None
        cleaned_data.append(cleaned_row)
    
    print("\nCleaned data:")
    print(json.dumps(cleaned_data, indent=2, default=str))
    
    # Verify no complex types remain
    print("\nVerification - checking for complex types:")
    for i, row in enumerate(cleaned_data):
        for key, value in row.items():
            if isinstance(value, (list, tuple, dict)):
                print(f"ERROR: Complex type still exists at row {i}, column '{key}': {type(value)} = {value}")
                return False
    
    print("SUCCESS: No complex types found in cleaned data!")
    return True

if __name__ == "__main__":
    test_complex_data_handling()
