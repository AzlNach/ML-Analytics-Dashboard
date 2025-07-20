import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(df, columns=None):
    """Preprocess data for ML algorithms with improved handling"""
    try:
        if columns:
            # Ensure columns is a list, not a Series or other type
            if not isinstance(columns, list):
                columns = list(columns)
            # Filter to only existing columns
            existing_columns = [col for col in columns if col in df.columns]
            if not existing_columns:
                raise ValueError("None of the selected columns exist in the dataframe")
            df = df[existing_columns].copy()
        
        # Handle complex data types (lists, tuples, arrays) before other processing
        for col in df.columns:
            try:
                # Check if column contains complex data types
                has_complex = False
                sample_data = df[col].dropna().head(100)  # Check first 100 non-null values
                
                for val in sample_data:
                    if isinstance(val, (list, tuple, np.ndarray)):
                        has_complex = True
                        break
                
                if has_complex:
                    print(f"Processing complex data in column: {col}")
                    
                    def handle_complex_value(x):
                        try:
                            if pd.isna(x) or x is None:
                                return np.nan
                            elif isinstance(x, (list, tuple, np.ndarray)):
                                if len(x) == 0:
                                    return np.nan
                                # Take first element if it's numeric
                                first_val = x[0]
                                if isinstance(first_val, (int, float, np.integer, np.floating)):
                                    return float(first_val)
                                else:
                                    # Convert to string if not numeric
                                    return str(first_val)
                            else:
                                return x
                        except Exception as e:
                            print(f"Error handling value in {col}: {e}")
                            return np.nan
                    
                    df[col] = df[col].apply(handle_complex_value)
                    print(f"Processed complex data in column: {col}")
            
            except Exception as e:
                print(f"Error checking column {col} for complex data: {e}")
                continue
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Fill missing numeric values with median
        for col in numeric_columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    # If median is NaN (all values are NaN), fill with 0
                    df[col].fillna(0, inplace=True)
                else:
                    df[col].fillna(median_val, inplace=True)
        
        # Fill missing categorical values with mode
        for col in categorical_columns:
            if df[col].isnull().any():
                mode_value = df[col].mode()
                fill_value = mode_value[0] if not mode_value.empty else 'Unknown'
                df[col].fillna(fill_value, inplace=True)
        
        # Ensure all data is numeric for ML algorithms
        cols_to_remove = []
        for col in categorical_columns:
            if col in df.columns:
                try:
                    # Try to convert to numeric first
                    numeric_converted = pd.to_numeric(df[col], errors='coerce')
                    
                    # If most values converted successfully, use numeric version
                    if not numeric_converted.isna().all() and numeric_converted.notna().sum() > len(df) * 0.5:
                        df[col] = numeric_converted.fillna(0)
                        print(f"Converted column {col} to numeric")
                    else:
                        # If still categorical, encode it
                        unique_vals = df[col].dropna().unique()
                        # Only encode if we have reasonable number of unique values
                        if len(unique_vals) <= 100:  # Avoid encoding columns with too many categories
                            le = LabelEncoder()
                            # Handle any remaining complex objects
                            df[col] = df[col].astype(str)
                            df[col] = le.fit_transform(df[col])
                            print(f"Label encoded column {col} with {len(unique_vals)} categories")
                        else:
                            cols_to_remove.append(col)
                            print(f"Removing column {col} - too many categories: {len(unique_vals)}")
                except Exception as e:
                    print(f"Warning: Could not process column {col}: {e}")
                    cols_to_remove.append(col)
        
        # Remove problematic columns
        if cols_to_remove:
            df = df.drop(columns=cols_to_remove)
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # Ensure we have at least some numeric data
        if df.empty or df.select_dtypes(include=[np.number]).empty:
            raise ValueError("No numeric data available after preprocessing")
        
        print(f"Preprocessing completed. Shape: {df.shape}")
        return df
    
    except Exception as e:
        print(f"Error in preprocess_data: {e}")
        raise

def test_decision_tree():
    print("Loading FinanKu dataset...")
    df = pd.read_csv('training_data/FinanKu Data All.csv')
    print(f"Original dataset shape: {df.shape}")
    
    # Test with typical column selection
    target_column = 'City'  # Using City as categorical target like the frontend might
    selected_features = ['Age', 'Avg. Annual Income/Month', 'Balance Q1', 'NumOfProducts Q1']
    
    print(f"Target column: {target_column}")
    print(f"Selected features: {selected_features}")
    
    # Check data types in target column
    print(f"Target column unique values: {df[target_column].unique()}")
    print(f"Target column data type: {df[target_column].dtype}")
    
    # Sample the data like the backend does
    max_rows = 5000
    if len(df) > max_rows:
        df_sample = df.sample(n=max_rows, random_state=42)
    else:
        df_sample = df.copy()
    
    print(f"Working with {len(df_sample)} rows")
    
    # Process features
    feature_cols = [col for col in selected_features if col in df_sample.columns]
    print(f"Valid feature columns: {feature_cols}")
    
    X_df = df_sample[feature_cols].copy()
    print(f"X_df shape before preprocessing: {X_df.shape}")
    print(f"X_df dtypes: {X_df.dtypes}")
    
    # Check for any complex types
    for col in X_df.columns:
        sample_vals = X_df[col].dropna().head(5)
        for i, val in enumerate(sample_vals):
            if isinstance(val, (list, tuple, dict, np.ndarray)):
                print(f"WARNING: Complex type in feature column '{col}' at index {i}: {type(val)} = {val}")
    
    # Preprocess features
    X = preprocess_data(X_df)
    print(f"X shape after preprocessing: {X.shape}")
    print(f"X dtypes after preprocessing: {X.dtypes}")
    
    # Process target
    y = df_sample[target_column].copy()
    print(f"y shape: {y.shape}, dtype: {y.dtype}")
    print(f"y sample values: {y.head().tolist()}")
    
    # Clean target column
    y_cleaned = []
    for i, val in enumerate(y):
        try:
            if isinstance(val, (list, tuple, np.ndarray)):
                clean_val = val[0] if len(val) > 0 else 'Unknown'
                print(f"Converted list/array value at index {i}: {val} -> {clean_val}")
                y_cleaned.append(clean_val)
            elif pd.isna(val) or val is None:
                y_cleaned.append('Unknown')
            else:
                y_cleaned.append(str(val))
        except Exception as e:
            print(f"Error processing target value at index {i}: {val}, error: {e}")
            y_cleaned.append('Unknown')
    
    y = pd.Series(y_cleaned, index=y.index)
    print(f"y after cleaning: unique values = {y.nunique()}, sample = {y.head().tolist()}")
    
    # Check for complex types in y
    for i, val in enumerate(y.head(10)):
        if isinstance(val, (list, tuple, dict, np.ndarray)):
            print(f"ERROR: Still have complex type in target at index {i}: {type(val)} = {val}")
            return
    
    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str))
    y = pd.Series(y_encoded, index=y.index)
    print(f"Target encoded with {len(label_encoder.classes_)} classes")
    
    # Align indices
    common_indices = X.index.intersection(y.index)
    X = X.loc[common_indices]
    y = y.loc[common_indices]
    
    print(f"Final shapes: X={X.shape}, y={y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train/test split: {len(X_train)}/{len(X_test)}")
    
    # Train decision tree
    print("Training decision tree...")
    print(f"X_train dtypes: {X_train.dtypes}")
    print(f"y_train dtype: {y_train.dtype}")
    
    # Check final data before training
    print("Checking final data before training...")
    for col in X_train.columns:
        sample_vals = X_train[col].head(5)
        for i, val in enumerate(sample_vals):
            if isinstance(val, (list, tuple, dict, np.ndarray)):
                print(f"ERROR: Complex type in X_train column '{col}' at index {i}: {type(val)} = {val}")
                return
    
    for i, val in enumerate(y_train.head(5)):
        if isinstance(val, (list, tuple, dict, np.ndarray)):
            print(f"ERROR: Complex type in y_train at index {i}: {type(val)} = {val}")
            return
    
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    
    print("Decision tree trained successfully!")
    print(f"Feature importances: {dict(zip(X.columns, dt.feature_importances_))}")

if __name__ == "__main__":
    test_decision_tree()
