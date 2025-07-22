from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import warnings
warnings.filterwarnings('ignore')

def clean_complex_types(value, row_index=None, column_key=None):
    """
    Clean complex data types and convert lists to tuples for hashability.
    This addresses 'unhashable type: list' errors in ML algorithms.
    """
    try:
        # Handle list values that might cause "unhashable type" errors
        if isinstance(value, list):
            if len(value) > 0:
                # For single values, extract the first element
                if len(value) == 1:
                    first_val = value[0]
                    if isinstance(first_val, (int, float, np.integer, np.floating)):
                        return float(first_val) if np.isfinite(first_val) else 0.0
                    else:
                        return str(first_val)
                else:
                    # For multiple values, convert to tuple (hashable)
                    # For ML purposes, take the first numeric value or convert to string
                    if isinstance(value[0], (int, float, np.integer, np.floating)):
                        return float(value[0]) if np.isfinite(value[0]) else 0.0
                    else:
                        # Convert to tuple for hashability, then to string for ML
                        return str(tuple(value))
            else:
                return None
        elif isinstance(value, tuple):
            # Tuples are already hashable, but extract first value for ML
            if len(value) > 0:
                first_val = value[0]
                if isinstance(first_val, (int, float, np.integer, np.floating)):
                    return float(first_val) if np.isfinite(first_val) else 0.0
                else:
                    return str(first_val)
            else:
                return None
        elif isinstance(value, np.ndarray):
            # Convert numpy arrays - extract first value
            if len(value) > 0:
                first_val = value[0]
                if isinstance(first_val, (int, float, np.integer, np.floating)):
                    return float(first_val) if np.isfinite(first_val) else 0.0
                else:
                    return str(first_val)
            else:
                return None
        elif isinstance(value, dict):
            # Convert dict to string representation (hashable)
            return str(value)
        elif isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
            # Handle string representations of complex types
            try:
                import ast
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    # Convert to tuple for hashability
                    if len(parsed) > 0 and isinstance(parsed[0], (int, float)):
                        return float(parsed[0])
                    else:
                        return str(tuple(parsed))
                elif isinstance(parsed, dict):
                    return str(parsed)
                else:
                    return parsed
            except:
                # If parsing fails, keep as string
                return value
        else:
            # Handle regular values
            if value is None or (isinstance(value, float) and (np.isnan(value) or not np.isfinite(value))):
                return None
            else:
                return value
    except Exception as e:
        if row_index is not None and column_key is not None:
            print(f"Error processing value at row {row_index}, column '{column_key}': {e}")
        return None

# Custom JSON encoder to handle numpy types and NaN values
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

def make_hashable_key(value):
    """
    Convert any value to a hashable key for use in dictionaries.
    This specifically addresses 'unhashable type: list' errors.
    """
    if isinstance(value, list):
        # Convert list to tuple for hashability
        return tuple(make_hashable_key(item) for item in value)
    elif isinstance(value, dict):
        # Convert dict to sorted tuple of key-value pairs
        return tuple(sorted((k, make_hashable_key(v)) for k, v in value.items()))
    elif isinstance(value, set):
        # Convert set to sorted tuple
        return tuple(sorted(make_hashable_key(item) for item in value))
    elif isinstance(value, np.ndarray):
        # Convert numpy array to tuple
        return tuple(value.flatten().tolist())
    else:
        # Value is already hashable or None
        return value

def safe_json_response(data):
    """Create a safe JSON response handling numpy types"""
    return app.response_class(
        response=json.dumps(data, cls=NumpyEncoder, ensure_ascii=False),
        status=200,
        mimetype='application/json'
    )

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:3001', 'http://127.0.0.1:3001'])

# Global variables for models
scaler = StandardScaler()
trained_models = {}
model_history = []

# Ensure models directory exists
MODELS_DIR = 'models'
TRAINING_DATA_DIR = '../training_data'
os.makedirs(MODELS_DIR, exist_ok=True)

def load_training_data():
    """Load and prepare training data for better ML models"""
    training_files = []
    
    if os.path.exists(TRAINING_DATA_DIR):
        for file in os.listdir(TRAINING_DATA_DIR):
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(TRAINING_DATA_DIR, file))
                    training_files.append({
                        'filename': file,
                        'shape': df.shape,
                        'columns': list(df.columns),
                        'data_types': df.dtypes.astype(str).to_dict(),
                        'missing_values': df.isnull().sum().to_dict(),
                        'file_size_mb': round(os.path.getsize(os.path.join(TRAINING_DATA_DIR, file)) / 1024 / 1024, 2)
                    })
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")
                    continue
    
    return training_files

def get_model_algorithms():
    """Get available ML algorithms"""
    return {
        'decision_tree': {
            'name': 'Decision Tree',
            'description': 'Tree-based classification with high interpretability',
            'best_for': 'Classification with clear decision rules'
        },
        'random_forest': {
            'name': 'Random Forest',
            'description': 'Ensemble method with multiple decision trees',
            'best_for': 'High accuracy classification tasks'
        },
        'logistic_regression': {
            'name': 'Logistic Regression',
            'description': 'Linear model for binary/multiclass classification',
            'best_for': 'Linear relationships and probability estimation'
        },
        'svm': {
            'name': 'Support Vector Machine',
            'description': 'Powerful classification using support vectors',
            'best_for': 'High-dimensional data and complex boundaries'
        }
    }

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

def create_model_by_type(model_type, **params):
    """Create ML model based on type with optimized parameters"""
    if model_type == 'decision_tree':
        return DecisionTreeClassifier(
            max_depth=params.get('max_depth', 10),
            min_samples_split=params.get('min_samples_split', 5),
            min_samples_leaf=params.get('min_samples_leaf', 2),
            random_state=42
        )
    elif model_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            min_samples_split=params.get('min_samples_split', 5),
            random_state=42
        )
    elif model_type == 'logistic_regression':
        return LogisticRegression(
            max_iter=params.get('max_iter', 1000),
            random_state=42
        )
    elif model_type == 'svm':
        return SVC(
            kernel=params.get('kernel', 'rbf'),
            probability=True,
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def evaluate_model_performance(model, X_test, y_test, model_type):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get probabilities if available
    prediction_proba = None
    if hasattr(model, 'predict_proba'):
        prediction_proba = model.predict_proba(X_test)
    
    # Classification report
    try:
        class_report = classification_report(y_test, y_pred, output_dict=True)
    except:
        class_report = None
    
    # Feature importance (if available)
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
    
    return {
        'accuracy': accuracy,
        'classification_report': class_report,
        'feature_importance': feature_importance.tolist() if feature_importance is not None else None,
        'prediction_probabilities': prediction_proba.tolist() if prediction_proba is not None else None
    }

@app.route('/api/algorithms', methods=['GET'])
def get_algorithms():
    """Get available ML algorithms with details"""
    return safe_json_response(get_model_algorithms())

@app.route('/api/model-history', methods=['GET'])
def get_model_history():
    """Get training history and model performance"""
    return safe_json_response({
        'history': convert_numpy_types(model_history),
        'total_models': len(model_history),
        'active_models': len(trained_models)
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint"""
    print("Loading training data...")
    training_data = load_training_data()
    print(f"Training data loaded: {len(training_data)} files")
    algorithms = get_model_algorithms()
    
    response_data = {
        'status': 'healthy',
        'message': 'ML Analytics API is running',
        'version': '2.0.0',
        'timestamp': datetime.now().isoformat(),
        'training_data': convert_numpy_types(training_data),
        'available_algorithms': algorithms,
        'trained_models_count': len(trained_models),
        'model_history_count': len(model_history)
    }
    print(f"Response training_data length: {len(response_data['training_data'])}")
    
    return safe_json_response(response_data)

def clean_dataframe(df):
    """Clean DataFrame by handling NaN values and converting types properly"""
    print(f"Cleaning DataFrame with shape: {df.shape}")
    
    # First, handle any complex data types that might have come from JSON
    for col in df.columns:
        # Check for lists, tuples, or other complex objects
        if df[col].dtype == 'object':
            # Check if any values are complex types
            sample_values = df[col].dropna().head(100)
            has_complex = any(isinstance(val, (list, tuple, dict)) for val in sample_values)
            
            if has_complex:
                print(f"Found complex data in column '{col}', converting with tuple casting...")
                def clean_value(x):
                    if pd.isna(x) or x is None:
                        return None
                    elif isinstance(x, (list, tuple)):
                        # Use tuple casting for hashability - take first element for ML
                        if len(x) > 0:
                            first_val = x[0]
                            if isinstance(first_val, (int, float, np.integer, np.floating)):
                                return float(first_val) if np.isfinite(first_val) else 0.0
                            else:
                                return str(first_val)
                        else:
                            return None
                    elif isinstance(x, dict):
                        # Convert dict to string representation (hashable)
                        return str(x)
                    else:
                        return x
                
                df[col] = df[col].apply(clean_value)
                print(f"Cleaned column '{col}' with tuple-based conversion")
    
    # Replace NaN values with None for JSON serialization
    df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    
    # Convert numpy types to native Python types
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('Int64')  # Nullable integer
        elif df[col].dtype == 'float64':
            # Keep as float64 but handle NaN properly
            pass
    
    print(f"DataFrame cleaning completed with shape: {df.shape}")
    return df

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif pd.isna(obj):
        return None
    return obj

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """Analyze uploaded CSV data"""
    try:
        data = request.json
        
        if not data or 'csv_data' not in data:
            return jsonify({'error': 'No CSV data provided'}), 400
        
        csv_data = data['csv_data']
        
        # Clean data for complex types before DataFrame creation with tuple casting
        if csv_data:
            cleaned_csv_data = []
            for i, row in enumerate(csv_data):
                cleaned_row = {}
                for key, value in row.items():
                    cleaned_row[key] = clean_complex_types(value, i, key)
                cleaned_csv_data.append(cleaned_row)
            df = pd.DataFrame(cleaned_csv_data)
        else:
            df = pd.DataFrame(csv_data)
        
        df = clean_dataframe(df)
        
        # Basic statistics
        stats = {}
        for column in df.columns:
            # Check for numeric columns (including nullable Int64)
            if df[column].dtype in ['int64', 'float64', 'Int64']:
                # Handle NaN values in calculations
                col_data = df[column].dropna()
                if len(col_data) > 0:
                    stats[column] = {
                        'type': 'numeric',
                        'mean': float(col_data.mean()) if len(col_data) > 0 else None,
                        'std': float(col_data.std()) if len(col_data) > 1 else None,
                        'min': float(col_data.min()) if len(col_data) > 0 else None,
                        'max': float(col_data.max()) if len(col_data) > 0 else None,
                        'median': float(col_data.median()) if len(col_data) > 0 else None,
                        'count': int(df[column].count())
                    }
                else:
                    stats[column] = {
                        'type': 'numeric',
                        'mean': None,
                        'std': None,
                        'min': None,
                        'max': None,
                        'median': None,
                        'count': 0
                    }
            else:
                value_counts = df[column].value_counts()
                most_common_dict = {}
                for k, v in value_counts.head(5).items():
                    # Convert numpy types to native Python types
                    if isinstance(k, (np.integer, np.floating)):
                        k = k.item()
                    if isinstance(v, (np.integer, np.floating)):
                        v = v.item()
                    most_common_dict[str(k)] = int(v)
                
                stats[column] = {
                    'type': 'categorical',
                    'unique': int(df[column].nunique()),
                    'most_common': most_common_dict,
                    'count': int(df[column].count())
                }
        
        # Correlation matrix for numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = {}
        if not numeric_df.empty:
            corr_df = numeric_df.corr()
            # Convert to dict with proper handling of NaN values
            for col1 in corr_df.columns:
                correlation_matrix[col1] = {}
                for col2 in corr_df.columns:
                    corr_val = corr_df.loc[col1, col2]
                    correlation_matrix[col1][col2] = float(corr_val) if not pd.isna(corr_val) else None
        
        return safe_json_response({
            'stats': convert_numpy_types(stats),
            'correlation_matrix': convert_numpy_types(correlation_matrix),
            'shape': [int(df.shape[0]), int(df.shape[1])],
            'columns': list(df.columns),
            'data_types': convert_numpy_types(df.dtypes.astype(str).to_dict())
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clustering', methods=['POST'])
def perform_clustering():
    """Perform DBSCAN clustering with improved data handling"""
    try:
        data = request.json
        csv_data = data['csv_data']
        
        # Clean data for complex types before DataFrame creation with tuple casting
        if csv_data:
            cleaned_csv_data = []
            for i, row in enumerate(csv_data):
                cleaned_row = {}
                for key, value in row.items():
                    cleaned_row[key] = clean_complex_types(value, i, key)
                cleaned_csv_data.append(cleaned_row)
            df = pd.DataFrame(cleaned_csv_data)
        else:
            df = pd.DataFrame(csv_data)
            
        df = clean_dataframe(df)
        selected_columns = data.get('selected_columns', [])
        eps = data.get('eps', 0.5)
        min_samples = data.get('min_samples', 5)
        
        # Ensure selected_columns is a list
        if not isinstance(selected_columns, list):
            selected_columns = list(selected_columns) if selected_columns else []
        
        if not selected_columns:
            selected_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Preprocess data
        df_processed = preprocess_data(df, selected_columns)
        
        # Handle large datasets by sampling if necessary
        max_rows = 10000  # Limit for performance
        if len(df_processed) > max_rows:
            sample_size = min(max_rows, len(df_processed))
            df_sample = df_processed.sample(n=sample_size, random_state=42)
            sample_indices = df_sample.index
        else:
            df_sample = df_processed
            sample_indices = df_processed.index
        
        # Scale the data
        scaler_local = StandardScaler()
        scaled_data = scaler_local.fit_transform(df_sample)
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(scaled_data)
        
        # Create result data
        result_data = df.loc[sample_indices].copy()
        result_data['cluster'] = clusters
        
        # Calculate cluster statistics
        cluster_counts = pd.Series(clusters).value_counts()
        cluster_sizes = {}
        for k, v in cluster_counts.items():
            cluster_sizes[str(k)] = int(v)  # Convert to string key to avoid JSON issues
        
        cluster_stats = {
            'num_clusters': int(len(set(clusters)) - (1 if -1 in clusters else 0)),
            'num_noise_points': int(sum(1 for x in clusters if x == -1)),
            'cluster_sizes': cluster_sizes,
            'total_points': len(clusters),
            'sample_size': len(df_sample) if len(df) > max_rows else len(df)
        }
        
        # Clean the result data for JSON serialization
        result_data = clean_dataframe(result_data)
        
        return safe_json_response({
            'clustered_data': convert_numpy_types(result_data.to_dict('records')),
            'cluster_stats': convert_numpy_types(cluster_stats),
            'parameters': {'eps': eps, 'min_samples': min_samples},
            'info': {
                'original_size': len(df),
                'processed_size': len(df_sample),
                'was_sampled': len(df) > max_rows
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/anomaly-detection', methods=['POST'])
def detect_anomalies():
    """Perform anomaly detection using Isolation Forest with improved data handling"""
    try:
        data = request.json
        csv_data = data['csv_data']
        
        # Clean data for complex types before DataFrame creation with tuple casting
        if csv_data:
            cleaned_csv_data = []
            for i, row in enumerate(csv_data):
                cleaned_row = {}
                for key, value in row.items():
                    cleaned_row[key] = clean_complex_types(value, i, key)
                cleaned_csv_data.append(cleaned_row)
            df = pd.DataFrame(cleaned_csv_data)
        else:
            df = pd.DataFrame(csv_data)
            
        df = clean_dataframe(df)
        selected_columns = data.get('selected_columns', [])
        contamination = data.get('contamination', 0.1)
        
        # Ensure selected_columns is a list
        if not isinstance(selected_columns, list):
            selected_columns = list(selected_columns) if selected_columns else []
        
        if not selected_columns:
            selected_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Preprocess data
        df_processed = preprocess_data(df, selected_columns)
        
        # Handle large datasets by sampling if necessary
        max_rows = 10000  # Limit for performance
        if len(df_processed) > max_rows:
            sample_size = min(max_rows, len(df_processed))
            df_sample = df_processed.sample(n=sample_size, random_state=42)
            sample_indices = df_sample.index
        else:
            df_sample = df_processed
            sample_indices = df_processed.index
        
        # Scale the data
        scaler_local = StandardScaler()
        scaled_data = scaler_local.fit_transform(df_sample)
        
        # Perform anomaly detection
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(scaled_data)
        anomaly_scores = iso_forest.decision_function(scaled_data)
        
        # Add results to original data
        result_data = df.loc[sample_indices].copy()
        result_data['is_anomaly'] = (anomaly_labels == -1)
        result_data['anomaly_score'] = anomaly_scores
        
        # Clean the result data for JSON serialization
        result_data = clean_dataframe(result_data)
        
        # Calculate statistics
        anomaly_stats = {
            'total_anomalies': int(sum(anomaly_labels == -1)),
            'anomaly_percentage': float((sum(anomaly_labels == -1) / len(anomaly_labels)) * 100),
            'contamination_used': float(contamination),
            'total_points': len(anomaly_labels),
            'sample_size': len(df_sample) if len(df) > max_rows else len(df)
        }
        
        return safe_json_response({
            'data_with_anomalies': convert_numpy_types(result_data.to_dict('records')),
            'anomaly_stats': convert_numpy_types(anomaly_stats),
            'info': {
                'original_size': len(df),
                'processed_size': len(df_sample),
                'was_sampled': len(df) > max_rows
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/decision-tree', methods=['POST'])
def build_decision_tree():
    """Build and train decision tree model with improved data handling"""
    try:
        data = request.json
        print(f"Raw input data keys: {list(data.keys())}")
        print(f"CSV data type: {type(data.get('csv_data'))}")
        print(f"CSV data length: {len(data.get('csv_data', []))}")
        
        # Create DataFrame with additional debugging
        csv_data = data['csv_data']
        if not csv_data:
            return jsonify({'error': 'No CSV data provided'}), 400
        
        # Enhanced preprocessing: Clean problematic data types BEFORE DataFrame creation
        print("Preprocessing input data for complex types with tuple casting...")
        cleaned_csv_data = []
        for i, row in enumerate(csv_data):
            cleaned_row = {}
            for key, value in row.items():
                # Use the comprehensive clean_complex_types function
                cleaned_value = clean_complex_types(value, i, key)
                cleaned_row[key] = cleaned_value
            cleaned_csv_data.append(cleaned_row)
        
        print(f"Preprocessed {len(cleaned_csv_data)} rows with tuple casting for hashability")
        
        # Create DataFrame with cleaned data
        df = pd.DataFrame(cleaned_csv_data)
        print(f"DataFrame created with shape: {df.shape}")
        
        # Additional cleaning with existing function
        df = clean_dataframe(df)
        print(f"DataFrame after cleaning: {df.shape}")
        
        target_column = data.get('target_column')
        selected_features = data.get('selected_features', [])
        max_depth = data.get('max_depth', 5)
        
        # Ensure target_column is a string, not a list
        if isinstance(target_column, list):
            if len(target_column) > 0:
                target_column = str(target_column[0])
            else:
                target_column = None
        elif target_column is not None:
            target_column = str(target_column)
        
        print(f"Decision tree request: target='{target_column}', features={selected_features[:3]}...")
        
        if not target_column or target_column not in df.columns:
            return jsonify({'error': 'Target column not specified or not found'}), 400
        
        # Ensure selected_features is a list and contains only valid column names
        if not isinstance(selected_features, list):
            selected_features = list(selected_features) if selected_features else []
        
        # Ensure all selected features are strings and exist in dataframe
        selected_features = [str(col) for col in selected_features if str(col) in df.columns]
        
        if not selected_features:
            selected_features = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in selected_features:
                selected_features.remove(target_column)
        
        print(f"Processing {len(df)} rows with {len(selected_features)} features")
        
        # Handle large datasets by sampling if necessary
        max_rows = 5000  # Decision trees can handle less data efficiently
        if len(df) > max_rows:
            sample_size = min(max_rows, len(df))
            df_sample = df.sample(n=sample_size, random_state=42)
            print(f"Sampled {len(df_sample)} rows from {len(df)} original rows")
        else:
            df_sample = df
            print(f"Using full dataset of {len(df_sample)} rows")
        
        # Clean features data - ensure we only have selected features and they exist
        feature_cols = [col for col in selected_features if col in df_sample.columns]
        if not feature_cols:
            return jsonify({'error': 'No valid feature columns found'}), 400
            
        try:
            # Preprocess features - pass only the feature columns
            X_df = df_sample[feature_cols].copy()
            print(f"Feature columns shape before preprocessing: {X_df.shape}")
            
            # Check for complex types in features before preprocessing
            for col in X_df.columns:
                sample_vals = X_df[col].dropna().head(5)
                for i, val in enumerate(sample_vals):
                    if isinstance(val, (list, tuple, dict, np.ndarray)):
                        print(f"WARNING: Complex type in feature column '{col}' at index {i}: {type(val)} = {val}")
            
            X = preprocess_data(X_df)
            print(f"Feature columns shape after preprocessing: {X.shape}")
            
            # Final validation - ensure no complex types remain
            for col in X.columns:
                sample_vals = X[col].dropna().head(5)
                for i, val in enumerate(sample_vals):
                    if isinstance(val, (list, tuple, dict, np.ndarray)):
                        print(f"ERROR: Complex type still exists in processed feature column '{col}' at index {i}: {type(val)} = {val}")
                        return jsonify({'error': f'Unable to process feature column {col} - complex data type remains: {type(val)}'}), 400
                        
        except Exception as e:
            print(f"Feature preprocessing error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Feature preprocessing failed: {str(e)}'}), 400
        
        # Handle target column
        if target_column not in df_sample.columns:
            return jsonify({'error': f'Target column "{target_column}" not found in data'}), 400
            
        y = df_sample[target_column].copy()
        print(f"Target column shape: {y.shape}, dtype: {y.dtype}")
        print(f"Target column sample values: {y.head().tolist()}")
        
        # Clean target column - handle any lists or complex objects
        y_cleaned = []
        for i, val in enumerate(y):
            try:
                if isinstance(val, (list, tuple, np.ndarray)):
                    # If value is a list/array, take the first element
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
        print(f"Target column after cleaning: unique values = {y.nunique()}, sample = {y.head().tolist()}")
        
        # Validate that we don't have any complex types in y
        for i, val in enumerate(y.head(10)):
            if isinstance(val, (list, tuple, dict, np.ndarray)):
                print(f"ERROR: Still have complex type in target at index {i}: {type(val)} = {val}")
                return jsonify({'error': f'Unable to clean target column - complex data type remains: {type(val)}'}), 400
        
        # Handle categorical target
        label_encoder = None
        if y.dtype == 'object' or y.nunique() < len(y) * 0.5:  # If categorical
            label_encoder = LabelEncoder()
            try:
                y_encoded = label_encoder.fit_transform(y.astype(str))
                y = pd.Series(y_encoded, index=y.index)
                print(f"Target encoded with {len(label_encoder.classes_)} classes")
            except Exception as e:
                print(f"Label encoding error: {e}")
                return jsonify({'error': f'Target encoding failed: {str(e)}'}), 400
        
        # Ensure X and y have the same indices
        common_indices = X.index.intersection(y.index)
        X = X.loc[common_indices]
        y = y.loc[common_indices]
        
        if len(X) == 0 or len(y) == 0:
            return jsonify({'error': 'No data remaining after preprocessing'}), 400
        
        print(f"Final data shapes: X={X.shape}, y={y.shape}")
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, 
                stratify=y if y.nunique() > 1 and y.nunique() < len(y) * 0.5 else None
            )
            print(f"Train/test split: {len(X_train)}/{len(X_test)}")
        except Exception as e:
            print(f"Train/test split error: {e}")
            return jsonify({'error': f'Data splitting failed: {str(e)}'}), 400
        
        # Train decision tree
        try:
            print("Starting decision tree training...")
            dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            
            # Additional validation before training
            print(f"Training data types - X: {X.dtypes.to_dict()}")
            print(f"Training data types - y: {y.dtype}")
            
            # Check for any remaining problematic values
            if X.isnull().any().any():
                print("WARNING: X contains null values")
                X = X.fillna(0)
            
            if y.isnull().any():
                print("WARNING: y contains null values")
                y = y.fillna('Unknown')
            
            dt.fit(X, y)
            print("Decision tree trained successfully")
        except TypeError as e:
            if "unhashable type" in str(e):
                print(f"Unhashable type error: {e}")
                print(f"X data types: {X.dtypes}")
                print(f"y data type: {y.dtype}")
                print(f"Sample X values: {X.head(2).to_dict()}")
                print(f"Sample y values: {y.head(5).tolist()}")
                return jsonify({'error': 'Data contains unhashable types (lists/arrays). Please check your CSV data for malformed cells.'}), 400
            else:
                raise e
        except Exception as e:
            print(f"Decision tree training error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Model training failed: {str(e)}'}), 400
        
        # Make predictions
        try:
            y_pred = dt.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 400
        
        # Get tree structure as text (limit depth for readability)
        try:
            tree_rules = export_text(dt, feature_names=list(X.columns), max_depth=3)
        except Exception as e:
            print(f"Tree export error: {e}")
            tree_rules = "Tree structure export failed"
        
        # Feature importance
        try:
            feature_importance = {}
            for feature, importance in zip(X.columns, dt.feature_importances_):
                feature_importance[str(feature)] = float(importance)
        except Exception as e:
            print(f"Feature importance error: {e}")
            feature_importance = {}
        
        # Store model for future use
        model_id = f"dt_{len(trained_models)}_{int(datetime.now().timestamp())}"
        trained_models[model_id] = {
            'model': dt,
            'label_encoder': label_encoder,
            'features': list(X.columns),
            'model_type': 'decision_tree',
            'target_column': target_column
        }
        
        return safe_json_response({
            'accuracy': float(accuracy),
            'tree_rules': tree_rules,
            'feature_importance': convert_numpy_types(feature_importance),
            'model_id': model_id,
            'test_size': len(X_test),
            'train_size': len(X_train),
            'info': {
                'original_size': len(df),
                'processed_size': len(df_sample),
                'was_sampled': len(df) > max_rows
            }
        })
        
    except Exception as e:
        print(f"Decision tree endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/training-data', methods=['GET'])
def get_training_data():
    """Get list of available training datasets"""
    return safe_json_response(convert_numpy_types(load_training_data()))

@app.route('/api/train-from-data', methods=['POST'])
def train_from_data():
    """Train model using data sent from frontend (cleaned data)"""
    try:
        data = request.json
        csv_data = data.get('csv_data')
        target_column = data.get('target_column')
        model_type = data.get('model_type', 'decision_tree')
        test_size = data.get('test_size', 0.2)
        cross_validation = data.get('cross_validation', True)
        dataset_name = data.get('dataset_name', 'uploaded_data')
        
        if not csv_data:
            return jsonify({'error': 'CSV data not provided'}), 400
        
        if not target_column:
            return jsonify({'error': 'Target column not provided'}), 400
            
        # Convert CSV data to DataFrame
        df = pd.DataFrame(csv_data)
        
        if target_column not in df.columns:
            return jsonify({'error': f'Target column "{target_column}" not found in data'}), 400
        
        # Clean the data
        df = clean_dataframe(df)
        
        # Prepare features (exclude target column)
        all_columns = list(df.columns)
        all_columns.remove(target_column)
        
        # Select only numeric columns for features
        numeric_columns = df[all_columns].select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = numeric_columns
        
        if not feature_columns:
            return jsonify({'error': 'No numeric features found for training'}), 400
        
        X = preprocess_data(df[feature_columns])
        y = df[target_column]
        
        # Encode target if categorical
        label_encoder = None
        original_classes = None
        if y.dtype == 'object' or y.dtype.name == 'category':
            label_encoder = LabelEncoder()
            original_classes = y.unique().tolist()
            y = label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Create and train model
        model = create_model_by_type(model_type)
        model.fit(X_train, y_train)
        
        # Evaluate model
        performance = evaluate_model_performance(model, X_test, y_test, model_type)
        
        # Cross-validation if requested
        cv_scores = None
        if cross_validation and len(X) > 50:
            cv_scores = cross_val_score(model, X, y, cv=5)
        
        # Generate unique model ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{model_type}_{dataset_name.replace('.csv', '').replace(' ', '_')}_{timestamp}"
        
        # Generate predictions for the training data
        y_pred_all = model.predict(X)
        
        # Convert predictions back to original classes if label encoder was used
        if label_encoder:
            y_pred_all = label_encoder.inverse_transform(y_pred_all)
            y_original = label_encoder.inverse_transform(y)
        else:
            y_original = y
        
        # Store model information
        model_info = {
            'model': model,
            'label_encoder': label_encoder,
            'features': feature_columns,
            'training_file': dataset_name,
            'model_type': model_type,
            'target_column': target_column,
            'original_classes': original_classes,
            'created_at': datetime.now().isoformat(),
            'performance': performance
        }
        
        trained_models[model_id] = model_info
        
        # Save model to disk
        model_path = os.path.join(MODELS_DIR, f'{model_id}.pkl')
        joblib.dump({
            'model': model,
            'label_encoder': label_encoder,
            'features': feature_columns,
            'metadata': {
                'model_type': model_type,
                'target_column': target_column,
                'original_classes': original_classes,
                'training_file': dataset_name,
                'created_at': datetime.now().isoformat()
            }
        }, model_path)
        
        # Add to history
        history_entry = {
            'model_id': model_id,
            'model_type': model_type,
            'training_file': dataset_name,
            'target_column': target_column,
            'accuracy': performance['accuracy'],
            'cv_mean_score': float(cv_scores.mean()) if cv_scores is not None else None,
            'cv_std_score': float(cv_scores.std()) if cv_scores is not None else None,
            'feature_count': len(feature_columns),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'created_at': datetime.now().isoformat()
        }
        model_history.append(history_entry)
        
        return safe_json_response({
            'model_id': model_id,
            'accuracy': convert_numpy_types(performance['accuracy']),
            'cv_scores': convert_numpy_types(cv_scores.tolist() if cv_scores is not None else None),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': feature_columns,
            'feature_importance': convert_numpy_types(performance['feature_importance']),
            'classification_report': convert_numpy_types(performance['classification_report']),
            'model_type': model_type,
            'target_column': target_column,
            'original_classes': convert_numpy_types(original_classes),
            'predictions': convert_numpy_types(y_pred_all.tolist()),
            'created_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Training from data error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

@app.route('/api/train-from-file', methods=['POST'])
def train_from_file():
    """Enhanced model training using data from training_data folder"""
    try:
        data = request.json
        filename = data.get('filename')
        target_column = data.get('target_column')
        model_type = data.get('model_type', 'decision_tree')
        test_size = data.get('test_size', 0.2)
        cross_validation = data.get('cross_validation', True)
        
        if not filename:
            return jsonify({'error': 'Filename not provided'}), 400
        
        # Load training data
        filepath = os.path.join(TRAINING_DATA_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Training file not found'}), 404
        
        df = pd.read_csv(filepath)
        
        if target_column not in df.columns:
            return jsonify({'error': f'Target column "{target_column}" not found in training data'}), 400
        
        # Prepare features (exclude target column)
        all_columns = list(df.columns)
        all_columns.remove(target_column)
        
        # Select only numeric columns for features (can be enhanced later for categorical)
        numeric_columns = df[all_columns].select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = numeric_columns
        
        if not feature_columns:
            return jsonify({'error': 'No numeric features found for training'}), 400
        
        X = preprocess_data(df[feature_columns])
        y = df[target_column]
        
        # Encode target if categorical
        label_encoder = None
        original_classes = None
        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            original_classes = y.unique().tolist()
            y = label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Create and train model
        model = create_model_by_type(model_type)
        model.fit(X_train, y_train)
        
        # Evaluate model
        performance = evaluate_model_performance(model, X_test, y_test, model_type)
        
        # Cross-validation if requested
        cv_scores = None
        if cross_validation and len(X) > 50:  # Only if enough data
            cv_scores = cross_val_score(model, X, y, cv=5)
        
        # Generate unique model ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{model_type}_{filename.replace('.csv', '')}_{timestamp}"
        
        # Store model information
        model_info = {
            'model': model,
            'label_encoder': label_encoder,
            'features': feature_columns,
            'training_file': filename,
            'model_type': model_type,
            'target_column': target_column,
            'original_classes': original_classes,
            'created_at': datetime.now().isoformat(),
            'performance': performance
        }
        
        trained_models[model_id] = model_info
        
        # Save model to disk
        model_path = os.path.join(MODELS_DIR, f'{model_id}.pkl')
        joblib.dump({
            'model': model,
            'label_encoder': label_encoder,
            'features': feature_columns,
            'metadata': {
                'model_type': model_type,
                'target_column': target_column,
                'original_classes': original_classes,
                'training_file': filename,
                'created_at': datetime.now().isoformat()
            }
        }, model_path)
        
        # Add to history
        history_entry = {
            'model_id': model_id,
            'model_type': model_type,
            'training_file': filename,
            'target_column': target_column,
            'accuracy': performance['accuracy'],
            'cv_mean_score': float(cv_scores.mean()) if cv_scores is not None else None,
            'cv_std_score': float(cv_scores.std()) if cv_scores is not None else None,
            'feature_count': len(feature_columns),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'created_at': datetime.now().isoformat()
        }
        model_history.append(history_entry)
        
        return safe_json_response({
            'model_id': model_id,
            'accuracy': convert_numpy_types(performance['accuracy']),
            'cv_scores': convert_numpy_types(cv_scores.tolist() if cv_scores is not None else None),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': feature_columns,
            'feature_importance': convert_numpy_types(performance['feature_importance']),
            'classification_report': convert_numpy_types(performance['classification_report']),
            'model_type': model_type,
            'target_column': target_column,
            'original_classes': convert_numpy_types(original_classes),
            'created_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

@app.route('/api/models', methods=['GET'])
def get_trained_models():
    """Get list of all trained models"""
    models_info = []
    for model_id, model_info in trained_models.items():
        models_info.append({
            'model_id': model_id,
            'model_type': model_info.get('model_type', 'unknown'),
            'training_file': model_info.get('training_file'),
            'target_column': model_info.get('target_column'),
            'features': model_info.get('features', []),
            'accuracy': model_info.get('performance', {}).get('accuracy'),
            'created_at': model_info.get('created_at'),
            'feature_count': len(model_info.get('features', []))
        })
    
    return safe_json_response({
        'models': convert_numpy_types(models_info),
        'total_count': len(models_info)
    })

@app.route('/api/models/<model_id>', methods=['GET'])
def get_model_details(model_id):
    """Get detailed information about a specific model"""
    if model_id not in trained_models:
        return jsonify({'error': 'Model not found'}), 404
    
    model_info = trained_models[model_id]
    
    # Don't include the actual model object in the response
    response_data = {
        'model_id': model_id,
        'model_type': model_info.get('model_type'),
        'training_file': model_info.get('training_file'),
        'target_column': model_info.get('target_column'),
        'features': model_info.get('features', []),
        'original_classes': model_info.get('original_classes'),
        'created_at': model_info.get('created_at'),
        'performance': model_info.get('performance', {})
    }
    
    return safe_json_response(convert_numpy_types(response_data))

@app.route('/api/models/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a trained model"""
    if model_id not in trained_models:
        return jsonify({'error': 'Model not found'}), 404
    
    try:
        # Remove from memory
        del trained_models[model_id]
        
        # Remove from disk
        model_path = os.path.join(MODELS_DIR, f'{model_id}.pkl')
        if os.path.exists(model_path):
            os.remove(model_path)
        
        return jsonify({'message': f'Model {model_id} deleted successfully'})
    
    except Exception as e:
        return jsonify({'error': f'Failed to delete model: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def make_prediction():
    """Enhanced prediction endpoint with detailed results"""
    try:
        data = request.json
        model_id = data.get('model_id')
        input_data = data.get('input_data')
        
        if not model_id or not input_data:
            return jsonify({'error': 'Model ID and input data are required'}), 400
        
        if model_id not in trained_models:
            return jsonify({'error': f'Model {model_id} not found'}), 404
        
        model_info = trained_models[model_id]
        model = model_info['model']
        features = model_info['features']
        label_encoder = model_info.get('label_encoder')
        
        # Validate input features
        missing_features = [f for f in features if f not in input_data]
        if missing_features:
            return jsonify({
                'error': f'Missing required features: {missing_features}',
                'required_features': features
            }), 400
        
        # Prepare input data
        input_df = pd.DataFrame([input_data])
        input_processed = preprocess_data(input_df[features])
        
        # Make prediction
        prediction = model.predict(input_processed)[0]
        
        # Get prediction probabilities if available
        prediction_proba = None
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_processed)[0]
            prediction_proba = proba.tolist()
            confidence = float(max(proba))
        
        # Decode prediction if label encoder was used
        original_prediction = prediction
        if label_encoder:
            prediction = label_encoder.inverse_transform([prediction])[0]
        
        result = {
            'prediction': prediction,
            'prediction_encoded': int(original_prediction) if label_encoder else None,
            'prediction_probability': prediction_proba,
            'confidence': confidence,
            'model_id': model_id,
            'model_type': model_info.get('model_type'),
            'features_used': features,
            'input_data': input_data
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/data-quality-report', methods=['POST'])
def generate_data_quality_report():
    """Generate comprehensive data quality report"""
    try:
        data = request.json
        
        if not data or 'csv_data' not in data:
            return jsonify({'error': 'No CSV data provided'}), 400
        
        csv_data = data['csv_data']
        
        # Clean data and create DataFrame
        cleaned_csv_data = []
        for i, row in enumerate(csv_data):
            cleaned_row = {}
            for key, value in row.items():
                cleaned_row[key] = clean_complex_types(value, i, key)
            cleaned_csv_data.append(cleaned_row)
        
        df = pd.DataFrame(cleaned_csv_data)
        df = clean_dataframe(df)
        
        # Generate quality report
        report = {
            'totalRows': len(df),
            'totalColumns': len(df.columns),
            'missingValues': {},
            'duplicates': 0,
            'outliers': {},
            'dataTypes': {},
            'memoryUsage': df.memory_usage(deep=True).sum(),
            'summary': {}
        }
        
        # Count missing values per column
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            report['missingValues'][col] = int(missing_count)
            report['dataTypes'][col] = str(df[col].dtype)
        
        # Count duplicate rows
        report['duplicates'] = int(df.duplicated().sum())
        
        # Detect outliers for numeric columns using IQR method
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64', 'Int64']:
                values = df[col].dropna()
                if len(values) > 0:
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = values[(values < lower_bound) | (values > upper_bound)]
                    if len(outliers) > 0:
                        report['outliers'][col] = int(len(outliers))
        
        # Generate summary statistics
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64', 'Int64']:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    report['summary'][col] = {
                        'type': 'numeric',
                        'count': int(len(col_data)),
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'unique': int(col_data.nunique())
                    }
            else:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    report['summary'][col] = {
                        'type': 'categorical',
                        'count': int(len(col_data)),
                        'unique': int(col_data.nunique()),
                        'top': str(col_data.mode().iloc[0]) if len(col_data.mode()) > 0 else 'N/A',
                        'freq': int(col_data.value_counts().iloc[0]) if len(col_data.value_counts()) > 0 else 0
                    }
        
        return safe_json_response(convert_numpy_types(report))
        
    except Exception as e:
        print(f"Data quality report error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Quality report generation failed: {str(e)}'}), 500

@app.route('/api/clean-data', methods=['POST'])
def clean_data():
    """Clean data based on user preferences"""
    try:
        data = request.json
        
        if not data or 'csv_data' not in data:
            return jsonify({'error': 'No CSV data provided'}), 400
        
        csv_data = data['csv_data']
        cleaning_options = data.get('cleaning_options', {})
        
        # Clean data and create DataFrame
        cleaned_csv_data = []
        for i, row in enumerate(csv_data):
            cleaned_row = {}
            for key, value in row.items():
                cleaned_row[key] = clean_complex_types(value, i, key)
            cleaned_csv_data.append(cleaned_row)
        
        df = pd.DataFrame(cleaned_csv_data)
        df = clean_dataframe(df)
        
        # Apply cleaning operations based on user choices
        original_size = len(df)
        
        # Handle missing values
        missing_strategy = cleaning_options.get('missingValues', 'fill_mean')
        if missing_strategy == 'remove_rows':
            df = df.dropna()
        elif missing_strategy == 'fill_mean':
            # Fill numeric columns with mean
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mean(), inplace=True)
            
            # Fill categorical columns with mode
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if df[col].isnull().any():
                    mode_value = df[col].mode()
                    fill_value = mode_value[0] if not mode_value.empty else 'Unknown'
                    df[col].fillna(fill_value, inplace=True)
        elif missing_strategy == 'fill_mode':
            # Fill all columns with mode
            for col in df.columns:
                if df[col].isnull().any():
                    mode_value = df[col].mode()
                    fill_value = mode_value[0] if not mode_value.empty else 'Unknown'
                    df[col].fillna(fill_value, inplace=True)
        
        # Handle duplicates
        duplicates_strategy = cleaning_options.get('duplicates', 'remove')
        if duplicates_strategy == 'remove':
            df = df.drop_duplicates()
        
        # Handle outliers
        outliers_strategy = cleaning_options.get('outliers', 'keep')
        if outliers_strategy == 'remove':
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Remove outliers
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        # Convert cleaned DataFrame back to list of dictionaries
        cleaned_result = df.to_dict('records')
        
        # Convert numpy types to Python native types for JSON serialization
        for row in cleaned_result:
            for key, value in row.items():
                if isinstance(value, (np.integer, np.floating, np.bool_)):
                    row[key] = value.item()
                elif pd.isna(value):
                    row[key] = None
        
        result = {
            'cleaned_data': cleaned_result,
            'original_size': original_size,
            'cleaned_size': len(df),
            'columns': list(df.columns),
            'cleaning_summary': {
                'missing_values_strategy': missing_strategy,
                'duplicates_strategy': duplicates_strategy,
                'outliers_strategy': outliers_strategy,
                'rows_removed': original_size - len(df)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Data cleaning failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
