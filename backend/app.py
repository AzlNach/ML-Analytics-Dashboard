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
import re
import ast

from ydata_profiling import ProfileReport
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

def convert_numpy_types(obj):
    """Convert numpy types to native Python types recursively"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

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
# Use absolute path to training_data directory - check both possible locations
current_dir = os.getcwd()
backend_dir = os.path.dirname(os.path.abspath(__file__))

# First try: training_data relative to current working directory (when run from project root)
training_data_option1 = os.path.join(current_dir, 'training_data')
# Second try: training_data relative to backend parent directory (when run from backend)
training_data_option2 = os.path.join(os.path.dirname(backend_dir), 'training_data')

if os.path.exists(training_data_option1):
    TRAINING_DATA_DIR = training_data_option1
elif os.path.exists(training_data_option2):
    TRAINING_DATA_DIR = training_data_option2
else:
    # Fallback to relative path
    TRAINING_DATA_DIR = 'training_data'

print(f"Using TRAINING_DATA_DIR: {TRAINING_DATA_DIR}")
print(f"TRAINING_DATA_DIR exists: {os.path.exists(TRAINING_DATA_DIR)}")

os.makedirs(MODELS_DIR, exist_ok=True)

def load_training_data():
    """Load and prepare training data for better ML models"""
    print(f"Loading training data from: {TRAINING_DATA_DIR}")
    print(f"Directory exists: {os.path.exists(TRAINING_DATA_DIR)}")
    
    training_files = []
    
    if os.path.exists(TRAINING_DATA_DIR):
        files_in_dir = os.listdir(TRAINING_DATA_DIR)
        print(f"Files in directory: {files_in_dir}")
        
        for file in files_in_dir:
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
                    print(f"Successfully loaded: {file}")
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")
                    continue
    else:
        print(f"Training data directory does not exist: {TRAINING_DATA_DIR}")
    
    print(f"Total training files loaded: {len(training_files)}")
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

def detect_data_types(df):
    """
    Deteksi tipe data otomatis menggunakan YData Profiling
    
    Returns:
        dict: Dictionary berisi informasi lengkap tentang setiap kolom termasuk:
              - Tipe data (numeric, categorical, boolean, datetime, unique)
              - Analisis kardinalitas
              - Identifikasi kolom unik/identifier
              - Statistik deskriptif
    """
    print("🔍 Menggunakan YData Profiling untuk deteksi tipe data otomatis...")
    
    try:
        # Buat profile report dengan konfigurasi minimal untuk performa
        profile = ProfileReport(
            df, 
            title="Data Type Detection with YData Profiling",
            minimal=True,  # Mode minimal untuk performa lebih cepat
            explorative=False,  # Nonaktifkan analisis eksploratif
            dark_mode=False,
            progress_bar=False,
            infer_dtypes=True,  # Aktifkan inferensi tipe data otomatis
            vars={
                'num': {'low_categorical_threshold': 0},  # Hindari false positive categorical
                'cat': {'cardinality_threshold': 50},     # Threshold untuk kategori
                'bool': {'imbalance_threshold': 0.9},     # Threshold untuk boolean imbalance
            }
        )
        
        # Ekstrak informasi tipe data dari profile
        result = {
            'numeric_columns': [],
            'categorical_columns': [],
            'binary_columns': [],
            'date_columns': [],
            'string_columns': [],
            'identifier_columns': [],
            'foreign_key_columns': [],
            'summary': {
                'total_columns': len(df.columns),
                'data_types_detected': {},
                'recommendations': []
            }
        }
        
        # Dapatkan informasi dari profile
        variables_info = profile.get_description()['variables']
        
        for column_name, column_info in variables_info.items():
            col_type = column_info.get('type', 'unknown')
            unique_count = column_info.get('n_unique', 0)
            total_rows = len(df)
            uniqueness_ratio = unique_count / total_rows if total_rows > 0 else 0
            
            # Dapatkan statistik tambahan
            stats = {
                'unique_count': unique_count,
                'uniqueness_ratio': uniqueness_ratio,
                'missing_count': column_info.get('n_missing', 0),
                'missing_percentage': (column_info.get('n_missing', 0) / total_rows) * 100 if total_rows > 0 else 0,
                'data_type': str(df[column_name].dtype),
                'memory_size': column_info.get('memory_size', 0)
            }
            
            # Deteksi naming patterns untuk membantu klasifikasi
            col_lower = column_name.lower().strip()
            
            # Pattern untuk identifikasi foreign key
            fk_patterns = [
                r'.*_id$', r'^id_.*', r'.*code$', r'^code.*',
                r'.*ref$', r'^ref.*', r'.*key$', r'^key.*',
                r'.*partner.*id.*', r'.*store.*id.*', r'.*customer.*id.*'
            ]
            has_fk_pattern = any(re.match(pattern, col_lower) for pattern in fk_patterns)
            
            # Pattern untuk identifikasi identifier/primary key
            id_patterns = [
                r'.*id$', r'^id.*', r'.*_id$', r'^id_.*',
                r'.*no$', r'^no.*', r'.*number$', r'^number.*'
            ]
            has_id_pattern = any(re.match(pattern, col_lower) for pattern in id_patterns)
            
            # Pattern untuk identifikasi date
            date_patterns = [
                r'.*date.*', r'.*time.*', r'.*created.*', r'.*updated.*',
                r'.*delivery.*', r'.*timestamp.*', r'.*datetime.*'
            ]
            has_date_pattern = any(re.match(pattern, col_lower) for pattern in date_patterns)
            
            # Klasifikasi berdasarkan YData Profiling type dan analisis tambahan
            if col_type == 'Numeric':
                # Deteksi apakah ini identifier/unique column
                if uniqueness_ratio > 0.95 and has_id_pattern:
                    result['identifier_columns'].append({
                        'column': column_name,
                        'identifier_type': 'primary_key' if uniqueness_ratio == 1.0 else 'unique_identifier',
                        'confidence_score': min(0.95, uniqueness_ratio + 0.05),
                        'reasons': ['High uniqueness ratio from YData Profiling', f'Uniqueness: {uniqueness_ratio:.3f}', 'ID naming pattern'],
                        'recommendation': 'Primary key or unique identifier - exclude from ML features',
                        **stats
                    })
                elif has_fk_pattern and 0.01 < uniqueness_ratio < 0.8:
                    # Foreign key detection dengan YData Profiling
                    confidence = 0.7 + (0.2 if has_fk_pattern else 0)
                    result['foreign_key_columns'].append({
                        'column': column_name,
                        'confidence_score': confidence,
                        'uniqueness_ratio': uniqueness_ratio,
                        'has_fk_pattern': has_fk_pattern,
                        'reasons': ['Numeric with moderate cardinality from YData Profiling', 'FK naming pattern detected'],
                        'recommendation': 'Foreign key reference - encode for ML if needed',
                        **stats
                    })
                else:
                    # True numeric column
                    numeric_type = 'continuous'
                    reasons = ['Detected as Numeric by YData Profiling']
                    
                    # Enhanced pattern recognition
                    if any(pattern in col_lower for pattern in ['price', 'amount', 'total', 'balance', 'income', 'cost', 'salary']):
                        numeric_type = 'monetary'
                        reasons.append('Monetary pattern detected')
                    elif any(pattern in col_lower for pattern in ['age', 'year', 'count', 'quantity', 'weight', 'height']):
                        numeric_type = 'measured_value'
                        reasons.append('Measured value pattern detected')
                    elif any(pattern in col_lower for pattern in ['percent', 'rate', 'ratio', 'score']):
                        numeric_type = 'percentage_or_ratio'
                        reasons.append('Percentage/ratio pattern detected')
                    
                    result['numeric_columns'].append({
                        'column': column_name,
                        'numeric_type': numeric_type,
                        'reasons': reasons,
                        'is_continuous': unique_count > 20,
                        'stats': {
                            'mean': float(df[column_name].mean()) if not df[column_name].isna().all() else None,
                            'std': float(df[column_name].std()) if not df[column_name].isna().all() else None,
                            'min': float(df[column_name].min()) if not df[column_name].isna().all() else None,
                            'max': float(df[column_name].max()) if not df[column_name].isna().all() else None,
                            'median': float(df[column_name].median()) if not df[column_name].isna().all() else None
                        },
                        'recommendation': 'Numeric column suitable for mathematical operations and regression',
                        **stats
                    })
            
            elif col_type == 'Categorical':
                # Enhanced binary detection
                if unique_count == 2:
                    unique_values = df[column_name].dropna().unique().tolist()
                    binary_type = determine_binary_type_ydata(set(unique_values))
                    
                    result['binary_columns'].append({
                        'column': column_name,
                        'binary_type': binary_type,
                        'unique_values': unique_values,
                        'is_strict_binary': True,
                        'reasons': ['Detected as binary by YData Profiling', f'Exactly 2 unique values: {unique_values}'],
                        'recommendation': 'Binary column - encode as 0/1 for ML',
                        **stats
                    })
                else:
                    # Enhanced categorical analysis
                    categorical_type = 'low_cardinality' if unique_count <= 10 else 'moderate_cardinality' if unique_count <= 50 else 'high_cardinality'
                    value_counts = df[column_name].value_counts().head(10)
                    
                    # Check for ordinal patterns
                    is_ordinal = False
                    ordinal_patterns = ['low', 'medium', 'high', 'small', 'large', 'good', 'better', 'best']
                    values_str = [str(v).lower() for v in unique_values[:10]]
                    if any(pattern in ' '.join(values_str) for pattern in ordinal_patterns):
                        is_ordinal = True
                    
                    result['categorical_columns'].append({
                        'column': column_name,
                        'categorical_type': categorical_type,
                        'is_ordinal': is_ordinal,
                        'cardinality': unique_count,
                        'reasons': ['Detected as Categorical by YData Profiling'],
                        'top_values': value_counts.to_dict(),
                        'mode_value': df[column_name].mode().iloc[0] if len(df[column_name].mode()) > 0 else None,
                        'recommendation': f'Categorical column ({categorical_type}) - suitable for encoding and classification',
                        **stats
                    })
            
            elif col_type == 'DateTime' or has_date_pattern:
                # Enhanced date/time detection
                sample_values = df[column_name].dropna().head(3).astype(str).tolist()
                
                # Detect date format
                date_format = 'unknown'
                if any('T' in str(val) for val in sample_values):
                    date_format = 'ISO_8601'
                elif any('/' in str(val) for val in sample_values):
                    date_format = 'MM/DD/YYYY_or_DD/MM/YYYY'
                elif any('-' in str(val) for val in sample_values):
                    date_format = 'YYYY-MM-DD'
                
                result['date_columns'].append({
                    'column': column_name,
                    'date_format_type': date_format,
                    'has_time_component': any(':' in str(val) for val in sample_values),
                    'reasons': ['Detected as DateTime by YData Profiling'] if col_type == 'DateTime' else ['Date pattern in column name'],
                    'sample_values': sample_values,
                    'recommendation': 'Date/time column - extract features like year, month, day for ML',
                    **stats
                })
            
            elif col_type == 'Text':
                # Enhanced text analysis
                sample_values = df[column_name].dropna().head(5).astype(str).tolist()
                avg_length = df[column_name].astype(str).str.len().mean()
                max_length = df[column_name].astype(str).str.len().max()
                
                # Check for URL pattern
                is_url = any('http' in str(val) or 'www.' in str(val) for val in sample_values[:3])
                
                # Check for email pattern
                is_email = any('@' in str(val) and '.' in str(val) for val in sample_values[:3])
                
                if is_url:
                    result['string_columns'].append({
                        'column': column_name,
                        'string_type': 'url',
                        'avg_length': avg_length,
                        'max_length': max_length,
                        'sample_values': sample_values,
                        'reasons': ['URL pattern detected by YData Profiling analysis'],
                        'recommendation': 'URL column - extract domain features if needed for ML',
                        **stats
                    })
                elif is_email:
                    result['string_columns'].append({
                        'column': column_name,
                        'string_type': 'email',
                        'avg_length': avg_length,
                        'max_length': max_length,
                        'sample_values': sample_values,
                        'reasons': ['Email pattern detected by YData Profiling analysis'],
                        'recommendation': 'Email column - extract domain features if needed for ML',
                        **stats
                    })
                elif avg_length > 50:  # Long text
                    result['string_columns'].append({
                        'column': column_name,
                        'string_type': 'long_text',
                        'avg_length': avg_length,
                        'max_length': max_length,
                        'sample_values': sample_values,
                        'reasons': ['Long text detected by YData Profiling', f'Average length: {avg_length:.1f}'],
                        'recommendation': 'Long text column - suitable for NLP analysis',
                        **stats
                    })
                else:
                    # Short text - likely categorical
                    result['categorical_columns'].append({
                        'column': column_name,
                        'categorical_type': 'text_based',
                        'cardinality': unique_count,
                        'reasons': ['Short text detected by YData Profiling'],
                        'top_values': df[column_name].value_counts().head(10).to_dict(),
                        'recommendation': 'Short text categorical - encode for ML',
                        **stats
                    })
            
            elif col_type == 'Boolean':
                # Boolean columns
                unique_values = df[column_name].dropna().unique().tolist()
                
                result['binary_columns'].append({
                    'column': column_name,
                    'binary_type': 'boolean',
                    'unique_values': unique_values,
                    'is_strict_binary': True,
                    'reasons': ['Detected as Boolean by YData Profiling'],
                    'recommendation': 'Boolean column - already suitable for ML (0/1)',
                    **stats
                })
            
            else:
                # Unknown type - enhanced fallback analysis
                if uniqueness_ratio > 0.95 and has_id_pattern:
                    result['identifier_columns'].append({
                        'column': column_name,
                        'identifier_type': 'unique_identifier',
                        'confidence_score': uniqueness_ratio,
                        'reasons': ['High uniqueness ratio', 'Unknown type by YData Profiling', 'ID naming pattern'],
                        'recommendation': 'Likely identifier column - exclude from ML features',
                        **stats
                    })
                elif unique_count == 2:
                    # Treat as binary
                    unique_values = df[column_name].dropna().unique().tolist()
                    result['binary_columns'].append({
                        'column': column_name,
                        'binary_type': 'unknown_binary',
                        'unique_values': unique_values,
                        'is_strict_binary': True,
                        'reasons': ['2 unique values', 'Unknown type by YData Profiling'],
                        'recommendation': 'Binary column - encode as 0/1 for ML',
                        **stats
                    })
                else:
                    # Treat as categorical
                    result['categorical_columns'].append({
                        'column': column_name,
                        'categorical_type': 'unknown',
                        'cardinality': unique_count,
                        'reasons': ['Unknown type by YData Profiling', 'Treated as categorical'],
                        'recommendation': 'Review manually - may need special encoding',
                        **stats
                    })
        
        # Enhanced summary with YData Profiling insights
        result['summary']['data_types_detected'] = {
            'numeric': len(result['numeric_columns']),
            'categorical': len(result['categorical_columns']),
            'binary': len(result['binary_columns']),
            'date': len(result['date_columns']),
            'string': len(result['string_columns']),
            'identifier': len(result['identifier_columns']),
            'foreign_key': len(result['foreign_key_columns'])
        }
        
        result['summary']['recommendations'] = [
            f"✅ YData Profiling successfully analyzed {len(df.columns)} columns",
            f"📊 Found {len(result['numeric_columns'])} numeric columns (suitable for regression/correlation)",
            f"🏷️ Found {len(result['categorical_columns'])} categorical columns (need encoding for ML)",
            f"⚡ Found {len(result['binary_columns'])} binary columns (encode as 0/1)",
            f"📅 Found {len(result['date_columns'])} date columns (extract temporal features)",
            f"📝 Found {len(result['string_columns'])} text columns (consider NLP techniques)",
            f"🔍 Found {len(result['identifier_columns'])} identifier columns (exclude from ML)",
            f"🔗 Found {len(result['foreign_key_columns'])} foreign key columns (consider for joins)"
        ]
        
        result['summary']['ydata_profiling_info'] = {
            'version': 'YData Profiling v4+',
            'analysis_method': 'Automated statistical analysis with pattern recognition',
            'confidence_level': 'High - based on statistical properties and naming patterns'
        }
        
        print("✅ YData Profiling analysis completed successfully!")
        return result
        
    except Exception as e:
        print(f"❌ Error in YData Profiling analysis: {e}")
        # Fallback ke metode manual jika YData Profiling gagal
        print("🔄 Falling back to manual detection method...")
        return detect_data_types_fallback(df)

def determine_binary_type_ydata(unique_set):
    """Determine the type of binary column for YData Profiling"""
    unique_list = list(unique_set)
    unique_lower = [str(v).lower() for v in unique_list if v is not None]
    
    if unique_set.issubset({0, 1, '0', '1', 0.0, 1.0}):
        return 'numeric_binary'
    elif unique_set.issubset({True, False}):
        return 'boolean'
    elif any(v in ['yes', 'no', 'y', 'n'] for v in unique_lower):
        return 'yes_no'
    elif any(v in ['true', 'false', 't', 'f'] for v in unique_lower):
        return 'true_false'
    elif any(v in ['male', 'female', 'm', 'f'] for v in unique_lower):
        return 'gender'
    elif any(v in ['active', 'inactive'] for v in unique_lower):
        return 'status'
    else:
        return 'custom_binary'

def detect_data_types_fallback(df):
    """Fallback method jika YData Profiling gagal - menggunakan deteksi sederhana"""
    print("🔄 Using fallback method for data type detection...")
    
    # Deteksi sederhana berdasarkan tipe data pandas
    numeric_columns = []
    categorical_columns = []
    binary_columns = []
    date_columns = []
    string_columns = []
    identifier_columns = []
    foreign_key_columns = []
    
    for col in df.columns:
        unique_count = df[col].nunique()
        total_rows = len(df)
        uniqueness_ratio = unique_count / total_rows if total_rows > 0 else 0
        
        # Numeric columns
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            if uniqueness_ratio > 0.95:
                identifier_columns.append({
                    'column': col,
                    'identifier_type': 'unique_identifier',
                    'confidence_score': uniqueness_ratio,
                    'reasons': ['High uniqueness ratio (fallback method)'],
                    'recommendation': 'Potential identifier column'
                })
            elif unique_count == 2:
                binary_columns.append({
                    'column': col,
                    'binary_type': 'numeric_binary',
                    'unique_values': df[col].dropna().unique().tolist(),
                    'reasons': ['Exactly 2 unique values (fallback method)'],
                    'recommendation': 'Binary numeric column'
                })
            else:
                numeric_columns.append({
                    'column': col,
                    'numeric_type': 'continuous',
                    'reasons': ['Numeric data type (fallback method)'],
                    'recommendation': 'Numeric column for calculations'
                })
        
        # Object/string columns
        elif df[col].dtype == 'object':
            if unique_count == 2:
                binary_columns.append({
                    'column': col,
                    'binary_type': 'string_binary',
                    'unique_values': df[col].dropna().unique().tolist(),
                    'reasons': ['Exactly 2 unique values (fallback method)'],
                    'recommendation': 'Binary string column'
                })
            elif uniqueness_ratio > 0.95:
                identifier_columns.append({
                    'column': col,
                    'identifier_type': 'string_identifier',
                    'confidence_score': uniqueness_ratio,
                    'reasons': ['High uniqueness ratio (fallback method)'],
                    'recommendation': 'Potential string identifier'
                })
            elif unique_count <= 50:  # Low cardinality
                categorical_columns.append({
                    'column': col,
                    'categorical_type': 'low_cardinality',
                    'reasons': ['Low cardinality string (fallback method)'],
                    'recommendation': 'Categorical column'
                })
            else:
                string_columns.append({
                    'column': col,
                    'string_type': 'text',
                    'reasons': ['High cardinality string (fallback method)'],
                    'recommendation': 'Text column'
                })
        
        # Boolean columns
        elif df[col].dtype == 'bool':
            binary_columns.append({
                'column': col,
                'binary_type': 'boolean',
                'unique_values': df[col].dropna().unique().tolist(),
                'reasons': ['Boolean data type (fallback method)'],
                'recommendation': 'Boolean column'
            })
        
        # Datetime columns
        elif 'datetime' in str(df[col].dtype):
            date_columns.append({
                'column': col,
                'date_format_type': 'datetime',
                'reasons': ['Datetime data type (fallback method)'],
                'recommendation': 'Date/time column'
            })
    
    return {
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'binary_columns': binary_columns,
        'date_columns': date_columns,
        'string_columns': string_columns,
        'identifier_columns': identifier_columns,
        'foreign_key_columns': foreign_key_columns,
        'summary': {
            'total_columns': len(df.columns),
            'data_types_detected': {
                'numeric': len(numeric_columns),
                'categorical': len(categorical_columns),
                'binary': len(binary_columns),
                'date': len(date_columns),
                'string': len(string_columns),
                'identifier': len(identifier_columns),
                'foreign_key': len(foreign_key_columns)
            },
            'recommendations': ['Used simple fallback detection method']
        }
    }





def perform_data_integration(df):
    """Perform data integration techniques"""
    integration_report = {
        'record_linkage_opportunities': [],
        'data_fusion_suggestions': [],
        'duplicate_records': 0,
        'integration_strategies': {
            'record_linkage': {
                'description': 'Integrasi data melibatkan penggabungan data dari berbagai sumber menjadi satu set data terpadu',
                'techniques': [
                    'Penautan data (data linking)',
                    'Fusi data (data fusion)',
                    'Identifikasi rekaman duplikat'
                ]
            },
            'data_fusion': {
                'description': 'Menggabungkan data dari berbagai sumber untuk menciptakan kumpulan data yang lebih komprehensif',
                'benefits': [
                    'Konsistensi data meningkat',
                    'Akurasi data lebih baik',
                    'Dataset lebih kaya untuk analisis'
                ]
            }
        }
    }
    
    # Record Linkage - Find potential duplicate records
    text_columns = df.select_dtypes(include=['object']).columns
    if len(text_columns) > 0:
        # Simple duplicate detection based on text similarity
        duplicates = df.duplicated(subset=text_columns, keep=False)
        integration_report['duplicate_records'] = duplicates.sum()
        
        if duplicates.sum() > 0:
            integration_report['record_linkage_opportunities'].append({
                'type': 'exact_match_duplicates',
                'count': duplicates.sum(),
                'columns': list(text_columns),
                'recommendation': 'Keterkaitan Rekaman - Identifikasi dan cocokkan rekaman yang merujuk entitas sama'
            })
    
    # Data Fusion - Suggest columns that could be merged or consolidated
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_count = df[col].nunique()
            total_count = len(df)
            
            # If many unique values relative to total, might need fusion
            if unique_count > total_count * 0.5:
                integration_report['data_fusion_suggestions'].append({
                    'column': col,
                    'unique_values': unique_count,
                    'suggestion': 'Fusi Data - Standardisasi atau kategorisasi nilai untuk integrasi yang lebih baik',
                    'technique': 'Data consolidation and normalization'
                })
    
    return integration_report

def perform_data_transformation(df, transformation_options=None):
    """Perform advanced data transformation"""
    if transformation_options is None:
        transformation_options = {}
    
    transformed_df = df.copy()
    transformation_report = {
        'normalization': {},
        'discretization': {},
        'aggregation': {},
        'hierarchy_creation': {},
        'transformation_strategies': {
            'normalization': {
                'description': 'Normalisasi Data - Proses penskalaan data ke rentang umum untuk memastikan konsistensi',
                'techniques': ['Min-Max Scaling', 'Z-Score Standardization', 'Robust Scaling']
            },
            'discretization': {
                'description': 'Diskritisasi - Mengubah data kontinyu menjadi kategori diskrit untuk memudahkan analisis',
                'methods': ['Equal-width binning', 'Equal-frequency binning', 'Custom thresholds']
            },
            'aggregation': {
                'description': 'Agregasi Data - Menggabungkan beberapa titik data ke dalam bentuk ringkasan',
                'types': ['Sum', 'Average', 'Count', 'Min/Max', 'Standard Deviation']
            },
            'hierarchy_creation': {
                'description': 'Pembuatan Hirarki Konsep - Mengorganisasikan data ke dalam hierarki konsep',
                'benefits': ['Tampilan tingkat tinggi', 'Pemahaman yang lebih baik', 'Analisis multi-level']
            }
        }
    }
    
    # Normalization
    if transformation_options.get('normalize', True):
        numeric_cols = transformed_df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        
        for col in numeric_cols:
            if transformed_df[col].std() > 0:  # Avoid division by zero
                original_values = transformed_df[col].copy()
                transformed_df[col] = scaler.fit_transform(transformed_df[[col]]).flatten()
                transformation_report['normalization'][col] = {
                    'method': 'Z-Score Standardization',
                    'mean': float(original_values.mean()),
                    'std': float(original_values.std()),
                    'min': float(original_values.min()),
                    'max': float(original_values.max()),
                    'transformed_mean': float(transformed_df[col].mean()),
                    'transformed_std': float(transformed_df[col].std())
                }
    
    # Discretization
    if transformation_options.get('discretize', True):
        numeric_cols = transformed_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if transformed_df[col].nunique() > 10:  # Only discretize if many unique values
                try:
                    transformed_df[f'{col}_discrete'] = pd.cut(
                        transformed_df[col], 
                        bins=5, 
                        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
                    )
                    transformation_report['discretization'][col] = {
                        'bins': 5,
                        'new_column': f'{col}_discrete',
                        'method': 'Equal-width binning',
                        'categories': ['Very Low', 'Low', 'Medium', 'High', 'Very High']
                    }
                except:
                    pass
    
    # Data Aggregation - Group similar records
    if transformation_options.get('aggregate', False):
        categorical_cols = transformed_df.select_dtypes(include=['object']).columns
        numeric_cols = transformed_df.select_dtypes(include=[np.number]).columns
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            # Try to aggregate by the first categorical column
            first_cat_col = categorical_cols[0]
            if transformed_df[first_cat_col].nunique() < len(transformed_df) * 0.5:
                agg_result = transformed_df.groupby(first_cat_col)[numeric_cols].agg(['mean', 'sum', 'count'])
                transformation_report['aggregation'][first_cat_col] = {
                    'grouped_by': first_cat_col,
                    'aggregated_columns': list(numeric_cols),
                    'operations': ['mean', 'sum', 'count'],
                    'description': 'Agregasi berdasarkan kategori untuk ringkasan data'
                }
    
    # Concept Hierarchy Creation
    if transformation_options.get('create_hierarchy', False):
        categorical_cols = transformed_df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            unique_values = transformed_df[col].nunique()
            if 5 < unique_values < 50:  # Good candidate for hierarchy
                # Create simplified hierarchy levels
                try:
                    # Group similar values or create levels based on frequency
                    value_counts = transformed_df[col].value_counts()
                    
                    # Create 3-level hierarchy: High, Medium, Low frequency
                    thresholds = value_counts.quantile([0.33, 0.67])
                    
                    def create_hierarchy_level(value):
                        count = value_counts.get(value, 0)
                        if count >= thresholds.iloc[1]:
                            return 'High_Frequency'
                        elif count >= thresholds.iloc[0]:
                            return 'Medium_Frequency'
                        else:
                            return 'Low_Frequency'
                    
                    transformed_df[f'{col}_hierarchy'] = transformed_df[col].apply(create_hierarchy_level)
                    
                    transformation_report['hierarchy_creation'][col] = {
                        'new_column': f'{col}_hierarchy',
                        'levels': ['High_Frequency', 'Medium_Frequency', 'Low_Frequency'],
                        'method': 'Frequency-based hierarchy',
                        'description': 'Hirarki berdasarkan frekuensi kemunculan nilai'
                    }
                except:
                    pass
    
    return transformed_df, transformation_report

def perform_data_reduction(df, reduction_options=None):
    """Perform data reduction techniques"""
    if reduction_options is None:
        reduction_options = {}
    
    reduced_df = df.copy()
    reduction_report = {
        'dimensionality_reduction': {},
        'numerosity_reduction': {},
        'data_compression': {},
        'reduction_strategies': {
            'dimensionality_reduction': {
                'description': 'Pengurangan Dimensionalitas - Mengurangi jumlah variabel sambil mempertahankan informasi penting',
                'techniques': ['PCA (Principal Component Analysis)', 'Feature Selection', 'Variance Threshold']
            },
            'numerosity_reduction': {
                'description': 'Pengurangan Jumlah - Mengurangi jumlah titik data dengan sampling',
                'methods': ['Random Sampling', 'Stratified Sampling', 'Cluster-based Sampling']
            },
            'data_compression': {
                'description': 'Kompresi Data - Mengurangi ukuran data dengan encoding yang lebih padat',
                'approaches': ['Lossless Compression', 'Lossy Compression', 'Dictionary Encoding']
            }
        }
    }
    
    # Feature Selection (Simple variance-based)
    if reduction_options.get('feature_selection', True):
        numeric_cols = reduced_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            from sklearn.feature_selection import VarianceThreshold
            
            # Remove low-variance features
            selector = VarianceThreshold(threshold=0.01)
            selected_features = selector.fit_transform(reduced_df[numeric_cols])
            selected_feature_names = [numeric_cols[i] for i in range(len(numeric_cols)) if selector.get_support()[i]]
            
            # Update dataframe with selected features
            reduced_df = reduced_df[selected_feature_names + list(reduced_df.select_dtypes(include=['object']).columns)]
            
            reduction_report['dimensionality_reduction'] = {
                'method': 'Variance Threshold Feature Selection',
                'original_features': len(numeric_cols),
                'selected_features': len(selected_feature_names),
                'removed_features': list(set(numeric_cols) - set(selected_feature_names)),
                'variance_threshold': 0.01,
                'description': 'Menghapus fitur dengan varians rendah untuk reduksi dimensi'
            }
    
    # Principal Component Analysis (if requested)
    if reduction_options.get('apply_pca', False):
        numeric_cols = reduced_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 2:
            from sklearn.decomposition import PCA
            
            # Apply PCA to reduce to 95% variance
            pca = PCA(n_components=0.95)
            pca_features = pca.fit_transform(reduced_df[numeric_cols])
            
            # Create new column names for PCA components
            pca_columns = [f'PC{i+1}' for i in range(pca_features.shape[1])]
            
            # Replace numeric columns with PCA components
            reduced_df = reduced_df.drop(columns=numeric_cols)
            pca_df = pd.DataFrame(pca_features, columns=pca_columns, index=reduced_df.index)
            reduced_df = pd.concat([reduced_df, pca_df], axis=1)
            
            reduction_report['dimensionality_reduction']['pca'] = {
                'method': 'Principal Component Analysis',
                'original_features': len(numeric_cols),
                'pca_components': len(pca_columns),
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': float(pca.explained_variance_ratio_.cumsum()[-1]),
                'description': 'Reduksi dimensi menggunakan PCA dengan mempertahankan 95% varians'
            }
    
    # Sampling for numerosity reduction
    if reduction_options.get('sampling', False):
        original_size = len(reduced_df)
        sample_size = min(1000, original_size)  # Sample up to 1000 records
        
        if original_size > sample_size:
            # Stratified sampling if categorical columns exist
            categorical_cols = reduced_df.select_dtypes(include=['object']).columns
            
            if len(categorical_cols) > 0:
                # Stratified sampling based on first categorical column
                first_cat = categorical_cols[0]
                reduced_df = reduced_df.groupby(first_cat, group_keys=False).apply(
                    lambda x: x.sample(min(len(x), max(1, int(sample_size * len(x) / original_size))), 
                                     random_state=42)
                )
                reduction_report['numerosity_reduction'] = {
                    'method': 'Stratified Sampling',
                    'original_size': original_size,
                    'reduced_size': len(reduced_df),
                    'reduction_ratio': len(reduced_df) / original_size,
                    'stratified_by': first_cat,
                    'description': 'Sampling berdasarkan strata untuk mempertahankan distribusi'
                }
            else:
                # Random sampling
                reduced_df = reduced_df.sample(n=sample_size, random_state=42)
                reduction_report['numerosity_reduction'] = {
                    'method': 'Random Sampling',
                    'original_size': original_size,
                    'reduced_size': sample_size,
                    'reduction_ratio': sample_size / original_size,
                    'description': 'Sampling acak untuk reduksi jumlah data'
                }
    
    # Data Compression through encoding
    if reduction_options.get('compress', False):
        # Dictionary encoding for categorical columns
        categorical_cols = reduced_df.select_dtypes(include=['object']).columns
        compression_info = {}
        
        for col in categorical_cols:
            if reduced_df[col].nunique() < len(reduced_df) * 0.8:  # Suitable for encoding
                # Create dictionary encoding
                unique_values = reduced_df[col].unique()
                encoding_dict = {val: idx for idx, val in enumerate(unique_values)}
                
                # Apply encoding
                reduced_df[f'{col}_encoded'] = reduced_df[col].map(encoding_dict)
                
                compression_info[col] = {
                    'encoding_dict': encoding_dict,
                    'unique_values': len(unique_values),
                    'compression_ratio': len(unique_values) / len(reduced_df),
                    'new_column': f'{col}_encoded'
                }
        
        if compression_info:
            reduction_report['data_compression'] = {
                'method': 'Dictionary Encoding',
                'compressed_columns': compression_info,
                'description': 'Encoding kamus untuk kompresi data kategorikal'
            }
    
    return reduced_df, reduction_report

def generate_enhanced_data_quality_report(df):
    """Generate comprehensive data quality report with advanced insights"""
    # Gunakan YData Profiling untuk deteksi tipe data
    ydata_result = detect_data_types(df)
    
    report = {
        'basic_info': {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'dtypes': df.dtypes.astype(str).to_dict()
        },
        'data_types': {
            'numeric_columns': ydata_result['numeric_columns'],
            'categorical_columns': ydata_result['categorical_columns'],
            'binary_columns': ydata_result['binary_columns'],
            'identifier_columns': ydata_result['identifier_columns'],
            'foreign_key_columns': ydata_result['foreign_key_columns'],
            'string_columns': ydata_result['string_columns'],
            'date_columns': ydata_result['date_columns'],
            'primary_key_candidates': ydata_result['identifier_columns']
        },
        'data_quality': {
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'unique_values_per_column': df.nunique().to_dict()
        },
        'statistical_summary': {},
        'data_distribution': {},
        'outliers': {},
        'integration_analysis': perform_data_integration(df),
        'transformation_opportunities': {
            'normalization_candidates': [],
            'discretization_candidates': [],
            'aggregation_opportunities': [],
            'hierarchy_creation_candidates': []
        },
        'reduction_opportunities': {
            'high_variance_features': [],
            'low_variance_features': [],
            'correlation_analysis': {},
            'sampling_recommendations': {}
        }
    }
    
    # Statistical summary for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats = df[numeric_cols].describe()
        report['statistical_summary'] = stats.to_dict()
        
        # Outlier detection using IQR method
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            report['outliers'][col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(df) * 100,
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
            
            # Transformation opportunities
            col_range = df[col].max() - df[col].min()
            col_std = df[col].std()
            
            if col_range > 1000 or col_std > 100:
                report['transformation_opportunities']['normalization_candidates'].append({
                    'column': col,
                    'reason': 'Wide range or high standard deviation',
                    'range': float(col_range),
                    'std': float(col_std)
                })
            
            if df[col].nunique() > 20:
                report['transformation_opportunities']['discretization_candidates'].append({
                    'column': col,
                    'unique_values': df[col].nunique(),
                    'reason': 'High cardinality numeric column suitable for binning'
                })
            
            # Variance analysis for reduction
            variance = df[col].var()
            if variance < 0.01:
                report['reduction_opportunities']['low_variance_features'].append({
                    'column': col,
                    'variance': float(variance),
                    'recommendation': 'Consider removing due to low variance'
                })
            else:
                report['reduction_opportunities']['high_variance_features'].append({
                    'column': col,
                    'variance': float(variance)
                })
    
    # Distribution analysis for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        value_counts = df[col].value_counts().head(10)
        report['data_distribution'][col] = {
            'top_values': value_counts.to_dict(),
            'unique_count': df[col].nunique(),
            'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None
        }
        
        # Hierarchy creation opportunities
        unique_count = df[col].nunique()
        if 5 < unique_count < 50:
            report['transformation_opportunities']['hierarchy_creation_candidates'].append({
                'column': col,
                'unique_values': unique_count,
                'reason': 'Suitable number of categories for hierarchy creation'
            })
    
    # Aggregation opportunities
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        for cat_col in categorical_cols:
            if df[cat_col].nunique() < len(df) * 0.3:  # Not too many unique values
                report['transformation_opportunities']['aggregation_opportunities'].append({
                    'groupby_column': cat_col,
                    'unique_groups': df[cat_col].nunique(),
                    'numeric_columns_for_aggregation': list(numeric_cols),
                    'reason': 'Suitable categorical column for grouping numeric data'
                })
    
    # Correlation analysis for reduction
    if len(numeric_cols) > 1:
        try:
            correlation_matrix = df[numeric_cols].corr()
            high_correlations = []
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:  # High correlation
                        high_correlations.append({
                            'column1': correlation_matrix.columns[i],
                            'column2': correlation_matrix.columns[j],
                            'correlation': float(corr_value),
                            'recommendation': 'Consider removing one of these highly correlated features'
                        })
            
            report['reduction_opportunities']['correlation_analysis'] = {
                'high_correlations': high_correlations,
                'description': 'Highly correlated features that might be redundant'
            }
        except:
            pass
    
    # Sampling recommendations
    dataset_size = len(df)
    if dataset_size > 10000:
        report['reduction_opportunities']['sampling_recommendations'] = {
            'current_size': dataset_size,
            'recommended_sample_size': min(5000, dataset_size),
            'sampling_ratio': min(5000, dataset_size) / dataset_size,
            'reason': 'Large dataset - sampling can improve processing speed',
            'suggested_method': 'Stratified sampling if categorical columns exist, random sampling otherwise'
        }
    
    return report

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
        
        # Detect and exclude primary key columns to avoid bias using YData Profiling
        all_data_types = detect_data_types(df)
        excluded_pk_columns = []
        
        for pk_info in all_data_types['identifier_columns']:
            col = pk_info['column']
            if pk_info['uniqueness_ratio'] > 0.95:  # Highly unique columns
                excluded_pk_columns.append(col)
                print(f"Excluding potential primary key column: {col} (uniqueness: {pk_info['uniqueness_ratio']:.3f})")
        
        # Remove primary key columns from dataset for ML processing
        if excluded_pk_columns:
            df = df.drop(columns=excluded_pk_columns)
            print(f"Excluded primary key columns: {excluded_pk_columns}")
        
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

@app.route('/api/detect-data-types', methods=['POST'])
def detect_all_data_types():
    """API endpoint to detect all data types using YData Profiling"""
    try:
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert data to DataFrame
        df = pd.DataFrame(data['data'])
        
        if df.empty:
            return jsonify({'error': 'Empty dataset provided'}), 400
        
        print(f"🔍 Starting YData Profiling analysis for {len(df.columns)} columns, {len(df)} rows")
        
        # Gunakan YData Profiling untuk deteksi tipe data otomatis
        result = detect_data_types(df)
        
        # Tambahkan primary key candidates dari identifier columns
        result['primary_key_candidates'] = result['identifier_columns']
        
        # Update summary dengan informasi YData Profiling
        result['summary']['description'] = 'Advanced data type detection using YData Profiling'
        result['summary']['detection_methods'] = [
            'YData Profiling: Automated intelligent data type detection',
            'Numeric: Continuous values, monetary patterns, measured values',
            'Categorical: Text-based categories, low/moderate cardinality',
            'Binary: Boolean, yes/no, true/false, gender, status patterns',
            'Identifier: Primary keys, unique identifiers (>95% uniqueness)',
            'Foreign Key: Reference columns with moderate cardinality',
            'String: Long text suitable for NLP analysis',
            'Date: Temporal data with datetime patterns'
        ]
        
        print("✅ YData Profiling analysis completed successfully")
        return safe_json_response(convert_numpy_types(result))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error in YData Profiling analysis: {e}")
        return jsonify({'error': f'Data type detection failed: {str(e)}'}), 500

@app.route('/api/detect-string', methods=['POST'])
def detect_string():
    """API endpoint to detect string/text columns using YData Profiling"""
    try:
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert data to DataFrame
        df = pd.DataFrame(data['data'])
        
        if df.empty:
            return jsonify({'error': 'Empty dataset provided'}), 400
        
        # Gunakan YData Profiling untuk deteksi
        ydata_result = detect_data_types(df)
        string_columns = ydata_result['string_columns']
        
        result = {
            'string_columns': string_columns,
            'total_columns': len(df.columns),
            'string_count': len(string_columns),
            'summary': {
                'description': 'String/text column detection using YData Profiling',
                'criteria': [
                    'Detected as Text type by YData Profiling',
                    'Long text content (>50 characters average)',
                    'Suitable for NLP analysis',
                    'Contains descriptive text patterns'
                ]
            }
        }
        
        return safe_json_response(convert_numpy_types(result))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'String detection failed: {str(e)}'}), 500

@app.route('/api/detect-foreign-key', methods=['POST'])
def detect_foreign_key():
    """API endpoint to detect foreign key columns using YData Profiling"""
    try:
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert data to DataFrame
        df = pd.DataFrame(data['data'])
        
        if df.empty:
            return jsonify({'error': 'Empty dataset provided'}), 400
        
        # Gunakan YData Profiling untuk deteksi
        ydata_result = detect_data_types(df)
        foreign_key_columns = ydata_result['foreign_key_columns']
        
        result = {
            'foreign_key_columns': foreign_key_columns,
            'total_columns': len(df.columns),
            'foreign_key_count': len(foreign_key_columns),
            'summary': {
                'description': 'Foreign key detection using YData Profiling',
                'criteria': [
                    'Moderate cardinality detected by YData Profiling',
                    'Numeric columns with reference patterns',
                    'Medium uniqueness ratio (10-80%)',
                    'Potential foreign key relationships'
                ]
            }
        }
        
        return safe_json_response(convert_numpy_types(result))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Foreign key detection failed: {str(e)}'}), 500

@app.route('/api/detect-date', methods=['POST'])
def detect_date():
    """API endpoint to detect date columns using YData Profiling"""
    try:
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert data to DataFrame
        df = pd.DataFrame(data['data'])
        
        if df.empty:
            return jsonify({'error': 'Empty dataset provided'}), 400
        
        # Gunakan YData Profiling untuk deteksi
        ydata_result = detect_data_types(df)
        date_columns = ydata_result['date_columns']
        
        result = {
            'date_columns': date_columns,
            'total_columns': len(df.columns),
            'date_count': len(date_columns),
            'summary': {
                'description': 'Date/time column detection using YData Profiling',
                'formats_detected': [
                    'DateTime type detected by YData Profiling',
                    'Automatic temporal pattern recognition',
                    'ISO date, US date, and custom formats',
                    'Suitable for time series analysis'
                ]
            }
        }
        
        return safe_json_response(convert_numpy_types(result))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Date detection failed: {str(e)}'}), 500

@app.route('/api/detect-categorical', methods=['POST'])
def detect_categorical():
    """API endpoint to detect categorical columns using YData Profiling"""
    try:
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert data to DataFrame
        df = pd.DataFrame(data['data'])
        
        if df.empty:
            return jsonify({'error': 'Empty dataset provided'}), 400
        
        # Gunakan YData Profiling untuk deteksi
        ydata_result = detect_data_types(df)
        categorical_columns = ydata_result['categorical_columns']
        binary_columns = ydata_result['binary_columns']
        
        result = {
            'categorical_columns': categorical_columns,
            'binary_columns': binary_columns,
            'total_columns': len(df.columns),
            'categorical_count': len(categorical_columns),
            'binary_count': len(binary_columns),
            'summary': {
                'description': 'Categorical column detection using YData Profiling',
                'methodology': [
                    'Categorical type detected by YData Profiling',
                    'Intelligent cardinality analysis',
                    'Text-based and numeric categorical detection',
                    'Binary subset identification (exactly 2 values)',
                    'Automatic encoding recommendations'
                ]
            }
        }
        
        return safe_json_response(convert_numpy_types(result))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Categorical detection failed: {str(e)}'}), 500

@app.route('/api/detect-binary', methods=['POST'])
def detect_binary():
    """API endpoint to detect binary columns using YData Profiling"""
    try:
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert data to DataFrame
        df = pd.DataFrame(data['data'])
        
        if df.empty:
            return jsonify({'error': 'Empty dataset provided'}), 400
        
        # Gunakan YData Profiling untuk deteksi
        ydata_result = detect_data_types(df)
        binary_columns = ydata_result['binary_columns']
        
        result = {
            'binary_columns': binary_columns,
            'total_columns': len(df.columns),
            'binary_count': len(binary_columns),
            'summary': {
                'description': 'Binary column detection using YData Profiling',
                'types_detected': [
                    'Boolean type detected by YData Profiling',
                    'Categorical with exactly 2 unique values',
                    'Numeric binary (0/1), Boolean (True/False)',
                    'Yes/No, Gender, Status patterns',
                    'Automatic binary pattern recognition'
                ]
            }
        }
        
        return safe_json_response(convert_numpy_types(result))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Binary detection failed: {str(e)}'}), 500

@app.route('/api/detect-primary-key', methods=['POST'])
def detect_primary_key():
    """API endpoint to detect primary key columns using YData Profiling"""
    try:
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert data to DataFrame
        df = pd.DataFrame(data['data'])
        
        if df.empty:
            return jsonify({'error': 'Empty dataset provided'}), 400
        
        # Gunakan YData Profiling untuk deteksi
        ydata_result = detect_data_types(df)
        primary_key_candidates = ydata_result['identifier_columns']
        
        result = {
            'primary_key_candidates': primary_key_candidates,
            'total_columns': len(df.columns),
            'candidate_count': len(primary_key_candidates),
            'summary': {
                'description': 'Primary key detection using YData Profiling',
                'criteria': [
                    'Unique identifier detection by YData Profiling',
                    'High uniqueness ratio (>95%)',
                    'Automatic ID pattern recognition',
                    'Primary key and unique identifier classification',
                    'Confidence score based analysis'
                ]
            }
        }
        
        return safe_json_response(convert_numpy_types(result))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Primary key detection failed: {str(e)}'}), 500

@app.route('/api/data-integration', methods=['POST'])
def perform_data_integration_api():
    """API endpoint for data integration analysis"""
    try:
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert data to DataFrame
        df = pd.DataFrame(data['data'])
        
        if df.empty:
            return jsonify({'error': 'Empty dataset provided'}), 400
        
        # Perform data integration analysis
        integration_report = perform_data_integration(df)
        
        result = {
            'integration_report': integration_report,
            'dataset_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns)
            }
        }
        
        return safe_json_response(convert_numpy_types(result))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Data integration analysis failed: {str(e)}'}), 500

@app.route('/api/data-transformation', methods=['POST'])
def perform_data_transformation_api():
    """API endpoint for data transformation"""
    try:
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert data to DataFrame
        df = pd.DataFrame(data['data'])
        
        if df.empty:
            return jsonify({'error': 'Empty dataset provided'}), 400
        
        # Get transformation options from request
        transformation_options = data.get('transformation_options', {})
        
        # Perform data transformation
        transformed_df, transformation_report = perform_data_transformation(df, transformation_options)
        
        # Convert transformed data back to records
        transformed_data = transformed_df.to_dict('records')
        
        # Convert numpy types for JSON serialization
        for row in transformed_data:
            for key, value in row.items():
                if isinstance(value, (np.integer, np.floating, np.bool_)):
                    row[key] = value.item()
                elif pd.isna(value):
                    row[key] = None
        
        result = {
            'transformed_data': transformed_data,
            'transformation_report': transformation_report,
            'original_shape': df.shape,
            'transformed_shape': transformed_df.shape,
            'columns': list(transformed_df.columns)
        }
        
        return safe_json_response(convert_numpy_types(result))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Data transformation failed: {str(e)}'}), 500

@app.route('/api/data-reduction', methods=['POST'])
def perform_data_reduction_api():
    """API endpoint for data reduction"""
    try:
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert data to DataFrame
        df = pd.DataFrame(data['data'])
        
        if df.empty:
            return jsonify({'error': 'Empty dataset provided'}), 400
        
        # Get reduction options from request
        reduction_options = data.get('reduction_options', {})
        
        # Perform data reduction
        reduced_df, reduction_report = perform_data_reduction(df, reduction_options)
        
        # Convert reduced data back to records
        reduced_data = reduced_df.to_dict('records')
        
        # Convert numpy types for JSON serialization
        for row in reduced_data:
            for key, value in row.items():
                if isinstance(value, (np.integer, np.floating, np.bool_)):
                    row[key] = value.item()
                elif pd.isna(value):
                    row[key] = None
        
        result = {
            'reduced_data': reduced_data,
            'reduction_report': reduction_report,
            'original_shape': df.shape,
            'reduced_shape': reduced_df.shape,
            'columns': list(reduced_df.columns),
            'reduction_summary': {
                'rows_reduced': df.shape[0] - reduced_df.shape[0],
                'columns_reduced': df.shape[1] - reduced_df.shape[1],
                'size_reduction_percentage': ((df.shape[0] * df.shape[1] - reduced_df.shape[0] * reduced_df.shape[1]) / (df.shape[0] * df.shape[1])) * 100
            }
        }
        
        return safe_json_response(convert_numpy_types(result))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Data reduction failed: {str(e)}'}), 500

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
    """Analyze uploaded CSV data with enhanced type detection"""
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
        
        # Enhanced column type detection using YData Profiling
        all_data_types = detect_data_types(df)
        binary_columns = all_data_types['binary_columns']
        primary_key_columns = all_data_types['identifier_columns']
        
        # Create lookup sets for faster checking
        binary_column_names = {col['column'] for col in binary_columns}
        primary_key_column_names = {col['column'] for col in primary_key_columns}
        
        # Enhanced statistics with proper type classification
        stats = {}
        for column in df.columns:
            col_data = df[column].dropna()
            
            # Determine column type with priority: identifier > binary > numeric > categorical
            if column in primary_key_column_names:
                # Point 9: Primary keys don't have statistical measures
                pk_info = next(pk for pk in primary_key_columns if pk['column'] == column)
                stats[column] = {
                    'type': 'identifier',
                    'subtype': pk_info.get('identifier_type', 'primary_key'),
                    'unique_count': int(pk_info['unique_count']),
                    'uniqueness_ratio': float(pk_info['uniqueness_ratio']),
                    'confidence_score': pk_info.get('confidence_score', 1.0),
                    'count': int(df[column].count()),
                    'null_count': int(pk_info.get('missing_count', df[column].isnull().sum())),
                    'recommendation': pk_info['recommendation']
                }
            elif column in binary_column_names:
                # Point 10: Binary columns don't have Mean, Std Dev, Min, Max
                binary_info = next(bin_col for bin_col in binary_columns if bin_col['column'] == column)
                stats[column] = {
                    'type': 'binary',
                    'subtype': binary_info['binary_type'],
                    'values': binary_info['unique_values'],
                    'is_strict_binary': binary_info.get('is_strict_binary', True),
                    'unique_count': int(binary_info['unique_count']),
                    'count': int(df[column].count()),
                    'null_count': int(binary_info.get('missing_count', df[column].isnull().sum())),
                    'recommendation': binary_info['recommendation']
                }
            elif df[column].dtype in ['int64', 'float64', 'Int64']:
                # Point 11: Enhanced numeric statistics with percentiles
                if len(col_data) > 0:
                    stats[column] = {
                        'type': 'numeric',
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()) if len(col_data) > 1 else 0.0,
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'median': float(col_data.median()),
                        'q1': float(col_data.quantile(0.25)),  # 25th percentile
                        'q3': float(col_data.quantile(0.75)),  # 75th percentile
                        'count': int(df[column].count()),
                        'null_count': int(df[column].isnull().sum()),
                        'unique_count': int(df[column].nunique())
                    }
                else:
                    stats[column] = {
                        'type': 'numeric',
                        'mean': None, 'std': None, 'min': None, 'max': None,
                        'median': None, 'q1': None, 'q3': None,
                        'count': 0, 'null_count': int(df[column].isnull().sum()),
                        'unique_count': 0
                    }
            else:
                # Categorical columns
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
                    'unique_count': int(df[column].nunique()),
                    'most_common': most_common_dict,
                    'count': int(df[column].count()),
                    'null_count': int(df[column].isnull().sum())
                }
        
        # Correlation matrix for numeric columns only (excluding binary and identifier)
        numeric_columns = [col for col in df.columns 
                          if col not in binary_column_names 
                          and col not in primary_key_column_names 
                          and df[col].dtype in ['int64', 'float64', 'Int64']]
        
        correlation_matrix = {}
        if numeric_columns:
            numeric_df = df[numeric_columns]
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
            'data_types': convert_numpy_types(df.dtypes.astype(str).to_dict()),
            'binary_columns': convert_numpy_types(binary_columns),
            'primary_key_columns': convert_numpy_types(primary_key_columns),
            'type_summary': {
                'numeric': len([col for col, stat in stats.items() if stat['type'] == 'numeric']),
                'categorical': len([col for col, stat in stats.items() if stat['type'] == 'categorical']),
                'binary': len([col for col, stat in stats.items() if stat['type'] == 'binary']),
                'identifier': len([col for col, stat in stats.items() if stat['type'] == 'identifier'])
            }
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

@app.route('/api/save-cleaned-data', methods=['POST'])
def save_cleaned_data():
    """Save cleaned data to training_data directory"""
    try:
        data = request.json
        
        if not data or 'csv_data' not in data or 'filename' not in data:
            return jsonify({'error': 'Missing csv_data or filename'}), 400
        
        csv_data = data['csv_data']
        filename = data['filename']
        
        # Ensure filename has proper format - fix naming convention
        if not filename.endswith('_cleaned.csv'):
            # Remove .csv first, then add _cleaned.csv to avoid .csv_cleaned format
            if filename.endswith('.csv'):
                filename = filename[:-4] + '_cleaned.csv'
            else:
                filename = filename + '_cleaned.csv'
        
        # Convert to DataFrame and save
        df = pd.DataFrame(csv_data)
        filepath = os.path.join(TRAINING_DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        
        print(f"Saved cleaned data to: {filepath}")
        
        return safe_json_response({
            'success': True,
            'message': f'Cleaned data saved as {filename}',
            'filepath': filepath,
            'shape': df.shape
        })
        
    except Exception as e:
        print(f"Error saving cleaned data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/train-from-data', methods=['POST'])
def train_from_data():
    """Train model using data sent from frontend"""
    try:
        data = request.json
        
        if not data or 'csv_data' not in data:
            return jsonify({'error': 'No CSV data provided'}), 400
        
        csv_data = data['csv_data']
        target_column = data.get('target_column')
        model_type = data.get('model_type', 'decision_tree')
        test_size = data.get('test_size', 0.2)
        cross_validation = data.get('cross_validation', True)
        
        if not target_column:
            return jsonify({'error': 'Target column not provided'}), 400
        
        # Clean data and create DataFrame
        cleaned_csv_data = []
        for i, row in enumerate(csv_data):
            cleaned_row = {}
            for key, value in row.items():
                cleaned_row[key] = clean_complex_types(value, i, key)
            cleaned_csv_data.append(cleaned_row)
        
        df = pd.DataFrame(cleaned_csv_data)
        df = clean_dataframe(df)
        
        if target_column not in df.columns:
            return jsonify({'error': f'Target column "{target_column}" not found in data'}), 400
        
        # Prepare features (exclude target column)
        all_columns = list(df.columns)
        all_columns.remove(target_column)
        
        # Select only numeric columns for features
        feature_columns = []
        for col in all_columns:
            if df[col].dtype in ['int64', 'float64', 'Int64', 'Float64']:
                feature_columns.append(col)
        
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
        
        # Generate predictions for the entire dataset
        y_pred_full = model.predict(X)
        
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_name = data.get('dataset_name', 'unknown')
        model_id = f"{model_type}_{dataset_name}_{timestamp}"
        model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
        
        # Prepare model data for saving
        model_data = {
            'model': model,
            'scaler': scaler if 'scaler' in locals() else None,
            'label_encoder': label_encoder,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'model_type': model_type,
            'performance': performance,
            'original_classes': original_classes,
            'created_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, model_path)
        
        # Store in global registry
        trained_models[model_id] = {
            'model_id': model_id,
            'model_type': model_type,
            'features': feature_columns,
            'target_column': target_column,
            'performance': performance,
            'created_at': datetime.now().isoformat(),
            'training_file': dataset_name,
            'accuracy': performance.get('accuracy', 0)
        }
        
        # Add to model history
        model_history.append({
            'model_id': model_id,
            'model_type': model_type,
            'training_file': dataset_name,
            'accuracy': performance.get('accuracy', 0),
            'cv_mean_score': performance.get('cv_mean_score'),
            'cv_std_score': performance.get('cv_std_score'),
            'created_at': datetime.now().isoformat()
        })
        
        # Cross-validation if requested
        cv_scores = None
        if cross_validation and len(X_train) > 10:  # Only if we have enough data
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)//2))
                performance['cv_mean_score'] = float(cv_scores.mean())
                performance['cv_std_score'] = float(cv_scores.std())
            except Exception as cv_error:
                print(f"Cross-validation failed: {cv_error}")
        
        result = {
            'model_id': model_id,
            'accuracy': performance.get('accuracy', 0),
            'performance': convert_numpy_types(performance),
            'predictions': convert_numpy_types(y_pred_full.tolist()),
            'feature_columns': feature_columns,
            'target_column': target_column,
            'model_type': model_type,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        return safe_json_response(result)
        
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
        
        # Generate enhanced quality report
        enhanced_report = generate_enhanced_data_quality_report(df)
        
        # Create simplified report for frontend compatibility
        report = {
            'totalRows': len(df),
            'totalColumns': len(df.columns),
            'missingValues': enhanced_report['data_quality']['missing_values'],
            'duplicates': enhanced_report['data_quality']['duplicate_rows'],
            'outliers': {},
            'dataTypes': enhanced_report['basic_info']['dtypes'],
            'memoryUsage': enhanced_report['basic_info']['memory_usage'],
            'summary': {},
            # Enhanced features
            'binaryColumns': enhanced_report['data_types']['binary_columns'],
            'primaryKeyCandidates': enhanced_report['data_types']['primary_key_candidates'],
            'integrationAnalysis': enhanced_report['integration_analysis'],
            'transformationOpportunities': enhanced_report['transformation_opportunities'],
            'reductionOpportunities': enhanced_report['reduction_opportunities']
        }
        
        # Convert outliers to simple format
        for col, outlier_data in enhanced_report['outliers'].items():
            if outlier_data['count'] > 0:
                report['outliers'][col] = outlier_data['count']
        
        # Convert summary statistics to simple format
        for col in df.columns:
            if col in enhanced_report['statistical_summary']:
                # Numeric column
                stats = enhanced_report['statistical_summary'][col]
                report['summary'][col] = {
                    'type': 'numeric',
                    'count': int(stats['count']),
                    'mean': float(stats['mean']),
                    'std': float(stats['std']),
                    'min': float(stats['min']),
                    'max': float(stats['max']),
                    'unique': enhanced_report['data_quality']['unique_values_per_column'][col]
                }
            elif col in enhanced_report['data_distribution']:
                # Categorical column
                dist_data = enhanced_report['data_distribution'][col]
                report['summary'][col] = {
                    'type': 'categorical',
                    'count': len(df[col].dropna()),
                    'unique': dist_data['unique_count'],
                    'top': dist_data['most_frequent'],
                    'freq': list(dist_data['top_values'].values())[0] if dist_data['top_values'] else 0
                }
        
        return safe_json_response(convert_numpy_types(report))
        
    except Exception as e:
        print(f"Data quality report error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Quality report generation failed: {str(e)}'}), 500

@app.route('/api/clean-data', methods=['POST'])
def clean_data():
    """Clean data based on user preferences with advanced techniques"""
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
        
        # Generate comprehensive data quality report before cleaning
        pre_cleaning_report = generate_enhanced_data_quality_report(df)
        
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
        
        # Apply advanced data processing techniques
        transformation_report = {}
        reduction_report = {}
        
        # Data Integration
        if cleaning_options.get('apply_integration', False):
            integration_report = perform_data_integration(df)
        else:
            integration_report = pre_cleaning_report['integration_analysis']
        
        # Data Transformation
        if cleaning_options.get('apply_transformation', False):
            transformation_options = {
                'normalize': cleaning_options.get('normalize_data', False),
                'discretize': cleaning_options.get('discretize_data', False),
                'aggregate': cleaning_options.get('aggregate_data', False)
            }
            df, transformation_report = perform_data_transformation(df, transformation_options)
        
        # Data Reduction
        if cleaning_options.get('apply_reduction', False):
            reduction_options = {
                'feature_selection': cleaning_options.get('feature_selection', False),
                'sampling': cleaning_options.get('apply_sampling', False)
            }
            df, reduction_report = perform_data_reduction(df, reduction_options)
        
        # Generate post-cleaning data quality report
        post_cleaning_report = generate_enhanced_data_quality_report(df)
        
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
            },
            'advanced_processing': {
                'data_integration': integration_report,
                'data_transformation': transformation_report,
                'data_reduction': reduction_report
            },
            'data_quality_reports': {
                'before_cleaning': convert_numpy_types(pre_cleaning_report),
                'after_cleaning': convert_numpy_types(post_cleaning_report)
            }
        }
        
        return safe_json_response(convert_numpy_types(result))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Data cleaning failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
