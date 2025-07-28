# 🐍 Backend Setup & Configuration

## 📑 Daftar Isi

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Instalasi](#instalasi)
- [Struktur Backend](#struktur-backend)
- [Flask Application](#flask-application)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Database & Storage](#database--storage)
- [API Endpoints](#api-endpoints)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## Overview

Backend ML Analytics Dashboard dibangun menggunakan **Flask** sebagai web framework dengan **Scikit-learn** untuk machine learning capabilities. Sistem ini menyediakan RESTful API untuk data analysis, model training, dan prediction.

---

## Prerequisites

- **Python 3.11+** - Core programming language
- **pip** - Python package manager
- **Virtual Environment** - Isolated Python environment

---

## Instalasi

### 1. Virtual Environment Setup
```bash
# Buat virtual environment
python -m venv venv311

# Aktifkan virtual environment
# Windows:
venv311\Scripts\activate.bat
# Linux/Mac:
source venv311/bin/activate

# Verify aktivasi
which python  # Should point to venv311
```

### 2. Install Dependencies
```bash
cd backend

# Install dari requirements.txt
pip install -r requirements.txt

# Atau install manual
pip install flask flask-cors pandas scikit-learn numpy matplotlib seaborn ydata-profiling joblib
```

### 3. Verify Installation
```bash
# Test semua imports
python -c "
import flask
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ydata_profiling
import joblib
print('All dependencies installed successfully!')
"
```

---

## Struktur Backend

```
backend/
├── app.py                         # Main Flask application
├── requirements.txt               # Python dependencies
├── models/                        # Trained ML models storage
│   ├── decision_tree_*.pkl        # Decision tree models
│   ├── random_forest_*.pkl        # Random forest models
│   ├── logistic_regression_*.pkl  # Logistic regression models
│   └── svm_*.pkl                  # SVM models
├── data/                          # Temporary data storage
│   ├── uploads/                   # Uploaded files
│   ├── cleaned/                   # Cleaned datasets
│   └── reports/                   # Analysis reports
└── utils/                         # Utility functions (optional)
    ├── data_processor.py          # Data processing utilities
    ├── model_trainer.py           # ML training utilities
    └── validators.py              # Input validation
```

---

## Flask Application

### 1. **Main Application (app.py)**

```python
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from datetime import datetime
import json

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000'])

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODELS_FOLDER'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables untuk menyimpan data session
current_data = None
current_cleaned_data = None
analysis_results = None
```

### 2. **Error Handling**

```python
@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        'error': 'Bad Request',
        'message': 'Invalid input parameters'
    }), 400

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'Something went wrong on the server'
    }), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({
        'error': 'File Too Large',
        'message': 'File size exceeds 16MB limit'
    }), 413
```

### 3. **Logging Configuration**

```python
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/ml_dashboard.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('ML Analytics Dashboard startup')
```

---

## Machine Learning Pipeline

### 1. **Data Processing Module**

```python
def clean_data(df, missing_strategy='fill_mean', duplicates_strategy='remove', outliers_strategy='keep'):
    """
    Comprehensive data cleaning function
    """
    cleaned_df = df.copy()
    cleaning_summary = {}
    
    # Handle missing values
    if missing_strategy == 'fill_mean':
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
        cleaned_df[categorical_cols] = cleaned_df[categorical_cols].fillna(cleaned_df[categorical_cols].mode().iloc[0] if not cleaned_df[categorical_cols].mode().empty else 'Unknown')
    elif missing_strategy == 'remove_rows':
        cleaned_df = cleaned_df.dropna()
    
    # Handle duplicates
    if duplicates_strategy == 'remove':
        before_dup = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        cleaning_summary['duplicates_removed'] = before_dup - len(cleaned_df)
    
    # Handle outliers
    if outliers_strategy == 'remove':
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    
    return cleaned_df, cleaning_summary
```

### 2. **Model Training Pipeline**

```python
def train_model(X, y, algorithm='decision_tree', parameters=None):
    """
    Train ML model dengan algoritma yang dipilih
    """
    if parameters is None:
        parameters = {}
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize model
    if algorithm == 'decision_tree':
        model = DecisionTreeClassifier(
            max_depth=parameters.get('max_depth', None),
            min_samples_split=parameters.get('min_samples_split', 2),
            min_samples_leaf=parameters.get('min_samples_leaf', 1),
            random_state=42
        )
    elif algorithm == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=parameters.get('n_estimators', 100),
            max_depth=parameters.get('max_depth', None),
            min_samples_split=parameters.get('min_samples_split', 2),
            random_state=42
        )
    elif algorithm == 'logistic_regression':
        model = LogisticRegression(
            C=parameters.get('C', 1.0),
            max_iter=parameters.get('max_iter', 1000),
            random_state=42
        )
    elif algorithm == 'svm':
        model = SVC(
            C=parameters.get('C', 1.0),
            kernel=parameters.get('kernel', 'rbf'),
            probability=True,
            random_state=42
        )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'model': model,
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'predictions': y_pred.tolist(),
        'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None,
        'test_indices': y_test.index.tolist()
    }
```

### 3. **Model Persistence**

```python
def save_model(model, algorithm, dataset_name):
    """
    Save trained model dengan metadata
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{algorithm}_{dataset_name}_{timestamp}.pkl"
    model_path = os.path.join(app.config['MODELS_FOLDER'], model_filename)
    
    # Save model
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        'algorithm': algorithm,
        'dataset_name': dataset_name,
        'timestamp': timestamp,
        'filename': model_filename,
        'created_at': datetime.now().isoformat()
    }
    
    metadata_path = model_path.replace('.pkl', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model_filename

def load_model(model_filename):
    """
    Load saved model
    """
    model_path = os.path.join(app.config['MODELS_FOLDER'], model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_filename} not found")
    
    model = joblib.load(model_path)
    
    # Load metadata if available
    metadata_path = model_path.replace('.pkl', '_metadata.json')
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, metadata
```

---

## Database & Storage

### 1. **File Storage Structure**

```python
# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)
os.makedirs('data/cleaned', exist_ok=True)
os.makedirs('data/reports', exist_ok=True)
os.makedirs('logs', exist_ok=True)
```

### 2. **Session Management**

```python
# In-memory session storage (untuk development)
# Untuk production, gunakan Redis atau database
session_data = {
    'current_data': None,
    'analysis_results': None,
    'cleaned_data': None,
    'trained_models': []
}

def get_session_data(key):
    return session_data.get(key)

def set_session_data(key, value):
    session_data[key] = value
```

---

## API Endpoints

### 1. **Health Check**

```python
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })
```

### 2. **Data Upload & Analysis**

```python
@app.route('/analyze', methods=['POST'])
def analyze_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are supported'}), 400
    
    try:
        # Read data
        df = pd.read_csv(file)
        set_session_data('current_data', df)
        
        # Basic analysis
        analysis = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'preview': df.head().to_dict('records')
        }
        
        # Statistical summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis['statistics'] = df[numeric_cols].describe().to_dict()
        
        set_session_data('analysis_results', analysis)
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
        
    except Exception as e:
        app.logger.error(f"Analysis error: {str(e)}")
        return jsonify({'error': f"Analysis failed: {str(e)}"}), 500
```

### 3. **Model Training Endpoint**

```python
@app.route('/train_model', methods=['POST'])
def train_model_endpoint():
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    algorithm = data.get('algorithm')
    target_column = data.get('target_column')
    parameters = data.get('parameters', {})
    
    # Validate inputs
    if not algorithm or not target_column:
        return jsonify({'error': 'Algorithm and target_column are required'}), 400
    
    # Get current data
    df = get_session_data('cleaned_data') or get_session_data('current_data')
    if df is None:
        return jsonify({'error': 'No data available. Please upload data first.'}), 400
    
    if target_column not in df.columns:
        return jsonify({'error': f'Target column {target_column} not found'}), 400
    
    try:
        # Prepare features dan target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical variables
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # Train model
        results = train_model(X_encoded, y, algorithm, parameters)
        
        # Save model
        dataset_name = "uploaded_data"
        model_filename = save_model(results['model'], algorithm, dataset_name)
        
        # Add to session
        trained_models = get_session_data('trained_models') or []
        trained_models.append({
            'filename': model_filename,
            'algorithm': algorithm,
            'target_column': target_column,
            'accuracy': results['accuracy'],
            'created_at': datetime.now().isoformat()
        })
        set_session_data('trained_models', trained_models)
        
        return jsonify({
            'success': True,
            'model_filename': model_filename,
            'accuracy': results['accuracy'],
            'confusion_matrix': results['confusion_matrix'],
            'classification_report': results['classification_report']
        })
        
    except Exception as e:
        app.logger.error(f"Training error: {str(e)}")
        return jsonify({'error': f"Training failed: {str(e)}"}), 500
```

---

## Testing

### 1. **Unit Tests**

```python
# tests/test_app.py
import unittest
import json
from app import app

class MLDashboardTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        app.config['TESTING'] = True
    
    def test_health_check(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_analyze_no_file(self):
        response = self.app.post('/analyze')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()
```

### 2. **Integration Tests**

```python
# tests/test_integration.py
import unittest
import pandas as pd
from io import StringIO
from app import app

class IntegrationTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        app.config['TESTING'] = True
    
    def test_full_workflow(self):
        # Create test CSV
        test_data = "feature1,feature2,target\n1,2,A\n3,4,B\n5,6,A"
        test_file = StringIO(test_data)
        
        # Test upload
        response = self.app.post('/analyze', 
            data={'file': (test_file, 'test.csv')},
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 200)
        
        # Test training
        train_data = {
            'algorithm': 'decision_tree',
            'target_column': 'target'
        }
        response = self.app.post('/train_model',
            data=json.dumps(train_data),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
```

---

## Deployment

### 1. **Production Configuration**

```python
# config.py
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'uploads'
    MODELS_FOLDER = os.environ.get('MODELS_FOLDER') or 'models'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False
    
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
```

### 2. **WSGI Configuration**

```python
# wsgi.py
from app import app

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
```

### 3. **Docker Configuration**

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "wsgi.py"]
```

---

## Troubleshooting

### 1. **Common Issues**

**Problem**: Import errors
```bash
# Solution: Verify virtual environment
which python
pip list
```

**Problem**: Model training fails
```python
# Solution: Add debugging
try:
    model.fit(X_train, y_train)
except Exception as e:
    print(f"Training error: {e}")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Data types: {X_train.dtypes}")
```

**Problem**: Memory issues dengan large datasets
```python
# Solution: Process data in chunks
def process_large_dataset(df, chunk_size=1000):
    for chunk in pd.read_csv(file, chunksize=chunk_size):
        yield process_chunk(chunk)
```

### 2. **Performance Monitoring**

```python
import time
import psutil

def monitor_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        app.logger.info(f"{func.__name__} - Time: {end_time - start_time:.2f}s, Memory: {(end_memory - start_memory) / 1024 / 1024:.2f}MB")
        
        return result
    return wrapper

@monitor_performance
def train_model_monitored(*args, **kwargs):
    return train_model(*args, **kwargs)
```

---

**Happy Backend Development! 🐍🚀**
