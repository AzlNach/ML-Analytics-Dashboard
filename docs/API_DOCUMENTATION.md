# 🔌 API Documentation

## 📑 Daftar Isi

- [Base URL](#base-url)
- [Authentication](#authentication)
- [Response Format](#response-format)
- [Data Analysis Endpoints](#data-analysis-endpoints)
- [Data Cleaning Endpoints](#data-cleaning-endpoints)
- [Model Training Endpoints](#model-training-endpoints)
- [Prediction Endpoints](#prediction-endpoints)
- [Utility Endpoints](#utility-endpoints)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)

---

## Base URL

```
Development: http://localhost:5000
Production: https://your-backend-url.com
```

---

## Authentication

Currently, the API does not require authentication. For production deployment, consider implementing:
- JWT tokens
- API keys
- OAuth 2.0

---

## Response Format

All API responses follow this standard format:

### Success Response
```json
{
  "success": true,
  "data": {
    // Response data here
  },
  "message": "Operation completed successfully",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Error Response
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": {
      // Additional error details
    }
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

---

## Data Analysis Endpoints

### 1. Upload and Analyze Data

**Endpoint**: `POST /analyze`

**Description**: Upload CSV file dan lakukan analisis data dasar

**Request**:
```http
POST /analyze HTTP/1.1
Content-Type: multipart/form-data

file: [CSV file]
```

**Response**:
```json
{
  "success": true,
  "analysis": {
    "shape": [100, 5],
    "columns": ["feature1", "feature2", "target"],
    "dtypes": {
      "feature1": "float64",
      "feature2": "int64",
      "target": "object"
    },
    "missing_values": {
      "feature1": 2,
      "feature2": 0,
      "target": 1
    },
    "duplicates": 3,
    "memory_usage": 2048,
    "preview": [
      {"feature1": 1.5, "feature2": 10, "target": "A"},
      {"feature1": 2.3, "feature2": 15, "target": "B"}
    ],
    "statistics": {
      "feature1": {
        "count": 98.0,
        "mean": 5.2,
        "std": 1.8,
        "min": 1.0,
        "25%": 3.5,
        "50%": 5.0,
        "75%": 6.8,
        "max": 10.0
      }
    }
  }
}
```

**Error Codes**:
- `400`: No file provided or invalid file format
- `413`: File too large (>16MB)
- `500`: Server error during analysis

---

### 2. Get Analysis Results

**Endpoint**: `GET /analyze`

**Description**: Mendapatkan hasil analisis terakhir

**Response**:
```json
{
  "success": true,
  "analysis": {
    // Same structure as POST /analyze response
  }
}
```

---

### 3. Generate Comprehensive Report

**Endpoint**: `POST /generate_report`

**Description**: Generate comprehensive analysis report menggunakan YData Profiling

**Request**:
```json
{
  "report_type": "html|json",
  "include_correlations": true,
  "include_missing_values": true,
  "include_duplicates": true
}
```

**Response**:
```json
{
  "success": true,
  "report": {
    "report_url": "/download/report_20240101_120000.html",
    "summary": {
      "total_observations": 1000,
      "total_variables": 10,
      "missing_cells": 25,
      "duplicate_rows": 5
    }
  }
}
```

---

## Data Cleaning Endpoints

### 1. Clean Data

**Endpoint**: `POST /clean_data`

**Description**: Membersihkan data dengan strategi yang dipilih

**Request**:
```json
{
  "missing_strategy": "fill_mean|fill_median|fill_mode|forward_fill|remove_rows",
  "duplicates_strategy": "remove|keep_first|keep_last",
  "outliers_strategy": "remove|cap|keep",
  "outlier_method": "iqr|zscore",
  "zscore_threshold": 3.0
}
```

**Response**:
```json
{
  "success": true,
  "cleaning_results": {
    "original_shape": [1000, 10],
    "cleaned_shape": [950, 10],
    "rows_removed": 50,
    "missing_values_handled": {
      "feature1": "filled with mean (5.2)",
      "feature2": "filled with mode (Active)"
    },
    "duplicates_removed": 5,
    "outliers_removed": 45,
    "cleaning_summary": "Dataset successfully cleaned. 50 rows removed, 95% data retained."
  },
  "download_url": "/download/cleaned_data.csv"
}
```

---

### 2. Preview Cleaning

**Endpoint**: `POST /preview_cleaning`

**Description**: Preview hasil cleaning tanpa menyimpan perubahan

**Request**:
```json
{
  "missing_strategy": "fill_mean",
  "duplicates_strategy": "remove",
  "outliers_strategy": "cap"
}
```

**Response**:
```json
{
  "success": true,
  "preview": {
    "before": {
      "shape": [1000, 10],
      "missing_count": 25,
      "duplicates_count": 5
    },
    "after": {
      "shape": [950, 10],
      "missing_count": 0,
      "duplicates_count": 0
    },
    "changes": {
      "rows_to_remove": 50,
      "values_to_fill": 25,
      "outliers_to_handle": 45
    }
  }
}
```

---

## Model Training Endpoints

### 1. Train Model

**Endpoint**: `POST /train_model`

**Description**: Train machine learning model

**Request**:
```json
{
  "algorithm": "decision_tree|random_forest|logistic_regression|svm",
  "target_column": "target",
  "parameters": {
    "max_depth": 5,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "n_estimators": 100,
    "C": 1.0,
    "kernel": "rbf"
  },
  "cross_validation": {
    "enabled": true,
    "cv_folds": 5,
    "scoring": "accuracy"
  },
  "test_size": 0.2,
  "random_state": 42
}
```

**Response**:
```json
{
  "success": true,
  "model_info": {
    "model_id": "decision_tree_uploaded_data_20240101_120000",
    "algorithm": "decision_tree",
    "target_column": "target",
    "training_completed_at": "2024-01-01T12:00:00Z"
  },
  "performance": {
    "accuracy": 0.95,
    "precision": 0.94,
    "recall": 0.96,
    "f1_score": 0.95,
    "confusion_matrix": [
      [45, 2],
      [3, 50]
    ],
    "classification_report": {
      "A": {"precision": 0.94, "recall": 0.96, "f1-score": 0.95},
      "B": {"precision": 0.96, "recall": 0.94, "f1-score": 0.95}
    }
  },
  "cross_validation": {
    "mean_accuracy": 0.93,
    "std_accuracy": 0.02,
    "scores": [0.91, 0.94, 0.95, 0.92, 0.93]
  },
  "feature_importance": {
    "feature1": 0.45,
    "feature2": 0.30,
    "feature3": 0.25
  }
}
```

---

### 2. List Models

**Endpoint**: `GET /models`

**Description**: Mendapatkan daftar semua trained models

**Response**:
```json
{
  "success": true,
  "models": [
    {
      "model_id": "decision_tree_uploaded_data_20240101_120000",
      "algorithm": "decision_tree",
      "target_column": "target",
      "accuracy": 0.95,
      "created_at": "2024-01-01T12:00:00Z",
      "file_size": "2.5MB",
      "status": "ready"
    },
    {
      "model_id": "random_forest_uploaded_data_20240101_130000",
      "algorithm": "random_forest",
      "target_column": "target",
      "accuracy": 0.97,
      "created_at": "2024-01-01T13:00:00Z",
      "file_size": "15.2MB",
      "status": "ready"
    }
  ],
  "total_models": 2
}
```

---

### 3. Get Model Details

**Endpoint**: `GET /models/{model_id}`

**Description**: Mendapatkan detail specific model

**Response**:
```json
{
  "success": true,
  "model": {
    "model_id": "decision_tree_uploaded_data_20240101_120000",
    "algorithm": "decision_tree",
    "parameters": {
      "max_depth": 5,
      "min_samples_split": 2
    },
    "training_data": {
      "dataset_name": "uploaded_data",
      "features_count": 10,
      "samples_count": 800,
      "target_classes": ["A", "B", "C"]
    },
    "performance": {
      "accuracy": 0.95,
      "confusion_matrix": [[45, 2], [3, 50]]
    },
    "metadata": {
      "created_at": "2024-01-01T12:00:00Z",
      "training_duration": "45.2 seconds",
      "file_size": "2.5MB"
    }
  }
}
```

---

### 4. Delete Model

**Endpoint**: `DELETE /models/{model_id}`

**Description**: Hapus trained model

**Response**:
```json
{
  "success": true,
  "message": "Model decision_tree_uploaded_data_20240101_120000 deleted successfully"
}
```

---

## Prediction Endpoints

### 1. Single Prediction

**Endpoint**: `POST /predict`

**Description**: Prediksi untuk single data point

**Request**:
```json
{
  "model_id": "decision_tree_uploaded_data_20240101_120000",
  "data": {
    "feature1": 5.2,
    "feature2": 10,
    "feature3": "Active",
    "feature4": 25.5
  }
}
```

**Response**:
```json
{
  "success": true,
  "prediction": {
    "predicted_class": "A",
    "confidence": 0.85,
    "probabilities": {
      "A": 0.85,
      "B": 0.15
    },
    "model_used": "decision_tree_uploaded_data_20240101_120000",
    "prediction_timestamp": "2024-01-01T14:00:00Z"
  }
}
```

---

### 2. Batch Prediction

**Endpoint**: `POST /predict_batch`

**Description**: Prediksi untuk multiple data points via file upload

**Request**:
```http
POST /predict_batch HTTP/1.1
Content-Type: multipart/form-data

file: [CSV file]
model_id: decision_tree_uploaded_data_20240101_120000
```

**Response**:
```json
{
  "success": true,
  "predictions": {
    "total_predictions": 100,
    "results": [
      {
        "row_index": 0,
        "predicted_class": "A",
        "confidence": 0.85,
        "probabilities": {"A": 0.85, "B": 0.15}
      },
      {
        "row_index": 1,
        "predicted_class": "B",
        "confidence": 0.92,
        "probabilities": {"A": 0.08, "B": 0.92}
      }
    ],
    "summary": {
      "class_distribution": {
        "A": 45,
        "B": 55
      },
      "average_confidence": 0.87
    }
  },
  "download_url": "/download/predictions_20240101_140000.csv"
}
```

---

### 3. Model Comparison

**Endpoint**: `POST /compare_models`

**Description**: Compare predictions dari multiple models

**Request**:
```json
{
  "model_ids": [
    "decision_tree_uploaded_data_20240101_120000",
    "random_forest_uploaded_data_20240101_130000"
  ],
  "data": {
    "feature1": 5.2,
    "feature2": 10
  }
}
```

**Response**:
```json
{
  "success": true,
  "comparison": {
    "input_data": {"feature1": 5.2, "feature2": 10},
    "predictions": [
      {
        "model_id": "decision_tree_uploaded_data_20240101_120000",
        "algorithm": "decision_tree",
        "predicted_class": "A",
        "confidence": 0.85
      },
      {
        "model_id": "random_forest_uploaded_data_20240101_130000",
        "algorithm": "random_forest",
        "predicted_class": "A",
        "confidence": 0.92
      }
    ],
    "consensus": {
      "agreed_prediction": "A",
      "agreement_percentage": 100,
      "average_confidence": 0.885
    }
  }
}
```

---

## Utility Endpoints

### 1. Health Check

**Endpoint**: `GET /health`

**Description**: Check server status

**Response**:
```json
{
  "success": true,
  "status": "healthy",
  "server_info": {
    "timestamp": "2024-01-01T12:00:00Z",
    "version": "1.0.0",
    "python_version": "3.11.0",
    "uptime": "2 days, 5 hours, 30 minutes"
  },
  "resources": {
    "memory_usage": "45%",
    "disk_usage": "12%",
    "active_models": 3
  }
}
```

---

### 2. Get Datasets

**Endpoint**: `GET /datasets`

**Description**: List available training datasets

**Response**:
```json
{
  "success": true,
  "datasets": [
    {
      "name": "iris_dataset.csv",
      "description": "Classic iris classification dataset",
      "size": "4.5KB",
      "features": 4,
      "samples": 150,
      "target_classes": 3
    },
    {
      "name": "customer_behavior.csv",
      "description": "Customer behavior analysis dataset",
      "size": "1.2MB",
      "features": 15,
      "samples": 10000,
      "target_classes": 2
    }
  ]
}
```

---

### 3. Download File

**Endpoint**: `GET /download/{filename}`

**Description**: Download generated files (reports, cleaned data, predictions)

**Response**: File download with appropriate headers

---

### 4. System Statistics

**Endpoint**: `GET /stats`

**Description**: Get system usage statistics

**Response**:
```json
{
  "success": true,
  "statistics": {
    "total_uploads": 156,
    "total_models_trained": 45,
    "total_predictions": 12500,
    "most_used_algorithm": "random_forest",
    "average_accuracy": 0.87,
    "storage_used": "2.5GB",
    "uptime": "7 days, 14 hours"
  }
}
```

---

## Error Handling

### Common Error Codes

| Code | Description | Example |
|------|-------------|---------|
| `400` | Bad Request | Invalid input parameters |
| `401` | Unauthorized | Missing or invalid authentication |
| `403` | Forbidden | Insufficient permissions |
| `404` | Not Found | Model or file not found |
| `413` | Payload Too Large | File size exceeds limit |
| `422` | Unprocessable Entity | Valid JSON but invalid data |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Server-side error |

### Error Response Examples

**400 Bad Request**:
```json
{
  "success": false,
  "error": {
    "code": "INVALID_INPUT",
    "message": "Target column 'target' not found in dataset",
    "details": {
      "available_columns": ["feature1", "feature2", "feature3"],
      "provided_target": "target"
    }
  }
}
```

**404 Not Found**:
```json
{
  "success": false,
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model with ID 'invalid_model_123' not found",
    "details": {
      "available_models": ["decision_tree_data_20240101_120000"]
    }
  }
}
```

**500 Internal Server Error**:
```json
{
  "success": false,
  "error": {
    "code": "TRAINING_FAILED",
    "message": "Model training failed due to insufficient data",
    "details": {
      "minimum_samples_required": 10,
      "samples_provided": 5
    }
  }
}
```

---

## Rate Limiting

API menggunakan rate limiting untuk mencegah abuse:

- **Default**: 100 requests per minute per IP
- **Upload endpoints**: 10 requests per minute per IP
- **Training endpoints**: 5 requests per minute per IP

**Rate Limit Headers**:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 85
X-RateLimit-Reset: 1704110400
```

**Rate Limit Exceeded Response**:
```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Too many requests. Please try again later.",
    "details": {
      "limit": 100,
      "reset_time": "2024-01-01T12:00:00Z"
    }
  }
}
```

---

## Examples

### Complete Workflow Example

```javascript
// 1. Upload dan analyze data
const formData = new FormData();
formData.append('file', file);

const analysisResponse = await fetch('/analyze', {
  method: 'POST',
  body: formData
});

// 2. Clean data
const cleaningResponse = await fetch('/clean_data', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    missing_strategy: 'fill_mean',
    duplicates_strategy: 'remove',
    outliers_strategy: 'cap'
  })
});

// 3. Train model
const trainingResponse = await fetch('/train_model', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    algorithm: 'random_forest',
    target_column: 'target',
    parameters: {
      n_estimators: 100,
      max_depth: 10
    }
  })
});

// 4. Make prediction
const predictionResponse = await fetch('/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model_id: 'random_forest_uploaded_data_20240101_120000',
    data: {
      feature1: 5.2,
      feature2: 10
    }
  })
});
```

### Python Client Example

```python
import requests
import json

class MLDashboardClient:
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url
    
    def upload_data(self, file_path):
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f'{self.base_url}/analyze', files=files)
        return response.json()
    
    def train_model(self, algorithm, target_column, parameters=None):
        data = {
            'algorithm': algorithm,
            'target_column': target_column,
            'parameters': parameters or {}
        }
        response = requests.post(
            f'{self.base_url}/train_model',
            json=data
        )
        return response.json()
    
    def predict(self, model_id, data):
        payload = {
            'model_id': model_id,
            'data': data
        }
        response = requests.post(
            f'{self.base_url}/predict',
            json=payload
        )
        return response.json()

# Usage
client = MLDashboardClient()
analysis = client.upload_data('data.csv')
model = client.train_model('random_forest', 'target')
prediction = client.predict(model['model_info']['model_id'], {'feature1': 5.2})
```

---

**Happy API Integration! 🔌🚀**
