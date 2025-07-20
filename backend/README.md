# Setup Backend Python Flask

## Prerequisites
- Python 3.8 atau lebih tinggi
- pip (Python package installer)

## Installation

1. **Navigate to backend directory:**
```bash
cd backend
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv

# Aktivasi virtual environment:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Create models directory:**
```bash
mkdir models
```

## Running the Backend

1. **Start the Flask server:**
```bash
python app.py
```

Server akan berjalan di `http://localhost:5000`

## API Endpoints

### Health Check
- **GET** `/api/health` - Check server status and get training data info

### Data Analysis
- **POST** `/api/analyze` - Analyze uploaded CSV data
- **POST** `/api/clustering` - Perform DBSCAN clustering
- **POST** `/api/anomaly-detection` - Detect anomalies using Isolation Forest
- **POST** `/api/decision-tree` - Build decision tree model

### Model Management
- **POST** `/api/train-from-file` - Train model from training data files
- **POST** `/api/predict` - Make predictions using trained models
- **GET** `/api/training-data` - Get list of available training datasets

## Training Data

Place your training CSV files in the `training_data/` folder. The system will automatically detect them and make them available for model training.

## Model Storage

Trained models are automatically saved in the `backend/models/` directory using joblib format.

## Environment Variables

You can set the following environment variables:

- `FLASK_ENV`: Set to 'development' for debug mode
- `FLASK_PORT`: Port number (default: 5000)
- `FLASK_HOST`: Host address (default: 0.0.0.0)

## Troubleshooting

1. **Port already in use**: Change the port in `app.py` or kill the process using the port
2. **Module not found**: Make sure you're in the virtual environment and all dependencies are installed
3. **CORS issues**: The backend includes CORS headers, but make sure your frontend is running on the expected port
