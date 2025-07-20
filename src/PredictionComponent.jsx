import React, { useState, useEffect } from 'react';
import { Target, Play, TrendingUp, AlertCircle, CheckCircle } from 'lucide-react';
import MLAnalyticsAPI from './services/api';

const PredictionComponent = () => {
  const [trainedModels, setTrainedModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [modelDetails, setModelDetails] = useState(null);
  const [inputValues, setInputValues] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadTrainedModels();
  }, []);

  useEffect(() => {
    if (selectedModel) {
      loadModelDetails(selectedModel);
    } else {
      setModelDetails(null);
      setInputValues({});
      setPrediction(null);
    }
  }, [selectedModel]);

  const loadTrainedModels = async () => {
    try {
      const response = await MLAnalyticsAPI.getTrainedModels();
      setTrainedModels(response.models || []);
    } catch (error) {
      console.error('Failed to load trained models:', error);
    }
  };

  const loadModelDetails = async (modelId) => {
    try {
      const details = await MLAnalyticsAPI.getModelDetails(modelId);
      setModelDetails(details);
      
      // Initialize input values for all features
      const initialValues = {};
      details.features.forEach(feature => {
        initialValues[feature] = '';
      });
      setInputValues(initialValues);
      setPrediction(null);
    } catch (error) {
      console.error('Failed to load model details:', error);
    }
  };

  const handleInputChange = (feature, value) => {
    setInputValues(prev => ({
      ...prev,
      [feature]: value
    }));
  };

  const validateInputs = () => {
    if (!modelDetails) return false;
    
    for (const feature of modelDetails.features) {
      if (inputValues[feature] === '' || inputValues[feature] === null || inputValues[feature] === undefined) {
        return false;
      }
    }
    return true;
  };

  const handlePredict = async () => {
    if (!validateInputs()) {
      alert('Please fill in all feature values');
      return;
    }

    setLoading(true);
    try {
      // Convert string inputs to numbers where appropriate
      const processedInputs = {};
      modelDetails.features.forEach(feature => {
        const value = inputValues[feature];
        // Try to convert to number if it looks like a number
        const numValue = parseFloat(value);
        processedInputs[feature] = isNaN(numValue) ? value : numValue;
      });

      const result = await MLAnalyticsAPI.makePrediction(selectedModel, processedInputs);
      setPrediction(result);
    } catch (error) {
      alert(`Prediction failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const renderFeatureInput = (feature) => {
    return (
      <div key={feature} className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">
          {feature}
        </label>
        <input
          type="number"
          step="any"
          value={inputValues[feature] || ''}
          onChange={(e) => handleInputChange(feature, e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          placeholder={`Enter ${feature}`}
        />
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Model Selection */}
      <div className="bg-white p-6 rounded-lg border">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Target className="w-5 h-5" />
          Make Predictions
        </h3>

        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">Select Trained Model</label>
          <select 
            value={selectedModel} 
            onChange={(e) => setSelectedModel(e.target.value)}
            className="w-full px-3 py-2 border rounded-lg"
          >
            <option value="">Choose a model...</option>
            {trainedModels.map((model) => (
              <option key={model.model_id} value={model.model_id}>
                {model.model_id} - {model.model_type} (Accuracy: {(model.accuracy * 100).toFixed(2)}%)
              </option>
            ))}
          </select>
        </div>

        {trainedModels.length === 0 && (
          <div className="flex items-center gap-2 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <AlertCircle className="w-5 h-5 text-yellow-600" />
            <span className="text-yellow-800">
              No trained models available. Please train a model first.
            </span>
          </div>
        )}
      </div>

      {/* Model Details and Input Form */}
      {modelDetails && (
        <div className="bg-white p-6 rounded-lg border">
          <h4 className="text-lg font-semibold mb-4">Model Information</h4>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="space-y-2">
              <p><strong>Model Type:</strong> {modelDetails.model_type}</p>
              <p><strong>Training File:</strong> {modelDetails.training_file}</p>
              <p><strong>Target Column:</strong> {modelDetails.target_column}</p>
            </div>
            <div className="space-y-2">
              <p><strong>Features Count:</strong> {modelDetails.features.length}</p>
              <p><strong>Accuracy:</strong> 
                <span className="text-green-600 font-medium ml-1">
                  {(modelDetails.performance?.accuracy * 100).toFixed(2)}%
                </span>
              </p>
              {modelDetails.created_at && (
                <p><strong>Created:</strong> {new Date(modelDetails.created_at).toLocaleString()}</p>
              )}
            </div>
          </div>

          {/* Feature Importance */}
          {modelDetails.performance?.feature_importance && (
            <div className="mb-6">
              <h5 className="font-medium mb-3">Feature Importance</h5>
              <div className="space-y-2">
                {modelDetails.features.map((feature, index) => {
                  const importance = modelDetails.performance.feature_importance[index] || 0;
                  return (
                    <div key={feature} className="flex items-center gap-2">
                      <span className="w-24 text-sm font-medium">{feature}</span>
                      <div className="flex-1 bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-blue-600 h-2 rounded-full" 
                          style={{ width: `${importance * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-sm text-gray-600 w-12">
                        {(importance * 100).toFixed(1)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Input Form */}
          <div className="border-t pt-6">
            <h5 className="font-medium mb-4">Enter Feature Values for Prediction</h5>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
              {modelDetails.features.map(renderFeatureInput)}
            </div>

            <button
              onClick={handlePredict}
              disabled={loading || !validateInputs()}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-green-600 to-blue-600 text-white rounded-lg hover:from-green-700 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <TrendingUp className="w-5 h-5 animate-pulse" />
                  Predicting...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  Make Prediction
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Prediction Results */}
      {prediction && (
        <div className="bg-white p-6 rounded-lg border">
          <h4 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-green-600" />
            Prediction Results
          </h4>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="p-4 bg-blue-50 rounded-lg">
                <h5 className="font-medium text-blue-900 mb-2">Predicted Value</h5>
                <p className="text-2xl font-bold text-blue-700">
                  {prediction.prediction}
                </p>
              </div>

              {prediction.confidence && (
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h5 className="font-medium text-gray-900 mb-2">Confidence Score</h5>
                  <p className={`text-2xl font-bold ${getConfidenceColor(prediction.confidence)}`}>
                    {(prediction.confidence * 100).toFixed(2)}%
                  </p>
                </div>
              )}
            </div>

            <div className="space-y-4">
              <div>
                <h5 className="font-medium mb-2">Input Values Used</h5>
                <div className="bg-gray-50 p-3 rounded text-sm">
                  {Object.entries(prediction.input_data).map(([key, value]) => (
                    <div key={key} className="flex justify-between py-1">
                      <span className="font-medium">{key}:</span>
                      <span>{value}</span>
                    </div>
                  ))}
                </div>
              </div>

              {prediction.prediction_probability && (
                <div>
                  <h5 className="font-medium mb-2">Class Probabilities</h5>
                  <div className="space-y-2">
                    {prediction.prediction_probability.map((prob, index) => (
                      <div key={index} className="flex items-center gap-2">
                        <span className="w-16 text-sm">Class {index}:</span>
                        <div className="flex-1 bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-green-600 h-2 rounded-full" 
                            style={{ width: `${prob * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-sm w-12">
                          {(prob * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionComponent;
