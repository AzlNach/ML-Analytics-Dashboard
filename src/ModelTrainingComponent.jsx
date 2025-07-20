import React, { useState, useEffect } from 'react';
import { Brain, Database, Play, Trash2, Info, CheckCircle, Clock } from 'lucide-react';
import MLAnalyticsAPI from './services/api';

const ModelTrainingComponent = ({ trainingData, onModelTrained }) => {
  const [algorithms, setAlgorithms] = useState({});
  const [trainedModels, setTrainedModels] = useState([]);
  const [modelHistory, setModelHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState('');
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('decision_tree');
  const [targetColumn, setTargetColumn] = useState('');
  const [trainingOptions, setTrainingOptions] = useState({
    test_size: 0.2,
    cross_validation: true
  });

  useEffect(() => {
    loadAlgorithms();
    loadTrainedModels();
    loadModelHistory();
  }, []);

  const loadAlgorithms = async () => {
    try {
      const algos = await MLAnalyticsAPI.getAlgorithms();
      setAlgorithms(algos);
    } catch (error) {
      console.error('Failed to load algorithms:', error);
    }
  };

  const loadTrainedModels = async () => {
    try {
      const response = await MLAnalyticsAPI.getTrainedModels();
      setTrainedModels(response.models || []);
    } catch (error) {
      console.error('Failed to load trained models:', error);
    }
  };

  const loadModelHistory = async () => {
    try {
      const response = await MLAnalyticsAPI.getModelHistory();
      setModelHistory(response.history || []);
    } catch (error) {
      console.error('Failed to load model history:', error);
    }
  };

  const handleTrainModel = async () => {
    if (!selectedFile || !targetColumn) {
      alert('Please select a file and target column');
      return;
    }

    setLoading(true);
    try {
      const result = await MLAnalyticsAPI.trainFromFile(
        selectedFile, 
        targetColumn, 
        selectedAlgorithm,
        trainingOptions
      );
      
      alert(`Model trained successfully! Accuracy: ${(result.accuracy * 100).toFixed(2)}%`);
      
      // Reload models and history
      await loadTrainedModels();
      await loadModelHistory();
      
      // Notify parent component
      if (onModelTrained) {
        onModelTrained(result);
      }
      
    } catch (error) {
      alert(`Training failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteModel = async (modelId) => {
    if (!window.confirm('Are you sure you want to delete this model?')) return;
    
    try {
      await MLAnalyticsAPI.deleteModel(modelId);
      await loadTrainedModels();
      alert('Model deleted successfully');
    } catch (error) {
      alert(`Failed to delete model: ${error.message}`);
    }
  };

  const getSelectedFileColumns = () => {
    const file = trainingData.find(f => f.filename === selectedFile);
    return file ? file.columns : [];
  };

  const getAlgorithmColor = (modelType) => {
    const colors = {
      'decision_tree': 'bg-green-100 text-green-800',
      'random_forest': 'bg-blue-100 text-blue-800',
      'logistic_regression': 'bg-purple-100 text-purple-800',
      'svm': 'bg-orange-100 text-orange-800'
    };
    return colors[modelType] || 'bg-gray-100 text-gray-800';
  };

  return (
    <div className="space-y-6">
      {/* Training Section */}
      <div className="bg-white p-6 rounded-lg border">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Brain className="w-5 h-5" />
          Train New Model
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          {/* File Selection */}
          <div>
            <label className="block text-sm font-medium mb-2">Training Dataset</label>
            <select 
              value={selectedFile} 
              onChange={(e) => {
                setSelectedFile(e.target.value);
                setTargetColumn(''); // Reset target column
              }}
              className="w-full px-3 py-2 border rounded-lg"
            >
              <option value="">Select a dataset...</option>
              {trainingData.map((file, index) => (
                <option key={index} value={file.filename}>
                  {file.filename} ({file.shape[0]} rows × {file.shape[1]} cols)
                </option>
              ))}
            </select>
          </div>

          {/* Algorithm Selection */}
          <div>
            <label className="block text-sm font-medium mb-2">Algorithm</label>
            <select 
              value={selectedAlgorithm} 
              onChange={(e) => setSelectedAlgorithm(e.target.value)}
              className="w-full px-3 py-2 border rounded-lg"
            >
              {Object.entries(algorithms).map(([key, algo]) => (
                <option key={key} value={key}>
                  {algo.name}
                </option>
              ))}
            </select>
            {algorithms[selectedAlgorithm] && (
              <p className="text-xs text-gray-600 mt-1">
                {algorithms[selectedAlgorithm].description}
              </p>
            )}
          </div>

          {/* Target Column */}
          <div>
            <label className="block text-sm font-medium mb-2">Target Column</label>
            <select 
              value={targetColumn} 
              onChange={(e) => setTargetColumn(e.target.value)}
              className="w-full px-3 py-2 border rounded-lg"
              disabled={!selectedFile}
            >
              <option value="">Select target column...</option>
              {getSelectedFileColumns().map((col, index) => (
                <option key={index} value={col}>
                  {col}
                </option>
              ))}
            </select>
          </div>

          {/* Training Options */}
          <div>
            <label className="block text-sm font-medium mb-2">Test Size</label>
            <select 
              value={trainingOptions.test_size} 
              onChange={(e) => setTrainingOptions(prev => ({
                ...prev, 
                test_size: parseFloat(e.target.value)
              }))}
              className="w-full px-3 py-2 border rounded-lg"
            >
              <option value={0.1}>10% (90% training)</option>
              <option value={0.2}>20% (80% training)</option>
              <option value={0.3}>30% (70% training)</option>
            </select>
          </div>
        </div>

        <div className="flex items-center gap-4 mb-4">
          <label className="flex items-center gap-2">
            <input 
              type="checkbox" 
              checked={trainingOptions.cross_validation}
              onChange={(e) => setTrainingOptions(prev => ({
                ...prev,
                cross_validation: e.target.checked
              }))}
              className="rounded"
            />
            <span className="text-sm">Enable Cross-Validation</span>
          </label>
        </div>

        <button
          onClick={handleTrainModel}
          disabled={loading || !selectedFile || !targetColumn}
          className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? (
            <>
              <Clock className="w-5 h-5 animate-spin" />
              Training Model...
            </>
          ) : (
            <>
              <Play className="w-5 h-5" />
              Train Model
            </>
          )}
        </button>
      </div>

      {/* Trained Models */}
      {trainedModels.length > 0 && (
        <div className="bg-white p-6 rounded-lg border">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Database className="w-5 h-5" />
            Trained Models ({trainedModels.length})
          </h3>
          
          <div className="space-y-3">
            {trainedModels.map((model) => (
              <div key={model.model_id} className="p-4 border rounded-lg hover:bg-gray-50">
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="font-medium">{model.model_id}</span>
                      <span className={`px-2 py-1 rounded-full text-xs ${getAlgorithmColor(model.model_type)}`}>
                        {algorithms[model.model_type]?.name || model.model_type}
                      </span>
                      {model.accuracy && (
                        <span className="flex items-center gap-1 text-green-600 text-sm">
                          <CheckCircle className="w-4 h-4" />
                          {(model.accuracy * 100).toFixed(2)}%
                        </span>
                      )}
                    </div>
                    
                    <div className="text-sm text-gray-600 space-y-1">
                      <p><strong>Dataset:</strong> {model.training_file}</p>
                      <p><strong>Target:</strong> {model.target_column}</p>
                      <p><strong>Features:</strong> {model.feature_count} columns</p>
                      {model.created_at && (
                        <p><strong>Created:</strong> {new Date(model.created_at).toLocaleString()}</p>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleDeleteModel(model.model_id)}
                      className="p-2 text-red-600 hover:bg-red-50 rounded"
                      title="Delete Model"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Model History */}
      {modelHistory.length > 0 && (
        <div className="bg-white p-6 rounded-lg border">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Info className="w-5 h-5" />
            Training History
          </h3>
          
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="bg-gray-50">
                  <th className="px-4 py-2 text-left">Model</th>
                  <th className="px-4 py-2 text-left">Algorithm</th>
                  <th className="px-4 py-2 text-left">Dataset</th>
                  <th className="px-4 py-2 text-left">Accuracy</th>
                  <th className="px-4 py-2 text-left">CV Score</th>
                  <th className="px-4 py-2 text-left">Trained</th>
                </tr>
              </thead>
              <tbody>
                {modelHistory.slice(0, 10).map((entry, index) => (
                  <tr key={index} className="border-t">
                    <td className="px-4 py-2 font-medium">{entry.model_id}</td>
                    <td className="px-4 py-2">
                      <span className={`px-2 py-1 rounded-full text-xs ${getAlgorithmColor(entry.model_type)}`}>
                        {algorithms[entry.model_type]?.name || entry.model_type}
                      </span>
                    </td>
                    <td className="px-4 py-2">{entry.training_file}</td>
                    <td className="px-4 py-2">
                      <span className="text-green-600 font-medium">
                        {(entry.accuracy * 100).toFixed(2)}%
                      </span>
                    </td>
                    <td className="px-4 py-2">
                      {entry.cv_mean_score ? (
                        <span className="text-blue-600">
                          {(entry.cv_mean_score * 100).toFixed(2)}% ± {(entry.cv_std_score * 100).toFixed(2)}%
                        </span>
                      ) : '-'}
                    </td>
                    <td className="px-4 py-2 text-gray-600">
                      {new Date(entry.created_at).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelTrainingComponent;
