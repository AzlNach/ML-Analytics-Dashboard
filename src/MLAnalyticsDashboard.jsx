import React, { useState, useCallback, useEffect } from 'react';
import { Upload, FileText, BarChart3, Brain, AlertTriangle, Layers, TreePine, RefreshCw, Server, Target } from 'lucide-react';
import MLAnalyticsAPI from './services/api';
import ModelTrainingComponent from './ModelTrainingComponent';
import PredictionComponent from './PredictionComponent';

const MLAnalyticsDashboard = () => {
  const [data, setData] = useState(null);
  const [fileName, setFileName] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [mlResults, setMLResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('upload');
  const [columns, setColumns] = useState([]);
  const [selectedColumns, setSelectedColumns] = useState([]);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [trainingData, setTrainingData] = useState([]);
  const [algorithms, setAlgorithms] = useState({});

  // Check backend health on component mount
  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const health = await MLAnalyticsAPI.healthCheck();
      setBackendStatus('connected');
      setTrainingData(health.training_data || []);
      setAlgorithms(health.available_algorithms || {});
    } catch (error) {
      setBackendStatus('disconnected');
      console.error('Backend connection failed:', error);
    }
  };

  // Function to parse CSV
  const parseCSV = (text) => {
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
    const rows = [];
    
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',').map(v => v.trim().replace(/"/g, ''));
      if (values.length === headers.length) {
        const row = {};
        headers.forEach((header, index) => {
          const value = values[index];
          // Try to parse as number
          const numValue = parseFloat(value);
          row[header] = isNaN(numValue) ? value : numValue;
        });
        rows.push(row);
      }
    }
    
    return { headers, rows };
  };

  // Analyze data using backend API
  const analyzeDataWithAPI = async (csvData) => {
    try {
      const result = await MLAnalyticsAPI.analyzeData(csvData);
      return result;
    } catch (error) {
      console.error('API analysis failed:', error);
      throw error;
    }
  };

  // Handle file upload
  const handleFileUpload = useCallback(async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setFileName(file.name);
    setLoading(true);

    try {
      const text = await file.text();
      const parsed = parseCSV(text);
      
      setData(parsed.rows);
      setColumns(parsed.headers);
      setSelectedColumns(parsed.headers.filter(h => {
        const firstValue = parsed.rows[0]?.[h];
        return typeof firstValue === 'number';
      }));

      // Analyze data with API
      if (backendStatus === 'connected') {
        const analysisResult = await analyzeDataWithAPI(parsed.rows);
        setAnalysis(analysisResult);
      }

      setActiveTab('overview');
    } catch (error) {
      console.error('Error processing file:', error);
      alert('Error processing file: ' + error.message);
    } finally {
      setLoading(false);
    }
  }, [backendStatus]);

  // Run ML analysis using backend
  const runMLAnalysis = async () => {
    if (!data || selectedColumns.length === 0) {
      alert('Please upload data and select columns first');
      return;
    }
    
    setLoading(true);
    
    try {
      // Clustering
      const clusteringResult = await MLAnalyticsAPI.performClustering(
        data, 
        selectedColumns,
        0.5, // eps
        5    // min_samples
      );
      
      // Anomaly Detection
      const anomalyResult = await MLAnalyticsAPI.detectAnomalies(
        data,
        selectedColumns,
        0.1
      );
      
      // Decision Tree (if there are categorical columns)
      let decisionTreeResult = null;
      const categoricalColumns = columns.filter(col => 
        analysis?.stats[col]?.type === 'categorical'
      );
      
      if (categoricalColumns.length > 0) {
        decisionTreeResult = await MLAnalyticsAPI.buildDecisionTree(
          data,
          selectedColumns,
          categoricalColumns[0] // Use first categorical column as target
        );
      }
      
      setMLResults({
        clustering: clusteringResult,
        anomalies: anomalyResult,
        decisionTree: decisionTreeResult
      });
      
      setActiveTab('clustering');
    } catch (error) {
      console.error('ML Analysis failed:', error);
      alert('ML Analysis failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  // Render Upload Tab
  const renderUploadTab = () => (
    <div className="p-6">
      <div className="max-w-2xl mx-auto">
        <div className="mb-6 p-4 rounded-lg border">
          <div className="flex items-center gap-2 mb-2">
            <Server className="w-5 h-5" />
            <span className="font-medium">Backend Status</span>
          </div>
          <div className={`flex items-center gap-2 ${
            backendStatus === 'connected' ? 'text-green-600' : 
            backendStatus === 'disconnected' ? 'text-red-600' : 'text-yellow-600'
          }`}>
            <div className={`w-2 h-2 rounded-full ${
              backendStatus === 'connected' ? 'bg-green-500' : 
              backendStatus === 'disconnected' ? 'bg-red-500' : 'bg-yellow-500'
            }`}></div>
            <span>
              {backendStatus === 'connected' ? 'Connected to Flask API' :
               backendStatus === 'disconnected' ? 'Disconnected - API not available' :
               'Checking connection...'}
            </span>
          </div>
          {backendStatus === 'disconnected' && (
            <button 
              onClick={checkBackendHealth}
              className="mt-2 px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"
            >
              Retry Connection
            </button>
          )}
        </div>

        <div className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:border-blue-400 transition-colors">
          <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">Upload CSV Dataset</h3>
          <p className="text-gray-500 mb-4">
            Upload your CSV file to begin machine learning analysis
          </p>
          <input
            type="file"
            accept=".csv"
            onChange={handleFileUpload}
            className="hidden"
            id="file-upload"
          />
          <label
            htmlFor="file-upload"
            className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 cursor-pointer"
          >
            <Upload className="w-4 h-4 mr-2" />
            Choose CSV File
          </label>
        </div>

        {Object.keys(algorithms).length > 0 && (
          <div className="mt-8">
            <h3 className="text-lg font-medium mb-4 flex items-center gap-2">
              <Brain className="w-5 h-5" />
              Available ML Algorithms
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(algorithms).map(([key, algo]) => (
                <div key={key} className="p-4 border rounded-lg hover:bg-gray-50">
                  <h4 className="font-medium text-blue-600">{algo.name}</h4>
                  <p className="text-sm text-gray-600 mt-1">{algo.description}</p>
                  <p className="text-xs text-gray-500 mt-2">
                    <strong>Best for:</strong> {algo.best_for}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto py-6 px-4">
        <h1 className="text-3xl font-bold text-gray-900 mb-8 flex items-center">
          <Brain className="w-8 h-8 mr-3 text-blue-600" />
          ML Analytics Dashboard
        </h1>
        
        {/* Tab Navigation */}
        <div className="mb-6">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8">
              {[
                { id: 'upload', name: 'Upload Data', icon: Upload },
                { id: 'overview', name: 'Data Overview', icon: FileText },
                { id: 'visualization', name: 'Visualization', icon: BarChart3 },
                { id: 'clustering', name: 'Clustering', icon: Layers },
                { id: 'anomalies', name: 'Anomalies', icon: AlertTriangle },
                { id: 'decision-tree', name: 'Decision Tree', icon: TreePine },
                { id: 'training', name: 'Model Training', icon: Brain },
                { id: 'prediction', name: 'Predictions', icon: Target }
              ].map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  } whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm flex items-center gap-2`}
                >
                  <tab.icon className="w-4 h-4" />
                  {tab.name}
                </button>
              ))}
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'upload' && renderUploadTab()}
        
        {activeTab === 'training' && (
          <ModelTrainingComponent 
            trainingData={trainingData}
            onModelTrained={() => {
              // Refresh model history when a new model is trained
              checkBackendHealth();
            }}
          />
        )}
        
        {activeTab === 'prediction' && (
          <PredictionComponent />
        )}
        
        {/* Simple content for other tabs for now */}
        {activeTab !== 'upload' && activeTab !== 'training' && activeTab !== 'prediction' && (
          <div className="bg-white p-6 rounded-lg border">
            <h2 className="text-xl font-semibold mb-4">
              {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} Analysis
            </h2>
            <p className="text-gray-600">
              {data ? 
                `Data loaded: ${data.length} rows, ${columns.length} columns` :
                'Please upload a CSV file first to see analysis results here.'
              }
            </p>
            {data && (
              <div className="mt-4">
                <button
                  onClick={runMLAnalysis}
                  disabled={loading || selectedColumns.length === 0}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
                >
                  {loading ? 'Analyzing...' : 'Run ML Analysis'}
                </button>
              </div>
            )}
          </div>
        )}

        {/* Loading indicator */}
        {loading && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded-lg flex items-center gap-3">
              <RefreshCw className="w-6 h-6 animate-spin text-blue-600" />
              <span>Processing...</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MLAnalyticsDashboard;
