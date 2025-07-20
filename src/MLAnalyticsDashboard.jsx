import React, { useState, useCallback, useEffect } from 'react';
import { Upload, FileText, BarChart3, Brain, AlertTriangle, Layers, TreePine, RefreshCw, Server, Target, Download } from 'lucide-react';
import { ResponsiveContainer, BarChart, ScatterChart, CartesianGrid, XAxis, YAxis, Tooltip, Bar, Scatter } from 'recharts';
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

  // Function to clean data for API calls
  const cleanDataForAPI = (data) => {
    console.log('Cleaning data for API, input length:', data.length);
    
    const cleanedData = data.map((row, rowIndex) => {
      const cleanRow = {};
      Object.keys(row).forEach(key => {
        let value = row[key];
        
        try {
          // Handle various data types
          if (value === null || value === undefined) {
            cleanRow[key] = null;
          } else if (typeof value === 'number' && (isNaN(value) || !isFinite(value))) {
            cleanRow[key] = null;
          } else if (Array.isArray(value)) {
            // Convert arrays to strings or take first element
            if (value.length > 0) {
              const firstVal = value[0];
              // If first element is a number, use it; otherwise convert to string
              cleanRow[key] = typeof firstVal === 'number' ? firstVal : String(firstVal);
            } else {
              cleanRow[key] = null;
            }
          } else if (typeof value === 'object' && value !== null) {
            // Convert objects to string representation
            cleanRow[key] = String(value);
          } else if (typeof value === 'string') {
            // Handle string values that might represent NaN or empty
            const lowerValue = value.toLowerCase().trim();
            if (value === '' || lowerValue === 'nan' || lowerValue === 'null' || lowerValue === 'undefined') {
              cleanRow[key] = null;
            } else if (value.startsWith('[') || value.startsWith('{')) {
              // Handle string representations of arrays/objects
              try {
                const parsed = JSON.parse(value);
                if (Array.isArray(parsed)) {
                  cleanRow[key] = parsed.length > 0 ? parsed[0] : null;
                } else if (typeof parsed === 'object') {
                  cleanRow[key] = String(parsed);
                } else {
                  cleanRow[key] = parsed;
                }
              } catch {
                // If JSON parsing fails, keep as string
                cleanRow[key] = value;
              }
            } else {
              // Try to parse as number if it looks like one
              const numValue = parseFloat(value);
              if (!isNaN(numValue) && isFinite(numValue) && value.trim() === String(numValue)) {
                cleanRow[key] = numValue;
              } else {
                cleanRow[key] = value;
              }
            }
          } else {
            cleanRow[key] = value;
          }
        } catch (error) {
          console.warn(`Error cleaning value for ${key} in row ${rowIndex}:`, error);
          cleanRow[key] = null;
        }
      });
      return cleanRow;
    });
    
    console.log('Data cleaning completed, output length:', cleanedData.length);
    if (cleanedData.length > 0) {
      console.log('Sample cleaned row:', cleanedData[0]);
    }
    
    return cleanedData;
  };

  // Function to parse CSV
  const parseCSV = (text) => {
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
    const rows = [];
    
    for (let i = 1; i < lines.length; i++) {
      // Handle comma within quoted values properly
      const line = lines[i];
      const values = [];
      let current = '';
      let inQuotes = false;
      
      for (let j = 0; j < line.length; j++) {
        const char = line[j];
        if (char === '"') {
          inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
          values.push(current.trim().replace(/^"|"$/g, ''));
          current = '';
        } else {
          current += char;
        }
      }
      values.push(current.trim().replace(/^"|"$/g, ''));
      
      if (values.length === headers.length) {
        const row = {};
        headers.forEach((header, index) => {
          const value = values[index];
          
          // Handle empty values
          if (value === '' || value.toLowerCase() === 'nan' || value.toLowerCase() === 'null' || value.toLowerCase() === 'undefined') {
            row[header] = null;
            return;
          }
          
          // Try to parse as number
          const numValue = parseFloat(value);
          if (!isNaN(numValue) && isFinite(numValue)) {
            row[header] = numValue;
          } else {
            row[header] = value;
          }
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
        try {
          const cleanedData = cleanDataForAPI(parsed.rows);
          const analysisResult = await analyzeDataWithAPI(cleanedData);
          setAnalysis(analysisResult);
        } catch (analysisError) {
          console.warn('Analysis failed, but continuing with basic functionality:', analysisError);
          // Set basic analysis structure so the app doesn't break
          setAnalysis({
            stats: {},
            correlation_matrix: {},
            shape: [parsed.rows.length, parsed.headers.length],
            columns: parsed.headers,
            data_types: {}
          });
        }
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
      // Clean data before sending to API
      const cleanedData = cleanDataForAPI(data);
      
      // Ensure selectedColumns is an array and contains valid columns
      const cleanedSelectedColumns = Array.isArray(selectedColumns) ? 
        selectedColumns.filter(col => typeof col === 'string' && col.length > 0) : [];
      
      if (cleanedSelectedColumns.length === 0) {
        throw new Error('No valid columns selected for analysis');
      }
      
      console.log('Starting ML analysis with:', {
        dataRows: cleanedData.length,
        selectedColumns: cleanedSelectedColumns,
        sampleData: cleanedData.slice(0, 2)
      });
      
      // Clustering
      console.log('Starting clustering analysis...');
      const clusteringResult = await MLAnalyticsAPI.performClustering(
        cleanedData, 
        cleanedSelectedColumns,
        0.5, // eps
        5    // min_samples
      );
      console.log('Clustering result received:', clusteringResult);
      
      // Anomaly Detection
      console.log('Starting anomaly detection...');
      const anomalyResult = await MLAnalyticsAPI.detectAnomalies(
        cleanedData,
        cleanedSelectedColumns,
        0.1 // contamination
      );
      console.log('Anomaly detection result received:', anomalyResult);
      
      // Decision Tree - Use a more flexible approach for target column selection
      let decisionTreeResult = null;
      
      // Look for categorical columns first, but also consider high-cardinality numeric columns
      const categoricalColumns = columns.filter(col => 
        analysis?.stats[col]?.type === 'categorical'
      );
      
      // Also consider columns that might be IDs or have high cardinality (good targets for classification)
      const potentialTargetColumns = columns.filter(col => {
        const colLower = col.toLowerCase();
        const stats = analysis?.stats[col];
        return (
          // Explicitly categorical
          stats?.type === 'categorical' ||
          // ID columns (often good targets despite being numeric)
          colLower.includes('id') ||
          colLower.includes('tag') ||
          colLower.includes('class') ||
          colLower.includes('category') ||
          colLower.includes('type') ||
          // High cardinality numeric columns that might represent categories
          (stats?.type === 'numeric' && stats?.unique && stats.unique > 10 && stats.unique < data.length * 0.8)
        );
      });
      
      console.log('Analysis stats:', analysis?.stats);
      console.log('All columns:', columns);
      console.log('Categorical columns:', categoricalColumns);
      console.log('Potential target columns:', potentialTargetColumns);
      console.log('Selected columns for ML:', cleanedSelectedColumns);
      
      // Use potential target columns if available, otherwise try the first categorical
      const availableTargets = potentialTargetColumns.length > 0 ? potentialTargetColumns : categoricalColumns;
      
      if (availableTargets.length > 0) {
        try {
          // Prefer columns that are not in selectedColumns (avoid using features as targets)
          let targetColumn = availableTargets.find(col => !cleanedSelectedColumns.includes(col)) || availableTargets[0];
          
          console.log('Using target column:', targetColumn);
          console.log('Target column type:', typeof targetColumn);
          console.log('Target column value:', JSON.stringify(targetColumn));
          
          // Ensure targetColumn is a string
          const safeTargetColumn = String(targetColumn);
          console.log('Safe target column:', safeTargetColumn);
          
          // Validate that the selected features don't include the target
          const validFeatures = cleanedSelectedColumns.filter(col => col !== safeTargetColumn);
          
          if (validFeatures.length === 0) {
            console.warn('No valid features for decision tree after excluding target');
            // If no valid features, use all numeric columns except the target
            const numericColumns = columns.filter(col => {
              const stats = analysis?.stats[col];
              return stats?.type === 'numeric' && col !== safeTargetColumn;
            });
            if (numericColumns.length > 0) {
              console.log('Using all numeric columns as features:', numericColumns);
              decisionTreeResult = await MLAnalyticsAPI.buildDecisionTree(
                cleanedData,
                safeTargetColumn,
                numericColumns,
                5 // max_depth
              );
            }
          } else {
            console.log('Building decision tree with features:', validFeatures);
            
            decisionTreeResult = await MLAnalyticsAPI.buildDecisionTree(
              cleanedData,
              safeTargetColumn,
              validFeatures,
              5 // max_depth
            );
          }
          
          console.log('Decision tree result:', decisionTreeResult);
        } catch (error) {
          console.error('Decision tree failed:', error);
          // Don't throw the error, just log it and continue with other analyses
        }
      } else {
        console.log('No suitable target columns found for decision tree analysis');
        console.log('Available column types:', Object.fromEntries(
          columns.map(col => [col, analysis?.stats[col]?.type || 'unknown'])
        ));
      }
      
      console.log('Setting ML Results:', {
        clustering: clusteringResult,
        anomalies: anomalyResult,
        decisionTree: decisionTreeResult
      });
      
      setMLResults({
        clustering: clusteringResult,
        anomalies: anomalyResult,
        decisionTree: decisionTreeResult
      });
      
      console.log('ML Results set successfully:', {
        clustering: !!clusteringResult,
        anomalies: !!anomalyResult,
        decisionTree: !!decisionTreeResult,
        clusteringKeys: clusteringResult ? Object.keys(clusteringResult) : [],
        anomaliesKeys: anomalyResult ? Object.keys(anomalyResult) : [],
        decisionTreeKeys: decisionTreeResult ? Object.keys(decisionTreeResult) : []
      });
      
      setActiveTab('clustering');
    } catch (error) {
      console.error('ML Analysis failed:', error);
      const errorMessage = error.message || 'Unknown error occurred';
      alert(`ML Analysis failed: ${errorMessage}`);
    } finally {
      setLoading(false);
    }
  };

  // Export results
  const exportResults = () => {
    const exportData = {
      fileName,
      analysis,
      mlResults,
      timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ml_analysis_${fileName.replace('.csv', '')}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Render components
  const renderUploadTab = () => (
    <div className="p-6">
      <div className="max-w-2xl mx-auto">
        {/* Backend Status */}
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

        {/* File Upload */}
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

        {/* Available Algorithms */}
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

  const renderTrainingTab = () => (
    <div className="p-6">
      <ModelTrainingComponent 
        trainingData={trainingData}
        onModelTrained={(result) => {
          console.log('Model trained:', result);
          // Optionally refresh some state here
        }}
      />
    </div>
  );

  const renderPredictionTab = () => (
    <div className="p-6">
      <PredictionComponent />
    </div>
  );

  const renderOverviewTab = () => (
    <div className="p-6">
      {/* Debug information */}
      <div className="mb-4 p-3 bg-gray-100 rounded text-sm">
        <strong>Debug Info:</strong> data = {data ? `${data.length} rows` : 'null'}, 
        analysis = {analysis ? 'exists' : 'null'}, 
        mlResults = {mlResults ? 'exists' : 'null'}
        {analysis && (
          <div>Analysis keys: {Object.keys(analysis).join(', ')}</div>
        )}
        {analysis?.stats && (
          <div>Stats keys: {Object.keys(analysis.stats).slice(0, 5).join(', ')}...</div>
        )}
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div className="bg-white p-6 rounded-lg border">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <FileText className="w-5 h-5 mr-2" />
            Dataset Information
          </h3>
          <div className="space-y-2">
            <p><span className="font-medium">File:</span> {fileName}</p>
            <p><span className="font-medium">Rows:</span> {data?.length || 0}</p>
            <p><span className="font-medium">Columns:</span> {columns.length}</p>
            <p><span className="font-medium">Backend:</span> 
              <span className={backendStatus === 'connected' ? 'text-green-600' : 'text-red-600'}>
                {backendStatus === 'connected' ? ' Connected' : ' Disconnected'}
              </span>
            </p>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg border">
          <h3 className="text-lg font-semibold mb-4">Column Selection for ML</h3>
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {columns.map(col => (
              <label key={col} className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={selectedColumns.includes(col)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setSelectedColumns([...selectedColumns, col]);
                    } else {
                      setSelectedColumns(selectedColumns.filter(c => c !== col));
                    }
                  }}
                  className="rounded"
                />
                <span className="text-sm">{col}</span>
                <span className="text-xs text-gray-500">
                  ({analysis?.stats[col]?.type || 'unknown'})
                </span>
              </label>
            ))}
          </div>
        </div>
      </div>

      {analysis && (
        <div className="bg-white p-6 rounded-lg border mb-6">
          <h3 className="text-lg font-semibold mb-4">Statistical Summary</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full table-auto">
              <thead>
                <tr className="bg-gray-50">
                  <th className="px-4 py-2 text-left">Column</th>
                  <th className="px-4 py-2 text-left">Type</th>
                  <th className="px-4 py-2 text-left">Count</th>
                  <th className="px-4 py-2 text-left">Summary</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(analysis.stats).map(([col, stats]) => (
                  <tr key={col} className="border-t">
                    <td className="px-4 py-2 font-medium">{col}</td>
                    <td className="px-4 py-2">
                      <span className={`px-2 py-1 rounded-full text-xs ${
                        stats.type === 'numeric' ? 'bg-blue-100 text-blue-800' : 'bg-green-100 text-green-800'
                      }`}>
                        {stats.type}
                      </span>
                    </td>
                    <td className="px-4 py-2">{stats.count}</td>
                    <td className="px-4 py-2 text-sm">
                      {stats.type === 'numeric' ? (
                        `Mean: ${stats.mean ? stats.mean.toFixed(2) : 'N/A'}, Std: ${stats.std ? stats.std.toFixed(2) : 'N/A'}`
                      ) : (
                        `Unique: ${stats.unique || 0}, Most common: ${Object.keys(stats.most_common || {})[0] || 'N/A'}`
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="flex gap-4">
        <button
          onClick={runMLAnalysis}
          disabled={loading || selectedColumns.length === 0 || backendStatus !== 'connected'}
          className="flex items-center px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:from-purple-700 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? <RefreshCw className="w-5 h-5 mr-2 animate-spin" /> : <Brain className="w-5 h-5 mr-2" />}
          {loading ? 'Running Analysis...' : 'Run ML Analysis'}
        </button>
        
        {analysis && (
          <button
            onClick={exportResults}
            className="flex items-center px-4 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
          >
            <Download className="w-5 h-5 mr-2" />
            Export Results
          </button>
        )}
      </div>
    </div>
  );

  const renderVisualizationTab = () => (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">Data Visualization</h2>
      
      {analysis && data && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Correlation Heatmap */}
          {Object.keys(analysis.correlation_matrix).length > 0 && (
            <div className="bg-white p-6 rounded-lg border">
              <h3 className="text-lg font-semibold mb-4">Correlation Matrix</h3>
              <div className="text-sm text-gray-600">
                Correlation data available - visualization would be implemented with a heatmap library
              </div>
            </div>
          )}

          {/* Distribution charts for numeric columns */}
          {selectedColumns.slice(0, 4).map(col => {
            const columnStats = analysis.stats[col];
            if (columnStats?.type !== 'numeric') return null;

            return (
              <div key={col} className="bg-white p-6 rounded-lg border">
                <h3 className="text-lg font-semibold mb-4">{col} Distribution</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={data.slice(0, 20)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey={col} />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey={col} fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );

  const renderClusteringTab = () => (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">Clustering Analysis</h2>
      
      {/* Debug information */}
      <div className="mb-4 p-3 bg-gray-100 rounded text-sm">
        <strong>Debug Info:</strong> mlResults = {mlResults ? 'exists' : 'null'}, 
        clustering = {mlResults?.clustering ? 'exists' : 'null'}
        {mlResults?.clustering && (
          <div>Keys: {Object.keys(mlResults.clustering).join(', ')}</div>
        )}
      </div>
      
      {mlResults?.clustering ? (
        <div className="space-y-6">
          {/* Dataset Information */}
          {mlResults.clustering.info && mlResults.clustering.info.was_sampled && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex items-center">
                <AlertTriangle className="w-5 h-5 text-blue-600 mr-2" />
                <span className="text-blue-800">
                  Large dataset detected. Analysis performed on a sample of {mlResults.clustering.info.processed_size} out of {mlResults.clustering.info.original_size} rows for optimal performance.
                </span>
              </div>
            </div>
          )}

          <div className="bg-white p-6 rounded-lg border">
            <h3 className="text-lg font-semibold mb-4">Cluster Statistics</h3>
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {mlResults.clustering.cluster_stats?.num_clusters || 'N/A'}
                </div>
                <div className="text-sm text-gray-600">Clusters Found</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">
                  {mlResults.clustering.cluster_stats?.num_noise_points || 'N/A'}
                </div>
                <div className="text-sm text-gray-600">Noise Points</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {mlResults.clustering.clustered_data?.length || 'N/A'}
                </div>
                <div className="text-sm text-gray-600">Total Analyzed Points</div>
              </div>
            </div>
          </div>

          {mlResults.clustering.clustered_data && selectedColumns.length >= 2 && (
            <div className="bg-white p-6 rounded-lg border">
              <h3 className="text-lg font-semibold mb-4">Cluster Visualization</h3>
              <ResponsiveContainer width="100%" height={400}>
                <ScatterChart data={mlResults.clustering.clustered_data.slice(0, 100)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey={selectedColumns[0]} 
                    type="number"
                    domain={['dataMin', 'dataMax']}
                  />
                  <YAxis 
                    dataKey={selectedColumns[1] || selectedColumns[0]} 
                    type="number"
                    domain={['dataMin', 'dataMax']}
                  />
                  <Tooltip />
                  <Scatter dataKey="cluster" fill="#8884d8" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      ) : (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertTriangle className="w-5 h-5 text-yellow-600 mr-2" />
            <span className="text-yellow-800">
              No clustering results available. Please run ML analysis first by uploading data and clicking "Run ML Analysis" in the Overview tab.
            </span>
          </div>
        </div>
      )}
    </div>
  );

  const renderAnomalyTab = () => (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">Anomaly Detection</h2>
      
      {/* Debug information */}
      <div className="mb-4 p-3 bg-gray-100 rounded text-sm">
        <strong>Debug Info:</strong> mlResults = {mlResults ? 'exists' : 'null'}, 
        anomalies = {mlResults?.anomalies ? 'exists' : 'null'}
        {mlResults?.anomalies && (
          <div>Keys: {Object.keys(mlResults.anomalies).join(', ')}</div>
        )}
      </div>
      
      {mlResults?.anomalies ? (
        <div className="space-y-6">
          {/* Dataset Information */}
          {mlResults.anomalies.info && mlResults.anomalies.info.was_sampled && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex items-center">
                <AlertTriangle className="w-5 h-5 text-blue-600 mr-2" />
                <span className="text-blue-800">
                  Large dataset detected. Analysis performed on a sample of {mlResults.anomalies.info.processed_size} out of {mlResults.anomalies.info.original_size} rows for optimal performance.
                </span>
              </div>
            </div>
          )}

          <div className="bg-white p-6 rounded-lg border">
            <h3 className="text-lg font-semibold mb-4">Anomaly Statistics</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600">
                  {mlResults.anomalies.anomaly_stats?.total_anomalies || 'N/A'}
                </div>
                <div className="text-sm text-gray-600">Anomalies Detected</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {mlResults.anomalies.anomaly_stats?.anomaly_percentage?.toFixed(2) || 'N/A'}%
                </div>
                <div className="text-sm text-gray-600">Anomaly Rate</div>
              </div>
            </div>
          </div>

          {mlResults.anomalies.data_with_anomalies && selectedColumns.length >= 2 && (
            <div className="bg-white p-6 rounded-lg border">
              <h3 className="text-lg font-semibold mb-4">Anomaly Visualization</h3>
              <ResponsiveContainer width="100%" height={400}>
                <ScatterChart data={mlResults.anomalies.data_with_anomalies.slice(0, 100)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey={selectedColumns[0]} 
                    type="number"
                    domain={['dataMin', 'dataMax']}
                  />
                  <YAxis 
                    dataKey={selectedColumns[1] || selectedColumns[0]} 
                    type="number"
                    domain={['dataMin', 'dataMax']}
                  />
                  <Tooltip />
                  <Scatter 
                    dataKey="anomaly_score" 
                    fill={(entry) => entry.is_anomaly ? "#ef4444" : "#3b82f6"} 
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      ) : (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertTriangle className="w-5 h-5 text-yellow-600 mr-2" />
            <span className="text-yellow-800">
              No anomaly detection results available. Please run ML analysis first by uploading data and clicking "Run ML Analysis" in the Overview tab.
            </span>
          </div>
        </div>
      )}
    </div>
  );

  const renderDecisionTreeTab = () => (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">Decision Tree Analysis</h2>
      
      {/* Debug information */}
      <div className="mb-4 p-3 bg-gray-100 rounded text-sm">
        <strong>Debug Info:</strong> mlResults = {mlResults ? 'exists' : 'null'}, 
        decisionTree = {mlResults?.decisionTree ? 'exists' : 'null'}
        {mlResults?.decisionTree && (
          <div>Keys: {Object.keys(mlResults.decisionTree).join(', ')}</div>
        )}
      </div>
      
      {mlResults?.decisionTree ? (
        <div className="space-y-6">
          {/* Dataset Information */}
          {mlResults.decisionTree.info && mlResults.decisionTree.info.was_sampled && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex items-center">
                <AlertTriangle className="w-5 h-5 text-blue-600 mr-2" />
                <span className="text-blue-800">
                  Large dataset detected. Model trained on a sample of {mlResults.decisionTree.info.processed_size} out of {mlResults.decisionTree.info.original_size} rows for optimal performance.
                </span>
              </div>
            </div>
          )}

          <div className="bg-white p-6 rounded-lg border">
            <h3 className="text-lg font-semibold mb-4">Model Performance</h3>
            <div className="text-center">
              <div className="text-3xl font-bold text-green-600">
                {((mlResults.decisionTree.accuracy || 0) * 100).toFixed(2)}%
              </div>
              <div className="text-sm text-gray-600">Accuracy</div>
            </div>
          </div>

          {mlResults.decisionTree.feature_importance && (
            <div className="bg-white p-6 rounded-lg border">
              <h3 className="text-lg font-semibold mb-4">Feature Importance</h3>
              <div className="space-y-2">
                {Object.entries(mlResults.decisionTree.feature_importance)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 5)
                  .map(([feature, importance]) => (
                    <div key={feature} className="flex justify-between items-center">
                      <span className="font-medium">{feature}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-32 bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-blue-600 h-2 rounded-full" 
                            style={{ width: `${importance * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-sm text-gray-600">
                          {(importance * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {mlResults.decisionTree.tree_rules && (
            <div className="bg-white p-6 rounded-lg border">
              <h3 className="text-lg font-semibold mb-4">Tree Structure</h3>
              <pre className="text-xs bg-gray-50 p-4 rounded overflow-x-auto">
                {mlResults.decisionTree.tree_rules}
              </pre>
            </div>
          )}
        </div>
      ) : (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertTriangle className="w-5 h-5 text-yellow-600 mr-2" />
            <span className="text-yellow-800">
              No decision tree results available. Please run ML analysis first by uploading data and clicking "Run ML Analysis" in the Overview tab.
            </span>
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">ML Analytics Dashboard</h1>
              <p className="text-gray-600">Advanced Machine Learning Analysis with Python Backend</p>
            </div>
            <div className="flex items-center space-x-4">
              {fileName && (
                <span className="text-sm text-gray-500">
                  Current file: {fileName}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-white rounded-lg shadow">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8">
              {[
                { id: 'upload', name: 'Upload Data', icon: Upload },
                { id: 'training', name: 'Train Models', icon: Brain },
                { id: 'prediction', name: 'Predictions', icon: Target },
                { id: 'overview', name: 'Overview', icon: FileText },
                { id: 'visualization', name: 'Visualization', icon: BarChart3 },
                { id: 'clustering', name: 'Clustering', icon: Layers },
                { id: 'anomaly', name: 'Anomaly Detection', icon: AlertTriangle },
                { id: 'decision-tree', name: 'Decision Tree', icon: TreePine },
              ].map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center`}
                >
                  <tab.icon className="w-4 h-4 mr-2" />
                  {tab.name}
                </button>
              ))}
            </nav>
          </div>

          <div className="min-h-96">
            {activeTab === 'upload' && renderUploadTab()}
            {activeTab === 'training' && renderTrainingTab()}
            {activeTab === 'prediction' && renderPredictionTab()}
            {activeTab === 'overview' && renderOverviewTab()}
            {activeTab === 'visualization' && renderVisualizationTab()}
            {activeTab === 'clustering' && renderClusteringTab()}
            {activeTab === 'anomaly' && renderAnomalyTab()}
            {activeTab === 'decision-tree' && renderDecisionTreeTab()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MLAnalyticsDashboard;