import React, { useState, useCallback, useEffect } from 'react';
import { Upload, FileText, BarChart3, Brain, AlertTriangle, Layers, TreePine, RefreshCw, Server, Target, Download, CheckCircle, XCircle, Clock, Zap, TrendingUp, Database, Settings, Play } from 'lucide-react';
import { ResponsiveContainer, BarChart, ScatterChart, CartesianGrid, XAxis, YAxis, Tooltip, Bar, Scatter, LineChart, Line, PieChart, Pie, Cell } from 'recharts';
import MLAnalyticsAPI from './services/api';
import ModelTrainingComponent from './ModelTrainingComponent';
import PredictionComponent from './PredictionComponent';
import './dashboard.css';

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
  
  // Workflow states - 5 Step Interactive Data Analysis System
  const [workflowStep, setWorkflowStep] = useState(1); // 1-5 for each step
  const [cleanedData, setCleanedData] = useState(null);
  const [trainedModel, setTrainedModel] = useState(null);
  const [dataQualityReport, setDataQualityReport] = useState(null);
  const [cleaningOptions, setCleaningOptions] = useState({
    missingValues: 'fill_mean',
    duplicates: 'remove',
    outliers: 'keep'
  });
  
  // Step-specific states
  const [cleanedDataFile, setCleanedDataFile] = useState(null); // dataset_cleaned.csv
  const [trainedModelFile, setTrainedModelFile] = useState(null); // model.pkl
  const [trainingResultsFile, setTrainingResultsFile] = useState(null); // dataset_trained_results.csv

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

  // Generate data quality report
  const generateDataQualityReport = (rows, analysisResult) => {
    const report = {
      totalRows: rows.length,
      totalColumns: columns.length,
      missingValues: {},
      duplicates: 0,
      outliers: {},
      dataTypes: analysisResult?.data_types || {}
    };

    // Count missing values per column
    columns.forEach(col => {
      const missing = rows.filter(row => 
        row[col] === null || row[col] === undefined || row[col] === ''
      ).length;
      report.missingValues[col] = missing;
    });

    // Count duplicates (simple row comparison)
    const seen = new Set();
    report.duplicates = rows.filter(row => {
      const key = JSON.stringify(row);
      if (seen.has(key)) return true;
      seen.add(key);
      return false;
    }).length;

    // Simple outlier detection for numeric columns
    columns.forEach(col => {
      const values = rows.map(row => row[col]).filter(val => 
        typeof val === 'number' && !isNaN(val)
      );
      
      if (values.length > 0) {
        values.sort((a, b) => a - b);
        const q1 = values[Math.floor(values.length * 0.25)];
        const q3 = values[Math.floor(values.length * 0.75)];
        const iqr = q3 - q1;
        const lowerBound = q1 - 1.5 * iqr;
        const upperBound = q3 + 1.5 * iqr;
        
        const outlierCount = values.filter(val => 
          val < lowerBound || val > upperBound
        ).length;
        
        if (outlierCount > 0) {
          report.outliers[col] = outlierCount;
        }
      }
    });

    return report;
  };

  // Apply data cleaning based on user choices
  const applyDataCleaning = async () => {
    if (!data || !dataQualityReport) return;

    setLoading(true);
    try {
      let cleaned = [...data];

      // Handle missing values
      if (cleaningOptions.missingValues === 'remove_rows') {
        cleaned = cleaned.filter(row => 
          !Object.values(row).some(val => 
            val === null || val === undefined || val === ''
          )
        );
      } else if (cleaningOptions.missingValues === 'fill_mean') {
        // Calculate means for numeric columns
        const means = {};
        columns.forEach(col => {
          const values = cleaned.map(row => row[col]).filter(val => 
            typeof val === 'number' && !isNaN(val)
          );
          if (values.length > 0) {
            means[col] = values.reduce((a, b) => a + b, 0) / values.length;
          }
        });

        // Fill missing values
        cleaned = cleaned.map(row => {
          const newRow = {...row};
          columns.forEach(col => {
            if (newRow[col] === null || newRow[col] === undefined || newRow[col] === '') {
              if (means[col] !== undefined) {
                newRow[col] = means[col];
              } else {
                newRow[col] = 'Unknown';
              }
            }
          });
          return newRow;
        });
      } else if (cleaningOptions.missingValues === 'fill_mode') {
        // Calculate modes for each column
        const modes = {};
        columns.forEach(col => {
          const values = cleaned.map(row => row[col]).filter(val => 
            val !== null && val !== undefined && val !== ''
          );
          if (values.length > 0) {
            const counts = {};
            values.forEach(val => counts[val] = (counts[val] || 0) + 1);
            modes[col] = Object.keys(counts).reduce((a, b) => 
              counts[a] > counts[b] ? a : b
            );
          }
        });

        // Fill missing values with mode
        cleaned = cleaned.map(row => {
          const newRow = {...row};
          columns.forEach(col => {
            if (newRow[col] === null || newRow[col] === undefined || newRow[col] === '') {
              newRow[col] = modes[col] || 'Unknown';
            }
          });
          return newRow;
        });
      }

      // Handle duplicates
      if (cleaningOptions.duplicates === 'remove') {
        const seen = new Set();
        cleaned = cleaned.filter(row => {
          const key = JSON.stringify(row);
          if (seen.has(key)) return false;
          seen.add(key);
          return true;
        });
      }

      // Handle outliers
      if (cleaningOptions.outliers === 'remove') {
        const outlierBounds = {};
        
        // Calculate bounds for numeric columns
        columns.forEach(col => {
          const values = cleaned.map(row => row[col]).filter(val => 
            typeof val === 'number' && !isNaN(val)
          );
          
          if (values.length > 0) {
            values.sort((a, b) => a - b);
            const q1 = values[Math.floor(values.length * 0.25)];
            const q3 = values[Math.floor(values.length * 0.75)];
            const iqr = q3 - q1;
            outlierBounds[col] = {
              lower: q1 - 1.5 * iqr,
              upper: q3 + 1.5 * iqr
            };
          }
        });

        // Remove rows with outliers
        cleaned = cleaned.filter(row => {
          return !columns.some(col => {
            const val = row[col];
            const bounds = outlierBounds[col];
            return bounds && (val < bounds.lower || val > bounds.upper);
          });
        });
      }

      setCleanedData(cleaned);
      
      // Auto-generate and save cleaned dataset file
      const cleanedFileName = `${fileName.replace('.csv', '')}_cleaned.csv`;
      const csvContent = generateCSVContent(cleaned, columns);
      setCleanedDataFile({
        name: cleanedFileName,
        content: csvContent,
        data: cleaned
      });

      setWorkflowStep(4); // Move to modeling step
      setActiveTab('modeling');
      
      alert(`Data cleaning completed! Dataset size: ${data.length} → ${cleaned.length} rows\nCleaned dataset saved as: ${cleanedFileName}`);
      
    } catch (error) {
      console.error('Data cleaning failed:', error);
      alert('Data cleaning failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  // Helper function to get current dataset (cleaned or original)
  const getCurrentDataset = () => {
    return cleanedData || data;
  };

  // Helper function to get current dataset name
  const getCurrentDatasetName = () => {
    if (cleanedData && cleanedDataFile) {
      return cleanedDataFile.name;
    }
    return fileName;
  };

  // Generate CSV content from data
  const generateCSVContent = (data, headers) => {
    const csvRows = [headers.join(',')];
    data.forEach(row => {
      const values = headers.map(header => {
        const value = row[header];
        // Escape commas and quotes in values
        if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
          return `"${value.replace(/"/g, '""')}"`;
        }
        return value;
      });
      csvRows.push(values.join(','));
    });
    return csvRows.join('\n');
  };

  // Handle file upload - Step 1: Data Upload
  const handleFileUpload = useCallback(async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setFileName(file.name);
    setLoading(true);
    setWorkflowStep(1);

    try {
      const text = await file.text();
      const parsed = parseCSV(text);
      
      setData(parsed.rows);
      setColumns(parsed.headers);
      setSelectedColumns(parsed.headers.filter(h => {
        const firstValue = parsed.rows[0]?.[h];
        return typeof firstValue === 'number';
      }));

      // Automatically proceed to Step 2: Analysis & Understanding
      if (backendStatus === 'connected') {
        try {
          const cleanedData = cleanDataForAPI(parsed.rows);
          const analysisResult = await analyzeDataWithAPI(cleanedData);
          setAnalysis(analysisResult);
          
          // Generate data quality report
          const qualityReport = generateDataQualityReport(parsed.rows, analysisResult);
          setDataQualityReport(qualityReport);
          
          setWorkflowStep(2);
          setActiveTab('overview');
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
          
          // Generate basic quality report
          const qualityReport = generateDataQualityReport(parsed.rows, {});
          setDataQualityReport(qualityReport);
          
          setWorkflowStep(2);
          setActiveTab('overview');
        }
      }

    } catch (error) {
      console.error('Error processing file:', error);
      alert('Error processing file: ' + error.message);
    } finally {
      setLoading(false);
    }
  }, [backendStatus, columns]);

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
    <div className="p-8">
      <div className="max-w-4xl mx-auto">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            <h2 className="text-4xl font-bold mb-4">Step 1: Upload Your Dataset</h2>
          </div>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Start your interactive data analysis journey by uploading a CSV dataset. Our system will automatically analyze and prepare your data through 5 comprehensive steps.
          </p>
        </div>

        {/* 5-Step Workflow Overview */}
        <div className="mb-12 bg-gradient-to-r from-blue-50 to-purple-50 rounded-2xl p-8 border border-blue-200">
          <h3 className="text-2xl font-bold text-gray-900 mb-6 text-center">Interactive Data Analysis Workflow</h3>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {[
              { 
                step: 1, 
                title: 'Upload Data', 
                desc: 'Upload your CSV dataset', 
                icon: '📤',
                output: 'Raw dataset loaded'
              },
              { 
                step: 2, 
                title: 'Analyze & Understand', 
                desc: 'Automatic EDA and data profiling', 
                icon: '📊',
                output: 'Quality report generated'
              },
              { 
                step: 3, 
                title: 'Clean & Prepare', 
                desc: 'Interactive data cleaning', 
                icon: '🛠️',
                output: 'dataset_cleaned.csv'
              },
              { 
                step: 4, 
                title: 'Model & Train', 
                desc: 'Machine learning modeling', 
                icon: '🤖',
                output: 'model.pkl + results.csv'
              },
              { 
                step: 5, 
                title: 'Predict', 
                desc: 'Make predictions on new data', 
                icon: '🔮',
                output: 'Predictions ready'
              }
            ].map((step, index) => (
              <div key={step.step} className={`text-center p-4 rounded-xl ${
                workflowStep >= step.step ? 'bg-white border-2 border-green-300 shadow-md' : 'bg-gray-50 border border-gray-200'
              }`}>
                <div className="text-3xl mb-2">{step.icon}</div>
                <h4 className={`font-semibold mb-2 ${
                  workflowStep >= step.step ? 'text-green-700' : 'text-gray-600'
                }`}>
                  {step.title}
                </h4>
                <p className="text-sm text-gray-600 mb-2">{step.desc}</p>
                <div className={`text-xs px-2 py-1 rounded ${
                  workflowStep >= step.step ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'
                }`}>
                  {step.output}
                </div>
                {workflowStep >= step.step && (
                  <CheckCircle className="w-5 h-5 text-green-600 mx-auto mt-2" />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Backend Status Card */}
        <div className="mb-8 p-6 rounded-xl border border-gray-200 bg-white shadow-sm hover:shadow-md transition-shadow">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
                backendStatus === 'connected' ? 'bg-green-100' : 
                backendStatus === 'disconnected' ? 'bg-red-100' : 'bg-yellow-100'
              }`}>
                <Server className={`w-6 h-6 ${
                  backendStatus === 'connected' ? 'text-green-600' : 
                  backendStatus === 'disconnected' ? 'text-red-600' : 'text-yellow-600'
                }`} />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Backend Connection</h3>
                <div className="flex items-center gap-2">
                  {backendStatus === 'connected' ? (
                    <>
                      <CheckCircle className="w-4 h-4 text-green-500" />
                      <span className="text-green-600 font-medium">Connected to Flask API</span>
                    </>
                  ) : backendStatus === 'disconnected' ? (
                    <>
                      <XCircle className="w-4 h-4 text-red-500" />
                      <span className="text-red-600 font-medium">API not available</span>
                    </>
                  ) : (
                    <>
                      <Clock className="w-4 h-4 text-yellow-500 animate-pulse" />
                      <span className="text-yellow-600 font-medium">Checking connection...</span>
                    </>
                  )}
                </div>
              </div>
            </div>
            {backendStatus === 'disconnected' && (
              <button 
                onClick={checkBackendHealth}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
              >
                <RefreshCw className="w-4 h-4" />
                Retry
              </button>
            )}
          </div>
        </div>

        {/* File Upload Section */}
        <div className="mb-12">
          <div className="relative group">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl blur opacity-30 group-hover:opacity-50 transition duration-1000"></div>
            <div className="relative bg-white rounded-2xl p-12 border-2 border-dashed border-gray-300 hover:border-blue-400 transition-all duration-300">
              <div className="text-center">
                <div className="bg-blue-50 rounded-full w-20 h-20 flex items-center justify-center mx-auto mb-6">
                  <Upload className="w-10 h-10 text-blue-600" />
                </div>
                <h3 className="text-2xl font-semibold text-gray-900 mb-4">Upload Your Dataset</h3>
                <p className="text-gray-600 mb-8 max-w-md mx-auto">
                  Drag and drop your CSV file here, or click to browse. We support datasets up to 100MB.
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
                  className="inline-flex items-center px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl hover:shadow-lg transform hover:scale-105 transition-all duration-200 cursor-pointer"
                >
                  <Upload className="w-5 h-5 mr-3" />
                  Choose CSV File
                </label>
                <div className="mt-6 text-sm text-gray-500">
                  Supported formats: CSV • Maximum size: 100MB
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Available Algorithms Section */}
        {Object.keys(algorithms).length > 0 && (
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-8">
              <div className="bg-purple-100 rounded-lg p-2">
                <Brain className="w-6 h-6 text-purple-600" />
              </div>
              <h3 className="text-2xl font-semibold text-gray-900">Available ML Algorithms</h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Object.entries(algorithms).map(([key, algo]) => (
                <div key={key} className="bg-white p-6 rounded-xl border border-gray-200 hover:shadow-md hover:border-blue-300 transition-all duration-300 group">
                  <div className="flex items-start gap-4">
                    <div className="bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg p-2 group-hover:scale-110 transition-transform">
                      <Zap className="w-5 h-5 text-white" />
                    </div>
                    <div className="flex-1">
                      <h4 className="font-semibold text-lg text-gray-900 mb-2">{algo.name}</h4>
                      <p className="text-gray-600 text-sm mb-3 leading-relaxed">{algo.description}</p>
                      <div className="bg-gray-50 rounded-lg p-3">
                        <p className="text-xs text-gray-700">
                          <span className="font-medium text-blue-600">Best for:</span> {algo.best_for}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white p-6 rounded-xl border border-gray-200 hover:shadow-md transition-shadow">
            <div className="bg-blue-100 rounded-lg p-3 w-fit mb-4">
              <BarChart3 className="w-6 h-6 text-blue-600" />
            </div>
            <h4 className="text-lg font-semibold text-gray-900 mb-2">Data Visualization</h4>
            <p className="text-gray-600 text-sm">
              Interactive charts and graphs to explore your data patterns and relationships.
            </p>
          </div>
          <div className="bg-white p-6 rounded-xl border border-gray-200 hover:shadow-md transition-shadow">
            <div className="bg-green-100 rounded-lg p-3 w-fit mb-4">
              <TrendingUp className="w-6 h-6 text-green-600" />
            </div>
            <h4 className="text-lg font-semibold text-gray-900 mb-2">Predictive Analytics</h4>
            <p className="text-gray-600 text-sm">
              Build and train machine learning models to make predictions on new data.
            </p>
          </div>
          <div className="bg-white p-6 rounded-xl border border-gray-200 hover:shadow-md transition-shadow">
            <div className="bg-purple-100 rounded-lg p-3 w-fit mb-4">
              <Database className="w-6 h-6 text-purple-600" />
            </div>
            <h4 className="text-lg font-semibold text-gray-900 mb-2">Data Processing</h4>
            <p className="text-gray-600 text-sm">
              Automated data cleaning, preprocessing, and feature engineering capabilities.
            </p>
          </div>
        </div>
      </div>
    </div>
  );

  const renderTrainingTab = () => (
    <div className="p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Model Training</h2>
          <p className="text-gray-600">Train custom machine learning models on your datasets</p>
        </div>
        
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
          <ModelTrainingComponent 
            trainingData={trainingData}
            onModelTrained={(result) => {
              console.log('Model trained:', result);
              // Optionally refresh some state here
            }}
          />
        </div>
      </div>
    </div>
  );

  const renderPredictionTab = () => (
    <div className="p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Step 5: Make Predictions</h2>
          <p className="text-gray-600">Use your trained model to make predictions on new data using the cleaned dataset</p>
        </div>
        
        {trainedModel ? (
          <div className="space-y-6">
            {/* Model Info Card */}
            <div className="bg-green-50 border border-green-200 rounded-xl p-6">
              <div className="flex items-center gap-3 mb-4">
                <CheckCircle className="w-6 h-6 text-green-600" />
                <h3 className="text-lg font-semibold text-green-900">Model Ready for Predictions</h3>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="font-medium text-green-800">Model ID:</span>
                  <p className="text-green-700">{trainedModel.model_id}</p>
                </div>
                <div>
                  <span className="font-medium text-green-800">Accuracy:</span>
                  <p className="text-green-700">{(trainedModel.accuracy * 100).toFixed(2)}%</p>
                </div>
                <div>
                  <span className="font-medium text-green-800">Dataset Used:</span>
                  <p className="text-green-700">{cleanedDataFile?.name || `${fileName}_cleaned`}</p>
                </div>
              </div>
              
              {trainingResultsFile && (
                <div className="mt-4 pt-4 border-t border-green-200">
                  <p className="text-green-800 mb-2">
                    <strong>Training Results:</strong> Results with predictions saved as <code>{trainingResultsFile.name}</code>
                  </p>
                  <button
                    onClick={() => {
                      const blob = new Blob([trainingResultsFile.content], { type: 'text/csv' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = trainingResultsFile.name;
                      a.click();
                      URL.revokeObjectURL(url);
                    }}
                    className="px-4 py-2 bg-green-600 text-white text-sm font-medium rounded-lg hover:bg-green-700 transition-colors flex items-center gap-2"
                  >
                    <Download className="w-4 h-4" />
                    Download Training Results
                  </button>
                </div>
              )}
            </div>
            
            {/* Prediction Component */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
              <PredictionComponent />
            </div>
          </div>
        ) : (
          <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-12 text-center">
            <div className="text-6xl mb-6">🤖</div>
            <h3 className="text-xl font-semibold text-gray-900 mb-4">No Trained Model Available</h3>
            <p className="text-gray-600 mb-6 max-w-md mx-auto">
              You need to train a machine learning model first before making predictions. 
              Go to the modeling step to train your model.
            </p>
            <button
              onClick={() => {
                setActiveTab('modeling');
                setWorkflowStep(4);
              }}
              className="px-6 py-3 bg-purple-600 text-white font-semibold rounded-lg hover:bg-purple-700 transition-colors flex items-center gap-2 mx-auto"
            >
              <Brain className="w-5 h-5" />
              Go to Model Training
            </button>
          </div>
        )}
      </div>
    </div>
  );

  const renderOverviewTab = () => (
    <div className="p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Step 2: Data Analysis & Understanding</h2>
          <p className="text-gray-600">Explore your dataset structure, statistics, and data quality to understand your data better</p>
        </div>

        {/* Workflow Progress */}
        <div className="mb-8 bg-white rounded-xl border border-gray-200 shadow-sm p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Clock className="w-5 h-5 text-blue-600" />
            Workflow Progress
          </h3>
          <div className="flex items-center justify-between">
            {[
              { step: 1, name: 'Upload', icon: Upload, completed: workflowStep >= 1, file: fileName },
              { step: 2, name: 'Analysis', icon: FileText, completed: workflowStep >= 2, file: dataQualityReport ? 'quality_report.json' : null },
              { step: 3, name: 'Cleaning', icon: Settings, completed: workflowStep >= 3, file: cleanedDataFile?.name },
              { step: 4, name: 'Modeling', icon: Brain, completed: workflowStep >= 4, file: trainedModelFile?.name },
              { step: 5, name: 'Prediction', icon: Target, completed: workflowStep >= 5, file: trainingResultsFile?.name }
            ].map((item, index) => (
              <div key={item.step} className="flex items-center">
                <div className={`flex items-center justify-center w-10 h-10 rounded-full ${
                  item.completed ? 'bg-green-100 text-green-600' : 
                  workflowStep === item.step ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-400'
                }`}>
                  {item.completed ? (
                    <CheckCircle className="w-6 h-6" />
                  ) : (
                    <item.icon className="w-5 h-5" />
                  )}
                </div>
                <div className="ml-2">
                  <span className={`text-sm font-medium ${
                    item.completed ? 'text-green-600' : 
                    workflowStep === item.step ? 'text-blue-600' : 'text-gray-400'
                  }`}>
                    {item.name}
                  </span>
                  {item.file && (
                    <div className="text-xs text-gray-500 truncate max-w-20" title={item.file}>
                      📄 {item.file}
                    </div>
                  )}
                </div>
                {index < 4 && (
                  <div className={`mx-4 h-0.5 w-12 ${
                    workflowStep > item.step ? 'bg-green-300' : 'bg-gray-200'
                  }`}></div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Dataset Overview Cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-gradient-to-r from-blue-500 to-blue-600 rounded-xl p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-blue-100 text-sm font-medium">Total Rows</p>
                <p className="text-2xl font-bold">{data?.length?.toLocaleString() || 0}</p>
              </div>
              <FileText className="w-8 h-8 text-blue-200" />
            </div>
          </div>
          
          <div className="bg-gradient-to-r from-purple-500 to-purple-600 rounded-xl p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-purple-100 text-sm font-medium">Columns</p>
                <p className="text-2xl font-bold">{columns.length}</p>
              </div>
              <BarChart3 className="w-8 h-8 text-purple-200" />
            </div>
          </div>
          
          <div className="bg-gradient-to-r from-green-500 to-green-600 rounded-xl p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-green-100 text-sm font-medium">Data Quality</p>
                <p className="text-lg font-bold">
                  {dataQualityReport ? 
                    `${((1 - (Object.values(dataQualityReport.missingValues || {}).reduce((a, b) => a + b, 0) + 
                              dataQualityReport.duplicates) / (dataQualityReport.totalRows || 1)) * 100).toFixed(0)}%` 
                    : 'N/A'}
                </p>
              </div>
              <CheckCircle className="w-8 h-8 text-green-200" />
            </div>
          </div>
          
          <div className="bg-gradient-to-r from-orange-500 to-orange-600 rounded-xl p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-orange-100 text-sm font-medium">Backend Status</p>
                <p className="text-lg font-bold">
                  {backendStatus === 'connected' ? 'Connected' : 'Disconnected'}
                </p>
              </div>
              {backendStatus === 'connected' ? (
                <CheckCircle className="w-8 h-8 text-orange-200" />
              ) : (
                <XCircle className="w-8 h-8 text-orange-200" />
              )}
            </div>
          </div>
        </div>

        {/* Data Quality Summary */}
        {dataQualityReport && (
          <div className="mb-8 bg-white rounded-xl border border-gray-200 shadow-sm p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-yellow-600" />
              Data Quality Summary
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className={`p-4 rounded-lg ${Object.values(dataQualityReport.missingValues || {}).reduce((a, b) => a + b, 0) > 0 ? 'bg-orange-50 border border-orange-200' : 'bg-green-50 border border-green-200'}`}>
                <div className="flex items-center justify-between">
                  <span className="font-medium text-gray-900">Missing Values</span>
                  <span className={`font-bold ${Object.values(dataQualityReport.missingValues || {}).reduce((a, b) => a + b, 0) > 0 ? 'text-orange-600' : 'text-green-600'}`}>
                    {Object.values(dataQualityReport.missingValues || {}).reduce((a, b) => a + b, 0)}
                  </span>
                </div>
              </div>
              <div className={`p-4 rounded-lg ${dataQualityReport.duplicates > 0 ? 'bg-orange-50 border border-orange-200' : 'bg-green-50 border border-green-200'}`}>
                <div className="flex items-center justify-between">
                  <span className="font-medium text-gray-900">Duplicate Rows</span>
                  <span className={`font-bold ${dataQualityReport.duplicates > 0 ? 'text-orange-600' : 'text-green-600'}`}>
                    {dataQualityReport.duplicates}
                  </span>
                </div>
              </div>
              <div className={`p-4 rounded-lg ${Object.keys(dataQualityReport.outliers || {}).length > 0 ? 'bg-yellow-50 border border-yellow-200' : 'bg-green-50 border border-green-200'}`}>
                <div className="flex items-center justify-between">
                  <span className="font-medium text-gray-900">Outliers Detected</span>
                  <span className={`font-bold ${Object.keys(dataQualityReport.outliers || {}).length > 0 ? 'text-yellow-600' : 'text-green-600'}`}>
                    {Object.keys(dataQualityReport.outliers || {}).length} columns
                  </span>
                </div>
              </div>
            </div>
            {(Object.values(dataQualityReport.missingValues || {}).reduce((a, b) => a + b, 0) > 0 || 
              dataQualityReport.duplicates > 0 || 
              Object.keys(dataQualityReport.outliers || {}).length > 0) && (
              <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-blue-800 text-sm">
                  <strong>Recommendation:</strong> Your data has quality issues that could affect model performance. 
                  Consider proceeding to the data cleaning step to improve your dataset quality.
                </p>
                <button
                  onClick={() => {
                    setWorkflowStep(3);
                    setActiveTab('cleaning');
                  }}
                  className="mt-3 px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Go to Data Cleaning
                </button>
              </div>
            )}
          </div>
        )}

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
          {/* Column Selection Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
              <div className="p-6 border-b border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                  <Settings className="w-5 h-5 text-blue-600" />
                  Feature Selection
                </h3>
                <p className="text-sm text-gray-600 mt-1">
                  Select columns for ML analysis
                </p>
              </div>
              <div className="p-6">
                <div className="space-y-3 max-h-80 overflow-y-auto">
                  {columns.map(col => (
                    <label key={col} className="flex items-center space-x-3 p-3 rounded-lg hover:bg-gray-50 transition-colors cursor-pointer">
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
                        className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                      />
                      <div className="flex-1 min-w-0">
                        <span className="text-sm font-medium text-gray-900 block truncate">{col}</span>
                        <span className={`text-xs px-2 py-1 rounded-full ${
                          analysis?.stats[col]?.type === 'numeric' 
                            ? 'bg-blue-100 text-blue-800' 
                            : 'bg-green-100 text-green-800'
                        }`}>
                          {analysis?.stats[col]?.type || 'unknown'}
                        </span>
                      </div>
                    </label>
                  ))}
                </div>
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <p className="text-sm text-gray-600">
                    Selected: <span className="font-medium text-blue-600">{selectedColumns.length}</span> columns
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Statistical Summary */}
          <div className="lg:col-span-2">
            {analysis && (
              <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
                <div className="p-6 border-b border-gray-200">
                  <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-green-600" />
                    Statistical Summary
                  </h3>
                  <p className="text-sm text-gray-600 mt-1">
                    Detailed statistics for each column
                  </p>
                </div>
                <div className="p-6">
                  <div className="overflow-x-auto">
                    <table className="min-w-full">
                      <thead>
                        <tr className="border-b border-gray-200">
                          <th className="text-left py-3 px-4 font-semibold text-gray-900">Column</th>
                          <th className="text-left py-3 px-4 font-semibold text-gray-900">Type</th>
                          <th className="text-left py-3 px-4 font-semibold text-gray-900">Count</th>
                          <th className="text-left py-3 px-4 font-semibold text-gray-900">Summary</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200">
                        {Object.entries(analysis.stats).map(([col, stats]) => (
                          <tr key={col} className="hover:bg-gray-50 transition-colors">
                            <td className="py-3 px-4">
                              <span className="font-medium text-gray-900">{col}</span>
                            </td>
                            <td className="py-3 px-4">
                              <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                                stats.type === 'numeric' 
                                  ? 'bg-blue-100 text-blue-800' 
                                  : 'bg-green-100 text-green-800'
                              }`}>
                                {stats.type}
                              </span>
                            </td>
                            <td className="py-3 px-4 text-gray-600">
                              {stats.count?.toLocaleString()}
                            </td>
                            <td className="py-3 px-4 text-sm text-gray-600">
                              {stats.type === 'numeric' ? (
                                <div className="space-y-1">
                                  <div>Mean: <span className="font-medium">{stats.mean ? stats.mean.toFixed(2) : 'N/A'}</span></div>
                                  <div>Std: <span className="font-medium">{stats.std ? stats.std.toFixed(2) : 'N/A'}</span></div>
                                </div>
                              ) : (
                                <div className="space-y-1">
                                  <div>Unique: <span className="font-medium">{stats.unique || 0}</span></div>
                                  <div>Top: <span className="font-medium">{Object.keys(stats.most_common || {})[0] || 'N/A'}</span></div>
                                </div>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap gap-4 justify-center">
          <button
            onClick={() => {
              setWorkflowStep(3);
              setActiveTab('cleaning');
            }}
            disabled={!dataQualityReport}
            className="flex items-center px-8 py-4 bg-gradient-to-r from-yellow-500 to-orange-500 text-white font-semibold rounded-xl hover:shadow-lg transform hover:scale-105 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
          >
            <Settings className="w-5 h-5 mr-3" />
            Proceed to Data Cleaning
          </button>
          
          <button
            onClick={runMLAnalysis}
            disabled={loading || selectedColumns.length === 0 || backendStatus !== 'connected'}
            className="flex items-center px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 text-white font-semibold rounded-xl hover:shadow-lg transform hover:scale-105 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
          >
            {loading ? (
              <>
                <RefreshCw className="w-5 h-5 mr-3 animate-spin" />
                Running Analysis...
              </>
            ) : (
              <>
                <Play className="w-5 h-5 mr-3" />
                Quick ML Analysis
              </>
            )}
          </button>
          
          {analysis && (
            <button
              onClick={exportResults}
              className="flex items-center px-6 py-4 bg-gray-600 text-white font-semibold rounded-xl hover:bg-gray-700 hover:shadow-lg transform hover:scale-105 transition-all duration-200"
            >
              <Download className="w-5 h-5 mr-3" />
              Export Results
            </button>
          )}
        </div>

        {/* Loading State */}
        {loading && (
          <div className="mt-8 bg-blue-50 border border-blue-200 rounded-xl p-6">
            <div className="flex items-center justify-center">
              <RefreshCw className="w-6 h-6 text-blue-600 animate-spin mr-3" />
              <span className="text-blue-800 font-medium">
                Processing your data with advanced ML algorithms...
              </span>
            </div>
            <div className="mt-4 bg-white rounded-lg p-4">
              <div className="flex justify-between text-sm text-gray-600 mb-2">
                <span>Progress</span>
                <span>Analyzing patterns...</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-blue-600 h-2 rounded-full animate-pulse" style={{width: '60%'}}></div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const renderVisualizationTab = () => (
    <div className="p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Data Visualization</h2>
          <p className="text-gray-600">Interactive charts and insights from your dataset</p>
        </div>
        
        {analysis && data ? (
          <div className="space-y-8">
            {/* Quick Insights Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <div className="bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl p-6 text-white">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-blue-100 text-sm font-medium">Numeric Columns</p>
                    <p className="text-2xl font-bold">
                      {Object.values(analysis.stats).filter(stat => stat.type === 'numeric').length}
                    </p>
                  </div>
                  <BarChart3 className="w-8 h-8 text-blue-200" />
                </div>
              </div>
              
              <div className="bg-gradient-to-r from-emerald-500 to-teal-500 rounded-xl p-6 text-white">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-emerald-100 text-sm font-medium">Categorical Columns</p>
                    <p className="text-2xl font-bold">
                      {Object.values(analysis.stats).filter(stat => stat.type === 'categorical').length}
                    </p>
                  </div>
                  <FileText className="w-8 h-8 text-emerald-200" />
                </div>
              </div>
              
              <div className="bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl p-6 text-white">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-purple-100 text-sm font-medium">Data Quality</p>
                    <p className="text-2xl font-bold">
                      {Math.round((data.length - (data.filter(row => Object.values(row).some(val => val === null || val === undefined)).length)) / data.length * 100)}%
                    </p>
                  </div>
                  <CheckCircle className="w-8 h-8 text-purple-200" />
                </div>
              </div>
            </div>

            {/* Visualization Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Column Type Distribution */}
              <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-blue-600" />
                  Column Type Distribution
                </h3>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={[
                        {
                          name: 'Numeric',
                          value: Object.values(analysis.stats).filter(stat => stat.type === 'numeric').length,
                          fill: '#3B82F6'
                        },
                        {
                          name: 'Categorical',
                          value: Object.values(analysis.stats).filter(stat => stat.type === 'categorical').length,
                          fill: '#10B981'
                        }
                      ]}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={5}
                      dataKey="value"
                      label={({name, value}) => `${name}: ${value}`}
                    >
                      {['#3B82F6', '#10B981'].map((color, index) => (
                        <Cell key={`cell-${index}`} fill={color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* Data Distribution for first numeric column */}
              {selectedColumns.filter(col => analysis.stats[col]?.type === 'numeric').slice(0, 1).map(col => (
                <div key={col} className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-green-600" />
                    {col} Distribution
                  </h3>
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={data.slice(0, 20)}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis 
                        dataKey={col} 
                        tick={{fontSize: 12}}
                        stroke="#6B7280"
                      />
                      <YAxis 
                        tick={{fontSize: 12}}
                        stroke="#6B7280"
                      />
                      <Tooltip 
                        contentStyle={{
                          backgroundColor: 'white',
                          border: '1px solid #e5e7eb',
                          borderRadius: '8px'
                        }}
                      />
                      <Bar 
                        dataKey={col} 
                        fill="url(#colorGradient)"
                        radius={[4, 4, 0, 0]}
                      />
                      <defs>
                        <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#8B5CF6" stopOpacity={0.8}/>
                          <stop offset="95%" stopColor="#3B82F6" stopOpacity={0.8}/>
                        </linearGradient>
                      </defs>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ))}

              {/* Correlation Matrix Preview */}
              {Object.keys(analysis.correlation_matrix).length > 0 && (
                <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <Layers className="w-5 h-5 text-purple-600" />
                    Correlation Matrix
                  </h3>
                  <div className="bg-gradient-to-br from-purple-50 to-blue-50 rounded-lg p-6 text-center">
                    <div className="text-4xl mb-4">📊</div>
                    <p className="text-gray-600 mb-4">
                      Correlation analysis available with {Object.keys(analysis.correlation_matrix).length} variables
                    </p>
                    <button className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors">
                      View Detailed Correlation
                    </button>
                  </div>
                </div>
              )}

              {/* Additional numeric column visualizations */}
              {selectedColumns.filter(col => analysis.stats[col]?.type === 'numeric').slice(1, 4).map(col => (
                <div key={col} className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-indigo-600" />
                    {col} Trend
                  </h3>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={data.slice(0, 50)}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis 
                        dataKey="index"
                        tick={{fontSize: 12}}
                        stroke="#6B7280"
                      />
                      <YAxis 
                        tick={{fontSize: 12}}
                        stroke="#6B7280"
                      />
                      <Tooltip 
                        contentStyle={{
                          backgroundColor: 'white',
                          border: '1px solid #e5e7eb',
                          borderRadius: '8px'
                        }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey={col} 
                        stroke="#6366F1" 
                        strokeWidth={2}
                        dot={{fill: '#6366F1', strokeWidth: 0, r: 3}}
                        activeDot={{r: 5, fill: '#6366F1'}}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ))}
            </div>

            {/* Summary Statistics Table */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-6 flex items-center gap-2">
                <Database className="w-5 h-5 text-blue-600" />
                Statistical Summary
              </h3>
              <div className="overflow-x-auto">
                <table className="min-w-full">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200">
                      <th className="text-left py-3 px-4 font-semibold text-gray-900">Column</th>
                      <th className="text-left py-3 px-4 font-semibold text-gray-900">Type</th>
                      <th className="text-left py-3 px-4 font-semibold text-gray-900">Mean</th>
                      <th className="text-left py-3 px-4 font-semibold text-gray-900">Std Dev</th>
                      <th className="text-left py-3 px-4 font-semibold text-gray-900">Min</th>
                      <th className="text-left py-3 px-4 font-semibold text-gray-900">Max</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {Object.entries(analysis.stats)
                      .filter(([col, stats]) => stats.type === 'numeric')
                      .map(([col, stats]) => (
                      <tr key={col} className="hover:bg-gray-50 transition-colors">
                        <td className="py-3 px-4 font-medium text-gray-900">{col}</td>
                        <td className="py-3 px-4">
                          <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-medium">
                            Numeric
                          </span>
                        </td>
                        <td className="py-3 px-4 text-gray-600">
                          {stats.mean ? stats.mean.toFixed(2) : 'N/A'}
                        </td>
                        <td className="py-3 px-4 text-gray-600">
                          {stats.std ? stats.std.toFixed(2) : 'N/A'}
                        </td>
                        <td className="py-3 px-4 text-gray-600">
                          {stats.min !== undefined ? stats.min.toFixed(2) : 'N/A'}
                        </td>
                        <td className="py-3 px-4 text-gray-600">
                          {stats.max !== undefined ? stats.max.toFixed(2) : 'N/A'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl border border-blue-200 p-12 text-center">
            <div className="text-6xl mb-6">📊</div>
            <h3 className="text-xl font-semibold text-gray-900 mb-4">No Data Available</h3>
            <p className="text-gray-600 mb-6 max-w-md mx-auto">
              Upload a dataset and run analysis to see beautiful visualizations of your data patterns and insights.
            </p>
            <button
              onClick={() => setActiveTab('upload')}
              className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors"
            >
              Upload Dataset
            </button>
          </div>
        )}
      </div>
    </div>
  );

  const renderClusteringTab = () => (
    <div className="p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Clustering Analysis</h2>
          <p className="text-gray-600">Discover hidden patterns and group similar data points</p>
        </div>
        
        {mlResults?.clustering ? (
          <div className="space-y-8">
            {/* Sample Warning */}
            {mlResults.clustering.info && mlResults.clustering.info.was_sampled && (
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl p-6">
                <div className="flex items-center gap-3">
                  <div className="bg-blue-100 rounded-full p-2">
                    <AlertTriangle className="w-5 h-5 text-blue-600" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-blue-900">Large Dataset Optimization</h4>
                    <p className="text-blue-800 mt-1">
                      Analysis performed on a sample of <span className="font-semibold">{mlResults.clustering.info.processed_size.toLocaleString()}</span> out of <span className="font-semibold">{mlResults.clustering.info.original_size.toLocaleString()}</span> rows for optimal performance.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Statistics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl p-6 text-white">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-blue-100 text-sm font-medium">Clusters Found</p>
                    <p className="text-3xl font-bold">
                      {mlResults.clustering.cluster_stats?.num_clusters || 'N/A'}
                    </p>
                    <p className="text-blue-100 text-xs mt-1">Distinct groups identified</p>
                  </div>
                  <Layers className="w-10 h-10 text-blue-200" />
                </div>
              </div>
              
              <div className="bg-gradient-to-r from-orange-500 to-red-500 rounded-xl p-6 text-white">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-orange-100 text-sm font-medium">Noise Points</p>
                    <p className="text-3xl font-bold">
                      {mlResults.clustering.cluster_stats?.num_noise_points || 'N/A'}
                    </p>
                    <p className="text-orange-100 text-xs mt-1">Outlier data points</p>
                  </div>
                  <AlertTriangle className="w-10 h-10 text-orange-200" />
                </div>
              </div>
              
              <div className="bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl p-6 text-white">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-green-100 text-sm font-medium">Total Points</p>
                    <p className="text-3xl font-bold">
                      {mlResults.clustering.clustered_data?.length?.toLocaleString() || 'N/A'}
                    </p>
                    <p className="text-green-100 text-xs mt-1">Analyzed data points</p>
                  </div>
                  <Database className="w-10 h-10 text-green-200" />
                </div>
              </div>
            </div>

            {/* Visualization and Details */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Cluster Visualization */}
              {mlResults.clustering.clustered_data && selectedColumns.length >= 2 && (
                <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-blue-600" />
                    Cluster Visualization
                  </h3>
                  <div className="mb-4 text-sm text-gray-600">
                    Showing {selectedColumns[0]} vs {selectedColumns[1] || selectedColumns[0]}
                  </div>
                  <ResponsiveContainer width="100%" height={350}>
                    <ScatterChart data={mlResults.clustering.clustered_data.slice(0, 200)}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis 
                        dataKey={selectedColumns[0]} 
                        type="number"
                        domain={['dataMin', 'dataMax']}
                        tick={{fontSize: 12}}
                        stroke="#6B7280"
                      />
                      <YAxis 
                        dataKey={selectedColumns[1] || selectedColumns[0]} 
                        type="number"
                        domain={['dataMin', 'dataMax']}
                        tick={{fontSize: 12}}
                        stroke="#6B7280"
                      />
                      <Tooltip 
                        contentStyle={{
                          backgroundColor: 'white',
                          border: '1px solid #e5e7eb',
                          borderRadius: '8px'
                        }}
                      />
                      <Scatter 
                        dataKey="cluster" 
                        fill="#8B5CF6"
                        fillOpacity={0.7}
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Cluster Distribution */}
              <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-green-600" />
                  Cluster Distribution
                </h3>
                
                {mlResults.clustering.cluster_stats && (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4 text-center">
                      <div className="bg-blue-50 rounded-lg p-4">
                        <div className="text-2xl font-bold text-blue-600">
                          {((mlResults.clustering.cluster_stats.num_clusters / (mlResults.clustering.cluster_stats.num_clusters + mlResults.clustering.cluster_stats.num_noise_points)) * 100).toFixed(1)}%
                        </div>
                        <div className="text-sm text-gray-600">Clustered</div>
                      </div>
                      <div className="bg-red-50 rounded-lg p-4">
                        <div className="text-2xl font-bold text-red-600">
                          {((mlResults.clustering.cluster_stats.num_noise_points / (mlResults.clustering.cluster_stats.num_clusters + mlResults.clustering.cluster_stats.num_noise_points)) * 100).toFixed(1)}%
                        </div>
                        <div className="text-sm text-gray-600">Noise</div>
                      </div>
                    </div>
                    
                    <div className="mt-6">
                      <h4 className="font-medium text-gray-900 mb-3">Algorithm Parameters</h4>
                      <div className="bg-gray-50 rounded-lg p-4 text-sm">
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <span className="font-medium">Algorithm:</span> DBSCAN
                          </div>
                          <div>
                            <span className="font-medium">Features:</span> {selectedColumns.length}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Cluster Analysis Summary */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Brain className="w-5 h-5 text-purple-600" />
                Clustering Insights
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg p-4">
                    <h4 className="font-semibold text-gray-900 mb-2">Key Findings</h4>
                    <ul className="text-sm text-gray-700 space-y-1">
                      <li>• {mlResults.clustering.cluster_stats?.num_clusters || 0} distinct clusters identified</li>
                      <li>• {mlResults.clustering.cluster_stats?.num_noise_points || 0} outlier points detected</li>
                      <li>• Analysis based on {selectedColumns.length} features</li>
                    </ul>
                  </div>
                </div>
                <div className="space-y-4">
                  <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-4">
                    <h4 className="font-semibold text-gray-900 mb-2">Recommendations</h4>
                    <ul className="text-sm text-gray-700 space-y-1">
                      <li>• Use clusters for customer segmentation</li>
                      <li>• Investigate noise points for anomalies</li>
                      <li>• Consider feature engineering for better clustering</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-gradient-to-br from-purple-50 to-blue-50 rounded-xl border border-purple-200 p-12 text-center">
            <div className="text-6xl mb-6">🔍</div>
            <h3 className="text-xl font-semibold text-gray-900 mb-4">No Clustering Results</h3>
            <p className="text-gray-600 mb-6 max-w-md mx-auto">
              Run ML analysis first to discover hidden patterns and clusters in your data.
            </p>
            <button
              onClick={() => setActiveTab('overview')}
              className="px-6 py-3 bg-purple-600 text-white font-semibold rounded-lg hover:bg-purple-700 transition-colors"
            >
              Go to Analysis
            </button>
          </div>
        )}
      </div>
    </div>
  );

  const renderAnomalyTab = () => (
    <div className="p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Anomaly Detection</h2>
          <p className="text-gray-600">Identify unusual patterns and outliers in your data</p>
        </div>
        
        {mlResults?.anomalies ? (
          <div className="space-y-8">
            {/* Sample Warning */}
            {mlResults.anomalies.info && mlResults.anomalies.info.was_sampled && (
              <div className="bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200 rounded-xl p-6">
                <div className="flex items-center gap-3">
                  <div className="bg-amber-100 rounded-full p-2">
                    <AlertTriangle className="w-5 h-5 text-amber-600" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-amber-900">Large Dataset Optimization</h4>
                    <p className="text-amber-800 mt-1">
                      Analysis performed on a sample of <span className="font-semibold">{mlResults.anomalies.info.processed_size.toLocaleString()}</span> out of <span className="font-semibold">{mlResults.anomalies.info.original_size.toLocaleString()}</span> rows for optimal performance.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Statistics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-gradient-to-r from-red-500 to-pink-500 rounded-xl p-6 text-white">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-red-100 text-sm font-medium">Anomalies Detected</p>
                    <p className="text-3xl font-bold">
                      {mlResults.anomalies.anomaly_stats?.total_anomalies?.toLocaleString() || 'N/A'}
                    </p>
                    <p className="text-red-100 text-xs mt-1">Unusual data points</p>
                  </div>
                  <AlertTriangle className="w-10 h-10 text-red-200" />
                </div>
              </div>
              
              <div className="bg-gradient-to-r from-blue-500 to-indigo-500 rounded-xl p-6 text-white">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-blue-100 text-sm font-medium">Anomaly Rate</p>
                    <p className="text-3xl font-bold">
                      {mlResults.anomalies.anomaly_stats?.anomaly_percentage?.toFixed(2) || 'N/A'}%
                    </p>
                    <p className="text-blue-100 text-xs mt-1">Of total data points</p>
                  </div>
                  <TrendingUp className="w-10 h-10 text-blue-200" />
                </div>
              </div>
            </div>

            {/* Visualization and Analysis */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Anomaly Visualization */}
              {mlResults.anomalies.data_with_anomalies && selectedColumns.length >= 2 && (
                <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-red-600" />
                    Anomaly Visualization
                  </h3>
                  <div className="mb-4 text-sm text-gray-600">
                    Red points indicate anomalies • Blue points are normal
                  </div>
                  <ResponsiveContainer width="100%" height={350}>
                    <ScatterChart data={mlResults.anomalies.data_with_anomalies.slice(0, 200)}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis 
                        dataKey={selectedColumns[0]} 
                        type="number"
                        domain={['dataMin', 'dataMax']}
                        tick={{fontSize: 12}}
                        stroke="#6B7280"
                      />
                      <YAxis 
                        dataKey={selectedColumns[1] || selectedColumns[0]} 
                        type="number"
                        domain={['dataMin', 'dataMax']}
                        tick={{fontSize: 12}}
                        stroke="#6B7280"
                      />
                      <Tooltip 
                        contentStyle={{
                          backgroundColor: 'white',
                          border: '1px solid #e5e7eb',
                          borderRadius: '8px'
                        }}
                      />
                      <Scatter 
                        dataKey="anomaly_score" 
                        fill="#EF4444"
                        fillOpacity={0.7}
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Anomaly Analysis Summary */}
              <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                  <Brain className="w-5 h-5 text-purple-600" />
                  Detection Analysis
                </h3>
                
                <div className="space-y-6">
                  {/* Anomaly Distribution */}
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h4 className="font-medium text-gray-900 mb-3">Score Distribution</h4>
                    <div className="grid grid-cols-2 gap-4 text-center">
                      <div className="bg-green-100 rounded-lg p-3">
                        <div className="text-xl font-bold text-green-600">
                          {mlResults.anomalies.data_with_anomalies ? 
                            (mlResults.anomalies.data_with_anomalies.length - (mlResults.anomalies.anomaly_stats?.total_anomalies || 0)) : 0}
                        </div>
                        <div className="text-xs text-gray-600">Normal Points</div>
                      </div>
                      <div className="bg-red-100 rounded-lg p-3">
                        <div className="text-xl font-bold text-red-600">
                          {mlResults.anomalies.anomaly_stats?.total_anomalies || 0}
                        </div>
                        <div className="text-xs text-gray-600">Anomalous Points</div>
                      </div>
                    </div>
                  </div>

                  {/* Algorithm Info */}
                  <div className="bg-blue-50 rounded-lg p-4">
                    <h4 className="font-medium text-gray-900 mb-2">Algorithm Details</h4>
                    <div className="text-sm text-gray-700 space-y-1">
                      <div>Method: Isolation Forest</div>
                      <div>Features: {selectedColumns.length} columns</div>
                      <div>Contamination: 10% threshold</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Insights and Recommendations */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Zap className="w-5 h-5 text-yellow-600" />
                Anomaly Insights & Recommendations
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="bg-gradient-to-r from-red-50 to-pink-50 rounded-lg p-4">
                    <h4 className="font-semibold text-gray-900 mb-2">Key Findings</h4>
                    <ul className="text-sm text-gray-700 space-y-1">
                      <li>• {mlResults.anomalies.anomaly_stats?.total_anomalies || 0} anomalous data points identified</li>
                      <li>• {mlResults.anomalies.anomaly_stats?.anomaly_percentage?.toFixed(1) || 0}% of total dataset flagged</li>
                      <li>• Analysis based on {selectedColumns.length} selected features</li>
                    </ul>
                  </div>
                </div>
                <div className="space-y-4">
                  <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-4">
                    <h4 className="font-semibold text-gray-900 mb-2">Next Steps</h4>
                    <ul className="text-sm text-gray-700 space-y-1">
                      <li>• Investigate high-score anomalies for errors</li>
                      <li>• Consider domain expertise for validation</li>
                      <li>• Use for fraud detection or quality control</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-gradient-to-br from-red-50 to-pink-50 rounded-xl border border-red-200 p-12 text-center">
            <div className="text-6xl mb-6">🚨</div>
            <h3 className="text-xl font-semibold text-gray-900 mb-4">No Anomaly Detection Results</h3>
            <p className="text-gray-600 mb-6 max-w-md mx-auto">
              Run ML analysis first to identify unusual patterns and outliers in your dataset.
            </p>
            <button
              onClick={() => setActiveTab('overview')}
              className="px-6 py-3 bg-red-600 text-white font-semibold rounded-lg hover:bg-red-700 transition-colors"
            >
              Go to Analysis
            </button>
          </div>
        )}
      </div>
    </div>
  );

  const renderDecisionTreeTab = () => (
    <div className="p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Decision Tree Analysis</h2>
          <p className="text-gray-600">Understand decision patterns and feature importance in your data</p>
        </div>
        
        {mlResults?.decisionTree ? (
          <div className="space-y-8">
            {/* Sample Warning */}
            {mlResults.decisionTree.info && mlResults.decisionTree.info.was_sampled && (
              <div className="bg-gradient-to-r from-emerald-50 to-teal-50 border border-emerald-200 rounded-xl p-6">
                <div className="flex items-center gap-3">
                  <div className="bg-emerald-100 rounded-full p-2">
                    <AlertTriangle className="w-5 h-5 text-emerald-600" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-emerald-900">Large Dataset Optimization</h4>
                    <p className="text-emerald-800 mt-1">
                      Model trained on a sample of <span className="font-semibold">{mlResults.decisionTree.info.processed_size.toLocaleString()}</span> out of <span className="font-semibold">{mlResults.decisionTree.info.original_size.toLocaleString()}</span> rows for optimal performance.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Performance Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl p-6 text-white">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-green-100 text-sm font-medium">Model Accuracy</p>
                    <p className="text-3xl font-bold">
                      {((mlResults.decisionTree.accuracy || 0) * 100).toFixed(1)}%
                    </p>
                    <p className="text-green-100 text-xs mt-1">Prediction accuracy</p>
                  </div>
                  <CheckCircle className="w-10 h-10 text-green-200" />
                </div>
              </div>
              
              <div className="bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl p-6 text-white">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-blue-100 text-sm font-medium">Features Used</p>
                    <p className="text-3xl font-bold">
                      {mlResults.decisionTree.feature_importance ? Object.keys(mlResults.decisionTree.feature_importance).length : 'N/A'}
                    </p>
                    <p className="text-blue-100 text-xs mt-1">Input variables</p>
                  </div>
                  <TreePine className="w-10 h-10 text-blue-200" />
                </div>
              </div>
              
              <div className="bg-gradient-to-r from-purple-500 to-indigo-500 rounded-xl p-6 text-white">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-purple-100 text-sm font-medium">Tree Depth</p>
                    <p className="text-3xl font-bold">5</p>
                    <p className="text-purple-100 text-xs mt-1">Maximum levels</p>
                  </div>
                  <Layers className="w-10 h-10 text-purple-200" />
                </div>
              </div>
            </div>

            {/* Feature Importance and Tree Structure */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Feature Importance */}
              {mlResults.decisionTree.feature_importance && (
                <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-6 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-blue-600" />
                    Feature Importance
                  </h3>
                  <div className="space-y-4">
                    {Object.entries(mlResults.decisionTree.feature_importance)
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 8)
                      .map(([feature, importance], index) => (
                        <div key={feature} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-blue-50 transition-colors">
                          <div className="flex items-center gap-3">
                            <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold text-white ${
                              index < 3 ? 'bg-blue-500' : index < 6 ? 'bg-green-500' : 'bg-gray-500'
                            }`}>
                              {index + 1}
                            </div>
                            <span className="font-medium text-gray-900">{feature}</span>
                          </div>
                          <div className="flex items-center gap-3">
                            <div className="w-32 bg-gray-200 rounded-full h-3 overflow-hidden">
                              <div 
                                className={`h-3 rounded-full transition-all duration-500 ${
                                  index < 3 ? 'bg-blue-500' : index < 6 ? 'bg-green-500' : 'bg-gray-500'
                                }`}
                                style={{ width: `${importance * 100}%` }}
                              ></div>
                            </div>
                            <span className="text-sm font-semibold text-gray-600 min-w-[3rem] text-right">
                              {(importance * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* Model Performance Breakdown */}
              <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-6 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-green-600" />
                  Model Performance
                </h3>
                
                <div className="space-y-6">
                  {/* Accuracy Gauge */}
                  <div className="text-center">
                    <div className="relative w-32 h-32 mx-auto mb-4">
                      <svg className="w-32 h-32 transform -rotate-90">
                        <circle
                          cx="64"
                          cy="64"
                          r="56"
                          stroke="#E5E7EB"
                          strokeWidth="8"
                          fill="none"
                        />
                        <circle
                          cx="64"
                          cy="64"
                          r="56"
                          stroke="#10B981"
                          strokeWidth="8"
                          fill="none"
                          strokeLinecap="round"
                          strokeDasharray={`${2 * Math.PI * 56}`}
                          strokeDashoffset={`${2 * Math.PI * 56 * (1 - (mlResults.decisionTree.accuracy || 0))}`}
                          className="transition-all duration-1000"
                        />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <span className="text-2xl font-bold text-gray-900">
                          {((mlResults.decisionTree.accuracy || 0) * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                    <p className="text-sm text-gray-600">Overall Accuracy</p>
                  </div>

                  {/* Performance Metrics */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-green-50 rounded-lg p-4 text-center">
                      <div className="text-xl font-bold text-green-600">
                        {mlResults.decisionTree.accuracy ? (mlResults.decisionTree.accuracy > 0.8 ? 'High' : mlResults.decisionTree.accuracy > 0.6 ? 'Medium' : 'Low') : 'N/A'}
                      </div>
                      <div className="text-xs text-gray-600">Performance</div>
                    </div>
                    <div className="bg-blue-50 rounded-lg p-4 text-center">
                      <div className="text-xl font-bold text-blue-600">
                        {mlResults.decisionTree.feature_importance ? 
                          Object.values(mlResults.decisionTree.feature_importance).filter(imp => imp > 0.1).length : 0}
                      </div>
                      <div className="text-xs text-gray-600">Key Features</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Tree Structure */}
            {mlResults.decisionTree.tree_rules && (
              <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                  <TreePine className="w-5 h-5 text-green-600" />
                  Decision Tree Structure
                </h3>
                <div className="bg-gray-50 rounded-lg p-4 max-h-96 overflow-auto">
                  <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono leading-relaxed">
                    {mlResults.decisionTree.tree_rules}
                  </pre>
                </div>
                <div className="mt-4 text-sm text-gray-600">
                  <p>This tree structure shows the decision-making process of the model. Each branch represents a decision rule based on feature values.</p>
                </div>
              </div>
            )}

            {/* Insights and Applications */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Brain className="w-5 h-5 text-purple-600" />
                Model Insights & Applications
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-4">
                    <h4 className="font-semibold text-gray-900 mb-2">Key Insights</h4>
                    <ul className="text-sm text-gray-700 space-y-1">
                      <li>• Model achieved {((mlResults.decisionTree.accuracy || 0) * 100).toFixed(1)}% accuracy</li>
                      <li>• {mlResults.decisionTree.feature_importance ? Object.keys(mlResults.decisionTree.feature_importance).length : 0} features analyzed</li>
                      <li>• Decision rules extracted for interpretation</li>
                      <li>• Maximum tree depth limited to 5 levels</li>
                    </ul>
                  </div>
                </div>
                <div className="space-y-4">
                  <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-4">
                    <h4 className="font-semibold text-gray-900 mb-2">Practical Applications</h4>
                    <ul className="text-sm text-gray-700 space-y-1">
                      <li>• Classification and prediction tasks</li>
                      <li>• Business rule extraction</li>
                      <li>• Feature selection guidance</li>
                      <li>• Interpretable machine learning</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl border border-green-200 p-12 text-center">
            <div className="text-6xl mb-6">🌳</div>
            <h3 className="text-xl font-semibold text-gray-900 mb-4">No Decision Tree Results</h3>
            <p className="text-gray-600 mb-6 max-w-md mx-auto">
              Run ML analysis first to build a decision tree model and understand your data's decision patterns.
            </p>
            <button
              onClick={() => setActiveTab('overview')}
              className="px-6 py-3 bg-green-600 text-white font-semibold rounded-lg hover:bg-green-700 transition-colors"
            >
              Go to Analysis
            </button>
          </div>
        )}
      </div>
    </div>
  );

  // Step 3: Data Cleaning Tab
  const renderCleaningTab = () => (
    <div className="p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Step 3: Data Preparation & Cleaning</h2>
          <p className="text-gray-600">Review data quality issues and apply cleaning operations to improve your dataset</p>
        </div>

        {dataQualityReport ? (
          <div className="space-y-8">
            {/* Data Quality Report */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Database className="w-5 h-5 text-blue-600" />
                Data Quality Report
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-blue-50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-blue-600">{dataQualityReport.totalRows.toLocaleString()}</div>
                  <div className="text-sm text-gray-600">Total Rows</div>
                </div>
                <div className="bg-green-50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-green-600">{dataQualityReport.totalColumns}</div>
                  <div className="text-sm text-gray-600">Total Columns</div>
                </div>
                <div className="bg-orange-50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-orange-600">
                    {Object.values(dataQualityReport.missingValues || {}).reduce((a, b) => a + b, 0)}
                  </div>
                  <div className="text-sm text-gray-600">Missing Values</div>
                </div>
                <div className="bg-red-50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-red-600">{dataQualityReport.duplicates}</div>
                  <div className="text-sm text-gray-600">Duplicate Rows</div>
                </div>
              </div>

              {/* Missing Values Details */}
              {Object.keys(dataQualityReport.missingValues || {}).some(col => 
                dataQualityReport.missingValues[col] > 0
              ) && (
                <div className="mb-6">
                  <h4 className="font-medium text-gray-900 mb-3">Missing Values by Column</h4>
                  <div className="space-y-2">
                    {Object.entries(dataQualityReport.missingValues || {})
                      .filter(([col, count]) => count > 0)
                      .map(([col, count]) => (
                        <div key={col} className="flex items-center justify-between bg-orange-50 rounded-lg p-3">
                          <span className="font-medium text-gray-900">{col}</span>
                          <span className="text-orange-600 font-semibold">
                            {count} missing ({((count / dataQualityReport.totalRows) * 100).toFixed(1)}%)
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* Outliers Details */}
              {Object.keys(dataQualityReport.outliers || {}).length > 0 && (
                <div className="mb-6">
                  <h4 className="font-medium text-gray-900 mb-3">Outliers Detected</h4>
                  <div className="space-y-2">
                    {Object.entries(dataQualityReport.outliers || {}).map(([col, count]) => (
                      <div key={col} className="flex items-center justify-between bg-yellow-50 rounded-lg p-3">
                        <span className="font-medium text-gray-900">{col}</span>
                        <span className="text-yellow-600 font-semibold">
                          {count} outliers ({((count / dataQualityReport.totalRows) * 100).toFixed(1)}%)
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Cleaning Options */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Settings className="w-5 h-5 text-purple-600" />
                Data Cleaning Options
              </h3>

              <div className="space-y-6">
                {/* Missing Values Strategy */}
                <div>
                  <label className="block text-sm font-medium text-gray-900 mb-3">
                    Missing Values Strategy
                  </label>
                  <div className="space-y-2">
                    {[
                      { value: 'keep', label: 'Keep as is', desc: 'No changes to missing values' },
                      { value: 'remove_rows', label: 'Remove rows', desc: 'Delete rows containing missing values' },
                      { value: 'fill_mean', label: 'Fill with mean/mode', desc: 'Fill numeric columns with mean, categorical with mode' },
                      { value: 'fill_zero', label: 'Fill with zero/unknown', desc: 'Fill missing values with 0 or "Unknown"' }
                    ].map(option => (
                      <label key={option.value} className="flex items-start gap-3 p-3 border rounded-lg hover:bg-gray-50">
                        <input
                          type="radio"
                          name="missingValues"
                          value={option.value}
                          checked={cleaningOptions.missingValues === option.value}
                          onChange={(e) => setCleaningOptions(prev => ({
                            ...prev,
                            missingValues: e.target.value
                          }))}
                          className="mt-1"
                        />
                        <div>
                          <div className="font-medium text-gray-900">{option.label}</div>
                          <div className="text-sm text-gray-600">{option.desc}</div>
                        </div>
                      </label>
                    ))}
                  </div>
                </div>

                {/* Duplicates Strategy */}
                <div>
                  <label className="block text-sm font-medium text-gray-900 mb-3">
                    Duplicate Rows Strategy
                  </label>
                  <div className="space-y-2">
                    {[
                      { value: 'keep', label: 'Keep duplicates', desc: 'No changes to duplicate rows' },
                      { value: 'remove', label: 'Remove duplicates', desc: 'Keep only unique rows' }
                    ].map(option => (
                      <label key={option.value} className="flex items-start gap-3 p-3 border rounded-lg hover:bg-gray-50">
                        <input
                          type="radio"
                          name="duplicates"
                          value={option.value}
                          checked={cleaningOptions.duplicates === option.value}
                          onChange={(e) => setCleaningOptions(prev => ({
                            ...prev,
                            duplicates: e.target.value
                          }))}
                          className="mt-1"
                        />
                        <div>
                          <div className="font-medium text-gray-900">{option.label}</div>
                          <div className="text-sm text-gray-600">{option.desc}</div>
                        </div>
                      </label>
                    ))}
                  </div>
                </div>

                {/* Outliers Strategy */}
                <div>
                  <label className="block text-sm font-medium text-gray-900 mb-3">
                    Outliers Strategy
                  </label>
                  <div className="space-y-2">
                    {[
                      { value: 'keep', label: 'Keep outliers', desc: 'No changes to outlier values' },
                      { value: 'remove', label: 'Remove outliers', desc: 'Remove rows containing outlier values' },
                      { value: 'cap', label: 'Cap outliers', desc: 'Limit outliers to 95th percentile values' }
                    ].map(option => (
                      <label key={option.value} className="flex items-start gap-3 p-3 border rounded-lg hover:bg-gray-50">
                        <input
                          type="radio"
                          name="outliers"
                          value={option.value}
                          checked={cleaningOptions.outliers === option.value}
                          onChange={(e) => setCleaningOptions(prev => ({
                            ...prev,
                            outliers: e.target.value
                          }))}
                          className="mt-1"
                        />
                        <div>
                          <div className="font-medium text-gray-900">{option.label}</div>
                          <div className="text-sm text-gray-600">{option.desc}</div>
                        </div>
                      </label>
                    ))}
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-4 pt-4 border-t">
                  <button
                    onClick={applyDataCleaning}
                    disabled={loading}
                    className="flex-1 px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white font-semibold rounded-lg hover:from-purple-700 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center gap-2"
                  >
                    {loading ? (
                      <>
                        <Clock className="w-5 h-5 animate-spin" />
                        Cleaning Data...
                      </>
                    ) : (
                      <>
                        <Play className="w-5 h-5" />
                        Apply Cleaning
                      </>
                    )}
                  </button>
                  <button
                    onClick={() => setWorkflowStep(4)}
                    disabled={!cleanedData}
                    className="px-6 py-3 border border-gray-300 text-gray-700 font-semibold rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    Skip Cleaning
                  </button>
                </div>
              </div>
            </div>

            {/* Cleaning Results */}
            {cleanedData && (
              <div className="bg-green-50 border border-green-200 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <CheckCircle className="w-6 h-6 text-green-600" />
                  <h3 className="text-lg font-semibold text-green-900">Data Cleaning Completed</h3>
                </div>
                <p className="text-green-800 mb-4">
                  Your dataset has been successfully cleaned. Dataset size: {data?.length || 0} → {cleanedData.length} rows
                </p>
                <div className="flex gap-4">
                  <button
                    onClick={() => {
                      setWorkflowStep(4);
                      setActiveTab('modeling');
                    }}
                    className="px-6 py-3 bg-green-600 text-white font-semibold rounded-lg hover:bg-green-700 transition-colors flex items-center gap-2"
                  >
                    <Brain className="w-5 h-5" />
                    Proceed to Modeling
                  </button>
                  <button
                    onClick={() => {
                      if (cleanedDataFile) {
                        const blob = new Blob([cleanedDataFile.content], { type: 'text/csv' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = cleanedDataFile.name;
                        a.click();
                        URL.revokeObjectURL(url);
                      } else {
                        // Fallback: generate CSV content on the fly
                        const csvContent = generateCSVContent(cleanedData, columns);
                        const blob = new Blob([csvContent], { type: 'text/csv' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `${fileName.replace('.csv', '')}_cleaned.csv`;
                        a.click();
                        URL.revokeObjectURL(url);
                      }
                    }}
                    className="px-6 py-3 border border-green-600 text-green-600 font-semibold rounded-lg hover:bg-green-50 transition-colors flex items-center gap-2"
                  >
                    <Download className="w-5 h-5" />
                    Download Cleaned CSV
                  </button>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="bg-gray-50 rounded-xl border border-gray-200 p-12 text-center">
            <div className="text-6xl mb-6">🛠️</div>
            <h3 className="text-xl font-semibold text-gray-900 mb-4">No Data Quality Report</h3>
            <p className="text-gray-600 mb-6 max-w-md mx-auto">
              Upload and analyze your data first to see data quality issues and cleaning recommendations.
            </p>
            <button
              onClick={() => setActiveTab('upload')}
              className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors"
            >
              Upload Data
            </button>
          </div>
        )}
      </div>
    </div>
  );

  // Step 4: ML Modeling Tab
  const renderModelingTab = () => (
    <div className="p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Step 4: Machine Learning Modeling</h2>
          <p className="text-gray-600">Train ML models on your cleaned dataset to discover patterns and build predictive models</p>
        </div>

        {(cleanedData || data) ? (
          <div className="space-y-8">
            {/* Dataset Info */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Database className="w-5 h-5 text-green-600" />
                Dataset Ready for Modeling
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-green-50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {(cleanedData || data)?.length.toLocaleString()}
                  </div>
                  <div className="text-sm text-gray-600">Training Samples</div>
                </div>
                <div className="bg-blue-50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-blue-600">{columns.length}</div>
                  <div className="text-sm text-gray-600">Features Available</div>
                </div>
                <div className="bg-purple-50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {cleanedData ? 'Cleaned' : 'Original'}
                  </div>
                  <div className="text-sm text-gray-600">Dataset Type</div>
                </div>
              </div>
            </div>

            {/* Model Training Component */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
              <ModelTrainingComponent
                trainingData={[
                  {
                    filename: cleanedData ? `${fileName}_cleaned` : fileName,
                    shape: [(cleanedData || data).length, columns.length],
                    columns: columns
                  }
                ]}
                onModelTrained={(result) => {
                  setTrainedModel(result);
                  
                  // Save model and training results files
                  const modelFileName = `${fileName.replace('.csv', '')}_model.pkl`;
                  const trainingResultsFileName = `${fileName.replace('.csv', '')}_training_results.csv`;
                  
                  setTrainedModelFile({
                    name: modelFileName,
                    modelId: result.model_id,
                    accuracy: result.accuracy,
                    created_at: new Date().toISOString()
                  });
                  
                  // Generate training results CSV with predictions if available
                  if (result.predictions && (cleanedData || data)) {
                    const trainingDataWithPredictions = (cleanedData || data).map((row, index) => ({
                      ...row,
                      predicted_label: result.predictions[index] || null
                    }));
                    
                    const trainingResultsContent = generateCSVContent(
                      trainingDataWithPredictions, 
                      [...columns, 'predicted_label']
                    );
                    
                    setTrainingResultsFile({
                      name: trainingResultsFileName,
                      content: trainingResultsContent,
                      data: trainingDataWithPredictions
                    });
                  }
                  
                  setWorkflowStep(5);
                  alert(`Model trained successfully!\nModel saved as: ${modelFileName}\nAccuracy: ${(result.accuracy * 100).toFixed(2)}%\nYou can now make predictions.`);
                }}
              />
            </div>

            {/* Quick Analysis Options */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Clustering */}
              <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                  <Layers className="w-5 h-5 text-yellow-600" />
                  Clustering Analysis
                </h3>
                <p className="text-gray-600 mb-4">
                  Discover hidden groups and patterns in your data using unsupervised learning.
                </p>
                <button
                  onClick={() => {
                    runMLAnalysis();
                    setActiveTab('visualization');
                  }}
                  className="w-full px-4 py-2 bg-yellow-100 text-yellow-800 rounded-lg hover:bg-yellow-200 transition-colors"
                >
                  Run Clustering
                </button>
              </div>

              {/* Anomaly Detection */}
              <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-red-600" />
                  Anomaly Detection
                </h3>
                <p className="text-gray-600 mb-4">
                  Identify unusual patterns and outliers that might indicate data quality issues or interesting insights.
                </p>
                <button
                  onClick={() => {
                    runMLAnalysis();
                    setActiveTab('visualization');
                  }}
                  className="w-full px-4 py-2 bg-red-100 text-red-800 rounded-lg hover:bg-red-200 transition-colors"
                >
                  Detect Anomalies
                </button>
              </div>
            </div>

            {/* Progress to Next Step */}
            {trainedModel && (
              <div className="bg-green-50 border border-green-200 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <CheckCircle className="w-6 h-6 text-green-600" />
                  <h3 className="text-lg font-semibold text-green-900">Model Training Completed</h3>
                </div>
                <p className="text-green-800 mb-4">
                  Your machine learning model has been successfully trained and is ready for making predictions.
                </p>
                <button
                  onClick={() => {
                    setWorkflowStep(5);
                    setActiveTab('prediction');
                  }}
                  className="px-6 py-3 bg-green-600 text-white font-semibold rounded-lg hover:bg-green-700 transition-colors flex items-center gap-2"
                >
                  <Target className="w-5 h-5" />
                  Start Making Predictions
                </button>
              </div>
            )}
          </div>
        ) : (
          <div className="bg-gray-50 rounded-xl border border-gray-200 p-12 text-center">
            <div className="text-6xl mb-6">🤖</div>
            <h3 className="text-xl font-semibold text-gray-900 mb-4">No Dataset Available</h3>
            <p className="text-gray-600 mb-6 max-w-md mx-auto">
              Please upload and clean your data first before proceeding with machine learning modeling.
            </p>
            <button
              onClick={() => setActiveTab('cleaning')}
              className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors"
            >
              Go to Data Cleaning
            </button>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-purple-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center space-x-4">
              <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl p-3">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                  ML Analytics Platform
                </h1>
                <p className="text-gray-600 mt-1">Advanced Machine Learning Analysis & Intelligence</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              {fileName && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg px-4 py-2">
                  <span className="text-sm text-blue-800 font-medium">
                    📊 {fileName}
                  </span>
                </div>
              )}
              <div className={`flex items-center gap-2 px-3 py-2 rounded-lg ${
                backendStatus === 'connected' 
                  ? 'bg-green-50 border border-green-200' 
                  : 'bg-red-50 border border-red-200'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  backendStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'
                }`}></div>
                <span className={`text-sm font-medium ${
                  backendStatus === 'connected' ? 'text-green-700' : 'text-red-700'
                }`}>
                  {backendStatus === 'connected' ? 'API Connected' : 'API Offline'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-white rounded-2xl shadow-xl border border-gray-200 overflow-hidden">
          {/* Navigation Tabs - 5-Step Workflow */}
          <div className="border-b border-gray-200 bg-gray-50">
            <nav className="flex overflow-x-auto">
              {[
                { 
                  id: 'upload', 
                  name: 'Step 1: Upload Data', 
                  icon: Upload, 
                  color: 'blue',
                  step: 1,
                  enabled: true 
                },
                { 
                  id: 'overview', 
                  name: 'Step 2: Data Analysis', 
                  icon: FileText, 
                  color: 'indigo',
                  step: 2,
                  enabled: workflowStep >= 2 
                },
                { 
                  id: 'cleaning', 
                  name: 'Step 3: Data Cleaning', 
                  icon: Settings, 
                  color: 'yellow',
                  step: 3,
                  enabled: workflowStep >= 2 && dataQualityReport 
                },
                { 
                  id: 'modeling', 
                  name: 'Step 4: ML Modeling', 
                  icon: Brain, 
                  color: 'purple',
                  step: 4,
                  enabled: workflowStep >= 4 
                },
                { 
                  id: 'prediction', 
                  name: 'Step 5: Predictions', 
                  icon: Target, 
                  color: 'green',
                  step: 5,
                  enabled: workflowStep >= 5 
                },
                // Additional tabs for detailed analysis
                { 
                  id: 'visualization', 
                  name: 'Visualization', 
                  icon: BarChart3, 
                  color: 'pink',
                  step: null,
                  enabled: workflowStep >= 2 
                },
              ].map(tab => (
                <button
                  key={tab.id}
                  onClick={() => tab.enabled && setActiveTab(tab.id)}
                  disabled={!tab.enabled}
                  className={`group relative px-6 py-4 font-medium text-sm whitespace-nowrap flex items-center transition-all duration-300 ${
                    activeTab === tab.id
                      ? `text-${tab.color}-600 bg-white border-b-2 border-${tab.color}-500 shadow-sm`
                      : tab.enabled 
                        ? 'text-gray-700 hover:text-gray-900 hover:bg-white/50 cursor-pointer'
                        : 'text-gray-400 cursor-not-allowed opacity-50'
                  }`}
                >
                  <tab.icon className={`w-4 h-4 mr-2 transition-transform duration-300 ${
                    activeTab === tab.id ? 'scale-110' : tab.enabled ? 'group-hover:scale-105' : ''
                  }`} />
                  <span className="flex items-center gap-2">
                    {tab.name}
                    {tab.step && workflowStep >= tab.step && (
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    )}
                    {tab.step && workflowStep === tab.step && (
                      <Clock className="w-4 h-4 text-blue-500 animate-pulse" />
                    )}
                  </span>
                  {activeTab === tab.id && (
                    <div className={`absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-${tab.color}-500 to-${tab.color}-600`}></div>
                  )}
                </button>
              ))}
            </nav>
          </div>

          {/* Tab Content */}
          <div className="min-h-96 bg-white">
            <div className="relative">
              {/* Background Pattern */}
              <div className="absolute inset-0 opacity-5">
                <div className="absolute inset-0" style={{
                  backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23000000' fill-opacity='0.4'%3E%3Ccircle cx='30' cy='30' r='1.5'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
                }}></div>
              </div>
              
              {/* Content with fade transition */}
              <div className="relative z-10">
                {activeTab === 'upload' && renderUploadTab()}
                {activeTab === 'overview' && renderOverviewTab()}
                {activeTab === 'cleaning' && renderCleaningTab()}
                {activeTab === 'modeling' && renderModelingTab()}
                {activeTab === 'prediction' && renderPredictionTab()}
                {activeTab === 'visualization' && renderVisualizationTab()}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-gray-500 text-sm">
          <p>ML Analytics Platform • Created By Azel</p>
          <p className="mt-1">© 2025 • Built with React & Python Flask</p>
        </div>
      </div>
    </div>
  );
};

export default MLAnalyticsDashboard;