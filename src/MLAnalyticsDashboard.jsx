import React, { useState, useCallback, useEffect } from 'react';
import { Upload, FileText, BarChart3, Brain, AlertTriangle, Layers, TreePine, Eye, Download, RefreshCw } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import * as math from 'mathjs';

const MLAnalyticsDashboard = () => {
  const [data, setData] = useState(null);
  const [fileName, setFileName] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [mlResults, setMLResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('upload');
  const [columns, setColumns] = useState([]);
  const [selectedColumns, setSelectedColumns] = useState([]);

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

  // Basic statistics calculation
  const calculateBasicStats = (data, columns) => {
    const stats = {};
    const numericColumns = [];
    const categoricalColumns = [];
    
    columns.forEach(col => {
      const values = data.map(row => row[col]).filter(v => v !== null && v !== undefined);
      const numericValues = values.filter(v => typeof v === 'number' && !isNaN(v));
      
      if (numericValues.length > values.length * 0.8) {
        numericColumns.push(col);
        stats[col] = {
          type: 'numeric',
          mean: math.mean(numericValues),
          std: math.std(numericValues),
          min: math.min(numericValues),
          max: math.max(numericValues),
          median: math.median(numericValues),
          count: numericValues.length
        };
      } else {
        categoricalColumns.push(col);
        const uniqueValues = [...new Set(values)];
        const valueCounts = {};
        values.forEach(v => {
          valueCounts[v] = (valueCounts[v] || 0) + 1;
        });
        stats[col] = {
          type: 'categorical',
          unique: uniqueValues.length,
          most_common: Object.entries(valueCounts).sort((a, b) => b[1] - a[1]).slice(0, 5),
          count: values.length
        };
      }
    });
    
    return { stats, numericColumns, categoricalColumns };
  };

  // DBSCAN Implementation (simplified)
  const dbscan = (data, eps = 0.5, minPts = 5) => {
    const points = data.map((row, index) => ({ ...row, _id: index, cluster: -1, visited: false }));
    let clusterCount = 0;
    
    const distance = (p1, p2, columns) => {
      let sum = 0;
      columns.forEach(col => {
        if (typeof p1[col] === 'number' && typeof p2[col] === 'number') {
          sum += Math.pow(p1[col] - p2[col], 2);
        }
      });
      return Math.sqrt(sum);
    };
    
    const getNeighbors = (point, points, eps, columns) => {
      return points.filter(p => distance(point, p, columns) <= eps);
    };
    
    points.forEach(point => {
      if (point.visited) return;
      point.visited = true;
      
      const neighbors = getNeighbors(point, points, eps, selectedColumns);
      
      if (neighbors.length < minPts) {
        point.cluster = -1; // Noise/anomaly
      } else {
        point.cluster = clusterCount;
        clusterCount++;
        
        for (let i = 0; i < neighbors.length; i++) {
          const neighbor = neighbors[i];
          if (!neighbor.visited) {
            neighbor.visited = true;
            const neighborNeighbors = getNeighbors(neighbor, points, eps, selectedColumns);
            if (neighborNeighbors.length >= minPts) {
              neighbors.push(...neighborNeighbors.filter(n => !neighbors.includes(n)));
            }
          }
          if (neighbor.cluster === undefined) {
            neighbor.cluster = point.cluster;
          }
        }
      }
    });
    
    return points;
  };

  // Isolation Forest (simplified implementation)
  const isolationForest = (data, numTrees = 10, sampleSize = 256) => {
    const buildIsolationTree = (sample, depth = 0, maxDepth = Math.ceil(Math.log2(sampleSize))) => {
      if (sample.length <= 1 || depth >= maxDepth) {
        return { type: 'leaf', size: sample.length };
      }
      
      const randomCol = selectedColumns[Math.floor(Math.random() * selectedColumns.length)];
      const colValues = sample.map(row => row[randomCol]).filter(v => typeof v === 'number');
      
      if (colValues.length === 0) return { type: 'leaf', size: sample.length };
      
      const minVal = Math.min(...colValues);
      const maxVal = Math.max(...colValues);
      const splitPoint = Math.random() * (maxVal - minVal) + minVal;
      
      const left = sample.filter(row => typeof row[randomCol] === 'number' && row[randomCol] < splitPoint);
      const right = sample.filter(row => typeof row[randomCol] === 'number' && row[randomCol] >= splitPoint);
      
      return {
        type: 'internal',
        column: randomCol,
        splitPoint,
        left: buildIsolationTree(left, depth + 1, maxDepth),
        right: buildIsolationTree(right, depth + 1, maxDepth)
      };
    };
    
    const getPathLength = (point, tree, depth = 0) => {
      if (tree.type === 'leaf') {
        return depth + (tree.size > 1 ? Math.log2(tree.size) : 0);
      }
      
      if (typeof point[tree.column] === 'number' && point[tree.column] < tree.splitPoint) {
        return getPathLength(point, tree.left, depth + 1);
      } else {
        return getPathLength(point, tree.right, depth + 1);
      }
    };
    
    const trees = [];
    for (let i = 0; i < numTrees; i++) {
      const sample = data.sort(() => 0.5 - Math.random()).slice(0, Math.min(sampleSize, data.length));
      trees.push(buildIsolationTree(sample));
    }
    
    return data.map(point => {
      const avgPathLength = trees.reduce((sum, tree) => sum + getPathLength(point, tree), 0) / numTrees;
      const c = 2 * (Math.log(sampleSize - 1) + 0.5772156649) - (2 * (sampleSize - 1)) / sampleSize;
      const anomalyScore = Math.pow(2, -avgPathLength / c);
      return { ...point, anomalyScore, isAnomaly: anomalyScore > 0.6 };
    });
  };

  // Decision Tree (simplified for demonstration)
  const buildDecisionTree = (data, targetColumn, maxDepth = 5) => {
    if (!targetColumn || data.length === 0) return null;
    
    const calculateEntropy = (subset) => {
      const counts = {};
      subset.forEach(row => {
        const value = row[targetColumn];
        counts[value] = (counts[value] || 0) + 1;
      });
      
      const total = subset.length;
      let entropy = 0;
      Object.values(counts).forEach(count => {
        const p = count / total;
        if (p > 0) entropy -= p * Math.log2(p);
      });
      return entropy;
    };
    
    const findBestSplit = (data) => {
      let bestGain = 0;
      let bestColumn = null;
      let bestSplitValue = null;
      
      selectedColumns.forEach(col => {
        const values = [...new Set(data.map(row => row[col]))].filter(v => typeof v === 'number').sort((a, b) => a - b);
        
        values.forEach((value, index) => {
          if (index === values.length - 1) return;
          
          const left = data.filter(row => typeof row[col] === 'number' && row[col] <= value);
          const right = data.filter(row => typeof row[col] === 'number' && row[col] > value);
          
          if (left.length === 0 || right.length === 0) return;
          
          const parentEntropy = calculateEntropy(data);
          const leftEntropy = calculateEntropy(left);
          const rightEntropy = calculateEntropy(right);
          
          const weightedEntropy = (left.length / data.length) * leftEntropy + 
                                 (right.length / data.length) * rightEntropy;
          const informationGain = parentEntropy - weightedEntropy;
          
          if (informationGain > bestGain) {
            bestGain = informationGain;
            bestColumn = col;
            bestSplitValue = value;
          }
        });
      });
      
      return { bestColumn, bestSplitValue, bestGain };
    };
    
    const buildTree = (subset, depth = 0) => {
      if (subset.length === 0 || depth >= maxDepth) {
        const counts = {};
        subset.forEach(row => {
          const value = row[targetColumn];
          counts[value] = (counts[value] || 0) + 1;
        });
        const prediction = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
        return { type: 'leaf', prediction, samples: subset.length };
      }
      
      const { bestColumn, bestSplitValue, bestGain } = findBestSplit(subset);
      
      if (bestGain === 0) {
        const counts = {};
        subset.forEach(row => {
          const value = row[targetColumn];
          counts[value] = (counts[value] || 0) + 1;
        });
        const prediction = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
        return { type: 'leaf', prediction, samples: subset.length };
      }
      
      const left = subset.filter(row => typeof row[bestColumn] === 'number' && row[bestColumn] <= bestSplitValue);
      const right = subset.filter(row => typeof row[bestColumn] === 'number' && row[bestColumn] > bestSplitValue);
      
      return {
        type: 'internal',
        column: bestColumn,
        splitValue: bestSplitValue,
        left: buildTree(left, depth + 1),
        right: buildTree(right, depth + 1),
        samples: subset.length
      };
    };
    
    return buildTree(data);
  };

  const runMLAnalysis = async () => {
    if (!data || selectedColumns.length === 0) return;
    
    setLoading(true);
    
    // Simulate async processing
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    try {
      const numericData = data.filter(row => 
        selectedColumns.every(col => typeof row[col] === 'number' && !isNaN(row[col]))
      );
      
      // DBSCAN Clustering
      const clusteredData = dbscan(numericData, 1.0, 3);
      const clusterCounts = {};
      clusteredData.forEach(point => {
        const cluster = point.cluster === -1 ? 'Anomaly' : `Cluster ${point.cluster}`;
        clusterCounts[cluster] = (clusterCounts[cluster] || 0) + 1;
      });
      
      // Isolation Forest
      const isolationResults = isolationForest(numericData);
      const anomalies = isolationResults.filter(point => point.isAnomaly);
      
      // Decision Tree (using first categorical column as target if available)
      const { categoricalColumns } = calculateBasicStats(data, columns);
      const decisionTree = categoricalColumns.length > 0 ? 
        buildDecisionTree(numericData, categoricalColumns[0]) : null;
      
      setMLResults({
        clustering: {
          data: clusteredData,
          clusterCounts,
          totalClusters: Object.keys(clusterCounts).length
        },
        anomalyDetection: {
          data: isolationResults,
          anomalies,
          anomalyRate: (anomalies.length / isolationResults.length * 100).toFixed(2)
        },
        decisionTree: decisionTree ? {
          tree: decisionTree,
          targetColumn: categoricalColumns[0]
        } : null
      });
      
    } catch (error) {
      console.error('ML Analysis error:', error);
    }
    
    setLoading(false);
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'text/csv') {
      setFileName(file.name);
      const reader = new FileReader();
      reader.onload = (e) => {
        const { headers, rows } = parseCSV(e.target.result);
        setColumns(headers);
        setData(rows);
        setSelectedColumns(headers.slice(0, 3)); // Auto-select first 3 columns
        
        const basicAnalysis = calculateBasicStats(rows, headers);
        setAnalysis(basicAnalysis);
        setActiveTab('overview');
      };
      reader.readAsText(file);
    }
  };

  const renderOverview = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-blue-50 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <FileText className="h-5 w-5 text-blue-600" />
            <h3 className="font-semibold text-blue-800">Dataset Info</h3>
          </div>
          <p className="text-2xl font-bold text-blue-600">{data?.length || 0}</p>
          <p className="text-sm text-blue-600">Total Rows</p>
        </div>
        
        <div className="bg-green-50 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <BarChart3 className="h-5 w-5 text-green-600" />
            <h3 className="font-semibold text-green-800">Columns</h3>
          </div>
          <p className="text-2xl font-bold text-green-600">{columns.length}</p>
          <p className="text-sm text-green-600">Total Features</p>
        </div>
        
        <div className="bg-purple-50 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <Brain className="h-5 w-5 text-purple-600" />
            <h3 className="font-semibold text-purple-800">ML Ready</h3>
          </div>
          <p className="text-2xl font-bold text-purple-600">{analysis?.numericColumns?.length || 0}</p>
          <p className="text-sm text-purple-600">Numeric Features</p>
        </div>
      </div>

      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold mb-4">Column Selection for ML Analysis</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mb-4">
          {analysis?.numericColumns?.map(col => (
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
            </label>
          ))}
        </div>
        <button
          onClick={runMLAnalysis}
          disabled={loading || selectedColumns.length === 0}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 flex items-center space-x-2"
        >
          {loading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Brain className="h-4 w-4" />}
          <span>{loading ? 'Analyzing...' : 'Run ML Analysis'}</span>
        </button>
      </div>

      {analysis && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h3 className="text-lg font-semibold mb-4">Numeric Columns Statistics</h3>
            <div className="space-y-3">
              {analysis.numericColumns.map(col => (
                <div key={col} className="border-b pb-2">
                  <h4 className="font-medium">{col}</h4>
                  <div className="text-sm text-gray-600 grid grid-cols-2 gap-2">
                    <span>Mean: {analysis.stats[col].mean.toFixed(2)}</span>
                    <span>Std: {analysis.stats[col].std.toFixed(2)}</span>
                    <span>Min: {analysis.stats[col].min.toFixed(2)}</span>
                    <span>Max: {analysis.stats[col].max.toFixed(2)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h3 className="text-lg font-semibold mb-4">Categorical Columns Info</h3>
            <div className="space-y-3">
              {analysis.categoricalColumns.map(col => (
                <div key={col} className="border-b pb-2">
                  <h4 className="font-medium">{col}</h4>
                  <div className="text-sm text-gray-600">
                    <span>Unique Values: {analysis.stats[col].unique}</span>
                    <div className="mt-1">
                      Top Values: {analysis.stats[col].most_common.slice(0, 3).map(([val, count]) => `${val} (${count})`).join(', ')}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const renderClustering = () => {
    if (!mlResults?.clustering) return <div>Run ML analysis first</div>;
    
    const clusterData = Object.entries(mlResults.clustering.clusterCounts).map(([name, value]) => ({
      name, value
    }));
    
    const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];
    
    return (
      <div className="space-y-6">
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
            <Layers className="h-5 w-5" />
            <span>DBSCAN Clustering Results</span>
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-2">Cluster Distribution</h4>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={clusterData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({name, percent}) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {clusterData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div>
              <h4 className="font-medium mb-2">Cluster Summary</h4>
              <div className="space-y-2">
                {Object.entries(mlResults.clustering.clusterCounts).map(([cluster, count]) => (
                  <div key={cluster} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                    <span className="font-medium">{cluster}</span>
                    <span className="text-gray-600">{count} points</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          {selectedColumns.length >= 2 && (
            <div className="mt-6">
              <h4 className="font-medium mb-2">Cluster Visualization</h4>
              <ResponsiveContainer width="100%" height={400}>
                <ScatterChart data={mlResults.clustering.data}>
                  <CartesianGrid />
                  <XAxis dataKey={selectedColumns[0]} />
                  <YAxis dataKey={selectedColumns[1]} />
                  <Tooltip />
                  <Scatter dataKey={selectedColumns[1]} fill="#8884d8" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderAnomalyDetection = () => {
    if (!mlResults?.anomalyDetection) return <div>Run ML analysis first</div>;
    
    return (
      <div className="space-y-6">
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
            <AlertTriangle className="h-5 w-5" />
            <span>Isolation Forest Anomaly Detection</span>
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-red-50 p-4 rounded-lg">
              <h4 className="font-semibold text-red-800">Anomalies Found</h4>
              <p className="text-2xl font-bold text-red-600">{mlResults.anomalyDetection.anomalies.length}</p>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <h4 className="font-semibold text-green-800">Normal Points</h4>
              <p className="text-2xl font-bold text-green-600">
                {mlResults.anomalyDetection.data.length - mlResults.anomalyDetection.anomalies.length}
              </p>
            </div>
            <div className="bg-blue-50 p-4 rounded-lg">
              <h4 className="font-semibold text-blue-800">Anomaly Rate</h4>
              <p className="text-2xl font-bold text-blue-600">{mlResults.anomalyDetection.anomalyRate}%</p>
            </div>
          </div>
          
          {selectedColumns.length >= 2 && (
            <div>
              <h4 className="font-medium mb-2">Anomaly Visualization</h4>
              <ResponsiveContainer width="100%" height={400}>
                <ScatterChart data={mlResults.anomalyDetection.data}>
                  <CartesianGrid />
                  <XAxis dataKey={selectedColumns[0]} />
                  <YAxis dataKey={selectedColumns[1]} />
                  <Tooltip 
                    content={({active, payload}) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="bg-white p-2 border rounded shadow">
                            <p>{`${selectedColumns[0]}: ${data[selectedColumns[0]]}`}</p>
                            <p>{`${selectedColumns[1]}: ${data[selectedColumns[1]]}`}</p>
                            <p>{`Anomaly Score: ${data.anomalyScore?.toFixed(3)}`}</p>
                            <p className={data.isAnomaly ? 'text-red-600' : 'text-green-600'}>
                              {data.isAnomaly ? 'Anomaly' : 'Normal'}
                            </p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Scatter 
                    dataKey={selectedColumns[1]} 
                    fill={(entry) => entry.isAnomaly ? '#FF4444' : '#44FF44'}
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}
          
          {mlResults.anomalyDetection.anomalies.length > 0 && (
            <div className="mt-6">
              <h4 className="font-medium mb-2">Top Anomalies</h4>
              <div className="overflow-x-auto">
                <table className="min-w-full table-auto border-collapse border border-gray-300">
                  <thead>
                    <tr className="bg-gray-100">
                      <th className="border border-gray-300 px-4 py-2">Index</th>
                      {selectedColumns.map(col => (
                        <th key={col} className="border border-gray-300 px-4 py-2">{col}</th>
                      ))}
                      <th className="border border-gray-300 px-4 py-2">Anomaly Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {mlResults.anomalyDetection.anomalies.slice(0, 10).map((anomaly, index) => (
                      <tr key={index}>
                        <td className="border border-gray-300 px-4 py-2">{anomaly._id}</td>
                        {selectedColumns.map(col => (
                          <td key={col} className="border border-gray-300 px-4 py-2">
                            {typeof anomaly[col] === 'number' ? anomaly[col].toFixed(2) : anomaly[col]}
                          </td>
                        ))}
                        <td className="border border-gray-300 px-4 py-2">
                          {anomaly.anomalyScore.toFixed(3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

const renderDecisionTree = () => {
    if (!mlResults?.decisionTree) {
      return (
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
            <TreePine className="h-5 w-5" />
            <span>Decision Tree Analysis</span>
          </h3>
          <p className="text-gray-600">No categorical target variable found for decision tree analysis.</p>
          <p className="text-sm text-gray-500 mt-2">
            Decision trees require at least one categorical column to use as a target variable.
          </p>
        </div>
      );
    }

    const renderTreeNode = (node, depth = 0, position = 'root') => {
      if (!node) return null;
      
      const marginLeft = depth * 30;
      const nodeId = `${depth}-${position}-${Math.random()}`;
      
      if (node.type === 'leaf') {
        return (
          <div key={nodeId} style={{marginLeft}} className="my-2">
            <div className="bg-green-100 border-2 border-green-400 rounded-lg px-4 py-3 inline-block shadow-sm">
              <div className="flex items-center space-x-2">
                <Eye className="h-4 w-4 text-green-600" />
                <span className="font-semibold text-green-800">Leaf Node</span>
              </div>
              <div className="mt-2">
                <strong className="text-green-900">Prediction:</strong> 
                <span className="ml-2 bg-green-200 px-2 py-1 rounded text-green-900 font-medium">
                  {node.prediction}
                </span>
              </div>
              <div className="text-sm text-green-700 mt-1">
                Samples: {node.samples}
              </div>
            </div>
          </div>
        );
      }
      
      return (
        <div key={nodeId} style={{marginLeft}} className="my-3">
          <div className="bg-blue-50 border-2 border-blue-400 rounded-lg px-4 py-3 inline-block shadow-sm">
            <div className="flex items-center space-x-2 mb-2">
              <TreePine className="h-4 w-4 text-blue-600" />
              <span className="font-semibold text-blue-800">Decision Node</span>
            </div>
            <div className="bg-blue-100 px-3 py-2 rounded">
              <strong className="text-blue-900">{node.column}</strong> 
              <span className="text-blue-700"> ≤ </span>
              <span className="font-mono bg-blue-200 px-2 py-1 rounded text-blue-900">
                {typeof node.splitValue === 'number' ? node.splitValue.toFixed(3) : node.splitValue}
              </span>
            </div>
            <div className="text-sm text-blue-600 mt-2">
              Total Samples: {node.samples}
            </div>
          </div>
          
          <div className="ml-8 mt-4 border-l-2 border-gray-300 pl-4">
            <div className="mb-4">
              <div className="flex items-center space-x-2 mb-2">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <span className="text-sm font-medium text-green-700">TRUE Branch</span>
              </div>
              {renderTreeNode(node.left, depth + 1, 'left')}
            </div>
            
            <div className="mb-4">
              <div className="flex items-center space-x-2 mb-2">
                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                <span className="text-sm font-medium text-red-700">FALSE Branch</span>
              </div>
              {renderTreeNode(node.right, depth + 1, 'right')}
            </div>
          </div>
        </div>
      );
    };

    const getTreeStats = (node) => {
      if (!node) return { nodes: 0, leaves: 0, depth: 0 };
      
      if (node.type === 'leaf') {
        return { nodes: 1, leaves: 1, depth: 1 };
      }
      
      const leftStats = getTreeStats(node.left);
      const rightStats = getTreeStats(node.right);
      
      return {
        nodes: 1 + leftStats.nodes + rightStats.nodes,
        leaves: leftStats.leaves + rightStats.leaves,
        depth: 1 + Math.max(leftStats.depth, rightStats.depth)
      };
    };

    const treeStats = getTreeStats(mlResults.decisionTree.tree);

  };

  const renderVisualization = () => {
    if (!data || !analysis) return <div>No data available for visualization</div>;

    const numericData = analysis.numericColumns.slice(0, 5).map(col => ({
      name: col,
      ...data.slice(0, 50).reduce((acc, row, index) => {
        acc[`point_${index}`] = row[col];
        return acc;
      }, {})
    }));

    return (
      <div className="space-y-6">
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold mb-4">Data Visualization</h3>
          
          {analysis.numericColumns.length >= 2 && (
            <div className="mb-6">
              <h4 className="font-medium mb-2">Correlation Analysis</h4>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart data={data.slice(0, 100)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey={analysis.numericColumns[0]} />
                  <YAxis dataKey={analysis.numericColumns[1]} />
                  <Tooltip />
                  <Scatter dataKey={analysis.numericColumns[1]} fill="#8884d8" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}

          {analysis.numericColumns.length > 0 && (
            <div className="mb-6">
              <h4 className="font-medium mb-2">Distribution Analysis</h4>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={data.slice(0, 20)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="_index" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  {analysis.numericColumns.slice(0, 3).map((col, index) => (
                    <Bar key={col} dataKey={col} fill={`hsl(${index * 120}, 70%, 50%)`} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {analysis.numericColumns.length > 0 && (
            <div>
              <h4 className="font-medium mb-2">Trend Analysis</h4>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={data.slice(0, 50).map((row, index) => ({ ...row, _index: index }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="_index" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  {analysis.numericColumns.slice(0, 3).map((col, index) => (
                    <Line 
                      key={col} 
                      type="monotone" 
                      dataKey={col} 
                      stroke={`hsl(${index * 120}, 70%, 50%)`}
                      strokeWidth={2}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
    );
  };

  const exportResults = () => {
    if (!mlResults) return;
    
    const exportData = {
      timestamp: new Date().toISOString(),
      filename: fileName,
      dataInfo: {
        totalRows: data.length,
        totalColumns: columns.length,
        numericColumns: analysis.numericColumns.length,
        categoricalColumns: analysis.categoricalColumns.length
      },
      mlResults: {
        clustering: {
          totalClusters: mlResults.clustering?.totalClusters || 0,
          clusterDistribution: mlResults.clustering?.clusterCounts || {}
        },
        anomalyDetection: {
          totalAnomalies: mlResults.anomalyDetection?.anomalies?.length || 0,
          anomalyRate: mlResults.anomalyDetection?.anomalyRate || 0
        },
        decisionTree: {
          hasTree: !!mlResults.decisionTree,
          targetColumn: mlResults.decisionTree?.targetColumn || null
        }
      }
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ml_analysis_${fileName.replace('.csv', '')}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto p-6">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            ML Analytics Dashboard
          </h1>
          <p className="text-gray-600">
            Upload CSV data and automatically generate machine learning insights with DBSCAN clustering, 
            Isolation Forest anomaly detection, and Decision Tree analysis.
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="bg-white rounded-lg shadow-sm mb-6">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8 px-6">
              {[
                { id: 'upload', name: 'Upload Data', icon: Upload },
                { id: 'overview', name: 'Overview', icon: FileText },
                { id: 'visualization', name: 'Visualization', icon: BarChart3 },
                { id: 'clustering', name: 'Clustering', icon: Layers },
                { id: 'anomaly', name: 'Anomaly Detection', icon: AlertTriangle },
                { id: 'tree', name: 'Decision Tree', icon: TreePine }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2`}
                >
                  <tab.icon className="h-4 w-4" />
                  <span>{tab.name}</span>
                </button>
              ))}
            </nav>
          </div>
        </div>

        {/* Main Content */}
        <div className="min-h-96">
          {activeTab === 'upload' && (
            <div className="bg-white p-8 rounded-lg shadow-sm border-2 border-dashed border-gray-300 text-center">
              <Upload className="mx-auto h-16 w-16 text-gray-400 mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">Upload Your CSV File</h3>
              <p className="text-gray-500 mb-4">
                Select a CSV file to begin automatic machine learning analysis
              </p>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="hidden"
                id="csv-upload"
              />
              <label
                htmlFor="csv-upload"
                className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 cursor-pointer inline-flex items-center space-x-2"
              >
                <Upload className="h-4 w-4" />
                <span>Choose CSV File</span>
              </label>
              {fileName && (
                <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
                  <p className="text-green-800">
                    <strong>File loaded:</strong> {fileName}
                  </p>
                  <p className="text-green-600 text-sm">
                    Ready for analysis - switch to Overview tab to continue
                  </p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'overview' && renderOverview()}
          {activeTab === 'visualization' && renderVisualization()}
          {activeTab === 'clustering' && renderClustering()}
          {activeTab === 'anomaly' && renderAnomalyDetection()}
          {activeTab === 'tree' && renderDecisionTree()}
        </div>

        {/* Export Results */}
        {mlResults && (
          <div className="mt-8 bg-white p-6 rounded-lg shadow-sm border">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold">Analysis Complete</h3>
                <p className="text-gray-600">Export your ML analysis results</p>
              </div>
              <button
                onClick={exportResults}
                className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 flex items-center space-x-2"
              >
                <Download className="h-4 w-4" />
                <span>Export Results</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MLAnalyticsDashboard;