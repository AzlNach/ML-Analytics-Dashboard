const API_BASE_URL = 'http://localhost:5000/api';

class MLAnalyticsAPI {
    static async healthCheck() {
        try {
            const response = await fetch(`${API_BASE_URL}/health`);
            return await response.json();
        } catch (error) {
            console.error('Health check failed:', error);
            throw error;
        }
    }

    static async getAlgorithms() {
        try {
            const response = await fetch(`${API_BASE_URL}/algorithms`);
            return await response.json();
        } catch (error) {
            console.error('Failed to get algorithms:', error);
            throw error;
        }
    }

    static async getModelHistory() {
        try {
            const response = await fetch(`${API_BASE_URL}/model-history`);
            return await response.json();
        } catch (error) {
            console.error('Failed to get model history:', error);
            throw error;
        }
    }

    static async getTrainedModels() {
        try {
            const response = await fetch(`${API_BASE_URL}/models`);
            return await response.json();
        } catch (error) {
            console.error('Failed to get trained models:', error);
            throw error;
        }
    }

    static async getModelDetails(modelId) {
        try {
            const response = await fetch(`${API_BASE_URL}/models/${modelId}`);
            return await response.json();
        } catch (error) {
            console.error('Failed to get model details:', error);
            throw error;
        }
    }

    static async deleteModel(modelId) {
        try {
            const response = await fetch(`${API_BASE_URL}/models/${modelId}`, {
                method: 'DELETE'
            });
            return await response.json();
        } catch (error) {
            console.error('Failed to delete model:', error);
            throw error;
        }
    }

    static async analyzeData(csvData) {
        try {
            const response = await fetch(`${API_BASE_URL}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ csv_data: csvData }),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Analysis failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Data analysis failed:', error);
            throw error;
        }
    }

    static async performClustering(csvData, selectedColumns, eps = 0.5, minSamples = 5) {
        try {
            console.log('Performing clustering request with:', {
                dataLength: csvData.length,
                selectedColumns,
                eps,
                minSamples
            });

            const response = await fetch(`${API_BASE_URL}/clustering`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    csv_data: csvData,
                    selected_columns: selectedColumns,
                    eps,
                    min_samples: minSamples,
                }),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Clustering failed');
            }

            const result = await response.json();
            console.log('Clustering response received:', result);
            return result;
        } catch (error) {
            console.error('Clustering failed:', error);
            throw error;
        }
    }

    static async detectAnomalies(csvData, selectedColumns, contamination = 0.1) {
        try {
            console.log('Performing anomaly detection request with:', {
                dataLength: csvData.length,
                selectedColumns,
                contamination
            });

            const response = await fetch(`${API_BASE_URL}/anomaly-detection`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    csv_data: csvData,
                    selected_columns: selectedColumns,
                    contamination,
                }),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Anomaly detection failed');
            }

            const result = await response.json();
            console.log('Anomaly detection response received:', result);
            return result;
        } catch (error) {
            console.error('Anomaly detection failed:', error);
            throw error;
        }
    }

    static async buildDecisionTree(csvData, targetColumn, selectedFeatures, maxDepth = 5) {
        try {
            console.log('Building decision tree with parameters:', {
                targetColumn,
                targetColumnType: typeof targetColumn,
                selectedFeatures,
                maxDepth,
                dataLength: csvData.length
            });

            const response = await fetch(`${API_BASE_URL}/decision-tree`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    csv_data: csvData,
                    target_column: targetColumn,
                    selected_features: selectedFeatures,
                    max_depth: maxDepth,
                }),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Decision tree building failed');
            }

            const result = await response.json();
            console.log('Decision tree response received:', result);
            return result;
        } catch (error) {
            console.error('Decision tree building failed:', error);
            throw error;
        }
    }

    static async makePrediction(modelId, inputData) {
        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model_id: modelId,
                    input_data: inputData,
                }),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Prediction failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Prediction failed:', error);
            throw error;
        }
    }

    static async getTrainingData() {
        try {
            const response = await fetch(`${API_BASE_URL}/training-data`);
            return await response.json();
        } catch (error) {
            console.error('Failed to get training data:', error);
            throw error;
        }
    }

    static async trainFromFile(filename, targetColumn, modelType = 'decision_tree', options = {}) {
        try {
            const response = await fetch(`${API_BASE_URL}/train-from-file`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    filename,
                    target_column: targetColumn,
                    model_type: modelType,
                    test_size: options.test_size || 0.2,
                    cross_validation: options.cross_validation !== false,
                    ...options
                }),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Training failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Training failed:', error);
            throw error;
        }
    }

    static async generateDataQualityReport(csvData) {
        try {
            const response = await fetch(`${API_BASE_URL}/data-quality-report`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ csv_data: csvData }),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Quality report generation failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Quality report generation failed:', error);
            throw error;
        }
    }

    static async cleanData(csvData, cleaningOptions) {
        try {
            const response = await fetch(`${API_BASE_URL}/clean-data`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    csv_data: csvData,
                    cleaning_options: cleaningOptions
                }),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Data cleaning failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Data cleaning failed:', error);
            throw error;
        }
    }
}

export default MLAnalyticsAPI;