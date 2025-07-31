import React, { useState, useEffect } from 'react';
import MLAnalyticsAPI from './services/api';

const DataTypeCustomizer = ({ data, analysis, onUpdateAnalysis }) => {
  const [customTypes, setCustomTypes] = useState({});
  const [loading, setLoading] = useState(false);
  const [showCustomizer, setShowCustomizer] = useState(false);

  // Available data types for selection
  const availableTypes = [
    { value: 'numeric', label: '🔢 Numeric', description: 'Numbers, prices, quantities, measurements' },
    { value: 'categorical', label: '🏷️ Categorical', description: 'Categories, groups, labels' },
    { value: 'binary', label: '⚡ Binary', description: 'Yes/No, True/False, Male/Female' },
    { value: 'date', label: '📅 Date', description: 'Dates, timestamps, temporal data' },
    { value: 'string', label: '📝 String/Text', description: 'Long text, descriptions, sentences' },
    { value: 'identifier', label: '🔍 Identifier', description: 'IDs, primary keys, unique identifiers' },
    { value: 'foreign_key', label: '🔗 Foreign Key', description: 'Reference keys, lookup values' }
  ];

  // Initialize custom types from current analysis
  useEffect(() => {
    if (analysis) {
      const initialTypes = {};
      
      // Extract current column types from analysis
      Object.keys(analysis).forEach(categoryKey => {
        if (categoryKey !== 'summary' && Array.isArray(analysis[categoryKey])) {
          analysis[categoryKey].forEach(column => {
            const dataType = categoryKey.replace('_columns', '');
            initialTypes[column.column] = dataType;
          });
        }
      });
      
      setCustomTypes(initialTypes);
    }
  }, [analysis]);

  const handleTypeChange = (columnName, newType) => {
    setCustomTypes(prev => ({
      ...prev,
      [columnName]: newType
    }));
  };

  const applyCustomTypes = async () => {
    if (!data || Object.keys(customTypes).length === 0) {
      alert('No changes to apply');
      return;
    }

    setLoading(true);
    try {
      // Only send overrides for columns that have been changed
      const overrides = {};
      
      // Get original types from analysis
      const originalTypes = {};
      if (analysis) {
        Object.keys(analysis).forEach(categoryKey => {
          if (categoryKey !== 'summary' && Array.isArray(analysis[categoryKey])) {
            analysis[categoryKey].forEach(column => {
              const dataType = categoryKey.replace('_columns', '');
              originalTypes[column.column] = dataType;
            });
          }
        });
      }

      // Find columns that have been changed
      Object.keys(customTypes).forEach(columnName => {
        if (originalTypes[columnName] !== customTypes[columnName]) {
          overrides[columnName] = customTypes[columnName];
        }
      });

      if (Object.keys(overrides).length === 0) {
        alert('No changes detected');
        setLoading(false);
        return;
      }

      console.log('Applying custom type overrides:', overrides);
      
      const updatedAnalysis = await MLAnalyticsAPI.updateColumnTypes(data, overrides);
      
      // Update parent component with new analysis
      onUpdateAnalysis(updatedAnalysis);
      
      alert(`Successfully updated ${Object.keys(overrides).length} column types!`);
      setShowCustomizer(false);
      
    } catch (error) {
      console.error('Failed to apply custom types:', error);
      alert(`Failed to apply custom types: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const resetToOriginal = () => {
    if (analysis) {
      const originalTypes = {};
      Object.keys(analysis).forEach(categoryKey => {
        if (categoryKey !== 'summary' && Array.isArray(analysis[categoryKey])) {
          analysis[categoryKey].forEach(column => {
            const dataType = categoryKey.replace('_columns', '');
            originalTypes[column.column] = dataType;
          });
        }
      });
      setCustomTypes(originalTypes);
    }
  };

  const getAllColumns = () => {
    if (!data || data.length === 0) return [];
    return Object.keys(data[0]);
  };

  const getColumnInfo = (columnName) => {
    if (!analysis) return null;
    
    // Find column in analysis
    for (const categoryKey of Object.keys(analysis)) {
      if (categoryKey !== 'summary' && Array.isArray(analysis[categoryKey])) {
        const found = analysis[categoryKey].find(col => col.column === columnName);
        if (found) {
          return {
            ...found,
            category: categoryKey.replace('_columns', '')
          };
        }
      }
    }
    return null;
  };

  const getTypeDescription = (type) => {
    const typeInfo = availableTypes.find(t => t.value === type);
    return typeInfo ? typeInfo.description : '';
  };

  const getTypeLabel = (type) => {
    const typeInfo = availableTypes.find(t => t.value === type);
    return typeInfo ? typeInfo.label : type;
  };

  if (!data || !analysis) {
    return null;
  }

  const allColumns = getAllColumns();

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">
            🔧 Custom Data Type Selection
          </h3>
          <p className="text-sm text-gray-600">
            Customize column data types if the automatic detection needs adjustment
          </p>
        </div>
        <button
          onClick={() => setShowCustomizer(!showCustomizer)}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          {showCustomizer ? 'Hide Customizer' : 'Customize Types'}
        </button>
      </div>

      {showCustomizer && (
        <div className="space-y-4">
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <h4 className="font-medium text-yellow-800 mb-2">📋 Instructions</h4>
            <ul className="text-sm text-yellow-700 space-y-1">
              <li>• Review the automatically detected data types below</li>
              <li>• Change any column type that seems incorrect</li>
              <li>• Click "Apply Changes" to update the analysis</li>
              <li>• The system will re-analyze with your custom settings</li>
            </ul>
          </div>

          <div className="grid gap-4 max-h-96 overflow-y-auto">
            {allColumns.map(columnName => {
              const columnInfo = getColumnInfo(columnName);
              const currentType = customTypes[columnName] || 'categorical';
              const isChanged = columnInfo && columnInfo.category !== currentType;
              
              return (
                <div 
                  key={columnName} 
                  className={`border rounded-lg p-4 ${isChanged ? 'border-orange-300 bg-orange-50' : 'border-gray-200'}`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <h4 className="font-medium text-gray-900">{columnName}</h4>
                        {isChanged && (
                          <span className="px-2 py-1 bg-orange-100 text-orange-700 text-xs rounded">
                            Modified
                          </span>
                        )}
                      </div>
                      
                      {columnInfo && (
                        <div className="text-sm text-gray-600 mb-3">
                          <p><strong>Auto-detected:</strong> {getTypeLabel(columnInfo.category)}</p>
                          <p><strong>Sample values:</strong> {
                            columnInfo.sample_values ? 
                            columnInfo.sample_values.slice(0, 3).join(', ') : 
                            'N/A'
                          }</p>
                          {columnInfo.reasons && (
                            <p><strong>Reasons:</strong> {columnInfo.reasons.join(', ')}</p>
                          )}
                        </div>
                      )}
                      
                      <div className="text-xs text-gray-500">
                        {getTypeDescription(currentType)}
                      </div>
                    </div>
                    
                    <div className="ml-4">
                      <select
                        value={currentType}
                        onChange={(e) => handleTypeChange(columnName, e.target.value)}
                        className="w-48 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      >
                        {availableTypes.map(type => (
                          <option key={type.value} value={type.value}>
                            {type.label}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="flex gap-3 pt-4 border-t border-gray-200">
            <button
              onClick={applyCustomTypes}
              disabled={loading}
              className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Applying Changes...' : 'Apply Changes'}
            </button>
            
            <button
              onClick={resetToOriginal}
              disabled={loading}
              className="px-6 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Reset to Original
            </button>
            
            <button
              onClick={() => setShowCustomizer(false)}
              disabled={loading}
              className="px-6 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {!showCustomizer && (
        <div className="text-sm text-gray-600">
          <p>✅ Current analysis uses automatic data type detection</p>
          <p>Click "Customize Types" to manually adjust column data types if needed</p>
        </div>
      )}
    </div>
  );
};

export default DataTypeCustomizer;
