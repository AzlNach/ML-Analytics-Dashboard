import React, { useState } from 'react';
import { Upload, Brain, Target } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import MLAnalyticsAPI from './services/api';
import ModelTrainingComponent from './ModelTrainingComponent';

const SimpleDashboard = () => {
  const [activeTab, setActiveTab] = useState('upload');

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">ML Analytics Dashboard</h1>
              <p className="text-gray-600">Advanced Machine Learning Analysis with Python Backend</p>
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

          <div className="min-h-96 p-6">
            {activeTab === 'upload' && (
              <div>
                <h2 className="text-xl font-bold mb-4">Upload Data</h2>
                <p>File upload functionality will be here</p>
              </div>
            )}
            {activeTab === 'training' && (
              <div>
                <h2 className="text-xl font-bold mb-4">Train Models</h2>
                <p>Model training functionality will be here</p>
              </div>
            )}
            {activeTab === 'prediction' && (
              <div>
                <h2 className="text-xl font-bold mb-4">Predictions</h2>
                <p>Prediction functionality will be here</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SimpleDashboard;
