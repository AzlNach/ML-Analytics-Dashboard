# 🖥️ Frontend Setup & Configuration

## 📑 Daftar Isi

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Instalasi](#instalasi)
- [Struktur Frontend](#struktur-frontend)
- [Komponen Utama](#komponen-utama)
- [Styling & CSS](#styling--css)
- [Development Workflow](#development-workflow)
- [Build & Deployment](#build--deployment)
- [Troubleshooting](#troubleshooting)

---

## Overview

Frontend ML Analytics Dashboard dibangun menggunakan **React 18** dengan komponen modern dan responsive design. Aplikasi ini menyediakan interface interaktif untuk analisis data, training model ML, dan prediksi.

---

## Prerequisites

- **Node.js 16+** - JavaScript runtime
- **npm 7+** - Package manager
- **Modern Browser** - Chrome, Firefox, Safari, Edge

---

## Instalasi

### 1. Install Dependencies
```bash
# Install semua dependencies
npm install

# Atau dengan yarn
yarn install
```

### 2. Environment Variables (Optional)
Buat file `.env` di root directory:
```env
REACT_APP_API_URL=http://localhost:5000
REACT_APP_VERSION=1.0.0
```

### 3. Development Server
```bash
# Start development server
npm start

# Aplikasi akan berjalan di http://localhost:3000
```

---

## Struktur Frontend

```
src/
├── App.js                         # Main React App component
├── index.js                       # Entry point
├── index.css                      # Global styles
├── dashboard.css                  # Dashboard-specific styles
├── MLAnalyticsDashboard.jsx       # Main dashboard component
├── ModelTrainingComponent.jsx     # ML model training interface
├── PredictionComponent.jsx        # Prediction interface
├── SimpleDashboard.jsx           # Alternative simple dashboard
└── services/
    └── api.js                     # API service functions
```

---

## Komponen Utama

### 1. **MLAnalyticsDashboard.jsx**
Komponen utama dashboard dengan tab-based navigation:

**Features:**
- 5-step workflow navigation
- Data upload dan preview
- EDA visualization
- Model training interface
- Prediction functionality

**State Management:**
```javascript
const [currentStep, setCurrentStep] = useState(1);
const [uploadedData, setUploadedData] = useState(null);
const [analysisResults, setAnalysisResults] = useState(null);
const [trainedModel, setTrainedModel] = useState(null);
```

### 2. **ModelTrainingComponent.jsx**
Interface untuk training machine learning models:

**Features:**
- Algorithm selection (Decision Tree, Random Forest, etc.)
- Target column selection
- Hyperparameter tuning
- Cross-validation results
- Model evaluation metrics

### 3. **PredictionComponent.jsx**
Interface untuk melakukan prediksi:

**Features:**
- Individual prediction input
- Batch prediction via file upload
- Confidence scores display
- Results export functionality

### 4. **services/api.js**
Service layer untuk komunikasi dengan backend:

```javascript
// Contoh API functions
export const uploadData = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${API_URL}/analyze`, {
    method: 'POST',
    body: formData,
  });
  
  return response.json();
};

export const trainModel = async (config) => {
  const response = await fetch(`${API_URL}/train_model`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(config),
  });
  
  return response.json();
};
```

---

## Styling & CSS

### 1. **Global Styles (index.css)**
```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
}
```

### 2. **Dashboard Styles (dashboard.css)**
```css
.dashboard-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  background: white;
  border-radius: 15px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}

.step-navigation {
  display: flex;
  justify-content: space-between;
  margin-bottom: 30px;
  background: #f8f9fa;
  border-radius: 10px;
  padding: 10px;
}

.step {
  flex: 1;
  text-align: center;
  padding: 15px;
  margin: 0 5px;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.step.active {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
}
```

### 3. **Responsive Design**
```css
/* Mobile First Approach */
@media (max-width: 768px) {
  .dashboard-container {
    margin: 10px;
    padding: 15px;
  }
  
  .step-navigation {
    flex-direction: column;
  }
  
  .step {
    margin: 2px 0;
  }
}

@media (min-width: 1200px) {
  .dashboard-container {
    max-width: 1400px;
  }
}
```

---

## Development Workflow

### 1. **Component Development**
```bash
# Buat komponen baru
touch src/components/NewComponent.jsx

# Import dan gunakan dalam App.js
import NewComponent from './components/NewComponent';
```

### 2. **State Management**
Menggunakan React Hooks untuk state management:

```javascript
// Custom hook untuk API calls
const useApi = (url) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    try {
      const response = await fetch(url);
      const result = await response.json();
      setData(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return { data, loading, error, fetchData };
};
```

### 3. **Event Handling**
```javascript
// File upload handler
const handleFileUpload = async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  try {
    setUploading(true);
    const result = await uploadData(file);
    setUploadedData(result);
    setCurrentStep(2); // Move to next step
  } catch (error) {
    console.error('Upload failed:', error);
    setError('Upload failed. Please try again.');
  } finally {
    setUploading(false);
  }
};
```

---

## Build & Deployment

### 1. **Production Build**
```bash
# Build untuk production
npm run build

# Test production build locally
npx serve -s build
```

### 2. **Environment Configuration**
```javascript
// src/config/index.js
const config = {
  development: {
    API_URL: 'http://localhost:5000',
  },
  production: {
    API_URL: 'https://your-backend-url.com',
  }
};

export default config[process.env.NODE_ENV || 'development'];
```

### 3. **Deployment Options**
```bash
# Deploy to Netlify
npm run build
# Upload build folder to Netlify

# Deploy to Vercel
npm install -g vercel
vercel --prod

# Deploy to GitHub Pages
npm install --save-dev gh-pages
npm run build
npm run deploy
```

---

## Troubleshooting

### 1. **Common Issues**

**Problem**: `Module not found` errors
```bash
# Solution: Clear cache dan reinstall
rm -rf node_modules package-lock.json
npm install
```

**Problem**: CORS errors
```javascript
// Solution: Update API service dengan proper headers
const response = await fetch(url, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
  },
  body: JSON.stringify(data),
});
```

**Problem**: Build fails
```bash
# Solution: Check untuk syntax errors
npm run lint
npm run test
```

### 2. **Development Tools**

**React Developer Tools**: Browser extension untuk debugging
**Chrome DevTools**: Performance monitoring dan debugging
**VS Code Extensions**:
- ES7+ React/Redux/React-Native snippets
- Prettier - Code formatter
- Auto Rename Tag

### 3. **Performance Optimization**

```javascript
// Lazy loading components
const ModelTrainingComponent = React.lazy(() => 
  import('./ModelTrainingComponent')
);

// Memoization untuk expensive calculations
const expensiveCalculation = useMemo(() => {
  return heavyFunction(data);
}, [data]);

// Debouncing untuk search inputs
const debouncedSearch = useCallback(
  debounce((query) => {
    performSearch(query);
  }, 300),
  []
);
```

---

## Testing

### 1. **Unit Testing**
```bash
# Run tests
npm test

# Run tests dengan coverage
npm test -- --coverage
```

### 2. **Test Examples**
```javascript
// src/__tests__/MLAnalyticsDashboard.test.js
import { render, screen, fireEvent } from '@testing-library/react';
import MLAnalyticsDashboard from '../MLAnalyticsDashboard';

test('renders dashboard with steps', () => {
  render(<MLAnalyticsDashboard />);
  expect(screen.getByText('Step 1: Upload Data')).toBeInTheDocument();
});

test('file upload functionality', async () => {
  render(<MLAnalyticsDashboard />);
  const fileInput = screen.getByRole('button', { name: /upload/i });
  
  const file = new File(['test'], 'test.csv', { type: 'text/csv' });
  fireEvent.change(fileInput, { target: { files: [file] } });
  
  // Assert file was processed
  await screen.findByText('File uploaded successfully');
});
```

---

**Happy Frontend Development! ⚛️📊**
