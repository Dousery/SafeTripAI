# SafeTrip 🌍 - AI Travel Health Risk Analyzer

![safetrip](https://github.com/user-attachments/assets/3f689ee7-42a3-4e3d-afeb-43e6b0de59ae)

SafeTrip is an intelligent travel planning application that analyzes infectious disease risks across US states using AI-powered forecasting. It helps travelers make informed decisions by comparing health risks and recommending safer destinations based on CDC epidemiological data.

## 🚀 Live Demo
**Try it now:** [https://safetrip.streamlit.app](https://safetrip.streamlit.app)

## ✨ Key Features

### 🎯 Smart Analysis
- **AI State Matching** - Gemini AI finds similar destinations based on climate, geography, and culture
- **Disease Forecasting** - SARIMAX time series modeling for Septicemia and Influenza/Pneumonia predictions
- **Risk Assessment** - Calculates death rates per 1 million population for accurate comparison

### 📊 Interactive Visualizations
- **US Heat Map** - Visual representation of selected and recommended states
- **Comparative Charts** - Side-by-side risk analysis with interactive graphs
- **Real-time Metrics** - Color-coded risk indicators and forecasted case numbers

### 🤖 AI-Powered Recommendations
- **Personalized Advice** - LLM-generated travel recommendations based on risk analysis
- **Destination Insights** - Detailed information about climate, culture, and travel advantages
- **Safety Rankings** - Identifies the safest alternative destinations

## 🛠️ Technology Stack

- **Frontend**: Streamlit for interactive web interface
- **Machine Learning**: 
  - SARIMAX (Seasonal ARIMA with External Variables) for time series forecasting
  - Scipy for statistical analysis and outlier detection
  - Scikit-learn for data preprocessing
- **AI Integration**: Google Gemini AI for intelligent recommendations
- **Visualization**: Plotly for interactive maps and statistical charts
- **Data Processing**: Pandas, NumPy for efficient data manipulation

## 🚀 Local Installation

### Prerequisites
- Python 3.8+
- Google Gemini API key

### Setup Steps
```bash
# Clone repository
git clone https://github.com/yourusername/safetrip.git
cd safetrip

# Install dependencies
pip install -r requirements.txt

# Set up API key
export GEMINI_API_KEY="your_gemini_api_key_here"

# Run application
streamlit run main.py
```

## 📊 How It Works

### 1. Data Processing
- Loads CDC disease surveillance data for US states
- Handles missing values using intelligent imputation algorithms
- Applies outlier smoothing with Gaussian filtering

### 2. AI Analysis
- **State Similarity**: Gemini AI analyzes climate, geography, and cultural factors
- **Risk Forecasting**: SARIMAX models predict future disease cases
- **Safety Assessment**: Calculates comparative risk metrics

### 3. Recommendations
- Identifies the safest travel destination among similar states
- Provides detailed explanations for recommendations
- Offers state-specific travel insights and seasonal considerations

## ⚠️ Important Disclaimer

SafeTrip provides AI-generated predictions for informational purposes only. These recommendations should supplement, not replace, official health authority guidance and professional medical advice. Always consult current CDC travel advisories and healthcare professionals before making travel decisions.

---

**📧 Questions?** Open an issue or reach out for support
