# 🇮🇩 Indonesia Stock Prediction & Valuation Platform

> AI-Powered Stock Analysis menggunakan LSTM-GRU Hybrid Model dan Azure OpenAI untuk prediksi harga saham Indonesia dengan analisis valuasi komprehensif.

[![Deploy to Azure](https://img.shields.io/badge/Deploy%20to-Azure-0078d7.svg)](https://azure.microsoft.com/en-us/products/container-apps)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18+-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Deskripsi

Platform prediksi harga saham Indonesia yang mengintegrasikan machine learning dengan AI generative untuk memberikan analisis investasi yang komprehensif. Aplikasi ini menggunakan model LSTM-GRU Hybrid yang telah dilatih dengan data historis 7 saham blue-chip Indonesia dan Jakarta Composite Index (IHSG).

## ✨ Fitur Utama

### 🤖 AI & Machine Learning
- **Prediksi Harga Saham** menggunakan LSTM-GRU Hybrid Neural Network
- **22 Technical Indicators** (RSI, MACD, Bollinger Bands, Moving Averages, dll)
- **Multi-Stock Support** untuk 7 saham utama Indonesia + IHSG
- **Real-time Data Integration** dari Yahoo Finance

### 💰 Investment Analysis
- **AI-Powered Valuation Analysis** menggunakan Azure OpenAI GPT-4
- **Investment Recommendations** (Buy/Hold/Sell dengan confidence level)
- **Risk Assessment** dan price target predictions
- **Indonesian Market Context** analysis

### 📊 Interactive Dashboard
- **Real-time Stock Charts** dengan Plotly candlestick visualization
- **Technical Analysis Dashboard** dengan multiple indicators
- **Prediction Visualization** dengan confidence intervals
- **Responsive Web UI** menggunakan Streamlit

### ☁️ Cloud-Native
- **Azure Container Apps** deployment
- **Auto-scaling** berdasarkan traffic
- **GitHub Codespaces** development environment
- **CI/CD Pipeline** dengan GitHub Actions

## 🏗️ Architecture

graph TB
A[Yahoo Finance API] --> B[Data Collector]
B --> C[Technical Indicators]
C --> D[LSTM-GRU Model]
D --> E[Price Predictions]
E --> F[Azure OpenAI]
F --> G[Valuation Analysis]
G --> H[Streamlit Dashboard]

text
subgraph "Azure Cloud"
    I[Azure Container Apps]
    J[Azure OpenAI Service]
    K[Azure Container Registry]
end

H --> I
F --> J
I --> K
text

## 📈 Supported Stocks

| Stock | Symbol | Sector |
|-------|--------|--------|
| Bank BCA | BBCA.JK | Financial Services |
| Astra International | ASII.JK | Automotive |
| Indofood | INDF.JK | Consumer Goods |
| Telkom Indonesia | TLKM.JK | Telecommunications |
| Bank Mandiri | BMRI.JK | Financial Services |
| Bank BNI | BBNI.JK | Financial Services |
| Jakarta Composite | ^JKSE | Market Index |

## 🛠️ Technology Stack

### Machine Learning & Data
- **TensorFlow 2.18+** - Deep learning framework
- **scikit-learn** - Data preprocessing dan metrics
- **pandas & NumPy** - Data manipulation
- **yfinance** - Real-time stock data

### Web Application
- **Streamlit** - Interactive web application
- **Plotly** - Interactive visualizations
- **Python-dotenv** - Environment management

### Cloud & AI Services
- **Azure OpenAI** - GPT-4 untuk valuation analysis
- **Azure Container Apps** - Serverless container hosting
- **Azure Container Registry** - Docker image storage
- **GitHub Codespaces** - Cloud development environment

## 📁 Project Structure
indonesia-stock-prediction/

├── 📂 .devcontainer/ # GitHub Codespaces configuration </br >
│ ├── devcontainer.json</br >
│ └── Dockerfile</br >
├── 📂 .github/workflows/ # CI/CD pipeline</br >
│ └── deploy-azure.yml</br >
├── 📂 models/ # Pre-trained models (Git LFS)</br >
│ ├── stock_prediction_model.h5</br >
│ ├── scalers.pkl</br >
│ ├── model_config.pkl</br >
│ └── feature_columns.json</br >
├── 📂 src/ # Source code</br >
│ ├── config.py # Configuration settings</br >
│ ├── model_loader.py # Model loading utilities</br >
│ ├── predictor.py # Stock prediction engine</br >
│ ├── valuation_analyzer.py # Azure OpenAI integration</br >
│ ├── data_collector.py # Yahoo Finance data fetching</br >
│ └── utils.py # Utility functions</br >
├── 📂 scripts/ # Deployment scripts</br >
│ ├── setup-azure.sh</br >
│ ├── deploy.sh</br >
│ └── verify-models.py</br >
├── 📂 .streamlit/ # Streamlit configuration</br >
│ └── config.toml</br >
├── 📂 notebooks/ # Jupyter notebooks untuk training</br >
│ └── Stock_Prediction_Model.ipynb</br >
├── 🐳 Dockerfile # Container configuration</br >
├── 📄 app.py # Main Streamlit application</br >
├── 📋 requirements.txt # Python dependencies</br >
├── 🔧 .env.example # Environment variables template</br >
└── 📖 README.md # Project documentation


## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Azure account (untuk deployment)
- Azure OpenAI resource (untuk valuation analysis)

### 1. Clone Repository
git clone https://github.com/alvingustav/indonesia-stock-prediction.git
cd indonesia-stock-prediction


### 2. Setup Environment
Install dependencies
pip install -r requirements.txt

Setup environment variables
cp .env.example .env

Edit .env dengan Azure OpenAI credentials Anda
text

### 3. Upload Pre-trained Models
Upload model files ke folder `models/`:
- `stock_prediction_model.h5` (LSTM-GRU model)
- `scalers.pkl` (Feature dan target scalers)
- `model_config.pkl` (Model configuration, optional)

### 4. Run Locally
streamlit run app.py

Akses aplikasi di `http://localhost:8501`

## ☁️ Deploy to Azure

### Option 1: Quick Deploy (Recommended)
Login ke Azure
az login

One-command deploy</br >
az containerapp up
--name indonesia-stock-prediction
--resource-group stock-prediction-rg
--location japaneast
--source .
--target-port 8501
--ingress external


### Option 2: Manual Deploy
1. Create Azure resources
az group create --name stock-prediction-rg --location japaneast
az acr create --resource-group stock-prediction-rg --name stockpredictionacr --sku Basic --admin-enabled true
az containerapp env create --name stock-prediction-env --resource-group stock-prediction-rg --location japaneast

2. Build dan push image
az acr build --registry stockpredictionacr --image stock-prediction:latest .

3. Get ACR credentials
az acr credential show --name stockpredictionacr

4. Deploy container app
az containerapp create
--name indonesia-stock-prediction
--resource-group stock-prediction-rg
--environment stock-prediction-env
--image stockpredictionacr.azurecr.io/stock-prediction:latest
--target-port 8501
--ingress external
--cpu 1.0
--memory 2.0Gi
--min-replicas 1
--max-replicas 3
--registry-server stockpredictionacr.azurecr.io
--registry-username <acr-username>
--registry-password <acr-password>
--env-vars
AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
AZURE_OPENAI_API_KEY="your-api-key"
AZURE_OPENAI_DEPLOYMENT="your-deployment-name"


### 5. Access Your App
Get application URL
az containerapp show --name indonesia-stock-prediction --resource-group stock-prediction-rg --query "properties.configuration.ingress.fqdn" --output tsv


## 🔧 Configuration

### Environment Variables
Azure OpenAI Configuration (Required untuk valuation analysis)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=your-deployment-name

Optional: App Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0


### Azure OpenAI Setup
1. Create Azure OpenAI resource di Azure Portal
2. Deploy GPT-4 atau GPT-3.5-turbo model
3. Copy endpoint, API key, dan deployment name
4. Update environment variables

## 📊 Model Performance

### Architecture Details
- **Model Type**: LSTM-GRU Hybrid Neural Network
- **Input Features**: 22 technical indicators
- **Sequence Length**: 60 days
- **Training Data**: 39,865 sequences dari 7 saham Indonesia
- **Total Parameters**: 145,751

### Training Results
- **Training Stocks**: BBCA, ASII, INDF, TLKM, BMRI, BBNI, IHSG
- **Model Architecture**: Optimized untuk time series prediction
- **Validation**: Cross-validation dengan multiple stocks
- **Performance**: RMSE and MAE metrics per stock

## 🔍 API Usage

### Prediction Endpoint
Generate stock price predictions
predictions = predictor.predict_prices(
symbol='BBCA', # Stock symbol
days_ahead=7 # Prediction horizon
)


### Valuation Analysis
Get AI-powered valuation analysis
analysis = valuation_analyzer.analyze_stock_valuation(
stock_name='Bank BCA',
current_price=8650,
predicted_prices=[8700][8750][8800],
historical_data=stock_data
)


## 🐛 Troubleshooting

### Common Issues

#### Model Loading Errors
Error: Could not locate function 'mse'
Solution: Model loader menggunakan multiple fallback methods
text

#### Azure OpenAI Deployment Not Found
Error: DeploymentNotFound
Solution:
1. Verify deployment name di Azure OpenAI Studio
2. Use exact deployment name (bukan model name)
3. Check API version compatibility
text

#### IHSG Data Not Found
Error: No data found for JKSE.JK
Solution: IHSG uses symbol ^JKSE (sudah fixed in config)
text

### Debug Mode
Enable debug mode untuk troubleshooting:
Set environment variable
export DEBUG=true

Atau tambahkan di .env
DEBUG=true

## 🤝 Contributing

### Development Workflow
1. **Fork** repository ini
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Code Standards
- Follow **PEP 8** untuk Python code style
- Add **type hints** untuk function signatures
- Include **docstrings** untuk semua functions
- Write **unit tests** untuk new features

### Development Setup
Setup development environment
pip install -r requirements.txt

Install pre-commit hooks
pre-commit install

Run tests
pytest tests/

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yahoo Finance API** untuk real-time stock data
- **Azure OpenAI Service** untuk AI-powered analysis
- **TensorFlow Team** untuk machine learning framework
- **Streamlit Team** untuk web application framework
- **Indonesia Stock Exchange (IDX)** untuk market data

## 📞 Support

- **Documentation**: [Wiki](https://github.com/alvingustav/indonesia-stock-prediction/wiki)
- **Issues**: [GitHub Issues](https://github.com/alvingustav/indonesia-stock-prediction/issues)
- **Discussions**: [GitHub Discussions](https://github.com/alvingustav/indonesia-stock-prediction/discussions)

## 🗺️ Roadmap

### V2.0 Planned Features
- [ ] **Real-time Streaming** predictions
- [ ] **Portfolio Optimization** recommendations
- [ ] **News Sentiment Analysis** integration
- [ ] **Mobile App** dengan React Native
- [ ] **API Gateway** dengan rate limiting
- [ ] **More Stocks** coverage (LQ45 index)

### V1.1 Upcoming
- [ ] **Backtesting Framework** untuk strategy validation
- [ ] **Email Alerts** untuk price targets
- [ ] **PDF Reports** generation
- [ ] **Dark Mode** UI theme

---

**⚠️ Disclaimer**: This application is for educational and research purposes only. Stock predictions are not guaranteed and should not be used as the sole basis for investment decisions. Always consult with financial advisors before making investment choices.

---

Made with ❤️ in Indonesia 🇮🇩
