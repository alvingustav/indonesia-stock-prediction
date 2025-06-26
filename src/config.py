import os
from pathlib import Path

# Indonesian stocks configuration (sesuai dengan notebook)
INDONESIAN_STOCKS = {
    "Bank BCA (BBCA)": "BBCA.JK",
    "Astra International (ASII)": "ASII.JK", 
    "Indofood (INDF)": "INDF.JK",
    "Telkom (TLKM)": "TLKM.JK",
    "Bank Mandiri (BMRI)": "BMRI.JK",
    "Bank BNI (BBNI)": "BBNI.JK",
    "Jakarta Composite (IHSG)": "^JKSE"
}

# Azure OpenAI Configuration
AZURE_OPENAI_CONFIG = {
    'endpoint': os.getenv('AZURE_OPENAI_ENDPOINT', 'https://your-resource.openai.azure.com/'),
    'api_key': os.getenv('AZURE_OPENAI_API_KEY', 'your-api-key'),
    'api_version': '2024-02-15-preview',
    'deployment_name': os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-35-turbo')
}

# App settings
APP_SETTINGS = {
    'default_prediction_days': 7,
    'max_prediction_days': 30,
    'sequence_length': 60,  # Sesuai notebook
    'page_title': 'Indonesia Stock Prediction & Valuation',
    'page_icon': '📈'
}

# Model configuration (sesuai dengan notebook training)
MODEL_CONFIG = {
    'sequence_length': 60,
    'features_count': 22,  # Sesuai dengan feature_columns di notebook
    'model_architecture': 'LSTM-GRU Hybrid',
    'training_stocks': ['BBCA', 'ASII', 'INDF', 'TLKM', 'BMRI', 'BBNI', 'IHSG'],
    'total_params': 145751,  # Dari notebook summary
    'model_size': '569.34 KB'
}

# Feature columns (sama persis dengan notebook)
FEATURE_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'MA_5', 'MA_10', 'MA_20', 'MA_50',
    'EMA_12', 'EMA_26', 'MACD', 'MACD_signal',
    'RSI', 'BB_middle', 'BB_upper', 'BB_lower',
    'Volume_MA', 'Volume_ratio', 'Price_change',
    'High_Low_ratio', 'Open_Close_ratio'
]

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Data collection settings
DATA_CONFIG = {
    'default_period': '2y',
    'min_data_points': 100,
    'cache_ttl': 300,  # 5 minutes
    'timeout': 30
}

# Technical indicators settings
TECHNICAL_CONFIG = {
    'ma_periods': [5, 10, 20, 50],
    'ema_periods': [12, 26],
    'rsi_period': 14,
    'bb_period': 20,
    'bb_std_multiplier': 2,
    'macd_signal_period': 9,
    'volume_ma_period': 20
}

# Model prediction settings
PREDICTION_CONFIG = {
    'batch_size': 32,
    'confidence_threshold': 0.7,
    'max_prediction_days': 30,
    'min_historical_data': 60
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_handler': True,
    'console_handler': True
}
