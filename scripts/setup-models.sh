#!/bin/bash

echo "📦 Setting up pre-trained model files..."

# Check if model files exist
MODEL_DIR="models"
MODEL_FILE="$MODEL_DIR/stock_prediction_model.h5"
SCALERS_FILE="$MODEL_DIR/scalers.pkl"

if [ ! -d "$MODEL_DIR" ]; then
    mkdir -p "$MODEL_DIR"
    echo "📁 Created models directory"
fi

echo "🔍 Checking for required model files..."

# Check model file
if [ -f "$MODEL_FILE" ]; then
    echo "✅ Found: stock_prediction_model.h5 ($(du -h "$MODEL_FILE" | cut -f1))"
else
    echo "❌ Missing: stock_prediction_model.h5"
    echo "📋 Please upload your trained model file to models/ directory"
    MISSING_FILES=true
fi

# Check scalers file
if [ -f "$SCALERS_FILE" ]; then
    echo "✅ Found: scalers.pkl ($(du -h "$SCALERS_FILE" | cut -f1))"
else
    echo "❌ Missing: scalers.pkl" 
    echo "📋 Please upload your scalers file to models/ directory"
    MISSING_FILES=true
fi

# Check for feature columns (optional)
FEATURE_COLS_FILE="$MODEL_DIR/feature_columns.json"
if [ ! -f "$FEATURE_COLS_FILE" ]; then
    echo "📝 Creating feature_columns.json from notebook configuration..."
    cat > "$FEATURE_COLS_FILE" << 'EOF'
[
    "Open", "High", "Low", "Close", "Volume",
    "MA_5", "MA_10", "MA_20", "MA_50",
    "EMA_12", "EMA_26", "MACD", "MACD_signal",
    "RSI", "BB_middle", "BB_upper", "BB_lower",
    "Volume_MA", "Volume_ratio", "Price_change",
    "High_Low_ratio", "Open_Close_ratio"
]
EOF
    echo "✅ Created feature_columns.json"
fi

if [ "$MISSING_FILES" = true ]; then
    echo ""
    echo "⚠️  Model files missing! Please upload:"
    echo "   1. Copy stock_prediction_model.h5 to models/"
    echo "   2. Copy scalers.pkl to models/"
    echo ""
    echo "💡 You can drag & drop files directly in VS Code file explorer"
    echo "   or use: git add models/ && git commit -m 'Add model files'"
    exit 1
else
    echo ""
    echo "🎉 All model files ready!"
    echo "✅ Your LSTM-GRU Hybrid model is ready for deployment"
fi
