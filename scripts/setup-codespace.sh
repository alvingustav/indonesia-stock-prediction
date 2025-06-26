#!/bin/bash

echo "🚀 Setting up Indonesia Stock Prediction Codespace..."

# Activate conda environment
source ~/.bashrc
conda activate stockpred

# Install additional dependencies if needed
pip install --upgrade pip

# Create directories
mkdir -p models logs

# Set permissions
chmod +x scripts/*.sh

# Download models (if available)
echo "📦 Checking for model files..."
if [ ! -f "models/stock_prediction_model.h5" ]; then
    echo "⚠️  Model files not found. You'll need to upload them or run training."
    echo "📋 Expected files:"
    echo "   - models/stock_prediction_model.h5"
    echo "   - models/scalers.pkl"
fi

# Setup Azure CLI (if not already configured)
echo "🔧 Azure CLI available. Use 'az login' to authenticate."

# Create environment file
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "📝 Created .env file. Please configure your Azure settings."
fi

echo "✅ Codespace setup completed!"
echo ""
echo "🎯 Next steps:"
echo "1. Upload your model files to models/ directory"
echo "2. Configure Azure settings in .env file"
echo "3. Run: streamlit run app.py (for local testing)"
echo "4. Run: bash scripts/deploy.sh (for Azure deployment)"
