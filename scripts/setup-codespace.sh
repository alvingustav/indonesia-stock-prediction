#!/bin/bash

echo "🚀 Setting up Indonesia Stock Prediction Codespace with Pre-trained Models..."

# Activate conda environment
source ~/.bashrc
conda activate stockpred

# Install any additional dependencies
pip install --upgrade pip
pip install python-dotenv

# Create required directories
mkdir -p models logs temp

# Set proper permissions
chmod +x scripts/*.sh

# Initialize Git LFS (for large model files)
git lfs install

# Check if this is a fresh clone or existing workspace
if [ ! -f ".codespace_initialized" ]; then
    echo "🔧 First time setup..."
    
    # Track large model files with Git LFS
    if [ ! -f ".gitattributes" ]; then
        echo "*.h5 filter=lfs diff=lfs merge=lfs -text" > .gitattributes
        echo "*.pkl filter=lfs diff=lfs merge=lfs -text" >> .gitattributes
        echo "📝 Created .gitattributes for Git LFS"
    fi
    
    # Create initialization marker
    touch .codespace_initialized
fi

# Setup environment variables
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "📝 Created .env file from template"
    echo "⚠️  Please configure your Azure settings in .env"
fi

# Check for model files
echo "🔍 Checking for pre-trained model files..."
bash scripts/setup-models.sh

# Setup Azure CLI
echo "🔧 Azure CLI ready. Use 'az login' to authenticate for deployment."

# Display status
echo ""
echo "✅ Codespace setup completed!"
echo ""
echo "📋 Status Summary:"
echo "   🐍 Python environment: Ready"
echo "   📦 Dependencies: Installed"
echo "   🤖 Models: $([ -f "models/stock_prediction_model.h5" ] && echo "✅ Ready" || echo "❌ Missing")"
echo "   ☁️  Azure CLI: Ready"
echo ""
echo "🎯 Next Steps:"
echo "1. Upload model files if missing:"
echo "   - models/stock_prediction_model.h5"
echo "   - models/scalers.pkl"
echo "2. Configure Azure settings in .env file"
echo "3. Test locally: streamlit run app.py"
echo "4. Deploy to Azure: bash scripts/deploy.sh"
echo ""
echo "💡 Tip: Use Git LFS for large model files:"
echo "   git add models/ && git commit -m 'Add pre-trained models'"
