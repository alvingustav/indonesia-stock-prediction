#!/bin/bash

# Deploy script for Azure Container Apps
set -e

echo "🚀 Deploying Indonesia Stock Prediction to Azure Container Apps..."

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Required variables
RESOURCE_GROUP=${RESOURCE_GROUP:-"stock-prediction-rg"}
APP_NAME=${APP_NAME:-"indonesia-stock-prediction"}
ACR_NAME=${ACR_NAME:-""}
ENVIRONMENT_NAME=${ENVIRONMENT_NAME:-"stock-prediction-env"}
IMAGE_NAME="stock-prediction"

if [ -z "$ACR_NAME" ]; then
    echo "❌ ACR_NAME not set. Please run setup-azure.sh first."
    exit 1
fi

# Check if models exist
if [ ! -f "models/stock_prediction_model.h5" ]; then
    echo "❌ Model file not found: models/stock_prediction_model.h5"
    echo "📋 Please upload your trained model files first."
    exit 1
fi

# Get ACR login server
ACR_SERVER=$(az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query "loginServer" --output tsv)

# Build and push image to ACR
echo "🔨 Building and pushing Docker image..."
az acr build \
    --registry $ACR_NAME \
    --image $IMAGE_NAME:latest \
    --image $IMAGE_NAME:$(date +%Y%m%d-%H%M%S) \
    .

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query "username" --output tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" --output tsv)

# Check if app exists
if az containerapp show --name $APP_NAME --resource-group $RESOURCE_GROUP > /dev/null 2>&1; then
    echo "🔄 Updating existing container app..."
    az containerapp update \
        --name $APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --image $ACR_SERVER/$IMAGE_NAME:latest
else
    echo "🆕 Creating new container app..."
    az containerapp create \
        --name $APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --environment $ENVIRONMENT_NAME \
        --image $ACR_SERVER/$IMAGE_NAME:latest \
        --target-port 8501 \
        --ingress external \
        --registry-server $ACR_SERVER \
        --registry-username $ACR_USERNAME \
        --registry-password $ACR_PASSWORD \
        --cpu 1.0 \
        --memory 2.0Gi \
        --min-replicas 1 \
        --max-replicas 3 \
        --env-vars \
            "STREAMLIT_SERVER_PORT=8501" \
            "STREAMLIT_SERVER_ADDRESS=0.0.0.0" \
            "AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT" \
            "AZURE_OPENAI_API_KEY=$AZURE_OPENAI_API_KEY" \
            "AZURE_OPENAI_DEPLOYMENT=$AZURE_OPENAI_DEPLOYMENT"
fi

# Get the app URL
APP_URL=$(az containerapp show \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query "properties.configuration.ingress.fqdn" \
    --output tsv)

echo "✅ Deployment completed successfully!"
echo ""
echo "🌐 Your Stock Prediction App is available at:"
echo "   https://$APP_URL"
echo ""
echo "📊 Azure Portal:"
echo "   https://portal.azure.com/#@/resource/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.App/containerApps/$APP_NAME"
echo ""
echo "🔍 To check logs:"
echo "   az containerapp logs show --name $APP_NAME --resource-group $RESOURCE_GROUP --follow"
