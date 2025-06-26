#!/bin/bash

# Azure setup script for stock prediction app
set -e

echo "🔧 Setting up Azure resources for Stock Prediction App..."

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Configuration
RESOURCE_GROUP=${RESOURCE_GROUP:-"stock-prediction-rg"}
LOCATION=${LOCATION:-"japaneast"}
ENVIRONMENT_NAME=${ENVIRONMENT_NAME:-"stock-prediction-env"}
APP_NAME=${APP_NAME:-"indonesia-stock-prediction"}
ACR_NAME=${ACR_NAME:-"stockpredictionacr$(date +%s)"}

echo "📋 Configuration:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Location: $LOCATION"
echo "  App Name: $APP_NAME"
echo "  ACR Name: $ACR_NAME"

# Check if logged in to Azure
if ! az account show > /dev/null 2>&1; then
    echo "❌ Not logged in to Azure. Please run 'az login' first."
    exit 1
fi

# Create resource group
echo "📦 Creating resource group..."
az group create \
    --name $RESOURCE_GROUP \
    --location $LOCATION

# Create Azure Container Registry
echo "📦 Creating Azure Container Registry..."
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $ACR_NAME \
    --sku Basic \
    --admin-enabled true

# Create Container Apps environment
echo "🌍 Creating Container Apps environment..."
az containerapp env create \
    --name $ENVIRONMENT_NAME \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION

echo "✅ Azure resources created successfully!"
echo ""
echo "📝 Save these values to your .env file:"
echo "RESOURCE_GROUP=$RESOURCE_GROUP"
echo "ACR_NAME=$ACR_NAME"
echo "ENVIRONMENT_NAME=$ENVIRONMENT_NAME"
echo "APP_NAME=$APP_NAME"
