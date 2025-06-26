import joblib
import json
import os
from pathlib import Path
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

class ModelLoader:
    def __init__(self):
        self.model = None
        self.scalers = None
        self.feature_columns = None
        self.models_dir = Path(__file__).parent.parent / "models"
        self.model_info = {}
        
    def load_pretrained_model(self):
        """Load pre-trained LSTM-GRU model and scalers from dump files"""
        try:
            # Load TensorFlow/Keras model
            model_path = self.models_dir / "stock_prediction_model.h5"
            if not model_path.exists():
                st.error(f"❌ Model file tidak ditemukan: {model_path}")
                st.info("📋 Pastikan file 'stock_prediction_model.h5' ada di folder models/")
                return False
                
            self.model = load_model(str(model_path))
            st.success("✅ LSTM-GRU Hybrid model berhasil dimuat!")
            
            # Load scalers (joblib dump dari notebook)
            scalers_path = self.models_dir / "scalers.pkl"
            if not scalers_path.exists():
                st.error(f"❌ File scalers tidak ditemukan: {scalers_path}")
                st.info("📋 Pastikan file 'scalers.pkl' ada di folder models/")
                return False
                
            self.scalers = joblib.load(str(scalers_path))
            st.success("✅ Scalers berhasil dimuat!")
            
            # Load feature columns configuration
            self._load_feature_configuration()
            
            # Extract and store model information
            self._extract_model_metadata()
            
            # Validate model compatibility
            if self._validate_model_components():
                st.success("🎯 Model validation passed!")
                return True
            else:
                return False
            
        except Exception as e:
            st.error(f"❌ Error loading pre-trained model: {e}")
            return False
    
    def _load_feature_configuration(self):
        """Load feature columns from configuration file"""
        feature_cols_path = self.models_dir / "feature_columns.json"
        
        if feature_cols_path.exists():
            with open(feature_cols_path, 'r') as f:
                self.feature_columns = json.load(f)
            st.info(f"📋 Loaded {len(self.feature_columns)} feature columns from config")
        else:
            # Default feature columns from notebook (sesuai dengan yang di-training)
            self.feature_columns = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'MA_5', 'MA_10', 'MA_20', 'MA_50',
                'EMA_12', 'EMA_26', 'MACD', 'MACD_signal',
                'RSI', 'BB_middle', 'BB_upper', 'BB_lower',
                'Volume_MA', 'Volume_ratio', 'Price_change',
                'High_Low_ratio', 'Open_Close_ratio'
            ]
            
            # Save for future use
            try:
                with open(feature_cols_path, 'w') as f:
                    json.dump(self.feature_columns, f, indent=2)
                st.info("📝 Created feature_columns.json from default configuration")
            except:
                pass  # Ignore save errors in production
    
    def _extract_model_metadata(self):
        """Extract model information for display"""
        if self.model:
            self.model_info = {
                'model_loaded': True,
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape,
                'total_params': self.model.count_params(),
                'trainable_params': sum([layer.count_params() for layer in self.model.layers if hasattr(layer, 'count_params')]),
                'scalers_available': self.scalers is not None,
                'feature_count': len(self.feature_columns) if self.feature_columns else 0,
                'architecture': 'LSTM-GRU Hybrid',
                'training_stocks': ['BBCA', 'ASII', 'INDF', 'TLKM', 'BMRI', 'BBNI', 'IHSG'],
                'sequence_length': 60,
                'model_size': self._get_model_size()
            }
        else:
            self.model_info = {'model_loaded': False}
    
    def _get_model_size(self):
        """Get model file size"""
        try:
            model_path = self.models_dir / "stock_prediction_model.h5"
            if model_path.exists():
                size_bytes = model_path.stat().st_size
                if size_bytes > 1024**3:  # GB
                    return f"{size_bytes / (1024**3):.1f} GB"
                elif size_bytes > 1024**2:  # MB
                    return f"{size_bytes / (1024**2):.1f} MB"
                else:  # KB
                    return f"{size_bytes / 1024:.1f} KB"
            return "Unknown"
        except:
            return "Unknown"
    
    def _validate_model_components(self):
        """Validate that all components are compatible"""
        try:
            # Check model input shape matches expected features
            expected_features = len(self.feature_columns)
            model_features = self.model.input_shape[-1]
            
            if model_features != expected_features:
                st.warning(f"⚠️ Feature count mismatch: Model expects {model_features}, config has {expected_features}")
                return False
            
            # Check scalers structure (from notebook dump)
            if isinstance(self.scalers, dict):
                # Multi-stock scalers format
                sample_scaler = list(self.scalers.values())[0]
                if 'feature_scaler' not in sample_scaler or 'target_scaler' not in sample_scaler:
                    st.error("❌ Invalid scalers format - missing feature_scaler or target_scaler")
                    return False
            else:
                st.error("❌ Scalers should be in dictionary format")
                return False
            
            return True
            
        except Exception as e:
            st.error(f"❌ Model validation failed: {e}")
            return False
    
    def get_model_info(self):
        """Get comprehensive model information"""
        return self.model_info
    
    def get_scalers_info(self):
        """Get information about available scalers"""
        if not self.scalers:
            return {"scalers_loaded": False}
        
        scaler_info = {
            "scalers_loaded": True,
            "format": "multi-stock" if isinstance(self.scalers, dict) else "single",
            "available_stocks": []
        }
        
        if isinstance(self.scalers, dict):
            scaler_info["available_stocks"] = list(self.scalers.keys())
            scaler_info["total_scalers"] = len(self.scalers)
            
        return scaler_info
    
    def get_scaler_for_stock(self, stock_symbol):
        """Get appropriate scaler for specific stock"""
        try:
            if isinstance(self.scalers, dict):
                # Try exact match first
                if stock_symbol in self.scalers:
                    return self.scalers[stock_symbol]
                
                # Try uppercase
                stock_upper = stock_symbol.upper()
                if stock_upper in self.scalers:
                    return self.scalers[stock_upper]
                
                # Use first available scaler as fallback
                first_key = list(self.scalers.keys())[0]
                st.info(f"📋 Using {first_key} scaler for {stock_symbol}")
                return self.scalers[first_key]
            
            return self.scalers
            
        except Exception as e:
            st.error(f"❌ Error getting scaler for {stock_symbol}: {e}")
            return None
