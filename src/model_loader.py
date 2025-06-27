import joblib
import json
import os
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
import tensorflow as tf
import numpy as np
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

class ModelLoader:
    def __init__(self):
        self.model = None
        self.scalers = None
        self.model_config = None  # Support for model_config.pkl
        self.feature_columns = None
        self.models_dir = Path(__file__).parent.parent / "models"
        self.model_info = {}
        
    def load_pretrained_model(self):
        """Load pre-trained LSTM-GRU model dengan comprehensive error handling"""
        try:
            st.info(f"🐍 TensorFlow version: {tf.__version__}")
            
            # Step 1: Load model configuration jika ada
            self._load_model_config()
            
            # Step 2: Load model dengan multiple fallback methods
            if not self._load_keras_model():
                return False
            
            # Step 3: Build metrics untuk fix warning
            self._build_model_metrics()
            
            # Step 4: Load scalers dengan validation
            if not self._load_scalers_with_validation():
                return False
            
            # Step 5: Load feature configuration
            self._load_feature_configuration()
            
            # Step 6: Extract model metadata
            self._extract_model_metadata()
            
            # Step 7: Comprehensive validation
            if self._validate_all_components():
                st.success("🎯 All components loaded and validated successfully!")
                return True
            else:
                st.error("❌ Component validation failed")
                return False
            
        except Exception as e:
            st.error(f"❌ Error loading pre-trained model: {e}")
            return False
    
    def _load_model_config(self):
        """Load model_config.pkl jika tersedia"""
        try:
            config_path = self.models_dir / "model_config.pkl"
            if config_path.exists():
                self.model_config = joblib.load(str(config_path))
                st.success("✅ Model configuration loaded from model_config.pkl")
                
                # Debug model config structure
                if isinstance(self.model_config, dict):
                    config_keys = list(self.model_config.keys())
                    st.info(f"📋 Model config keys: {config_keys}")
                    
                    # Show available stocks if configured
                    if 'stocks' in self.model_config:
                        available_stocks = list(self.model_config['stocks'].keys())
                        st.info(f"📊 Stocks in config: {available_stocks}")
            else:
                st.info("📋 model_config.pkl not found, using fallback configuration")
                self.model_config = None
                
        except Exception as e:
            st.warning(f"⚠️ Error loading model config: {e}")
            self.model_config = None
    
    def _load_keras_model(self):
        """Load Keras model dengan multiple fallback methods"""
        model_path = self.models_dir / "stock_prediction_model.h5"
        if not model_path.exists():
            st.error(f"❌ Model file tidak ditemukan: {model_path}")
            return False
        
        # Method 1: Try with custom objects (Keras 3 compatible)
        if self._try_load_with_custom_objects(str(model_path)):
            return True
        
        # Method 2: Load without compile then recompile
        if self._try_load_and_recompile(str(model_path)):
            return True
        
        # Method 3: Custom object scope (Keras 2 compatible)
        if self._try_load_with_object_scope(str(model_path)):
            return True
        
        # Method 4: Force load with TF compatibility
        if self._try_force_load_compatible(str(model_path)):
            return True
        
        st.error("❌ All model loading methods failed")
        return False
    
    def _try_load_with_custom_objects(self, model_path):
        """Method 1: Load dengan custom objects (recommended untuk Keras 3)"""
        try:
            st.info("🔄 Method 1: Loading with custom objects...")
            
            # Define custom objects untuk resolve 'mse' issue
            custom_objects = {
                'mse': tf.keras.metrics.mean_squared_error,
                'mean_squared_error': tf.keras.metrics.mean_squared_error,
                'MeanSquaredError': tf.keras.metrics.MeanSquaredError,
                'adam': tf.keras.optimizers.Adam,
                'Adam': tf.keras.optimizers.Adam
            }
            
            self.model = load_model(model_path, custom_objects=custom_objects)
            st.success("✅ Method 1 successful - Model loaded with custom objects!")
            return True
            
        except Exception as e:
            st.warning(f"⚠️ Method 1 failed: {str(e)[:150]}...")
            return False
    
    def _try_load_and_recompile(self, model_path):
        """Method 2: Load tanpa compile, lalu recompile"""
        try:
            st.info("🔄 Method 2: Load without compile then recompile...")
            
            # Load tanpa compilation
            self.model = load_model(model_path, compile=False)
            
            # Get compilation parameters dari model_config atau use defaults
            compile_params = self._get_compilation_parameters()
            
            # Recompile dengan parameters yang benar
            self.model.compile(**compile_params)
            
            st.success("✅ Method 2 successful - Model recompiled!")
            return True
            
        except Exception as e:
            st.warning(f"⚠️ Method 2 failed: {str(e)[:150]}...")
            return False
    
    def _try_load_with_object_scope(self, model_path):
        """Method 3: Custom object scope untuk Keras 2 compatibility"""
        try:
            st.info("🔄 Method 3: Custom object scope...")
            
            with tf.keras.utils.custom_object_scope({
                'mse': tf.keras.metrics.mean_squared_error,
                'mean_squared_error': tf.keras.metrics.mean_squared_error,
                'MeanSquaredError': tf.keras.metrics.MeanSquaredError
            }):
                self.model = load_model(model_path)
            
            st.success("✅ Method 3 successful - Custom object scope!")
            return True
            
        except Exception as e:
            st.warning(f"⚠️ Method 3 failed: {str(e)[:150]}...")
            return False
    
    def _try_force_load_compatible(self, model_path):
        """Method 4: Force compatibility mode"""
        try:
            st.info("🔄 Method 4: Force compatibility mode...")
            
            # Load dengan safe_mode disabled untuk compatibility
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects=None,
                compile=False,
                safe_mode=False  # Disable safe mode untuk compatibility
            )
            
            # Manual compilation dengan standard parameters
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mean_squared_error', 'mean_absolute_error']
            )
            
            st.success("✅ Method 4 successful - Force compatibility!")
            return True
            
        except Exception as e:
            st.warning(f"⚠️ Method 4 failed: {str(e)[:150]}...")
            return False
    
    def _get_compilation_parameters(self):
        """Get compilation parameters dari model_config atau defaults"""
        default_params = {
            'optimizer': tf.keras.optimizers.Adam(learning_rate=0.001),
            'loss': 'mean_squared_error',
            'metrics': ['mean_squared_error', 'mean_absolute_error']
        }
        
        if self.model_config and 'training_params' in self.model_config:
            params = self.model_config['training_params']
            
            # Update parameters dari config
            if 'learning_rate' in params:
                default_params['optimizer'] = tf.keras.optimizers.Adam(
                    learning_rate=params['learning_rate']
                )
            if 'loss' in params:
                default_params['loss'] = params['loss']
            if 'metrics' in params:
                default_params['metrics'] = params['metrics']
        
        return default_params
    
    def _build_model_metrics(self):
        """Build model metrics untuk fix TensorFlow warning"""
        try:
            st.info("🔧 Building model metrics...")
            
            # Get input shape dari model
            input_shape = self.model.input_shape
            sequence_length = input_shape[1] if len(input_shape) > 1 else 60
            features_count = input_shape[2] if len(input_shape) > 2 else 22
            
            # Create dummy input untuk build metrics
            dummy_input = np.random.random((1, sequence_length, features_count))
            
            # Make dummy prediction untuk build internal metrics
            _ = self.model.predict(dummy_input, verbose=0)
            
            st.success("✅ Model metrics built successfully - TensorFlow warning fixed!")
            
        except Exception as e:
            st.warning(f"⚠️ Could not build metrics: {e}")
    
    def _load_scalers_with_validation(self):
        """Load scalers dengan comprehensive validation"""
        try:
            scalers_path = self.models_dir / "scalers.pkl"
            if not scalers_path.exists():
                st.error(f"❌ scalers.pkl tidak ditemukan: {scalers_path}")
                return False
            
            # Load scalers
            self.scalers = joblib.load(str(scalers_path))
            
            # Merge dengan scalers dari model_config jika ada
            if self.model_config and 'scalers' in self.model_config:
                self._merge_config_scalers()
            
            # Validate scaler structure
            if not self._validate_scalers_structure():
                return False
            
            # Debug available scalers
            self._debug_available_scalers()
            
            st.success("✅ Scalers loaded and validated!")
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading scalers: {e}")
            return False
    
    def _merge_config_scalers(self):
        """Merge scalers dari model_config"""
        try:
            config_scalers = self.model_config['scalers']
            
            if isinstance(self.scalers, dict) and isinstance(config_scalers, dict):
                # Add missing scalers dari config
                for stock, scaler_data in config_scalers.items():
                    if stock not in self.scalers:
                        self.scalers[stock] = scaler_data
                        st.info(f"➕ Added {stock} scaler from model_config")
                
                st.info("🔧 Scalers merged from model_config")
            
        except Exception as e:
            st.warning(f"⚠️ Could not merge config scalers: {e}")
    
    def _validate_scalers_structure(self):
        """Validate bahwa scalers memiliki structure yang benar"""
        try:
            if not isinstance(self.scalers, dict):
                st.error("❌ Scalers should be dictionary format")
                return False
            
            # Check each scaler
            valid_scalers = 0
            for stock, scaler_data in self.scalers.items():
                if isinstance(scaler_data, dict):
                    if 'feature_scaler' in scaler_data and 'target_scaler' in scaler_data:
                        valid_scalers += 1
                    else:
                        st.warning(f"⚠️ {stock}: Missing feature_scaler or target_scaler")
                else:
                    st.warning(f"⚠️ {stock}: Scaler data is not dictionary")
            
            if valid_scalers == 0:
                st.error("❌ No valid scalers found")
                return False
            
            st.info(f"✅ {valid_scalers}/{len(self.scalers)} scalers are valid")
            return True
            
        except Exception as e:
            st.error(f"❌ Scaler validation failed: {e}")
            return False
    
    def _debug_available_scalers(self):
        """Debug available scalers untuk troubleshooting"""
        try:
            if isinstance(self.scalers, dict):
                available_stocks = list(self.scalers.keys())
                st.info(f"📊 Available scalers: {available_stocks}")
                
                # Show structure dari first scaler
                if available_stocks:
                    sample_stock = available_stocks[0]
                    sample_scaler = self.scalers[sample_stock]
                    if isinstance(sample_scaler, dict):
                        scaler_keys = list(sample_scaler.keys())
                        st.info(f"🔧 Scaler structure example ({sample_stock}): {scaler_keys}")
            
        except Exception as e:
            st.warning(f"⚠️ Debug scalers failed: {e}")
    
    def _load_feature_configuration(self):
        """Load feature configuration dengan prioritas: model_config > JSON > default"""
        try:
            # Priority 1: Dari model_config
            if self.model_config and 'feature_columns' in self.model_config:
                self.feature_columns = self.model_config['feature_columns']
                st.success("✅ Feature columns loaded from model_config")
                return
            
            # Priority 2: Dari JSON file
            feature_cols_path = self.models_dir / "feature_columns.json"
            if feature_cols_path.exists():
                with open(feature_cols_path, 'r') as f:
                    self.feature_columns = json.load(f)
                st.info("📋 Feature columns loaded from JSON file")
                return
            
            # Priority 3: Default hardcoded (sama dengan notebook)
            self.feature_columns = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'MA_5', 'MA_10', 'MA_20', 'MA_50',
                'EMA_12', 'EMA_26', 'MACD', 'MACD_signal',
                'RSI', 'BB_middle', 'BB_upper', 'BB_lower',
                'Volume_MA', 'Volume_ratio', 'Price_change',
                'High_Low_ratio', 'Open_Close_ratio'
            ]
            
            # Save default configuration
            self._save_feature_columns_json()
            st.warning("⚠️ Using default feature columns - saved to JSON")
            
        except Exception as e:
            st.error(f"❌ Error loading feature configuration: {e}")
    
    def _save_feature_columns_json(self):
        """Save feature columns ke JSON file"""
        try:
            feature_cols_path = self.models_dir / "feature_columns.json"
            with open(feature_cols_path, 'w') as f:
                json.dump(self.feature_columns, f, indent=2)
        except:
            pass  # Silent fail
    
    def _extract_model_metadata(self):
        """Extract comprehensive model information"""
        if self.model:
            self.model_info = {
                'model_loaded': True,
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape,
                'total_params': self.model.count_params(),
                'scalers_available': self.scalers is not None,
                'feature_count': len(self.feature_columns) if self.feature_columns else 0,
                'architecture': 'LSTM-GRU Hybrid',
                'training_stocks': ['BBCA', 'ASII', 'INDF', 'TLKM', 'BMRI', 'BBNI', 'IHSG'],
                'sequence_length': 60,
                'tensorflow_version': tf.__version__,
                'model_config_loaded': self.model_config is not None,
                'available_scalers': list(self.scalers.keys()) if isinstance(self.scalers, dict) else []
            }
        else:
            self.model_info = {'model_loaded': False}
    
    def _validate_all_components(self):
        """Comprehensive validation semua components"""
        try:
            # Validate model
            if self.model is None:
                st.error("❌ Model not loaded")
                return False
            
            # Validate input/output shapes
            expected_features = len(self.feature_columns)
            model_features = self.model.input_shape[-1]
            
            if model_features != expected_features:
                st.error(f"❌ Feature mismatch: Model expects {model_features}, config has {expected_features}")
                return False
            
            # Validate scalers
            if not isinstance(self.scalers, dict) or len(self.scalers) == 0:
                st.error("❌ No valid scalers available")
                return False
            
            # Test model prediction
            if not self._test_model_prediction():
                return False
            
            st.success("✅ All component validations passed!")
            return True
            
        except Exception as e:
            st.error(f"❌ Validation failed: {e}")
            return False
    
    def _test_model_prediction(self):
        """Test model prediction untuk ensure it works"""
        try:
            st.info("🧪 Testing model prediction...")
            
            # Create dummy input
            sequence_length = self.model.input_shape[1]
            features_count = self.model.input_shape[2]
            test_input = np.random.random((1, sequence_length, features_count))
            
            # Test prediction
            prediction = self.model.predict(test_input, verbose=0)
            
            if prediction is not None and len(prediction) > 0:
                st.success("✅ Model prediction test passed!")
                return True
            else:
                st.error("❌ Model prediction returned invalid result")
                return False
                
        except Exception as e:
            st.error(f"❌ Model prediction test failed: {e}")
            return False
    
    def get_scaler_for_stock(self, stock_symbol):
        """Enhanced scaler retrieval dengan comprehensive matching"""
        try:
            if not isinstance(self.scalers, dict):
                st.error("❌ Scalers not available")
                return None
            
            # Clean symbol
            clean_symbol = stock_symbol.upper().replace('.JK', '')
            if clean_symbol == 'JKSE':
                clean_symbol == 'IHSG'
            
            # Debug info
            st.info(f"🔍 Looking for scaler: '{clean_symbol}'")
            
            # Try multiple variations
            variations = [
                clean_symbol,                    # BBCA
                clean_symbol.lower(),           # bbca
                f"{clean_symbol}.JK",           # BBCA.JK
                stock_symbol,                   # Original input
                stock_symbol.upper(),           # Uppercase original
                stock_symbol.lower()            # Lowercase original
            ]
            
            for variation in variations:
                if variation in self.scalers:
                    scaler_data = self.scalers[variation]
                    if self._validate_scaler_data(scaler_data):
                        st.success(f"✅ Found valid scaler for {clean_symbol} using: {variation}")
                        return scaler_data
            
            # Check model_config untuk stock-specific scaler
            if self.model_config and 'stocks' in self.model_config:
                stocks_config = self.model_config['stocks']
                if clean_symbol in stocks_config:
                    stock_config = stocks_config[clean_symbol]
                    if 'scaler' in stock_config:
                        st.info(f"📋 Using scaler from model_config for {clean_symbol}")
                        return stock_config['scaler']
            
            # Fallback: Use first valid scaler dengan warning
            for stock, scaler_data in self.scalers.items():
                if self._validate_scaler_data(scaler_data):
                    st.warning(f"⚠️ No specific scaler for {clean_symbol}, using {stock} as fallback")
                    return scaler_data
            
            st.error(f"❌ No valid scaler found for {clean_symbol}")
            return None
            
        except Exception as e:
            st.error(f"❌ Error getting scaler for {stock_symbol}: {e}")
            return None
    
    def _validate_scaler_data(self, scaler_data):
        """Validate bahwa scaler data valid"""
        try:
            if not isinstance(scaler_data, dict):
                return False
            
            required_keys = ['feature_scaler', 'target_scaler']
            return all(key in scaler_data for key in required_keys)
            
        except:
            return False
    
    def get_model_info(self):
        """Get comprehensive model information"""
        return self.model_info
    
    def get_scalers_info(self):
        """Get detailed scalers information"""
        if not self.scalers:
            return {"scalers_loaded": False}
        
        scaler_info = {
            "scalers_loaded": True,
            "format": "multi-stock" if isinstance(self.scalers, dict) else "single",
            "total_scalers": len(self.scalers) if isinstance(self.scalers, dict) else 1,
            "available_stocks": [],
            "valid_scalers": 0
        }
        
        if isinstance(self.scalers, dict):
            scaler_info["available_stocks"] = list(self.scalers.keys())
            
            # Count valid scalers
            for stock, scaler_data in self.scalers.items():
                if self._validate_scaler_data(scaler_data):
                    scaler_info["valid_scalers"] += 1
        
        return scaler_info
    
    def get_stock_specific_config(self, stock_symbol):
        """Get stock-specific configuration dari model_config"""
        try:
            if not self.model_config or 'stocks' not in self.model_config:
                return {}
            
            clean_symbol = stock_symbol.upper().replace('.JK', '')
            stocks_config = self.model_config['stocks']
            
            return stocks_config.get(clean_symbol, {})
            
        except Exception as e:
            st.error(f"❌ Error getting stock config: {e}")
            return {}