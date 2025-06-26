import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from model_loader import ModelLoader
from data_collector import DataCollector
from utils import add_technical_indicators
from config import INDONESIAN_STOCKS, MODEL_CONFIG

class StockPredictor:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.data_collector = DataCollector()
        self.sequence_length = MODEL_CONFIG['sequence_length']
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize predictor dengan pre-trained model"""
        try:
            success = self.model_loader.load_pretrained_model()
            self.is_initialized = success
            return success
        except Exception as e:
            st.error(f"Error initializing predictor: {str(e)}")
            return False
    
    def prepare_prediction_data(self, symbol: str, period: str = '2y') -> Tuple[Optional[np.ndarray], Optional[object], Optional[float]]:
        """
        Prepare data untuk prediction menggunakan pre-trained scalers
        
        Args:
            symbol: Stock symbol tanpa .JK suffix (e.g., 'BBCA')
            period: Data period untuk fetch
            
        Returns:
            Tuple of (sequence_data, target_scaler, current_price)
        """
        try:
            if not self.is_initialized:
                st.error("Predictor not initialized")
                return None, None, None
            
            # Add .JK suffix jika perlu
            stock_symbol = f"{symbol}.JK" if not symbol.endswith('.JK') and symbol != 'IHSG' else symbol
            if symbol == 'IHSG':
                stock_symbol = '^JKSE'
            
            # Fetch stock data
            stock_data = self.data_collector.get_stock_data(stock_symbol, period)
            if stock_data is None or stock_data.empty:
                st.error(f"Failed to fetch data for {symbol}")
                return None, None, None
            
            # Add technical indicators (sama dengan notebook)
            data_with_indicators = add_technical_indicators(stock_data)
            
            # Remove NaN values
            data_clean = data_with_indicators.dropna()
            
            if len(data_clean) < self.sequence_length:
                st.error(f"Insufficient data: need at least {self.sequence_length} records, got {len(data_clean)}")
                return None, None, None
            
            # Select features (sama order dengan training)
            feature_columns = self.model_loader.feature_columns
            features = data_clean[feature_columns].values
            
            # Get appropriate scaler untuk stock ini
            scaler_data = self.model_loader.get_scaler_for_stock(symbol.upper())
            if scaler_data is None:
                st.error(f"No scaler found for {symbol}")
                return None, None, None
            
            feature_scaler = scaler_data['feature_scaler']
            target_scaler = scaler_data['target_scaler']
            
            # Scale features
            scaled_features = feature_scaler.transform(features)
            
            # Create sequence untuk prediction (last sequence_length records)
            sequence = scaled_features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            current_price = data_clean['Close'].iloc[-1]
            
            return sequence, target_scaler, current_price
            
        except Exception as e:
            st.error(f"Error preparing prediction data: {str(e)}")
            return None, None, None
    
    def predict_prices(self, symbol: str, days_ahead: int = 7) -> Optional[List[float]]:
        """
        Predict stock prices untuk beberapa hari ke depan
        
        Args:
            symbol: Stock symbol (e.g., 'BBCA')
            days_ahead: Number of days to predict
            
        Returns:
            List of predicted prices atau None jika error
        """
        try:
            if not self.is_initialized:
                st.error("Predictor not initialized. Please check model files.")
                return None
            
            # Prepare data
            sequence, target_scaler, current_price = self.prepare_prediction_data(symbol)
            if sequence is None:
                return None
            
            # Generate predictions
            predictions = []
            current_sequence = sequence.copy()
            
            for day in range(days_ahead):
                # Predict next price
                pred_scaled = self.model_loader.model.predict(current_sequence, verbose=0)
                
                # Inverse transform prediction
                pred_price = target_scaler.inverse_transform(pred_scaled)[0][0]
                predictions.append(float(pred_price))
                
                # Update sequence untuk next prediction
                # Simple approach: gunakan predicted price untuk update features
                next_features = current_sequence[0, -1, :].copy()
                
                # Update Close price feature (index 3 dalam feature_columns)
                close_idx = self.model_loader.feature_columns.index('Close')
                next_features[close_idx] = pred_scaled[0][0]
                
                # Roll the sequence forward
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, :] = next_features
            
            return predictions
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            return None
    
    def predict_with_confidence(self, symbol: str, days_ahead: int = 7, n_samples: int = 10) -> Optional[dict]:
        """
        Generate predictions dengan confidence intervals
        
        Args:
            symbol: Stock symbol
            days_ahead: Days to predict
            n_samples: Number of prediction samples untuk confidence
            
        Returns:
            Dict dengan predictions, confidence intervals, dll
        """
        try:
            if not self.is_initialized:
                return None
            
            # Generate multiple predictions dengan slight noise
            all_predictions = []
            
            for _ in range(n_samples):
                predictions = self.predict_prices(symbol, days_ahead)
                if predictions:
                    all_predictions.append(predictions)
            
            if not all_predictions:
                return None
            
            # Calculate statistics
            predictions_array = np.array(all_predictions)
            mean_predictions = np.mean(predictions_array, axis=0)
            std_predictions = np.std(predictions_array, axis=0)
            
            # Calculate confidence intervals (95%)
            confidence_interval = 1.96 * std_predictions
            lower_bound = mean_predictions - confidence_interval
            upper_bound = mean_predictions + confidence_interval
            
            return {
                'predictions': mean_predictions.tolist(),
                'lower_bound': lower_bound.tolist(),
                'upper_bound': upper_bound.tolist(),
                'confidence_std': std_predictions.tolist(),
                'n_samples': n_samples
            }
            
        except Exception as e:
            st.error(f"Error in confidence prediction: {str(e)}")
            return None
    
    def validate_prediction_input(self, symbol: str, days_ahead: int) -> bool:
        """Validate input parameters"""
        if symbol.upper() not in [s.replace('.JK', '').replace('^', '') for s in INDONESIAN_STOCKS.values()]:
            st.error(f"Unsupported stock symbol: {symbol}")
            return False
        
        if days_ahead < 1 or days_ahead > 30:
            st.error(f"Invalid prediction period: {days_ahead} days. Must be 1-30.")
            return False
        
        return True
    
    def get_prediction_metadata(self, symbol: str) -> dict:
        """Get metadata about prediction"""
        return {
            'model_architecture': MODEL_CONFIG['model_architecture'],
            'sequence_length': self.sequence_length,
            'features_count': MODEL_CONFIG['features_count'],
            'training_stocks': MODEL_CONFIG['training_stocks'],
            'symbol_supported': symbol.upper() in [s.replace('.JK', '').replace('^', '') for s in INDONESIAN_STOCKS.values()]
        }
