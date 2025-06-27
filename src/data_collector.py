import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import warnings
from typing import Optional, Dict, Any
warnings.filterwarnings('ignore')

class DataCollector:
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    def get_stock_data(self, symbol: str, period: str = '2y') -> Optional[pd.DataFrame]:
        """
        Fetch stock data dari Yahoo Finance dengan caching
        
        Args:
            symbol: Stock symbol (e.g., 'BBCA.JK')
            period: Time period ('1y', '2y', '5y', etc.)
            
        Returns:
            DataFrame dengan OHLCV data atau None jika error
        """
        try:
            cache_key = f"{symbol}_{period}"
            current_time = datetime.now()
            
            # Check cache
            if cache_key in self.cache:
                cached_data, cache_time = self.cache[cache_key]
                if (current_time - cache_time).seconds < self.cache_ttl:
                    return cached_data
            
            if symbol == 'JKSE.JK':
                symbol = '^JKSE'
            # Fetch fresh data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                st.warning(f"No data found for {symbol}")
                return None
            
            # Clean data
            data = self._clean_data(data)
            
            # Cache the result
            self.cache[cache_key] = (data, current_time)
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate stock data"""
        # Remove any NaN values
        data = data.dropna()
        
        # Ensure positive values
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = data[col].abs()
        
        # Ensure High >= Low
        if 'High' in data.columns and 'Low' in data.columns:
            data.loc[data['High'] < data['Low'], ['High', 'Low']] = \
                data.loc[data['High'] < data['Low'], ['Low', 'High']].values
        
        return data
    
    def get_multiple_stocks_data(self, symbols: Dict[str, str], period: str = '2y') -> Dict[str, pd.DataFrame]:
        """
        Fetch data untuk multiple stocks
        
        Args:
            symbols: Dict mapping name to symbol
            period: Time period
            
        Returns:
            Dict mapping name to DataFrame
        """
        stock_data = {}
        
        for name, symbol in symbols.items():
            with st.spinner(f"Fetching {name} data..."):
                data = self.get_stock_data(symbol, period)
                if data is not None:
                    stock_data[name] = data
                    st.success(f"✅ {name}: {len(data)} records")
                else:
                    st.error(f"❌ Failed to fetch {name}")
        
        return stock_data
    
    def get_real_time_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote untuk stock"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'current_price': info.get('currentPrice', 0),
                'previous_close': info.get('previousClose', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0)
            }
            
        except Exception as e:
            st.error(f"Error getting real-time quote for {symbol}: {str(e)}")
            return None
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality dan return metrics"""
        if data is None or data.empty:
            return {'valid': False, 'reason': 'No data'}
        
        quality_metrics = {
            'valid': True,
            'total_records': len(data),
            'date_range': {
                'start': data.index.min(),
                'end': data.index.max()
            },
            'missing_values': data.isnull().sum().to_dict(),
            'zero_volume_days': (data['Volume'] == 0).sum() if 'Volume' in data.columns else 0,
            'price_anomalies': 0
        }
        
        # Check for price anomalies
        if 'Close' in data.columns:
            price_changes = data['Close'].pct_change()
            extreme_changes = (abs(price_changes) > 0.2).sum()  # >20% daily change
            quality_metrics['price_anomalies'] = extreme_changes
        
        # Determine if data is sufficient for training/prediction
        min_required_records = 100
        if len(data) < min_required_records:
            quality_metrics['valid'] = False
            quality_metrics['reason'] = f'Insufficient data: {len(data)} < {min_required_records}'
        
        return quality_metrics
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status"""
        try:
            # Use IHSG (Jakarta Composite) sebagai reference
            ihsg = yf.Ticker("^JKSE")
            hist = ihsg.history(period="1d")
            
            if not hist.empty:
                latest = hist.iloc[-1]
                return {
                    'market_open': True,
                    'last_trade': hist.index[-1],
                    'ihsg_level': latest['Close'],
                    'ihsg_change': latest['Close'] - latest['Open'],
                    'ihsg_volume': latest['Volume']
                }
            else:
                return {'market_open': False, 'last_trade': None}
                
        except Exception as e:
            return {'market_open': False, 'error': str(e)}
