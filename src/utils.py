import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to stock data (sama persis dengan notebook)
    
    Args:
        df: DataFrame dengan OHLCV data
        
    Returns:
        DataFrame dengan technical indicators ditambahkan
    """
    df = df.copy()
    
    try:
        # Moving Averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Average
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price change indicators
        df['Price_change'] = df['Close'].pct_change()
        df['High_Low_ratio'] = df['High'] / df['Low']
        df['Open_Close_ratio'] = df['Open'] / df['Close']
        
        return df
        
    except Exception as e:
        st.error(f"Error adding technical indicators: {e}")
        return df

def calculate_price_metrics(stock_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive price metrics untuk display
    
    Args:
        stock_data: DataFrame dengan stock price data
        
    Returns:
        Dict dengan berbagai price metrics
    """
    try:
        if stock_data is None or stock_data.empty:
            return {}
        
        current_price = stock_data['Close'].iloc[-1]
        prev_price = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
        
        # Basic price metrics
        daily_change = current_price - prev_price
        daily_change_pct = (daily_change / prev_price) * 100 if prev_price != 0 else 0
        
        # 52-week metrics
        if len(stock_data) >= 252:
            high_52w = stock_data['High'].tail(252).max()
            low_52w = stock_data['Low'].tail(252).min()
        else:
            high_52w = stock_data['High'].max()
            low_52w = stock_data['Low'].min()
        
        # Volatility (annualized)
        returns = stock_data['Close'].pct_change().dropna()
        if len(returns) > 1:
            volatility = returns.std() * np.sqrt(252) * 100
        else:
            volatility = 0
        
        # Volume metrics
        current_volume = stock_data['Volume'].iloc[-1]
        if len(stock_data) >= 20:
            avg_volume = stock_data['Volume'].tail(20).mean()
        else:
            avg_volume = stock_data['Volume'].mean()
        
        # Performance metrics
        if len(stock_data) >= 30:
            month_ago_price = stock_data['Close'].iloc[-30]
            monthly_return = ((current_price - month_ago_price) / month_ago_price) * 100
        else:
            monthly_return = 0
        
        if len(stock_data) >= 90:
            quarter_ago_price = stock_data['Close'].iloc[-90]
            quarterly_return = ((current_price - quarter_ago_price) / quarter_ago_price) * 100
        else:
            quarterly_return = 0
        
        return {
            'current_price': current_price,
            'previous_price': prev_price,
            'daily_change': daily_change,
            'daily_change_pct': daily_change_pct,
            'high_52w': high_52w,
            'low_52w': low_52w,
            'volatility': volatility,
            'current_volume': current_volume,
            'avg_volume': avg_volume,
            'monthly_return': monthly_return,
            'quarterly_return': quarterly_return,
            'market_cap_proxy': current_price * current_volume,  # Rough proxy
            'price_to_52w_high': ((current_price - high_52w) / high_52w) * 100,
            'price_to_52w_low': ((current_price - low_52w) / low_52w) * 100
        }
        
    except Exception as e:
        st.error(f"Error calculating price metrics: {e}")
        return {}

def format_currency(amount: float, currency: str = "Rp") -> str:
    """Format currency untuk display"""
    try:
        if currency == "Rp":
            return f"Rp {amount:,.0f}"
        else:
            return f"{currency} {amount:,.2f}"
    except:
        return str(amount)

def format_percentage(value: float, decimal_places: int = 2) -> str:
    """Format percentage dengan sign"""
    try:
        formatted = f"{value:+.{decimal_places}f}%"
        return formatted
    except:
        return str(value)

def validate_stock_symbol(symbol: str) -> bool:
    """Validate Indonesian stock symbol format"""
    from config import INDONESIAN_STOCKS
    
    valid_symbols = [
        s.replace('.JK', '').replace('^', '') 
        for s in INDONESIAN_STOCKS.values()
    ]
    
    return symbol.upper() in [s.upper() for s in valid_symbols]

def calculate_technical_signals(data: pd.DataFrame) -> Dict[str, str]:
    """
    Calculate trading signals dari technical indicators
    
    Args:
        data: DataFrame dengan technical indicators
        
    Returns:
        Dict dengan trading signals
    """
    try:
        if data.empty or len(data) < 2:
            return {}
        
        signals = {}
        latest = data.iloc[-1]
        previous = data.iloc[-2]
        
        # RSI Signal
        if 'RSI' in data.columns:
            rsi = latest['RSI']
            if rsi > 70:
                signals['RSI'] = "Overbought"
            elif rsi < 30:
                signals['RSI'] = "Oversold"
            else:
                signals['RSI'] = "Neutral"
        
        # MACD Signal
        if 'MACD' in data.columns and 'MACD_signal' in data.columns:
            macd = latest['MACD']
            macd_signal = latest['MACD_signal']
            prev_macd = previous['MACD']
            prev_signal = previous['MACD_signal']
            
            if macd > macd_signal and prev_macd <= prev_signal:
                signals['MACD'] = "Bullish Crossover"
            elif macd < macd_signal and prev_macd >= prev_signal:
                signals['MACD'] = "Bearish Crossover"
            elif macd > macd_signal:
                signals['MACD'] = "Bullish"
            else:
                signals['MACD'] = "Bearish"
        
        # Moving Average Signal
        if 'MA_20' in data.columns and 'MA_50' in data.columns:
            ma20 = latest['MA_20']
            ma50 = latest['MA_50']
            price = latest['Close']
            
            if price > ma20 > ma50:
                signals['MA_Trend'] = "Strong Bullish"
            elif price > ma20 and ma20 < ma50:
                signals['MA_Trend'] = "Weak Bullish"
            elif price < ma20 < ma50:
                signals['MA_Trend'] = "Strong Bearish"
            else:
                signals['MA_Trend'] = "Weak Bearish"
        
        # Bollinger Bands Signal
        if all(col in data.columns for col in ['BB_upper', 'BB_lower', 'BB_middle']):
            price = latest['Close']
            bb_upper = latest['BB_upper']
            bb_lower = latest['BB_lower']
            bb_middle = latest['BB_middle']
            
            if price > bb_upper:
                signals['Bollinger'] = "Overbought"
            elif price < bb_lower:
                signals['Bollinger'] = "Oversold"
            elif price > bb_middle:
                signals['Bollinger'] = "Above Middle"
            else:
                signals['Bollinger'] = "Below Middle"
        
        return signals
        
    except Exception as e:
        st.error(f"Error calculating technical signals: {e}")
        return {}

def create_prediction_summary(predictions: list, current_price: float) -> Dict[str, Any]:
    """Create summary statistics dari predictions"""
    try:
        if not predictions:
            return {}
        
        predictions_array = np.array(predictions)
        
        return {
            'count': len(predictions),
            'mean': np.mean(predictions_array),
            'median': np.median(predictions_array),
            'min': np.min(predictions_array),
            'max': np.max(predictions_array),
            'std': np.std(predictions_array),
            'range': np.max(predictions_array) - np.min(predictions_array),
            'expected_return': ((np.mean(predictions_array) - current_price) / current_price) * 100,
            'max_upside': ((np.max(predictions_array) - current_price) / current_price) * 100,
            'max_downside': ((np.min(predictions_array) - current_price) / current_price) * 100,
            'volatility': (np.std(predictions_array) / np.mean(predictions_array)) * 100,
            'trend': 'Bullish' if np.mean(predictions_array) > current_price else 'Bearish'
        }
        
    except Exception as e:
        st.error(f"Error creating prediction summary: {e}")
        return {}

def log_user_action(action: str, details: Dict[str, Any] = None):
    """Log user actions untuk monitoring (simple implementation)"""
    try:
        timestamp = pd.Timestamp.now()
        log_entry = {
            'timestamp': timestamp,
            'action': action,
            'details': details or {}
        }
        
        # Simple logging ke session state
        if 'user_logs' not in st.session_state:
            st.session_state['user_logs'] = []
        
        st.session_state['user_logs'].append(log_entry)
        
        # Keep only last 100 entries
        if len(st.session_state['user_logs']) > 100:
            st.session_state['user_logs'] = st.session_state['user_logs'][-100:]
            
    except Exception:
        pass  # Silent fail untuk logging
