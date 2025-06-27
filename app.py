import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set page config
st.set_page_config(
    page_title="🇮🇩 Indonesia Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Import modules with error handling
try:
    from model_loader import ModelLoader
    from predictor import StockPredictor
    from valuation_analyzer import ValuationAnalyzer
    from data_collector import DataCollector
    from config import INDONESIAN_STOCKS
    from utils import calculate_price_metrics
except ImportError as e:
    st.error(f"❌ Error importing modules: {e}")
    st.info("Please ensure all required files are in the src/ directory")
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #f0f2f6 0%, #e8f4fd 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource(show_spinner=False)
def init_components():
    """Initialize all components with proper error handling"""
    try:
        data_collector = DataCollector()
        predictor = StockPredictor()
        valuation_analyzer = ValuationAnalyzer()
        
        # Initialize predictor (loads model)
        model_loaded = predictor.initialize()
        
        return data_collector, predictor, valuation_analyzer, model_loaded
        
    except Exception as e:
        st.error(f"❌ Error initializing components: {e}")
        return None, None, None, False

def main():
    # Header
    st.markdown('<h1 class="main-header">🇮🇩 Indonesia Stock Prediction & Valuation</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;"><em>AI-Powered Stock Analysis using LSTM-GRU Hybrid Model & Azure OpenAI</em></p>', unsafe_allow_html=True)
    
    # Initialize components
    with st.spinner("🚀 Initializing AI Models..."):
        data_collector, predictor, valuation_analyzer, model_loaded = init_components()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Control Panel")
        
        # Model status
        if model_loaded:
            st.success("✅ LSTM-GRU Model Ready")
            
            if predictor:
                with st.expander("🤖 Model Details"):
                    st.write("**Architecture:** LSTM-GRU Hybrid")
                    st.write("**Features:** 22 technical indicators")
                    st.write("**Training:** Multi-stock Indonesian data")
                    st.write("**Sequence Length:** 60 days")
        else:
            st.error("❌ Model Not Loaded")
            st.info("Please check model files in models/ directory")
        
        # Azure OpenAI status
        azure_configured = bool(os.getenv('AZURE_OPENAI_API_KEY'))
        if azure_configured:
            st.success("✅ Azure OpenAI Ready")
        else:
            st.warning("⚠️ Azure OpenAI Not Configured")
        
        st.markdown("---")
        
        # Stock selection
        selected_stock = st.selectbox(
            "🏢 Select Stock:",
            list(INDONESIAN_STOCKS.keys()),
            help="Choose Indonesian stock for analysis"
        )
        
        prediction_days = st.slider(
            "📅 Prediction Days:",
            min_value=1,
            max_value=30,
            value=7,
            help="Number of days to predict"
        )
        
        st.markdown("---")
        st.markdown("**🎯 Features:**")
        st.markdown("- 📊 Real-time Data")
        st.markdown("- 🤖 AI Prediction")
        st.markdown("- 💰 Valuation Analysis")
        st.markdown("- 📈 Technical Indicators")
    
    # Main content
    if not all([data_collector, predictor, valuation_analyzer, model_loaded]):
        st.error("❌ Application not ready. Please check configuration.")
        return
    
    stock_code = INDONESIAN_STOCKS[selected_stock]
    
    # Fetch stock data
    @st.cache_data(ttl=300)
    def get_stock_data(symbol):
        return data_collector.get_stock_data(symbol, period='1y')
    
    with st.spinner(f"📡 Fetching data for {selected_stock}..."):
        stock_data = get_stock_data(stock_code)
    
    if stock_data is None or stock_data.empty:
        st.error(f"❌ Failed to fetch data for {selected_stock}")
        return
    
    # Calculate metrics
    try:
        metrics = calculate_price_metrics(stock_data)
    except Exception as e:
        st.error(f"❌ Error calculating metrics: {e}")
        return
    
    # Display current info
    st.subheader(f"📊 {selected_stock} ({stock_code})")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Current Price",
            f"Rp {metrics['current_price']:,.0f}",
            f"{metrics['daily_change']:+,.0f} ({metrics['daily_change_pct']:+.2f}%)"
        )
    
    with col2:
        st.metric("📈 52W High", f"Rp {metrics['high_52w']:,.0f}")
    
    with col3:
        st.metric("📉 52W Low", f"Rp {metrics['low_52w']:,.0f}")
    
    with col4:
        st.metric("📊 Volatility", f"{metrics['volatility']:.1f}%")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Price Analysis", "🤖 AI Prediction", "💰 Valuation Report"])
    
    import plotly.graph_objects as go
    with tab1:
        # Price chart and analysis code here
        st.subheader("📈 Price Chart")
        recent_data = stock_data.tail(360) if len(stock_data) > 360 else stock_data
    # Buat candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=recent_data.index,
            open=recent_data['Open'],
            high=recent_data['High'],
            low=recent_data['Low'],
            close=recent_data['Close'],
            name="Candlestick"
       )])

        fig.update_layout(
            title=f"{selected_stock} Price Chart (Last {len(recent_data)} Days)",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            height=500
       )

        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🤖 AI Price Prediction")
        
        if st.button("🚀 Generate Prediction", type="primary"):
            with st.spinner("🤖 Running AI prediction..."):
                try:
                    # Clean symbol
                    clean_symbol = stock_code.replace('.JK', '').replace('^', '')
                    
                    # Generate predictions
                    predictions = predictor.predict_prices(clean_symbol, days_ahead=prediction_days)
                    
                    if predictions:
                        st.session_state['predictions'] = predictions
                        st.session_state['current_price'] = metrics['current_price']
                        st.session_state['stock_name'] = selected_stock
                        
                        st.success("✅ Prediction completed!")
                        
                        # Display results
                        avg_pred = np.mean(predictions)
                        potential = ((avg_pred - metrics['current_price']) / metrics['current_price']) * 100
                        
                        pred_col1, pred_col2, pred_col3 = st.columns(3)
                        
                        with pred_col1:
                            st.metric("📊 Average Predicted", f"Rp {avg_pred:,.0f}")
                        
                        with pred_col2:
                            trend = "📈 Bullish" if avg_pred > metrics['current_price'] else "📉 Bearish"
                            st.metric("📈 Trend", trend)
                        
                        with pred_col3:
                            st.metric("💹 Expected Return", f"{potential:+.2f}%")
                        
                        # Predictions table
                        future_dates = pd.date_range(
                            start=stock_data.index[-1] + pd.Timedelta(days=1),
                            periods=prediction_days,
                            freq='D'
                        )
                        
                        pred_df = pd.DataFrame({
                            'Date': future_dates,
                            'Predicted Price': predictions,
                            'Change %': [((p - metrics['current_price']) / metrics['current_price']) * 100 for p in predictions]
                        })
                        
                        st.dataframe(pred_df, use_container_width=True)
                    
                    else:
                        st.error("❌ Prediction failed")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with tab3:
        st.subheader("💰 Investment Valuation")
        
        if st.button("🧠 Generate AI Analysis", type="primary"):
            if 'predictions' not in st.session_state:
                st.warning("⚠️ Please generate predictions first")
            else:
                with st.spinner("🧠 Analyzing with Azure OpenAI..."):
                    try:
                        analysis = valuation_analyzer.analyze_stock_valuation(
                            st.session_state['stock_name'],
                            st.session_state['current_price'],
                            st.session_state['predictions'],
                            stock_data
                        )
                        
                        if "Error" not in analysis:
                            st.success("✅ Analysis completed!")
                            st.markdown("### 📋 Investment Analysis Report")
                            st.markdown(analysis)
                        else:
                            st.error("❌ Analysis failed")
                            st.error(analysis)
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>🇮🇩 Indonesia Stock Prediction & Valuation Platform</strong></p>
        <p>Powered by LSTM-GRU Hybrid AI Model & Azure OpenAI | Data from Yahoo Finance</p>
        <p>⚠️ <em>Educational purposes only. Not financial advice.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
