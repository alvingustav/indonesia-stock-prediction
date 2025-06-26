from openai import AzureOpenAI
import json
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, List
from config import AZURE_OPENAI_CONFIG

class ValuationAnalyzer:
    def __init__(self):
        self.client = None
        self._setup_openai_client()
    
    def _setup_openai_client(self):
        """Setup Azure OpenAI client dengan SDK terbaru"""
        try:
            if AZURE_OPENAI_CONFIG['api_key'] and AZURE_OPENAI_CONFIG['api_key'] != 'your-api-key':
                self.client = AzureOpenAI(
                    azure_endpoint=AZURE_OPENAI_CONFIG['endpoint'],
                    api_key=AZURE_OPENAI_CONFIG['api_key'],
                    api_version=AZURE_OPENAI_CONFIG['api_version']
                )
                st.success("✅ Azure OpenAI client initialized")
            else:
                st.warning("⚠️ Azure OpenAI API key not configured")
                self.client = None
        except Exception as e:
            st.error(f"❌ Error setting up OpenAI client: {e}")
            self.client = None
    
    def analyze_stock_valuation(self, stock_name: str, current_price: float, 
                              predicted_prices: List[float], historical_data: Optional[pd.DataFrame] = None) -> str:
        """
        Analyze stock valuation menggunakan Azure OpenAI
        
        Args:
            stock_name: Nama saham
            current_price: Harga saat ini
            predicted_prices: List harga prediksi
            historical_data: Data historis untuk analisis tambahan
            
        Returns:
            String analisis valuation atau error message
        """
        if not self.client:
            return "❌ Azure OpenAI not configured. Please set up API credentials."
        
        try:
            # Create enhanced analysis prompt
            prompt = self._create_comprehensive_valuation_prompt(
                stock_name, current_price, predicted_prices, historical_data
            )
            
            # Call Azure OpenAI
            response = self.client.chat.completions.create(
                model=AZURE_OPENAI_CONFIG['deployment_name'],
                messages=[
                    {
                        "role": "system", 
                        "content": """You are a senior financial analyst specializing in Indonesian stock market (IDX). 
                        Provide detailed, actionable investment analysis with specific price targets and risk assessments.
                        Focus on Indonesian market context, regulations, and economic conditions.
                        Use Indonesian Rupiah (Rp) for all price references."""
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = f"Error in valuation analysis: {str(e)}"
            st.error(error_msg)
            return error_msg
    
    def _create_comprehensive_valuation_prompt(self, stock_name: str, current_price: float,
                                             predicted_prices: List[float], historical_data: Optional[pd.DataFrame]) -> str:
        """Create comprehensive prompt untuk valuation analysis"""
        
        # Basic prediction analysis
        avg_predicted = np.mean(predicted_prices)
        max_predicted = max(predicted_prices)
        min_predicted = min(predicted_prices)
        price_trend = "increasing" if avg_predicted > current_price else "decreasing"
        potential_return = ((avg_predicted - current_price) / current_price) * 100
        prediction_volatility = (np.std(predicted_prices) / np.mean(predicted_prices)) * 100
        
        # Enhanced analysis jika historical data tersedia
        market_context = ""
        if historical_data is not None and not historical_data.empty:
            try:
                # Historical performance metrics
                recent_data = historical_data.tail(252)  # Last year
                yearly_return = ((recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]) * 100
                volatility = recent_data['Close'].pct_change().std() * np.sqrt(252) * 100
                
                # Support/Resistance levels
                high_52w = recent_data['High'].max()
                low_52w = recent_data['Low'].min()
                
                # Volume analysis
                avg_volume = recent_data['Volume'].mean()
                recent_volume = recent_data['Volume'].tail(10).mean()
                volume_trend = "increasing" if recent_volume > avg_volume else "decreasing"
                
                market_context = f"""
HISTORICAL PERFORMANCE ANALYSIS:
- 1-Year Return: {yearly_return:+.2f}%
- Annual Volatility: {volatility:.2f}%
- 52-Week High: Rp {high_52w:,.0f}
- 52-Week Low: Rp {low_52w:,.0f}
- Volume Trend: {volume_trend}
- Current vs 52W High: {((current_price - high_52w) / high_52w) * 100:+.2f}%
- Current vs 52W Low: {((current_price - low_52w) / low_52w) * 100:+.2f}%
"""
            except Exception:
                market_context = "\nHistorical analysis: Data insufficient for detailed analysis."
        
        prompt = f"""
Conduct a comprehensive investment valuation analysis for {stock_name} on the Indonesian Stock Exchange (IDX):

CURRENT MARKET DATA:
- Current Price: Rp {current_price:,.0f}
- AI Predicted Average (7-day): Rp {avg_predicted:,.0f}
- Prediction Range: Rp {min_predicted:,.0f} - Rp {max_predicted:,.0f}
- Price Trend: {price_trend}
- Expected Return: {potential_return:+.2f}%
- Prediction Volatility: {prediction_volatility:.2f}%{market_context}

REQUIRED COMPREHENSIVE ANALYSIS:

**1. INVESTMENT THESIS**
- Overall valuation assessment (Undervalued/Fair Value/Overvalued)
- Key investment highlights
- Primary value drivers

**2. RECOMMENDATION & TARGETS**
- Investment recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
- Confidence level (High/Medium/Low)
- Target price (7-day): Rp X,XXX
- Target price (1-month): Rp X,XXX
- Stop-loss level: Rp X,XXX
- Position sizing recommendation

**3. RISK-RETURN ANALYSIS**
- Upside potential: +X.X%
- Downside risk: -X.X%
- Risk level (Low/Medium/High)
- Risk factors:
  • Company-specific risks
  • Sector/industry risks
  • Market/systemic risks

**4. INDONESIAN MARKET CONTEXT**
- IDX sector performance impact
- Indonesian economic factors (GDP, inflation, BI rate)
- Currency impact (USD/IDR)
- Regulatory environment
- Regional market correlation

**5. TRADING STRATEGY**
- Optimal entry points
- Exit strategy
- Time horizon (Short/Medium/Long term)
- Portfolio allocation suggestion

**6. MARKET SENTIMENT & CATALYSTS**
- Current market sentiment
- Upcoming catalysts (earnings, dividends, corporate actions)
- Technical support/resistance levels
- Institutional flow considerations

Provide specific, actionable insights with clear reasoning. Use Indonesian market terminology where appropriate.
All price targets must be in Indonesian Rupiah (Rp).
"""
        
        return prompt
    
    def get_quick_sentiment_analysis(self, stock_name: str, predicted_prices: List[float]) -> str:
        """Get quick sentiment analysis berdasarkan price predictions"""
        if not self.client:
            return "Azure OpenAI not available for sentiment analysis"
        
        try:
            # Analyze price trend
            if len(predicted_prices) < 2:
                return "Insufficient data for sentiment analysis"
            
            trend_direction = "bullish" if predicted_prices[-1] > predicted_prices[0] else "bearish"
            price_momentum = ((predicted_prices[-1] - predicted_prices[0]) / predicted_prices[0]) * 100
            
            prompt = f"""
Provide a brief market sentiment analysis for {stock_name}:

Price Prediction Trend: {trend_direction}
Expected 7-day momentum: {price_momentum:+.2f}%

Analyze in 2-3 sentences:
1. Current market sentiment (Bullish/Bearish/Neutral)
2. Key momentum drivers
3. Short-term outlook

Keep response under 150 words, focused on Indonesian market context.
"""
            
            response = self.client.chat.completions.create(
                model=AZURE_OPENAI_CONFIG['deployment_name'],
                messages=[
                    {"role": "system", "content": "You are a market sentiment analyst for Indonesian stocks."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.6
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error in sentiment analysis: {str(e)}"
    
    def analyze_risk_metrics(self, predicted_prices: List[float], current_price: float) -> dict:
        """Calculate risk metrics dari predictions"""
        try:
            predictions_array = np.array(predicted_prices)
            
            # Calculate various risk metrics
            expected_return = ((np.mean(predictions_array) - current_price) / current_price) * 100
            downside_risk = min(0, ((np.min(predictions_array) - current_price) / current_price) * 100)
            upside_potential = ((np.max(predictions_array) - current_price) / current_price) * 100
            prediction_vol = (np.std(predictions_array) / np.mean(predictions_array)) * 100
            
            # Risk-adjusted return
            sharpe_approx = expected_return / prediction_vol if prediction_vol > 0 else 0
            
            # VaR approximation (95% confidence)
            var_95 = np.percentile(predictions_array, 5)
            var_loss = ((var_95 - current_price) / current_price) * 100
            
            return {
                'expected_return_pct': expected_return,
                'downside_risk_pct': downside_risk,
                'upside_potential_pct': upside_potential,
                'volatility_pct': prediction_vol,
                'sharpe_ratio_approx': sharpe_approx,
                'var_95_loss_pct': var_loss,
                'risk_level': 'High' if prediction_vol > 15 else 'Medium' if prediction_vol > 8 else 'Low'
            }
            
        except Exception as e:
            st.error(f"Error calculating risk metrics: {e}")
            return {}
