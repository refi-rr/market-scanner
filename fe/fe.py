# app.py - Enhanced Streamlit Frontend
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import io
from fpdf import FPDF
import base64
import uuid

# Configuration
API_BASE_URL = "http://localhost:2401/api"

# Page config
st.set_page_config(
    page_title="Crypto Scanner Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_theme_css(theme: str = 'dark'):
    """Inject theme-specific CSS."""
    if theme == 'light':
        bg_app = '#ffffff'
        bg_main = '#f7f7f8'
        text_color = '#111111'
        metric_color = '#111111'
        card_bg = 'linear-gradient(135deg, #ffffff 0%, #f3f4f6 100%)'
        border_color = '#e5e7eb'
    else:
        bg_app = '#0f1318'
        bg_main = '#0f1318'
        text_color = '#ffffff'
        metric_color = '#ffffff'
        card_bg = 'linear-gradient(135deg, #1a1f2e 0%, #252b3d 100%)'
        border_color = '#2d3548'

    css = f"""
    <style>
        /* Force color scheme and ignore OS preference */
        :root, body, .stApp {{
            color-scheme: {'light' if theme == 'light' else 'dark'} !important;
        }}
        /* Neutralize prefers-color-scheme media queries by reapplying our colors */
        @media (prefers-color-scheme: dark), (prefers-color-scheme: light) {{
            .stApp, .main, div[data-testid="stSidebar"], div[data-testid="stSidebar"] * {{
                background: {bg_main} !important;
                color: {text_color} !important;
                background-image: none !important;
                box-shadow: none !important;
            }}
            .stApp input, .stApp textarea, .stApp select {{
                color: {text_color} !important;
                -webkit-text-fill-color: {text_color} !important;
            }}
        }}
        .main {{
            background-color: {bg_main};
            color: {text_color};
        }}
        .stApp {{
            background-color: {bg_app};
        }}
        /* Sidebar styling */
        /* Apply background/color deeply to override nested Streamlit wrappers */
        div[data-testid="stSidebar"],
        div[data-testid="stSidebar"] > div,
        div[data-testid="stSidebar"] > div > div,
        div[data-testid="stSidebar"] > div > div > div {{
            background: {bg_main} !important;
            background-image: none !important;
            box-shadow: none !important;
            color: {text_color} !important;
            border-right: 1px solid {border_color} !important;
        }}
        /* Force text color for common elements inside sidebar */
        div[data-testid="stSidebar"] h1, div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3,
        div[data-testid="stSidebar"] p, div[data-testid="stSidebar"] label, div[data-testid="stSidebar"] span,
        div[data-testid="stSidebar"] .stMarkdown, div[data-testid="stSidebar"] .stText,
        div[data-testid="stSidebar"] .css-1lsmgbg, div[data-testid="stSidebar"] .css-1d391kg {{
            color: {text_color} !important;
        }}
        /* Buttons and inputs in sidebar */
        div[data-testid="stSidebar"] button, div[data-testid="stSidebar"] .stButton>button {{
            color: {text_color} !important;
            background: {card_bg} !important;
            border: 1px solid {border_color} !important;
        }}
        div[data-testid="stSidebar"] input, div[data-testid="stSidebar"] textarea, div[data-testid="stSidebar"] select {{
            color: {text_color} !important;
            background-color: transparent !important;
            border: 1px solid {border_color} !important;
        }}
        /* Radio/select labels */
        div[data-testid="stSidebar"] [role="radiogroup"] label, div[data-testid="stSidebar"] .stRadio, div[data-testid="stSidebar"] .stSelectbox {{
            color: {text_color} !important;
        }}
        /* Generic catch-all to force readable text inside sidebar and remove opaque backgrounds */
        div[data-testid="stSidebar"] * {{
            color: {text_color} !important;
            background: transparent !important;
            background-color: transparent !important;
            box-shadow: none !important;
        }}
        div[data-testid="stMetricValue"] {{
            font-size: 28px;
            color: {metric_color};
        }}
        /* General content text styling */
        div[data-testid="stMarkdownContainer"], .stText, p, span, label, h1, h2, h3, h4 {{
            color: {text_color} !important;
        }}
        /* Expander and containers */
        div[data-testid="stExpander"] > div {{
            background: transparent !important;
            color: {text_color} !important;
        }}
        .metric-card {{
            background: {card_bg};
            border-radius: 12px;
            padding: 20px;
            border: 1px solid {border_color};
        }}
        .signal-long {{
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: bold;
        }}
        .signal-short {{
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: bold;
        }}
        .signal-neutral {{
            background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: bold;
        }}
        .indicator-bullish {{
            color: #10b981;
            font-weight: bold;
        }}
        .indicator-bearish {{
            color: #ef4444;
            font-weight: bold;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def render_modal_css():
    """Inject modal CSS for chart expansion."""
    modal_css = """
    <style>
        /* Modal styling */
        .modal-overlay {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.7);
            z-index: 9999;
            overflow-y: auto;
        }}
        
        .modal-overlay.show {{
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .modal-content {{
            background-color: #0f1318;
            color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            max-width: 95vw;
            max-height: 95vh;
            overflow-y: auto;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
            position: relative;
            border: 1px solid #2d3548;
        }}
        
        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #2d3548;
        }}
        
        .modal-close-btn {{
            background-color: #ef4444;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
        }}
        
        .modal-close-btn:hover {{
            background-color: #dc2626;
        }}
        
        .chart-controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 20px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
        }}
        
        .indicator-toggle {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            background: rgba(100, 100, 100, 0.2);
            border-radius: 6px;
            cursor: pointer;
            user-select: none;
            font-size: 13px;
        }}
        
        .indicator-toggle:hover {{
            background: rgba(100, 100, 100, 0.3);
        }}
        
        .indicator-toggle.active {{
            background: rgba(76, 175, 80, 0.3);
            border: 1px solid #4CAF50;
        }}
        
        /* Expand button styling */
        .expand-chart-btn {{
            background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
            color: white;
            border: none;
            padding: 10px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            font-size: 18px;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 44px;
            min-height: 44px;
            box-shadow: 0 2px 8px rgba(33, 150, 243, 0.3);
        }}
        
        .expand-chart-btn:hover {{
            background: linear-gradient(135deg, #1976D2 0%, #1565C0 100%);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(33, 150, 243, 0.5);
        }}
        
        .expand-chart-btn:active {{
            transform: translateY(0px);
        }}
        
        /* Responsive chart container */
        .chart-container {{
            position: relative;
            width: 100%;
            min-height: 400px;
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .chart-container.expanded {{
            min-height: 800px;
        }}
        
        /* Indicator summary cards */
        .indicator-summary-card {{
            background: linear-gradient(135deg, rgba(33, 150, 243, 0.1) 0%, rgba(100, 181, 246, 0.05) 100%);
            border-left: 4px solid #2196F3;
            padding: 12px;
            border-radius: 6px;
            margin: 5px 0;
            font-size: 13px;
        }}
        
        .indicator-summary-card.bullish {{
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(129, 199, 132, 0.05) 100%);
            border-left-color: #4CAF50;
        }}
        
        .indicator-summary-card.bearish {{
            background: linear-gradient(135deg, rgba(244, 67, 54, 0.1) 0%, rgba(248, 118, 113, 0.05) 100%);
            border-left-color: #ef4444;
        }}
    </style>
    """
    st.markdown(modal_css, unsafe_allow_html=True)

# Session state initialization
if 'token' not in st.session_state:
    st.session_state.token = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'page' not in st.session_state:
    st.session_state.page = 'login'
# Theme (dark / light)
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
# Chart modal state
if 'show_chart_modal' not in st.session_state:
    st.session_state.show_chart_modal = False
if 'modal_chart_data' not in st.session_state:
    st.session_state.modal_chart_data = None
if 'selected_indicators' not in st.session_state:
    st.session_state.selected_indicators = {
        'rsi': True,
        'macd': True,
        'volume': True,
        'stochastic': True,
        'smi': True,
        'bollinger_bands': True,
        'support_resistance': True,
        'order_blocks': True,
        'fair_value_gaps': True
    }

# Render CSS according to current theme (after theme is initialized)
render_theme_css(st.session_state.theme)
render_modal_css()

def show_chart_modal(chart, data, title):
    """Display chart in an expandable modal."""
    if st.session_state.show_chart_modal:
        modal_html = f"""
        <div class="modal-overlay show" id="chartModal">
            <div class="modal-content">
                <div class="modal-header">
                    <h2>{title} - Detailed Analysis</h2>
                    <button class="modal-close-btn" onclick="document.getElementById('chartModal').classList.remove('show')">✕ Close</button>
                </div>
            </div>
        </div>
        <script>
            var modal = document.getElementById('chartModal');
            modal.addEventListener('click', function(e) {{
                if (e.target === modal) {{
                    modal.classList.remove('show');
                }}
            }});
        </script>
        """
        st.markdown(modal_html, unsafe_allow_html=True)

def display_chart_with_controls(chart, result, show_ichimoku=False):
    """Display chart with expand button and indicator controls in collapsible section."""
    
    # Display chart with expand button
    col_expand, col_title = st.columns([0.5, 4])
    
    with col_expand:
        if st.button("🔍", key="expand_chart_btn", help="View chart in expanded layout", use_container_width=True):
            st.session_state.show_chart_modal = True
            st.session_state.modal_chart_data = {
                'chart': chart,
                'result': result,
                'show_ichimoku': show_ichimoku
            }
            st.rerun()
    
    with col_title:
        st.plotly_chart(chart, use_container_width=True)
    
    # Indicator controls in collapsible section
    with st.expander("⚙️ Chart Indicators Control", expanded=False):
        st.markdown("**Toggle indicators visibility:**")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        indicators = [
            ('rsi', '📊 RSI'),
            ('macd', '📈 MACD'),
            ('volume', '📉 Volume'),
            ('stochastic', '⚡ Stochastic'),
            ('smi', '🎯 SMI')
        ]
        
        for idx, (key, label) in enumerate(indicators):
            if idx % 5 == 0:
                cols = [col1, col2, col3, col4, col5]
            with cols[idx % 5]:
                st.session_state.selected_indicators[key] = st.checkbox(
                    label, 
                    value=st.session_state.selected_indicators.get(key, True), 
                    key=f"toggle_{key}"
                )
        
        st.markdown("**Advanced Indicators:**")
        col_adv1, col_adv2, col_adv3, col_adv4 = st.columns(4)
        adv_indicators = [
            ('bollinger_bands', '☁️ Bollinger Bands'),
            ('support_resistance', '📍 Support/Resistance'),
            ('order_blocks', '📦 Order Blocks'),
            ('fair_value_gaps', '💨 Fair Value Gaps')
        ]
        
        for idx, (key, label) in enumerate(adv_indicators):
            cols_adv = [col_adv1, col_adv2, col_adv3, col_adv4]
            with cols_adv[idx % 4]:
                st.session_state.selected_indicators[key] = st.checkbox(
                    label, 
                    value=st.session_state.selected_indicators.get(key, True), 
                    key=f"toggle_adv_{key}"
                )

def create_compact_indicator_display(result):
    """Create a compact display of key indicators for quick analysis."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    # RSI
    with col1:
        rsi = result.get('rsi')
        if rsi is not None:
            status = "🔴 Overbought" if rsi > 70 else "🟢 Oversold" if rsi < 30 else "⚪ Neutral"
            st.metric("RSI (14)", f"{rsi:.1f}", status)
        else:
            st.metric("RSI (14)", "N/A")
    
    # MACD
    with col2:
        macd = result.get('macd', {})
        if macd and macd.get('histogram') is not None:
            status = "🟢 Bullish" if macd['histogram'] > 0 else "🔴 Bearish"
            st.metric("MACD", f"{macd['histogram']:.4f}", status)
        else:
            st.metric("MACD", "N/A")
    
    # Stochastic
    with col3:
        stoch = result.get('stochastic', {})
        if stoch and stoch.get('k') is not None:
            status = "🔴 Overbought" if stoch['k'] > 80 else "🟢 Oversold" if stoch['k'] < 20 else "⚪ Neutral"
            st.metric("Stochastic %K", f"{stoch['k']:.0f}", status)
        else:
            st.metric("Stochastic", "N/A")
    
    # SMI
    with col4:
        smi = result.get('smi', {})
        if smi and smi.get('smi') is not None:
            status = "🟢 Bullish" if smi['smi'] > 0 else "🔴 Bearish"
            st.metric("SMI", f"{smi['smi']:.1f}", status)
        else:
            st.metric("SMI", "N/A")

def calculate_indicator_status(result):
    """Calculate which indicators are bullish, bearish, or neutral."""
    bullish = []
    bearish = []
    neutral = []
    
    indicators = result.get('indicators', {})
    
    # Check Moving Averages
    current_price = result.get('current_price', 0)
    mas = result.get('moving_averages', {})
    if mas:
        ma_bullish = []
        ma_bearish = []
        for ma_name, ma_value in mas.items():
            if ma_value and current_price > ma_value:
                ma_bullish.append(ma_name)
            elif ma_value and current_price < ma_value:
                ma_bearish.append(ma_name)
        
        if ma_bullish:
            bullish.append(f"Moving Averages ({len(ma_bullish)})")
        if ma_bearish:
            bearish.append(f"Moving Averages ({len(ma_bearish)})")
    
    # Check RSI
    rsi = result.get('rsi')
    if rsi is not None:
        if rsi > 70:
            bearish.append("RSI (Overbought)")
        elif rsi < 30:
            bullish.append("RSI (Oversold)")
        else:
            neutral.append("RSI (Neutral Zone)")
    
    # Check MACD
    macd = result.get('macd', {})
    if macd and macd.get('histogram') is not None:
        if macd['histogram'] > 0:
            bullish.append("MACD (Bullish Crossover)")
        else:
            bearish.append("MACD (Bearish Crossover)")
    
    # Check Volume
    volume = result.get('volume', {})
    if volume and volume.get('status'):
        if volume['status'] == 'High Volume' or volume.get('strength') == 'strong':
            if result.get('signal') == 'LONG':
                bullish.append("Volume (High & Confirming)")
            else:
                bearish.append("Volume (High & Confirming)")
        else:
            neutral.append("Volume (Low/Weak)")
    
    # Check Stochastic
    stoch = result.get('stochastic', {})
    if stoch and stoch.get('k') is not None:
        if stoch['k'] > 80:
            bearish.append("Stochastic (Overbought)")
        elif stoch['k'] < 20:
            bullish.append("Stochastic (Oversold)")
        else:
            neutral.append("Stochastic (Neutral Zone)")
    
    # Check Bollinger Bands
    bb = result.get('bollinger_bands', {})
    if bb and bb.get('upper') is not None:
        if current_price > bb.get('upper', 0):
            bearish.append("Bollinger Bands (Upper Band)")
        elif current_price < bb.get('lower', 0):
            bullish.append("Bollinger Bands (Lower Band)")
        else:
            neutral.append("Bollinger Bands (Middle Zone)")
    
    # Check SMI
    smi = result.get('smi', {})
    if smi and smi.get('smi') is not None:
        if smi['smi'] > 0:
            bullish.append("SMI (Positive)")
        else:
            bearish.append("SMI (Negative)")
    
    # Check Ichimoku
    ichimoku = result.get('ichimoku', {})
    if ichimoku and ichimoku.get('tenkan_sen') is not None:
        senkou_a = ichimoku.get('senkou_span_a')
        senkou_b = ichimoku.get('senkou_span_b')
        if senkou_a and senkou_b:
            if senkou_a > senkou_b:
                bullish.append("Ichimoku Cloud (Bullish)")
            else:
                bearish.append("Ichimoku Cloud (Bearish)")
    
    # Check Support & Resistance
    support_levels = result.get('support_levels', [])
    resistance_levels = result.get('resistance_levels', [])
    if support_levels or resistance_levels:
        if support_levels and current_price > support_levels[0]:
            bullish.append(f"Support Levels ({len(support_levels)})")
        if resistance_levels and current_price < resistance_levels[0]:
            neutral.append(f"Resistance Levels ({len(resistance_levels)})")
    
    # Check SMC indicators if available
    smc = result.get('smc', {})
    if smc:
        bos = smc.get('break_of_structure', {})
        if bos.get('bullish_bos'):
            bullish.append("Break of Structure (Bullish)")
        if bos.get('bearish_bos'):
            bearish.append("Break of Structure (Bearish)")
        
        ob = smc.get('order_blocks', {})
        if ob.get('bullish'):
            bullish.append("Order Blocks (Bullish)")
        if ob.get('bearish'):
            bearish.append("Order Blocks (Bearish)")
    
    return {
        'bullish': list(set(bullish)),  # Remove duplicates
        'bearish': list(set(bearish)),  # Remove duplicates
        'neutral': list(set(neutral))   # Remove duplicates
    }

def display_signal_reasoning(result):
    """Display signal with detailed reasoning and scoring breakdown."""
    signal = result.get('signal', 'NEUTRAL')
    confidence = result.get('confidence', 0)
    bullish_signals = result.get('bullish_signals', 0)
    bearish_signals = result.get('bearish_signals', 0)
    
    # Signal color mapping
    signal_colors = {
        'LONG': '#10b981',
        'SHORT': '#ef4444',
        'NEUTRAL': '#6b7280'
    }
    
    signal_emojis = {
        'LONG': '📈',
        'SHORT': '📉',
        'NEUTRAL': '⚪'
    }
    
    signal_color = signal_colors.get(signal, '#6b7280')
    signal_emoji = signal_emojis.get(signal, '⚪')
    
    # Main signal display
    col_signal, col_confidence = st.columns([2, 1])
    
    with col_signal:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {signal_color}20 0%, {signal_color}05 100%);
                border-left: 4px solid {signal_color};
                padding: 20px;
                border-radius: 8px;
                margin: 10px 0;
            ">
                <h2 style="color: {signal_color}; margin: 0 0 10px 0;">
                    {signal_emoji} Signal: {signal}
                </h2>
                <p style="margin: 5px 0; color: #ffffff;">
                    <strong>Confidence:</strong> {confidence}%
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col_confidence:
        # Confidence gauge
        if confidence >= 70:
            confidence_label = "🟢 Strong"
            confidence_color = "#10b981"
        elif confidence >= 50:
            confidence_label = "🟡 Moderate"
            confidence_color = "#f59e0b"
        else:
            confidence_label = "🔴 Weak"
            confidence_color = "#ef4444"
        
        st.metric("Confidence Level", confidence_label, f"{confidence}%")
    
    # Detailed reasoning
    st.markdown("### 📊 Signal Breakdown")
    
    breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)
    
    with breakdown_col1:
        st.markdown(
            f"""
            <div style="
                background: rgba(76, 175, 80, 0.1);
                border-left: 3px solid #4CAF50;
                padding: 15px;
                border-radius: 6px;
            ">
                <h4 style="margin-top: 0; color: #4CAF50;">Bullish Signals</h4>
                <p style="font-size: 24px; margin: 10px 0; color: #4CAF50;"><strong>{bullish_signals}</strong></p>
                <p style="margin: 0; font-size: 12px; color: #888;">Indicators supporting upward trend</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with breakdown_col2:
        st.markdown(
            f"""
            <div style="
                background: rgba(244, 67, 54, 0.1);
                border-left: 3px solid #ef4444;
                padding: 15px;
                border-radius: 6px;
            ">
                <h4 style="margin-top: 0; color: #ef4444;">Bearish Signals</h4>
                <p style="font-size: 24px; margin: 10px 0; color: #ef4444;"><strong>{bearish_signals}</strong></p>
                <p style="margin: 0; font-size: 12px; color: #888;">Indicators supporting downward trend</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Indicator Details Breakdown
    st.markdown("### 🎯 Indicator Breakdown")
    
    # Calculate indicator statuses
    indicators_status = calculate_indicator_status(result)
    
    if indicators_status['bullish'] or indicators_status['bearish'] or indicators_status['neutral']:
        ind_col1, ind_col2, ind_col3 = st.columns(3)
        
        # Bullish indicators
        with ind_col1:
            st.markdown("**🟢 Bullish Indicators**")
            if indicators_status['bullish']:
                for indicator in indicators_status['bullish']:
                    st.markdown(
                        f"""
                        <div style="
                            background: rgba(76, 175, 80, 0.15);
                            padding: 8px 12px;
                            border-radius: 4px;
                            margin: 4px 0;
                            border-left: 3px solid #4CAF50;
                            font-size: 13px;
                        ">
                            ✓ {indicator}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No bullish indicators")
        
        # Bearish indicators
        with ind_col2:
            st.markdown("**🔴 Bearish Indicators**")
            if indicators_status['bearish']:
                for indicator in indicators_status['bearish']:
                    st.markdown(
                        f"""
                        <div style="
                            background: rgba(244, 67, 54, 0.15);
                            padding: 8px 12px;
                            border-radius: 4px;
                            margin: 4px 0;
                            border-left: 3px solid #ef4444;
                            font-size: 13px;
                        ">
                            ✗ {indicator}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No bearish indicators")
        
        # Neutral indicators
        with ind_col3:
            st.markdown("**⚪ Neutral Indicators**")
            if indicators_status['neutral']:
                for indicator in indicators_status['neutral']:
                    st.markdown(
                        f"""
                        <div style="
                            background: rgba(158, 158, 158, 0.15);
                            padding: 8px 12px;
                            border-radius: 4px;
                            margin: 4px 0;
                            border-left: 3px solid #9E9E9E;
                            font-size: 13px;
                        ">
                            • {indicator}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No neutral indicators")
    
    with breakdown_col3:
        total_signals = bullish_signals + bearish_signals
        if total_signals > 0:
            bullish_percentage = (bullish_signals / total_signals) * 100
            bearish_percentage = (bearish_signals / total_signals) * 100
        else:
            bullish_percentage = 0
            bearish_percentage = 0
        
        st.markdown(
            f"""
            <div style="
                background: rgba(33, 150, 243, 0.1);
                border-left: 3px solid #2196F3;
                padding: 15px;
                border-radius: 6px;
            ">
                <h4 style="margin-top: 0; color: #2196F3;">Score Ratio</h4>
                <p style="margin: 10px 0 5px 0; font-size: 12px;">
                    <span style="color: #4CAF50;">🟢 {bullish_percentage:.0f}%</span> 
                    /
                    <span style="color: #ef4444;">🔴 {bearish_percentage:.0f}%</span>
                </p>
                <p style="margin: 0; font-size: 12px; color: #888;">Bullish vs Bearish ratio</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Detailed analysis if available
    if 'analysis' in result and result['analysis']:
        st.markdown("### 🔍 Detailed Analysis")
        
        analysis = result['analysis']
        with st.expander("📈 Indicator Details", expanded=False):
            analysis_cols = st.columns(2)
            
            with analysis_cols[0]:
                st.write("**Trend Analysis**")
                if 'trend' in analysis:
                    st.write(f"Current Trend: {analysis['trend']}")
                
                st.write("**Moving Averages**")
                if 'moving_averages' in analysis:
                    for ma, value in list(analysis['moving_averages'].items())[:3]:
                        if value:
                            st.write(f"• {ma}: ${value:,.2f}")
            
            with analysis_cols[1]:
                st.write("**Momentum Indicators**")
                if 'rsi' in analysis:
                    rsi_status = "Overbought" if analysis['rsi'] > 70 else "Oversold" if analysis['rsi'] < 30 else "Neutral"
                    st.write(f"• RSI: {analysis['rsi']:.2f} ({rsi_status})")
                
                st.write("**Volume & Volatility**")
                if 'volume' in analysis and 'status' in analysis['volume']:
                    st.write(f"• Volume: {analysis['volume']['status']}")

# Helper functions
def make_request(endpoint, method="GET", data=None, params=None):
    """Make API request with enhanced error handling for DataSectors API"""
    headers = {}
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=params, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=30)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=30)
        elif method == "PUT":
            response = requests.put(url, json=data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            st.error("⚠️ Session expired. Please login again.")
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.page = 'login'
            st.rerun()
        elif response.status_code == 429:
            st.error("⚠️ DataSectors API rate limit reached. Please try again in a moment.")
            return None
        elif response.status_code == 400:
            error_detail = "Bad request"
            try:
                error_detail = response.json().get('detail', 'Invalid request parameters')
            except:
                error_detail = response.text if response.text else "Bad request"
            st.error(f"❌ Invalid request: {error_detail}")
            return None
        elif response.status_code == 404:
            st.error("❌ Symbol not found in DataSectors API. Please try a different search term.")
            return None
        else:
            error_detail = "Unknown error"
            try:
                error_detail = response.json().get('detail', 'Unknown error')
            except:
                error_detail = response.text if response.text else "Unknown error"
            
            st.error(f"❌ API Error: {error_detail}")
            return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Server timeout. DataSectors API is slow to respond. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Cannot connect to backend server. Make sure it's running on http://localhost:2401")
        return None
    except requests.exceptions.JSONDecodeError:
        st.error("❌ Invalid response format from server. Please try again.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None

def make_request_with_retry(endpoint, retries=3):
    for attempt in range(retries):
        try:
            return make_request(endpoint)
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
                continue
            raise


def safe_rerun():
    """Attempt to rerun the Streamlit script with fallbacks for different Streamlit versions."""
    try:
        # Preferred API when available
        st.experimental_rerun()
        return
    except Exception:
        pass

    # Try raising the internal Rerun exception (different import paths across versions)
    RerunException = None
    try:
        from streamlit.runtime.scriptrunner import RerunException as RerunException
    except Exception:
        try:
            from streamlit.script_runner import RerunException as RerunException
        except Exception:
            RerunException = None

    if RerunException is not None:
        # Some Streamlit versions require a 'rerun_data' argument when instantiating
        try:
            rerun_data = {"_trigger": "safe_rerun"}
            raise RerunException(rerun_data)
        except TypeError:
            # Unexpected signature; try constructing without args then fallback
            try:
                raise RerunException()
            except TypeError:
                pass

    # Fallback: toggle a session_state flag and attempt to update query params to trigger a rerun
    st.session_state['_rerun_toggle'] = not st.session_state.get('_rerun_toggle', False)
    try:
        st.experimental_set_query_params(_rerun=int(st.session_state['_rerun_toggle']))
    except Exception:
        # Last resort: call st.rerun if available
        try:
            st.rerun()
        except Exception:
            # Give up silently — no reliable rerun mechanism available
            return

def create_candlestick_chart(data, indicators, show_ichimoku=False):
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Theme-aware colors
    theme = st.session_state.get('theme', 'dark')
    if theme == 'light':
        plot_bg = '#ffffff'
        paper_bg = '#f7f7f8'
        font_color = '#111111'
        grid_color = '#e5e7eb'
    else:
        plot_bg = '#1a1f2e'
        paper_bg = '#0f1318'
        font_color = '#ffffff'
        grid_color = '#2d3548'

    # Calculate indicator history from OHLC data
    # RSI calculation
    rsi_period = 14
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi_history = 100 - (100 / (1 + rs))
    
    # MACD calculation
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_history = ema12 - ema26
    macd_signal = macd_history.ewm(span=9, adjust=False).mean()
    macd_hist = macd_history - macd_signal
    
    # Stochastic calculation
    stoch_period = 14
    low_min = df['low'].rolling(window=stoch_period).min()
    high_max = df['high'].rolling(window=stoch_period).max()
    stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min))
    stoch_d = stoch_k.rolling(window=3).mean()
    
    # SMI calculation (Stochastic Momentum Index)
    smi_period = 14
    diff = df['close'] - ((df['high'] + df['low']) / 2)
    highest_high = df['high'].rolling(window=smi_period).max()
    lowest_low = df['low'].rolling(window=smi_period).min()
    smi_raw = diff / ((highest_high - lowest_low) / 2) * 100
    smi_history = smi_raw.ewm(span=3, adjust=False).mean()
    smi_ema = smi_history.ewm(span=3, adjust=False).mean()

    # Create subplots with adjusted heights for better visibility
    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.45, 0.12, 0.12, 0.12, 0.12, 0.12],
        subplot_titles=(
            '📈 Price Action & Indicators',
            '📊 Volume',
            '🎯 RSI (14)',
            '⚡ MACD',
            '🔄 Stochastic',
            '💫 SMI'
        )
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )

    # Chart Pattern Annotations
    patterns = indicators.get('patterns', [])
    if patterns:
        # Add pattern labels at the top of the chart
        for i, pattern in enumerate(patterns):
            pattern_name = pattern.get('pattern', '')
            signal = pattern.get('signal', '')
            strength = pattern.get('strength', '')
            
            # Color based on signal
            if signal == 'BULLISH':
                pattern_color = '#10b981'
                pattern_emoji = '📈'
            elif signal == 'BEARISH':
                pattern_color = '#ef4444'
                pattern_emoji = '📉'
            else:
                pattern_color = '#6b7280'
                pattern_emoji = '➡️'
            
            # Get recent high for annotation placement
            recent_high = df['high'].tail(20).max()
            
            # Add annotation at the top
            fig.add_annotation(
                x=df['timestamp'].iloc[-1],
                y=recent_high * (1.01 + i * 0.02),
                text=f"{pattern_emoji} {pattern_name}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=pattern_color,
                ax=40,
                ay=-20,
                bgcolor=f'rgba({int(pattern_color[1:3], 16)}, {int(pattern_color[3:5], 16)}, {int(pattern_color[5:7], 16)}, 0.8)',
                font=dict(color='white', size=10, family='Arial Black'),
                bordercolor=pattern_color,
                borderwidth=2,
                borderpad=4,
                row=1,
                col=1
            )

    # SMC Visualizations
    smc = indicators.get('smart_money_concepts', {})
    
    # Fair Value Gaps
    if smc and 'fair_value_gaps' in smc:
        fvg = smc.get('fair_value_gaps', {})
        bullish_fvgs = fvg.get('bullish', [])
        bearish_fvgs = fvg.get('bearish', [])
        
        # Bullish FVGs
        for i, level in enumerate(bullish_fvgs[-5:]):
            if isinstance(level, (int, float)):
                fig.add_hline(y=level, line=dict(color='rgba(76,175,80,0.3)', width=2, dash='dash'), 
                             annotation_text=f"FVG-B: {level:,.2f}", annotation_position='right', 
                             row=1, col=1, opacity=0.6)
        
        # Bearish FVGs
        for i, level in enumerate(bearish_fvgs[-5:]):
            if isinstance(level, (int, float)):
                fig.add_hline(y=level, line=dict(color='rgba(244,67,54,0.3)', width=2, dash='dash'), 
                             annotation_text=f"FVG-R: {level:,.2f}", annotation_position='right', 
                             row=1, col=1, opacity=0.6)
    
    # Order Blocks
    if smc and 'order_blocks' in smc:
        ob = smc.get('order_blocks', {})
        bullish_obs = ob.get('bullish', [])
        bearish_obs = ob.get('bearish', [])
        
        # Bullish Order Blocks
        for i, block in enumerate(bullish_obs[-3:]):
            if isinstance(block, (int, float)):
                fig.add_hline(y=block, line=dict(color='rgba(76,175,80,0.5)', width=2.5, dash='dot'), 
                             annotation_text=f"OB+: {block:,.2f}", annotation_position='left', 
                             row=1, col=1, opacity=0.7)
        
        # Bearish Order Blocks
        for i, block in enumerate(bearish_obs[-3:]):
            if isinstance(block, (int, float)):
                fig.add_hline(y=block, line=dict(color='rgba(244,67,54,0.5)', width=2.5, dash='dot'), 
                             annotation_text=f"OB-: {block:,.2f}", annotation_position='left', 
                             row=1, col=1, opacity=0.7)
    
    # Equal Highs/Lows
    if smc and 'equal_levels' in smc:
        equal = smc.get('equal_levels', {})
        equal_highs = equal.get('equal_highs', [])
        equal_lows = equal.get('equal_lows', [])
        
        # Equal Highs
        for level in equal_highs[-3:]:
            if isinstance(level, (int, float)):
                fig.add_hline(y=level, line=dict(color='rgba(255,152,0,0.5)', width=2, dash='dashdot'), 
                             annotation_text=f"EH: {level:,.2f}", annotation_position='right', 
                             row=1, col=1, opacity=0.6)
        
        # Equal Lows
        for level in equal_lows[-3:]:
            if isinstance(level, (int, float)):
                fig.add_hline(y=level, line=dict(color='rgba(156,39,176,0.5)', width=2, dash='dashdot'), 
                             annotation_text=f"EL: {level:,.2f}", annotation_position='right', 
                             row=1, col=1, opacity=0.6)

    # Bollinger Bands (if provided)
    bb = indicators.get('bollinger_bands', {})
    if bb and bb.get('middle') is not None:
        bb_middle = [bb['middle']] * len(df)
        bb_lower = [bb['lower']] * len(df)
        bb_upper = [bb['upper']] * len(df)

        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=bb_upper, name='BB Upper', line=dict(color='rgba(100,181,246,0.5)', width=1, dash='dot'), showlegend=False),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=bb_middle, name='BB Middle', line=dict(color='rgba(100,181,246,0.7)', width=1), showlegend=False),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=bb_lower, name='BB Lower', line=dict(color='rgba(100,181,246,0.5)', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(100,181,246,0.1)', showlegend=False),
            row=1, col=1
        )

    # Ichimoku Cloud (optional)
    if show_ichimoku and 'ichimoku' in indicators:
        ich = indicators['ichimoku']
        if ich.get('tenkan_sen') is not None:
            tenkan = [ich['tenkan_sen']] * len(df)
            kijun = [ich['kijun_sen']] * len(df)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=tenkan, name='Tenkan-sen', line=dict(color='#F44336', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=kijun, name='Kijun-sen', line=dict(color='#2196F3', width=1.5)), row=1, col=1)
            if ich.get('senkou_span_a') is not None:
                senkou_a = [ich['senkou_span_a']] * len(df)
                senkou_b = [ich['senkou_span_b']] * len(df)
                cloud_color = 'rgba(76,175,80,0.2)' if ich['senkou_span_a'] > ich['senkou_span_b'] else 'rgba(244,67,54,0.2)'
                fig.add_trace(go.Scatter(x=df['timestamp'], y=senkou_a, name='Senkou A', line=dict(color='rgba(76,175,80,0.5)', width=1), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['timestamp'], y=senkou_b, name='Senkou B', line=dict(color='rgba(244,67,54,0.5)', width=1), fill='tonexty', fillcolor=cloud_color, showlegend=False), row=1, col=1)

    # Support and Resistance
    if 'support_levels' in indicators and indicators['support_levels']:
        for i, support in enumerate(indicators['support_levels']):
            fig.add_hline(y=support, line=dict(color='#4CAF50', width=2, dash='dash'), annotation_text=f"S{i+1}: {support:,.2f}", annotation_position='right', row=1, col=1, opacity=0.7)

    if 'resistance_levels' in indicators and indicators['resistance_levels']:
        for i, resistance in enumerate(indicators['resistance_levels']):
            fig.add_hline(y=resistance, line=dict(color='#F44336', width=2, dash='dash'), annotation_text=f"R{i+1}: {resistance:,.2f}", annotation_position='right', row=1, col=1, opacity=0.7)

    # Volume
    colors_vol = ['#ef5350' if r['close'] < r['open'] else '#26a69a' for _, r in df.iterrows()]
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', marker_color=colors_vol, opacity=0.7, showlegend=True), row=2, col=1)
    if 'volume' in indicators and indicators['volume'].get('status'):
        vol_ma = df['volume'].rolling(20).mean()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=vol_ma, name='Vol MA (20)', line=dict(color='#FFC107', width=2), opacity=0.8, showlegend=True), row=2, col=1)

    # RSI - Actual historical values
    rsi_val = indicators.get('rsi')
    fig.add_trace(go.Scatter(x=df['timestamp'], y=rsi_history, name='RSI (14)', line=dict(color='#9C27B0', width=2.5), fill='tozeroy', fillcolor='rgba(156,39,176,0.15)', showlegend=True), row=3, col=1)
    fig.add_hline(y=70, line=dict(color='#F44336', width=1.5, dash='dash'), row=3, col=1, opacity=0.6)
    fig.add_hline(y=50, line=dict(color='#9E9E9E', width=1, dash='dot'), row=3, col=1, opacity=0.4)
    fig.add_hline(y=30, line=dict(color='#4CAF50', width=1.5, dash='dash'), row=3, col=1, opacity=0.6)
    if rsi_val is not None:
        fig.add_annotation(x=df['timestamp'].iloc[-1], y=rsi_val, text=f"{rsi_val:.1f}", showarrow=True, xanchor='left', bgcolor='rgba(156,39,176,0.9)', font=dict(color='white', size=11, family='Arial Black'), row=3, col=1, arrowsize=1, arrowwidth=2, arrowcolor='#9C27B0')

    # MACD - Actual historical values
    macd_val = indicators.get('macd', {}).get('macd')
    signal_val = indicators.get('macd', {}).get('signal')
    hist_val = indicators.get('macd', {}).get('histogram')
    
    hist_colors = ['#26a69a' if h > 0 else '#ef5350' for h in macd_hist.fillna(0)]
    
    fig.add_trace(go.Bar(x=df['timestamp'], y=macd_hist, name='MACD Hist', marker_color=hist_colors, opacity=0.4, showlegend=True), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=macd_history, name='MACD', line=dict(color='#2196F3', width=2.5), showlegend=True), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=macd_signal, name='Signal (9)', line=dict(color='#FF9800', width=2.5, dash='dot'), showlegend=True), row=4, col=1)
    fig.add_hline(y=0, line=dict(color='#9E9E9E', width=1, dash='dot'), row=4, col=1, opacity=0.5)
    
    if macd_val is not None:
        fig.add_annotation(x=df['timestamp'].iloc[-1], y=macd_val, text=f"{macd_val:.4f}", showarrow=True, xanchor='left', bgcolor='rgba(33,150,243,0.8)', font=dict(color='white', size=9), row=4, col=1)
    if signal_val is not None:
        fig.add_annotation(x=df['timestamp'].iloc[-2], y=signal_val, text=f"{signal_val:.4f}", showarrow=True, xanchor='left', bgcolor='rgba(255,152,0,0.8)', font=dict(color='white', size=9), row=4, col=1)

    # Stochastic - Actual historical values
    stoch_k_val = indicators.get('stochastic', {}).get('k')
    stoch_d_val = indicators.get('stochastic', {}).get('d')
    
    fig.add_trace(go.Scatter(x=df['timestamp'], y=stoch_k, name='%K', line=dict(color='#00BCD4', width=2.5), showlegend=True), row=5, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=stoch_d, name='%D (3)', line=dict(color='#FF5722', width=2.5, dash='dot'), showlegend=True), row=5, col=1)
    fig.add_hline(y=80, line=dict(color='#F44336', width=1.5, dash='dash'), row=5, col=1, opacity=0.6)
    fig.add_hline(y=50, line=dict(color='#9E9E9E', width=1, dash='dot'), row=5, col=1, opacity=0.4)
    fig.add_hline(y=20, line=dict(color='#4CAF50', width=1.5, dash='dash'), row=5, col=1, opacity=0.6)
    fig.add_hrect(y0=80, y1=100, fillcolor='rgba(244,67,54,0.05)', line_width=0, row=5, col=1)
    fig.add_hrect(y0=0, y1=20, fillcolor='rgba(76,175,80,0.05)', line_width=0, row=5, col=1)
    
    if stoch_k_val is not None:
        fig.add_annotation(x=df['timestamp'].iloc[-1], y=stoch_k_val, text=f"K:{stoch_k_val:.0f}", showarrow=True, xanchor='left', bgcolor='rgba(0,188,212,0.8)', font=dict(color='white', size=9), row=5, col=1)
    if stoch_d_val is not None:
        fig.add_annotation(x=df['timestamp'].iloc[-2], y=stoch_d_val, text=f"D:{stoch_d_val:.0f}", showarrow=True, xanchor='left', bgcolor='rgba(255,87,34,0.8)', font=dict(color='white', size=9), row=5, col=1)

    # SMI - Actual historical values
    smi_val = indicators.get('smi', {}).get('smi')
    smi_ema_val = indicators.get('smi', {}).get('smi_ema')
    
    smi_color = '#26a69a' if smi_val and smi_val > 0 else '#ef5350'
    fig.add_trace(go.Scatter(x=df['timestamp'], y=smi_history, name='SMI', line=dict(color=smi_color, width=2.5), fill='tozeroy', fillcolor=f'rgba({38 if smi_val and smi_val > 0 else 239}, {166 if smi_val and smi_val > 0 else 83}, {154 if smi_val and smi_val > 0 else 80}, 0.15)', showlegend=True), row=6, col=1)
    
    if smi_ema is not None:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=smi_ema, name='SMI EMA (3)', line=dict(color='#FF9800', width=2, dash='dot'), showlegend=True), row=6, col=1)
    
    fig.add_hline(y=40, line=dict(color='#F44336', width=1.5, dash='dash'), row=6, col=1, opacity=0.6)
    fig.add_hline(y=0, line=dict(color='#9E9E9E', width=1, dash='solid'), row=6, col=1, opacity=0.5)
    fig.add_hline(y=-40, line=dict(color='#4CAF50', width=1.5, dash='dash'), row=6, col=1, opacity=0.6)
    fig.add_hrect(y0=40, y1=120, fillcolor='rgba(244,67,54,0.05)', line_width=0, row=6, col=1)
    fig.add_hrect(y0=-120, y1=-40, fillcolor='rgba(76,175,80,0.05)', line_width=0, row=6, col=1)
    
    if smi_val is not None:
        fig.add_annotation(x=df['timestamp'].iloc[-1], y=smi_val, text=f"{smi_val:.1f}", showarrow=True, xanchor='left', bgcolor='rgba(156,39,176,0.8)', font=dict(color='white', size=10, family='Arial Black'), row=6, col=1)

    # Layout and axes - TradingView style
    fig.update_layout(
        height=1400, 
        showlegend=True, 
        legend=dict(
            orientation='h', 
            yanchor='top', 
            y=0.98, 
            xanchor='left', 
            x=0.01,
            font=dict(size=10),
            bgcolor='rgba(0,0,0,0.3)',
            bordercolor='rgba(100,100,100,0.5)',
            borderwidth=1
        ), 
        xaxis_rangeslider_visible=False, 
        plot_bgcolor=plot_bg, 
        paper_bgcolor=paper_bg, 
        font=dict(color=font_color, size=11, family='Arial'),
        hovermode='x unified', 
        margin=dict(l=70, r=100, t=100, b=80)
    )
    fig.update_xaxes(gridcolor=grid_color, showgrid=True, zeroline=False, gridwidth=0.5)
    fig.update_yaxes(gridcolor=grid_color, showgrid=True, zeroline=False, gridwidth=0.5)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=5, col=1)
    fig.update_yaxes(range=[-120, 120], row=6, col=1)
    fig.update_yaxes(title_text='Price (USD)', row=1, col=1)
    fig.update_yaxes(title_text='Volume', row=2, col=1)
    fig.update_yaxes(title_text='RSI', row=3, col=1)
    fig.update_yaxes(title_text='MACD', row=4, col=1)
    fig.update_yaxes(title_text='Stoch', row=5, col=1)
    fig.update_yaxes(title_text='SMI', row=6, col=1)
    fig.update_xaxes(title_text='Time', row=6, col=1)

    return fig

def generate_pdf_report(scan_data):
    """Generate PDF report from scan data"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    
    # Title
    pdf.cell(0, 10, f"Crypto Scanner Report - {scan_data['ticker']}", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)
    
    # Signal
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Signal: {scan_data['signal']}", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Confidence: {scan_data['confidence']}%", ln=True)
    pdf.cell(0, 10, f"Price: ${scan_data['price']}", ln=True)
    pdf.cell(0, 10, f"Timeframe: {scan_data['timeframe']}", ln=True)
    pdf.ln(5)
    
    # Analysis summary
    if 'analysis' in scan_data and scan_data['analysis']:
        analysis = scan_data['analysis']
        
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Technical Analysis", ln=True)
        pdf.set_font("Arial", "", 10)
        
        if 'trend' in analysis:
            pdf.cell(0, 8, f"Trend: {analysis['trend']}", ln=True)
        
        if 'rsi' in analysis:
            pdf.cell(0, 8, f"RSI: {analysis['rsi']:.2f}", ln=True)
        
        if 'moving_averages' in analysis:
            pdf.ln(5)
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 8, "Moving Averages:", ln=True)
            pdf.set_font("Arial", "", 10)
            for ma, value in analysis['moving_averages'].items():
                if value:
                    pdf.cell(0, 7, f"  {ma}: ${value:.2f}", ln=True)
    
    # Save to bytes
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return pdf_bytes

def export_to_excel(history_data):
    """Export history to Excel"""
    df = pd.DataFrame(history_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Scan History', index=False)
    return output.getvalue()

# Authentication Pages
def login_page():
    st.title("🔐 Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                result = make_request("/auth/login", "POST", {
                    "username": username,
                    "password": password
                })
                
                if result:
                    st.session_state.token = result['access_token']
                    st.session_state.username = result['username']
                    st.session_state.page = 'dashboard'
                    st.rerun()
        
        if st.button("Don't have an account? Register", use_container_width=True):
            st.session_state.page = 'register'
            st.rerun()

def register_page():
    st.title("📝 Register")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("register_form"):
            username = st.text_input("Username", placeholder="Choose a username")
            email = st.text_input("Email", placeholder="Enter your email")
            password = st.text_input("Password", type="password", placeholder="Choose a password")
            password_confirm = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            submit = st.form_submit_button("Register", use_container_width=True)
            
            if submit:
                if password != password_confirm:
                    st.error("Passwords don't match!")
                else:
                    result = make_request("/auth/register", "POST", {
                        "username": username,
                        "email": email,
                        "password": password
                    })
                    
                    if result:
                        st.success("Registration successful! Please login.")
                        time.sleep(1)
                        st.session_state.page = 'login'
                        st.rerun()
        
        if st.button("Already have an account? Login", use_container_width=True):
            st.session_state.page = 'login'
            st.rerun()

# Main Pages
def dashboard_page():
    st.title("📊 Dashboard")
    
    stats = make_request("/dashboard/stats")
    
    if stats:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Scans", stats['total_scans'])
        
        with col2:
            st.metric("Multi-TF Scans", stats.get('total_multi_timeframe_scans', 0))
        
        signal_dist = stats.get('signal_distribution', {})
        with col3:
            st.metric("Long Signals", signal_dist.get('LONG', 0))
        
        with col4:
            st.metric("Short Signals", signal_dist.get('SHORT', 0))
        
        with col5:
            st.metric("Neutral", signal_dist.get('NEUTRAL', 0))
        
        st.markdown("---")
        
        st.subheader("📈 Recent Scans")
        if stats['recent_scans']:
            for scan in stats['recent_scans']:
                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                
                with col1:
                    st.write(f"**{scan['ticker']}**")
                
                with col2:
                    signal_class = f"signal-{scan['signal'].lower()}"
                    st.markdown(f'<div class="{signal_class}">{scan["signal"]}</div>', unsafe_allow_html=True)
                
                with col3:
                    st.write(f"Confidence: {scan['confidence']:.1f}%")
                
                with col4:
                    st.write(scan['date'])
                
                st.markdown("---")
        else:
            st.info("No scans yet. Go to Scan page to start analyzing!")

def scan_page():
    st.title("🔍 Crypto Scanner")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Settings")
        
        # Search box for symbols
        search_term = st.text_input("Search Ticker", placeholder="e.g., BTC, ETH")
        
        @st.cache_data(ttl=1800, show_spinner="Fetching symbols from DataSectors...")
        def get_cached_symbols(search_term):
            """Cache symbol search results for 30 minutes from DataSectors API"""
            return make_request(f"/symbols?search={search_term}&limit=500" if search_term else "/symbols?limit=500")
        
        symbols_data = get_cached_symbols(search_term)
        symbols = [s['symbol'] for s in symbols_data['symbols']] if symbols_data else []
        
        if symbols_data and 'source' in symbols_data:
            total = symbols_data.get('total', 0)
            source = symbols_data.get('source', 'DataSectors')
            st.caption(f"📊 {total} tickers | 🔌 {source}")
        else:
            if symbols_data is None:
                st.info("💡 Try searching for a cryptocurrency (e.g., Bitcoin, Ethereum) powered by DataSectors", icon="ℹ️")
        
        selected_symbol = st.selectbox(
            "Select Ticker",
            options=symbols,
            index=0 if symbols else None
        )
        
        timeframe = st.selectbox(
            "Timeframe",
            options=["5m", "15m", "30m", "1h", "4h", "1d", "1w"],
            format_func=lambda x: {
                "5m": "5 Minutes",
                "15m": "15 Minutes",
                "30m": "30 Minutes",
                "1h": "1 Hour",
                "4h": "4 Hours",
                "1d": "1 Day",
                "1w": "1 Week"
            }[x]
        )
        
        show_ichimoku = st.checkbox("Show Ichimoku Cloud", value=False)
        
        # Candle range selection
        st.subheader("Chart Settings")
        candle_range = st.select_slider(
            "Select number of candles to display",
            options=[50, 100, 200, 500],
            value=100,
            help="Choose how many candles to display on the chart"
        )
        
        scan_button = st.button("🚀 Scan & Analyze", use_container_width=True, type="primary")
        
        if scan_button and selected_symbol:
            with st.spinner("Analyzing..."):
                result = make_request("/scan", "POST", {
                    "ticker": selected_symbol,
                    "timeframe": timeframe,
                    "candle_range": candle_range  # Send selected candle range
                })
                
                if result:
                    st.session_state.scan_result = result
                    st.session_state.scanned_symbol = selected_symbol
                    st.session_state.scanned_timeframe = timeframe
    
    with col2:
        if 'scan_result' in st.session_state:
            result = st.session_state.scan_result
            
            # Display signal with detailed reasoning
            display_signal_reasoning(result)
            
            st.markdown("---")
            
            # Additional metrics
            col_price, col_trend, col_candles = st.columns(3)
            with col_price:
                st.metric("Current Price", f"${result.get('current_price', 0):,.2f}")
            with col_trend:
                st.metric("Trend", result.get('trend', 'N/A'))
            with col_candles:
                total_candles = result.get('total_candles', 'N/A')
                st.metric("Candles Analyzed", total_candles)
                if total_candles == 1000:
                    st.caption("✅ Max historical data")
            
            st.markdown("---")
            
            # Export buttons
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                if st.button("📄 Export to PDF", use_container_width=True):
                    export_data = {
                        'ticker': st.session_state.scanned_symbol,
                        'timeframe': st.session_state.scanned_timeframe,
                        'signal': result['signal'],
                        'confidence': result['confidence'],
                        'price': result['current_price'],
                        'analysis': result
                    }
                    pdf_bytes = generate_pdf_report(export_data)
                    st.download_button(
                        "⬇️ Download PDF",
                        pdf_bytes,
                        file_name=f"scan_{st.session_state.scanned_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
            
            # Chart with expand modal
            st.subheader(f"📈 {st.session_state.scanned_symbol} Chart")
            with st.spinner("📊 Fetching chart data..."):
                chart = create_candlestick_chart(result['chart_data'], result, show_ichimoku)
                display_chart_with_controls(chart, result, show_ichimoku)
            
            # Compact Indicators Display
            st.markdown("---")
            st.subheader("📊 Key Indicators Snapshot")
            create_compact_indicator_display(result)
            
            # Technical Indicators in tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Core Indicators", "🎯 Levels", "☁️ Advanced", "📈 Oscillators", "💰 Smart Money"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Moving Averages**")
                    mas = result['moving_averages']
                    current_price = result['current_price']
                    for ma, value in mas.items():
                        if value is not None:
                            trend_class = "indicator-bullish" if current_price > value else "indicator-bearish"
                            st.markdown(f'<span class="{trend_class}">{ma}: ${value:,.2f}</span>', unsafe_allow_html=True)
                    
                    st.write("**Bollinger Bands**")
                    bb = result['bollinger_bands']
                    if bb['upper'] is not None:
                        st.write(f"Upper: ${bb['upper']:,.2f}")
                        st.write(f"Middle: ${bb['middle']:,.2f}")
                        st.write(f"Lower: ${bb['lower']:,.2f}")
                        
                        # BB Position
                        bb_position = (current_price - bb['lower']) / (bb['upper'] - bb['lower']) * 100
                        st.write(f"BB Position: {bb_position:.1f}%")
                    else:
                        st.info("BB data not available")
                
                with col2:
                    st.write("**MACD**")
                    macd = result['macd']
                    if macd['histogram'] is not None:
                        macd_trend = "🟢 Bullish" if macd['histogram'] > 0 else "🔴 Bearish"
                        st.write(f"Status: {macd_trend}")
                        st.write(f"MACD: {macd['macd']:.4f}")
                        st.write(f"Signal: {macd['signal']:.4f}")
                        st.write(f"Histogram: {macd['histogram']:.4f}")
                    else:
                        st.info("MACD data not available")
                    
                    st.write("**Volume Analysis**")
                    vol = result['volume']
                    st.write(f"Status: {vol['status']}")
                    st.write(f"Strength: {vol['strength']}")
                    if vol['obv'] is not None:
                        st.write(f"OBV: {vol['obv']:,.0f}")
                    else:
                        st.write("OBV: N/A")
            
            with tab2:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Support Levels**")
                    for i, support in enumerate(result['support_levels'], 1):
                        distance = ((current_price - support) / support) * 100
                        st.write(f"S{i}: ${support:,.2f} ({distance:+.2f}%)")
                
                with col2:
                    st.write("**Resistance Levels**")
                    for i, resistance in enumerate(result['resistance_levels'], 1):
                        distance = ((resistance - current_price) / current_price) * 100
                        st.write(f"R{i}: ${resistance:,.2f} ({distance:+.2f}%)")
                
                with col3:
                    st.write("**Fibonacci Levels**")
                    fib = result['fibonacci']
                    for level, price in list(fib.items())[:5]:
                        st.write(f"{level.replace('level_', '')}: ${price:,.2f}")
            
            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Ichimoku Cloud**")
                    ich = result['ichimoku']
                    if ich['tenkan_sen'] is not None:
                        st.write(f"Tenkan-sen: ${ich['tenkan_sen']:,.2f}")
                        st.write(f"Kijun-sen: ${ich['kijun_sen']:,.2f}")
                        if ich['senkou_span_a'] is not None:
                            st.write(f"Senkou A: ${ich['senkou_span_a']:,.2f}")
                            st.write(f"Senkou B: ${ich['senkou_span_b']:,.2f}")
                            
                            # Cloud color
                            if ich['senkou_span_a'] > ich['senkou_span_b']:
                                st.write("☁️ Bullish Cloud (Green)")
                            else:
                                st.write("☁️ Bearish Cloud (Red)")
                        else:
                            st.info("Cloud data not yet available")
                    else:
                        st.info("Insufficient data for Ichimoku")
                    
                    st.write("**ADX - Trend Strength**")
                    adx = result['adx']
                    if adx['adx'] is not None:
                        adx_val = adx['adx']
                        if adx_val > 25:
                            strength = "Strong Trend"
                        elif adx_val > 20:
                            strength = "Moderate Trend"
                        else:
                            strength = "Weak/No Trend"
                        st.write(f"ADX: {adx_val:.2f} ({strength})")
                        st.write(f"+DI: {adx['plus_di']:.2f}")
                        st.write(f"-DI: {adx['minus_di']:.2f}")
                    else:
                        st.info("ADX data not available")
                
                with col2:
                    st.write("**ATR - Volatility**")
                    atr = result['atr']
                    if atr is not None and atr > 0:
                        atr_percent = (atr / current_price) * 100
                        st.write(f"ATR: ${atr:.2f}")
                        st.write(f"ATR %: {atr_percent:.2f}%")
                        
                        if atr_percent > 3:
                            st.write("⚡ High Volatility")
                        elif atr_percent > 1.5:
                            st.write("📊 Normal Volatility")
                        else:
                            st.write("😴 Low Volatility")
                    else:
                        st.info("ATR data not available")
            
            with tab4:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**RSI - Relative Strength**")
                    rsi = result.get('rsi')
                    if rsi is not None:
                        if rsi > 70:
                            rsi_status = "🔴 Overbought"
                        elif rsi < 30:
                            rsi_status = "🟢 Oversold"
                        else:
                            rsi_status = "⚪ Neutral"
                        st.write(f"RSI: {rsi:.2f}")
                        st.write(f"Status: {rsi_status}")
                    else:
                        st.info("RSI data not available")
                    
                    st.write("**SMI - Stochastic Momentum**")
                    smi_data = result.get('smi', {})
                    smi_val = smi_data.get('smi')
                    smi_ema = smi_data.get('smi_ema')
                    
                    if smi_val is not None:
                        if smi_val > 40:
                            smi_status = "🔴 Overbought"
                        elif smi_val < -40:
                            smi_status = "🟢 Oversold"
                        else:
                            smi_status = "⚪ Neutral"
                        
                        st.write(f"SMI: {smi_val:.2f}")
                        if smi_ema is not None:
                            st.write(f"SMI EMA: {smi_ema:.2f}")
                        st.write(f"Status: {smi_status}")
                        
                        # Signal interpretation
                        if smi_val is not None and smi_ema is not None:
                            if smi_val > smi_ema:
                                st.write("📈 Bullish crossover")
                            elif smi_val < smi_ema:
                                st.write("📉 Bearish crossover")
                    else:
                        st.info("SMI data not available")
                
                with col2:
                    st.write("**Stochastic Oscillator**")
                    stoch = result.get('stochastic', {})
                    stoch_k = stoch.get('k')
                    stoch_d = stoch.get('d')
                    
                    if stoch_k is not None:
                        st.write(f"%K: {stoch_k:.2f}")
                        st.write(f"%D: {stoch_d:.2f}")
                        
                        if stoch_k > 80:
                            st.write("Status: 🔴 Overbought")
                        elif stoch_k < 20:
                            st.write("Status: 🟢 Oversold")
                        else:
                            st.write("Status: ⚪ Neutral")
                    else:
                        st.info("Stochastic data not available")
            
            with tab5:
                if 'smart_money_concepts' in result:
                    smc = result['smart_money_concepts']
                    
                    # SMC Overview
                    col_smc1, col_smc2, col_smc3 = st.columns(3)
                    
                    with col_smc1:
                        smc_bias = smc.get('smc_bias', 'NEUTRAL')
                        bias_emoji = "🟢" if smc_bias == "BULLISH" else "🔴" if smc_bias == "BEARISH" else "⚪"
                        st.markdown(f'<div style="text-align: center; padding: 10px; background: rgba(100,100,100,0.1); border-radius: 5px;"><h4>{bias_emoji} {smc_bias}</h4><small>SMC Bias</small></div>', unsafe_allow_html=True)
                    
                    with col_smc2:
                        smc_strength = smc.get('smc_signal_strength', 50)
                        st.metric("SMC Strength", f"{smc_strength}%")
                    
                    with col_smc3:
                        smc_count = smc.get('smc_signal_count', 0)
                        st.metric("Signal Count", smc_count)
                    
                    st.markdown("---")
                    
                    # Swing Structure
                    with st.expander("📊 Swing Structure", expanded=False):
                        col_swing1, col_swing2 = st.columns(2)
                        
                        with col_swing1:
                            st.write("**Swing Highs**")
                            swing_highs = smc.get('swing_structure', {}).get('highs', [])
                            if swing_highs:
                                for h in swing_highs[-5:]:
                                    st.write(f"Bar {h['index']}: ${h['price']:,.2f}")
                            else:
                                st.write("No swing highs detected")
                        
                    with col_swing2:
                        st.write("**Swing Lows**")
                        swing_lows = smc.get('swing_structure', {}).get('lows', [])
                        if swing_lows:
                            for l in swing_lows[-5:]:
                                st.write(f"Bar {l['index']}: ${l['price']:,.2f}")
                        else:
                            st.write("No swing lows detected")
                    
                    # Break of Structure & Change of Character
                    with st.expander("🔪 Break of Structure & Change of Character (BOS/CHoCH)", expanded=False):
                        col_bos1, col_bos2 = st.columns(2)
                        
                        bos = smc.get('break_of_structure', {})
                        
                        with col_bos1:
                            st.write("**Bullish BOS**")
                            bullish_bos = bos.get('bullish_bos', [])
                            if bullish_bos:
                                st.write(f"🟢 {len(bullish_bos)} detected")
                            else:
                                st.write("🔵 None detected")
                            
                            st.write("**Bullish CHoCH**")
                            bullish_choch = bos.get('bullish_choch', [])
                            if bullish_choch:
                                st.write(f"🟢 {len(bullish_choch)} detected")
                            else:
                                st.write("🔵 None detected")
                        
                        with col_bos2:
                            st.write("**Bearish BOS**")
                            bearish_bos = bos.get('bearish_bos', [])
                            if bearish_bos:
                                st.write(f"🔴 {len(bearish_bos)} detected")
                            else:
                                st.write("🔵 None detected")
                            
                            st.write("**Bearish CHoCH**")
                            bearish_choch = bos.get('bearish_choch', [])
                            if bearish_choch:
                                st.write(f"🔴 {len(bearish_choch)} detected")
                            else:
                                st.write("🔵 None detected")
                    
                    # Order Blocks
                    with st.expander("📦 Order Blocks", expanded=False):
                        col_ob1, col_ob2 = st.columns(2)
                        
                        ob = smc.get('order_blocks', {})
                        
                        with col_ob1:
                            st.write("**Bullish Order Blocks**")
                            bullish_ob = ob.get('bullish', [])
                            if bullish_ob:
                                for block in bullish_ob[-3:]:
                                    high = block.get('high', 'N/A')
                                    low = block.get('low', 'N/A')
                                    if high != 'N/A' and low != 'N/A':
                                        st.write(f"🟢 ${high:,.2f} - ${low:,.2f}")
                                    else:
                                        st.write(f"🟢 Order Block Found")
                            else:
                                st.write("🔵 No bullish order blocks")
                        
                        with col_ob2:
                            st.write("**Bearish Order Blocks**")
                            bearish_ob = ob.get('bearish', [])
                            if bearish_ob:
                                for block in bearish_ob[-3:]:
                                    high = block.get('high', 'N/A')
                                    low = block.get('low', 'N/A')
                                    if high != 'N/A' and low != 'N/A':
                                        st.write(f"🔴 ${high:,.2f} - ${low:,.2f}")
                                    else:
                                        st.write(f"🔴 Order Block Found")
                            else:
                                st.write("🔵 No bearish order blocks")
                    
                    # Fair Value Gaps
                    with st.expander("💨 Fair Value Gaps (FVG)", expanded=False):
                        col_fvg1, col_fvg2 = st.columns(2)
                        
                        fvg = smc.get('fair_value_gaps', {})
                        
                        with col_fvg1:
                            st.write("**Bullish FVG**")
                            bullish_fvg = fvg.get('bullish', [])
                            if bullish_fvg:
                                for gap in bullish_fvg[-3:]:
                                    if isinstance(gap, dict):
                                        level = gap.get('level', gap.get('bottom', 'N/A'))
                                        st.write(f"🟢 ${level:,.2f}" if level != 'N/A' else f"🟢 FVG Detected")
                                    else:
                                        st.write(f"🟢 Level: ${gap:,.2f}")
                            else:
                                st.write("🔵 No bullish FVGs")
                        
                        with col_fvg2:
                            st.write("**Bearish FVG**")
                            bearish_fvg = fvg.get('bearish', [])
                            if bearish_fvg:
                                for gap in bearish_fvg[-3:]:
                                    if isinstance(gap, dict):
                                        level = gap.get('level', gap.get('bottom', 'N/A'))
                                        st.write(f"🔴 ${level:,.2f}" if level != 'N/A' else f"🔴 FVG Detected")
                                    else:
                                        st.write(f"🔴 Level: ${gap:,.2f}")
                            else:
                                st.write("🔵 No bearish FVGs")
                    
                    # Equal Highs/Lows
                    with st.expander("📏 Equal Highs & Lows", expanded=False):
                        col_eq1, col_eq2 = st.columns(2)
                        
                        equal = smc.get('equal_levels', {})
                        
                        with col_eq1:
                            st.write("**Equal Highs**")
                            equal_highs = equal.get('equal_highs', [])
                            if equal_highs:
                                for h in equal_highs:
                                    if isinstance(h, dict):
                                        st.write(f"📌 ${h.get('level', 'N/A'):,.2f}")
                                    elif isinstance(h, (int, float)):
                                        st.write(f"📌 ${h:,.2f}")
                                    else:
                                        st.write(f"📌 {h}")
                            else:
                                st.write("🔵 No equal highs detected")
                        
                        with col_eq2:
                            st.write("**Equal Lows**")
                            equal_lows = equal.get('equal_lows', [])
                            if equal_lows:
                                for l in equal_lows:
                                    if isinstance(l, dict):
                                        st.write(f"📌 ${l.get('level', 'N/A'):,.2f}")
                                    elif isinstance(l, (int, float)):
                                        st.write(f"📌 ${l:,.2f}")
                                    else:
                                        st.write(f"📌 {l}")
                            else:
                                st.write("🔵 No equal lows detected")
                    
                    # Premium & Discount Zones
                    with st.expander("🎯 Premium & Discount Zones", expanded=False):
                        pdz = smc.get('premium_discount_zones', {})
                        
                        if pdz:
                            st.write("**Premium Zones** (Price needs to pull back)")
                            premium = pdz.get('premium_zones', [])
                            if premium:
                                for zone in premium:
                                    if isinstance(zone, dict):
                                        st.write(f"🔴 ${zone.get('high', 'N/A')} - ${zone.get('low', 'N/A')}")
                                    else:
                                        st.write(f"🔴 {zone}")
                            else:
                                st.write("No premium zones")
                            
                            st.write("**Discount Zones** (Price needs to push up)")
                            discount = pdz.get('discount_zones', [])
                            if discount:
                                for zone in discount:
                                    if isinstance(zone, dict):
                                        st.write(f"🟢 ${zone.get('high', 'N/A')} - ${zone.get('low', 'N/A')}")
                                    else:
                                        st.write(f"🟢 {zone}")
                            else:
                                st.write("No discount zones")
                        else:
                            st.write("No premium/discount zones detected")
                    
                    # SMC Guide
                    with st.expander("📚 SMC Guide", expanded=False):
                        st.write("""
                        **Swing Structure**: Identifies market structure through swing highs and lows
                        
                        **Break of Structure (BOS)**: Price breaks below recent swing low (bearish) or above swing high (bullish)
                        
                        **Change of Character (CHoCH)**: Sequence of lower lows & highs changes to higher lows & highs (or vice versa)
                        
                        **Order Blocks**: Significant candles where institutional orders may be resting
                        
                        **Fair Value Gaps (FVG)**: Imbalances between candles that price often returns to fill
                        
                        **Equal Highs/Lows**: Price levels that match exactly, showing market awareness of key prices
                        
                        **Premium Zones**: Areas where price is overextended and likely to pull back
                        
                        **Discount Zones**: Areas where price is underextended and likely to push higher
                        """)
                else:
                    st.info("SMC data not available for this ticker")
            
            # Pattern Recognition Section
            if 'patterns' in result and result['patterns']:
                st.markdown("---")
                st.subheader("🔍 Chart Patterns Detected")
                
                for pattern in result['patterns']:
                    col_p1, col_p2, col_p3 = st.columns([2, 1, 1])
                    
                    with col_p1:
                        st.write(f"**{pattern['pattern']}**")
                    
                    with col_p2:
                        signal_color = "🟢" if pattern['signal'] == "BULLISH" else "🔴" if pattern['signal'] == "BEARISH" else "⚪"
                        st.write(f"{signal_color} {pattern['signal']}")
                    
                    with col_p3:
                        strength_emoji = "⭐⭐⭐" if pattern['strength'] == "HIGH" else "⭐⭐"
                        st.write(f"{strength_emoji} {pattern['strength']}")
                    
                    st.markdown("---")
                
                # Pattern explanation
                with st.expander("📚 Pattern Guide"):
                    st.write("""
                    **Head and Shoulders**: Strong reversal pattern indicating trend change
                    **Inverse H&S**: Bullish reversal after downtrend
                    **Ascending Triangle**: Bullish continuation, breakout expected upward
                    **Descending Triangle**: Bearish continuation, breakout expected downward
                    **Symmetrical Triangle**: Neutral, can break either direction
                    **Double Top/Bottom**: Reversal patterns at support/resistance
                    **Rising Wedge**: Bearish reversal despite upward movement
                    **Falling Wedge**: Bullish reversal despite downward movement
                    """)
        else:
            st.info("👈 Select a ticker and click 'Scan & Analyze' to start")

def watchlist_page():
    st.title("👁️ Watchlist & Auto-Scan")
    
    tab1, tab2 = st.tabs(["📋 My Watchlist", "➕ Add New"])
    
    with tab1:
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔄 Scan All", use_container_width=True, type="primary"):
                with st.spinner("Scanning watchlist..."):
                    result = make_request("/watchlist/scan-all", "POST")
                    
                    if result:
                        st.success(f"✅ Scanned {result['total_scanned']} tickers")
                        st.session_state.watchlist_results = result['results']
                        time.sleep(1)
                        st.rerun()
        
        # watchlist_data = make_request("/watchlist") # ❌ Ketika di watchlist_page, jika ada scan baru, data tidak update otomatis
        
        # Polling every 10 seconds (adjust as needed)
        if 'last_watchlist_update' not in st.session_state or (time.time() - st.session_state.last_watchlist_update) > 10:
            watchlist_data = make_request("/watchlist")
            if watchlist_data:
                st.session_state.watchlist_data = watchlist_data
                st.session_state.last_watchlist_update = time.time()
        else:
            watchlist_data = st.session_state.watchlist_data
        
        if watchlist_data and watchlist_data['watchlist']:
            st.subheader("📊 Watchlist Tickers")
            
            for item in watchlist_data['watchlist']:
                with st.expander(f"{item['ticker']} - {item['timeframe']}", expanded=False):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.write(f"**Added:** {item['added_at']}")
                        if item['alert_price']:
                            st.write(f"**Alert Price:** ${item['alert_price']:,.2f}")
                    
                    with col2:
                        if item['last_scan']:
                            scan = item['last_scan']
                            signal_class = f"signal-{scan['signal'].lower()}"
                            st.markdown(f'<div class="{signal_class}">{scan["signal"]}</div>', 
                                      unsafe_allow_html=True)
                            st.write(f"Price: ${scan['price']:,.2f}")
                            st.write(f"Confidence: {scan['confidence']}%")
                            st.write(f"RSI: {scan['rsi']:.1f}")
                            
                            if 'patterns' in scan and scan['patterns']:
                                st.write("**Patterns:**")
                                for p in scan['patterns']:
                                    st.write(f"- {p['pattern']} ({p['signal']})")
                        else:
                            st.info("Not scanned yet")
                    
                    with col3:
                        if st.button("🗑️ Remove", key=f"remove_{item['ticker']}"):
                            result = make_request(f"/watchlist/{item['ticker']}", "DELETE")
                            if result:
                                st.success("Removed!")
                                time.sleep(0.5)
                                st.rerun()
            
            # Display scan results if available
            if 'watchlist_results' in st.session_state:
                st.markdown("---")
                st.subheader("🎯 Latest Scan Results")
                
                results = st.session_state.watchlist_results
                
                # Filter by signal
                signal_filter = st.selectbox(
                    "Filter by Signal",
                    options=["ALL", "LONG", "SHORT", "NEUTRAL"]
                )
                
                filtered_results = results if signal_filter == "ALL" else [
                    r for r in results if r['signal'] == signal_filter
                ]
                
                for scan in filtered_results:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Ticker", scan['ticker'])
                    
                    with col2:
                        signal_class = f"signal-{scan['signal'].lower()}"
                        st.markdown(f'<div class="{signal_class}">{scan["signal"]}</div>', 
                                  unsafe_allow_html=True)
                    
                    with col3:
                        st.metric("Confidence", f"{scan['confidence']}%")
                    
                    with col4:
                        st.metric("Price", f"${scan['price']:,.2f}")
                    
                    if scan['patterns']:
                        st.write("**Patterns:**", ", ".join([p['pattern'] for p in scan['patterns']]))
                    
                    st.markdown("---")
        else:
            st.info("Your watchlist is empty. Add tickers in the 'Add New' tab.")
    
    with tab2:
        st.subheader("➕ Add to Watchlist")
        
        col1, col2 = st.columns(2)
        
        with col1:
            search_term = st.text_input("Search Ticker", placeholder="e.g., BTC, ETH", key="wl_search")
            symbols_data = make_request(f"/symbols?search={search_term}&limit=500" if search_term else "/symbols?limit=500")
            symbols = [s['symbol'] for s in symbols_data['symbols']] if symbols_data else []
            
            # Show data source
            if symbols_data and symbols:
                st.caption(f"🔌 Powered by {symbols_data.get('source', 'DataSectors')}")
            
            selected_ticker = st.selectbox("Select Ticker", options=symbols, help="Data from DataSectors API")
        
        with col2:
            timeframe = st.selectbox(
                "Timeframe",
                options=["5m", "15m", "30m", "1h", "4h", "1d", "1w"],
                key="wl_tf"
            )
            
            alert_price = st.number_input(
                "Alert Price (Optional)",
                min_value=0.0,
                value=0.0,
                help="Get notified when price reaches this level"
            )
        
        if st.button("✅ Add to Watchlist", use_container_width=True, type="primary"):
            if selected_ticker:
                result = make_request("/watchlist/add", "POST", {
                    "ticker": selected_ticker,
                    "timeframe": timeframe,
                    "alert_price": alert_price if alert_price > 0 else None
                })
                
                if result:
                    st.success(f"✅ {selected_ticker} added to watchlist!")
                    time.sleep(1)
                    st.rerun()

def analytics_page():
    st.title("📊 Performance Analytics")
    
    # Get analytics data
    overview = make_request("/analytics/overview")
    accuracy = make_request("/analytics/accuracy")
    
    if overview:
        # Key Metrics
        st.subheader("📈 Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_scans = sum([s['count'] for s in overview['signal_distribution']])
        
        with col1:
            st.metric("Total Scans", total_scans)
        
        with col2:
            if overview['signal_distribution']:
                avg_conf = sum([s['avg_confidence'] for s in overview['signal_distribution']]) / len(overview['signal_distribution'])
                st.metric("Avg Confidence", f"{avg_conf:.1f}%")
        
        with col3:
            st.metric("Patterns Detected", len(overview.get('pattern_distribution', {})))
        
        with col4:
            if overview['trade_performance']:
                total_trades = sum([t['trades'] for t in overview['trade_performance']])
                st.metric("Total Trades", total_trades)
        
        st.markdown("---")
        
        # Charts
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Signals", "📅 Timeline", "🎯 Top Tickers", "🎨 Patterns"])
        
        with tab1:
            st.subheader("Signal Distribution")
            
            if overview['signal_distribution']:
                signal_df = pd.DataFrame(overview['signal_distribution'])
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    # Pie chart
                    fig = go.Figure(data=[go.Pie(
                        labels=signal_df['signal'],
                        values=signal_df['count'],
                        marker=dict(colors=['#10b981', '#ef4444', '#6b7280'])
                    )])
                    fig.update_layout(
                        title="Signal Count",
                        plot_bgcolor='#1a1f2e',
                        paper_bgcolor='#0f1318',
                        font=dict(color='#ffffff')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_chart2:
                    # Bar chart - Average Confidence
                    fig = go.Figure(data=[go.Bar(
                        x=signal_df['signal'],
                        y=signal_df['avg_confidence'],
                        marker_color=['#10b981', '#ef4444', '#6b7280']
                    )])
                    fig.update_layout(
                        title="Average Confidence by Signal",
                        yaxis_title="Confidence (%)",
                        plot_bgcolor='#1a1f2e',
                        paper_bgcolor='#0f1318',
                        font=dict(color='#ffffff')
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Scan Activity Timeline")
            
            if overview['scans_timeline']:
                timeline_df = pd.DataFrame(overview['scans_timeline'])
                
                fig = go.Figure(data=[go.Scatter(
                    x=timeline_df['date'],
                    y=timeline_df['count'],
                    mode='lines+markers',
                    line=dict(color='#3b82f6', width=2),
                    marker=dict(size=8)
                )])
                fig.update_layout(
                    title="Scans Over Time (Last 30 Days)",
                    xaxis_title="Date",
                    yaxis_title="Number of Scans",
                    plot_bgcolor='#1a1f2e',
                    paper_bgcolor='#0f1318',
                    font=dict(color='#ffffff')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No scan timeline data yet")
        
        with tab3:
            st.subheader("Most Scanned Tickers")
            
            if overview['top_tickers']:
                top_df = pd.DataFrame(overview['top_tickers'])
                
                fig = go.Figure(data=[go.Bar(
                    y=top_df['ticker'],
                    x=top_df['scans'],
                    orientation='h',
                    marker_color='#8b5cf6'
                )])
                fig.update_layout(
                    title="Top 10 Tickers",
                    xaxis_title="Scan Count",
                    plot_bgcolor='#1a1f2e',
                    paper_bgcolor='#0f1318',
                    font=dict(color='#ffffff'),
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No ticker data yet")
        
        with tab4:
            st.subheader("Pattern Distribution")
            
            if overview['pattern_distribution']:
                pattern_df = pd.DataFrame([
                    {"pattern": k, "count": v} 
                    for k, v in overview['pattern_distribution'].items()
                ])
                pattern_df = pattern_df.sort_values('count', ascending=True)
                
                fig = go.Figure(data=[go.Bar(
                    y=pattern_df['pattern'],
                    x=pattern_df['count'],
                    orientation='h',
                    marker_color='#f59e0b'
                )])
                fig.update_layout(
                    title="Chart Patterns Detected",
                    xaxis_title="Frequency",
                    plot_bgcolor='#1a1f2e',
                    paper_bgcolor='#0f1318',
                    font=dict(color='#ffffff'),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No patterns detected yet. Run more scans to see patterns!")
        
        st.markdown("---")
        
        # Trade Performance
        if accuracy and accuracy['accuracy']:
            st.subheader("🎯 Signal Accuracy")
            
            acc_df = pd.DataFrame(accuracy['accuracy'])
            
            col_acc1, col_acc2 = st.columns(2)
            
            with col_acc1:
                # Accuracy by signal
                fig = go.Figure(data=[go.Bar(
                    x=acc_df['signal'],
                    y=acc_df['accuracy'],
                    text=acc_df['accuracy'].apply(lambda x: f"{x:.1f}%"),
                    textposition='auto',
                    marker_color=['#10b981', '#ef4444', '#6b7280']
                )])
                fig.update_layout(
                    title="Signal Accuracy (%)",
                    yaxis_title="Accuracy",
                    plot_bgcolor='#1a1f2e',
                    paper_bgcolor='#0f1318',
                    font=dict(color='#ffffff')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_acc2:
                # Win/Loss breakdown
                fig = go.Figure(data=[go.Bar(
                    x=acc_df['signal'],
                    y=acc_df['wins'],
                    name='Wins',
                    marker_color='#10b981'
                ), go.Bar(
                    x=acc_df['signal'],
                    y=acc_df['total_trades'] - acc_df['wins'],
                    name='Losses',
                    marker_color='#ef4444'
                )])
                fig.update_layout(
                    title="Win/Loss Breakdown",
                    yaxis_title="Trades",
                    barmode='stack',
                    plot_bgcolor='#1a1f2e',
                    paper_bgcolor='#0f1318',
                    font=dict(color='#ffffff')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance table
            st.dataframe(
                acc_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "signal": "Signal",
                    "accuracy": st.column_config.NumberColumn("Accuracy", format="%.2f%%"),
                    "wins": "Wins",
                    "total_trades": "Total Trades",
                    "avg_profit_loss": st.column_config.NumberColumn("Avg P/L", format="%.2f")
                }
            )
    else:
        st.info("No analytics data available yet. Start scanning tickers to build your analytics!")

def multi_timeframe_page():
    st.title("⏱️ Multi-Timeframe Analysis")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Settings")
        
        search_term = st.text_input("Search Ticker", placeholder="e.g., BTC, ETH", key="mtf_search")
        symbols_data = make_request(f"/symbols?search={search_term}&limit=500" if search_term else "/symbols?limit=500")
        symbols = [s['symbol'] for s in symbols_data['symbols']] if symbols_data else []
        
        # Show data source info
        if symbols_data and symbols:
            source = symbols_data.get('source', 'DataSectors')
            total = symbols_data.get('total', len(symbols))
            st.caption(f"🔌 {total} symbols from {source}")
        
        selected_symbol = st.selectbox(
            "Select Ticker",
            options=symbols,
            index=0 if symbols else None,
            help="Data sourced from DataSectors API"
        )
        
        st.write("**Select Timeframes**")
        tf_options = ["5m", "15m", "30m", "1h", "4h", "1d", "1w"]
        selected_timeframes = []
        
        for tf in tf_options:
            if st.checkbox(tf, value=tf in ["15m", "1h", "4h", "1d"]):
                selected_timeframes.append(tf)
        
        scan_button = st.button("🚀 Multi-TF Scan", use_container_width=True, type="primary")
        
        if scan_button and selected_symbol and selected_timeframes:
            with st.spinner("Analyzing multiple timeframes..."):
                result = make_request("/scan/multi-timeframe", "POST", {
                    "ticker": selected_symbol,
                    "timeframes": selected_timeframes
                })
                
                if result:
                    st.session_state.mtf_result = result
    
    with col2:
        if 'mtf_result' in st.session_state:
            result = st.session_state.mtf_result
            
            # Overall signal
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                signal = result['overall_signal']
                signal_class = f"signal-{signal.lower()}"
                st.markdown(f'<div class="{signal_class}">Overall: {signal}</div>', unsafe_allow_html=True)
            
            with col_b:
                st.metric("Avg Confidence", f"{result['average_confidence']:.1f}%")
            
            with col_c:
                st.metric("Ticker", result['ticker'])
            
            st.markdown("---")
            
            # Signal distribution
            st.subheader("📊 Signal Distribution")
            dist = result['signal_distribution']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("LONG", dist.get('LONG', 0), delta="Bullish" if dist.get('LONG', 0) > 0 else None)
            with col2:
                st.metric("SHORT", dist.get('SHORT', 0), delta="Bearish" if dist.get('SHORT', 0) > 0 else None)
            with col3:
                st.metric("NEUTRAL", dist.get('NEUTRAL', 0))
            
            st.markdown("---")
            
            # Timeframe analysis table
            st.subheader("🔍 Timeframe Breakdown")
            
            tf_data = []
            for tf, analysis in result['timeframe_analysis'].items():
                tf_data.append({
                    "Timeframe": tf,
                    "Signal": analysis['signal'],
                    "Confidence": f"{analysis['confidence']:.1f}%",
                    "Trend": analysis['trend'],
                    "Price": f"${analysis['price']:,.2f}",
                    "RSI": f"{analysis['rsi']:.1f}",
                    "ADX": f"{analysis['adx']:.1f}"
                })
            
            df_tf = pd.DataFrame(tf_data)
            
            # Display table without styling (simpler approach)
            st.dataframe(
                df_tf,
                use_container_width=True,
                hide_index=True
            )
            
            # Show signal summary separately
            st.markdown("---")
            st.subheader("📊 Signal Summary")
            
            for idx, row in df_tf.iterrows():
                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                
                with col1:
                    st.write(f"**{row['Timeframe']}**")
                
                with col2:
                    signal = row['Signal']
                    if signal == 'LONG':
                        st.markdown('🟢 **LONG**')
                    elif signal == 'SHORT':
                        st.markdown('🔴 **SHORT**')
                    else:
                        st.markdown('⚪ **NEUTRAL**')
                
                with col3:
                    st.write(f"Conf: {row['Confidence']}")
                
                with col4:
                    st.write(f"Trend: {row['Trend']}")
            
            st.markdown("---")
            
            # Interpretation
            st.subheader("💡 Interpretation")
            
            long_count = dist.get('LONG', 0)
            short_count = dist.get('SHORT', 0)
            total = long_count + short_count + dist.get('NEUTRAL', 0)
            
            if long_count > total * 0.6:
                st.success("🟢 **Strong Bullish Alignment**: Multiple timeframes show LONG signals. Consider bullish positions.")
            elif short_count > total * 0.6:
                st.error("🔴 **Strong Bearish Alignment**: Multiple timeframes show SHORT signals. Consider bearish positions.")
            elif long_count == short_count:
                st.warning("⚪ **Mixed Signals**: Timeframes are conflicting. Wait for clearer alignment or use caution.")
            else:
                st.info("📊 **Moderate Signal**: Some alignment present but not overwhelming. Consider risk management.")
            
            # Export
            if st.button("📄 Export Multi-TF Report", use_container_width=True):
                excel_data = export_to_excel(tf_data)
                st.download_button(
                    "⬇️ Download Excel",
                    excel_data,
                    file_name=f"mtf_scan_{result['ticker']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.info("👈 Select a ticker and timeframes, then click 'Multi-TF Scan' to start")

def history_page():
    st.title("📜 Scan History")
    
    tab1, tab2 = st.tabs(["Single Timeframe", "Multi-Timeframe"])
    
    with tab1:
        history_data = make_request("/history")
        
        if history_data and history_data['history']:
            df = pd.DataFrame(history_data['history'])
            
            # Export button
            if st.button("📊 Export to Excel", key="export_single"):
                excel_data = export_to_excel(history_data['history'])
                st.download_button(
                    "⬇️ Download Excel",
                    excel_data,
                    file_name=f"scan_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ticker": "Ticker",
                    "timeframe": "Timeframe",
                    "signal": st.column_config.TextColumn("Signal"),
                    "confidence": st.column_config.NumberColumn("Confidence", format="%.2f%%"),
                    "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "created_at": "Date"
                }
            )
        else:
            st.info("No scan history yet.")
    
    with tab2:
        mtf_history = make_request("/history/multi-timeframe")
        
        if mtf_history and mtf_history['history']:
            for entry in mtf_history['history']:
                with st.expander(f"{entry['ticker']} - {entry['overall_signal']} ({entry['created_at']})"):
                    st.write(f"**Timeframes**: {entry['timeframes']}")
                    st.write(f"**Overall Signal**: {entry['overall_signal']}")
                    
                    if entry['analysis']:
                        analysis = entry['analysis']
                        st.write(f"**Average Confidence**: {analysis.get('average_confidence', 0):.1f}%")
                        
                        if 'timeframe_analysis' in analysis:
                            st.write("**Breakdown**:")
                            for tf, data in analysis['timeframe_analysis'].items():
                                st.write(f"- {tf}: {data['signal']} ({data['confidence']:.1f}%)")
        else:
            st.info("No multi-timeframe scan history yet.")

def indonesia_stocks_page():
    st.title("🇮🇩 Indonesia Stock Analysis")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Settings")
        
        # Fetch available stocks
        stocks_list = make_request("/stocks/indonesia/list")
        if stocks_list and 'stocks' in stocks_list:
            stock_options = {s['ticker']: f"{s['ticker']} - {s['name']}" for s in stocks_list['stocks']}
            selected_stock = st.selectbox("Select Stock", options=list(stock_options.keys()), format_func=lambda x: stock_options[x])
        else:
            st.error("Failed to load stock list")
            return
        
        # Timeframe selection (DataSectors supported intervals)
        interval = st.selectbox(
            "Timeframe",
            options=["1m", "5m", "15m", "30m", "1h", "1d", "1w", "1mo"],
            format_func=lambda x: {
                "1m": "1 Minute",
                "5m": "5 Minutes",
                "15m": "15 Minutes",
                "30m": "30 Minutes",
                "1h": "1 Hour",
                "1d": "Daily",
                "1w": "Weekly",
                "1mo": "Monthly"
            }[x]
        )
        
        if st.button("📊 Analyze", use_container_width=True, type="primary"):
            with st.spinner("Analyzing..."):
                result = make_request(
                    f"/stocks/indonesia/{selected_stock}/analyze",
                    params={'interval': interval}
                )
                
                if result and 'error' not in result:
                    st.session_state.indonesia_stock_result = result
                    st.success("Analysis complete!")
                    st.rerun()
                else:
                    error_msg = result.get('detail', result.get('error', 'Unknown error')) if result else 'Request failed'
                    st.error(f"❌ {error_msg}")
    
    with col2:
        if 'indonesia_stock_result' in st.session_state:
            result = st.session_state.indonesia_stock_result
            
            # Display signal with reasoning
            display_signal_reasoning(result)
            
            st.markdown("---")
            
            # Additional metrics
            col_price, col_trend, col_sector = st.columns(3)
            with col_price:
                st.metric("Current Price", f"Rp {result.get('current_price', 0):,.0f}")
            with col_trend:
                st.metric("Trend", result.get('trend', 'N/A'))
            with col_sector:
                st.metric("Sector", result.get('sector', 'N/A'))
            
            st.markdown("---")
            
            # Chart with controls
            st.subheader(f"📈 {result['ticker']} ({result['symbol']}) Chart")
            
            with st.spinner("📊 Rendering chart..."):
                # Prepare data for chart
                chart_data = result.get('chart_data', [])
                if chart_data:
                    chart = create_candlestick_chart(chart_data, result, show_ichimoku=False)
                    display_chart_with_controls(chart, result, show_ichimoku=False)
                else:
                    st.warning("No chart data available")
            
            # Compact indicators
            st.markdown("---")
            st.subheader("📊 Key Indicators Snapshot")
            create_compact_indicator_display(result)
            
            # Technical Indicators in tabs
            st.markdown("---")
            st.subheader("📈 Detailed Technical Analysis")
            
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Core Indicators", "🎯 Levels", "📈 Oscillators", "💰 SMC"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Moving Averages**")
                    mas = result['indicators']['moving_averages']
                    current_price = result['current_price']
                    for ma, value in mas.items():
                        if value is not None:
                            trend_class = "indicator-bullish" if current_price > value else "indicator-bearish"
                            st.markdown(f'<span class="{trend_class}">{ma}: Rp {value:,.0f}</span>', unsafe_allow_html=True)
                    
                    st.write("**Bollinger Bands**")
                    bb = result['indicators']['bollinger_bands']
                    if bb['upper'] is not None:
                        st.write(f"Upper: Rp {bb['upper']:,.0f}")
                        st.write(f"Middle: Rp {bb['middle']:,.0f}")
                        st.write(f"Lower: Rp {bb['lower']:,.0f}")
                    else:
                        st.info("BB data not available")
                
                with col2:
                    st.write("**RSI (14)**")
                    rsi = result['indicators']['rsi']
                    if rsi is not None:
                        rsi_status = "🔴 Overbought" if rsi > 70 else "🟢 Oversold" if rsi < 30 else "⚪ Neutral"
                        st.write(f"Value: {rsi:.1f}")
                        st.write(f"Status: {rsi_status}")
                    else:
                        st.write("RSI: N/A")
                    
                    st.write("**MACD**")
                    macd = result['indicators']['macd']
                    if macd['histogram'] is not None:
                        macd_trend = "🟢 Bullish" if macd['histogram'] > 0 else "🔴 Bearish"
                        st.write(f"Status: {macd_trend}")
                        st.write(f"Value: {macd['histogram']:.6f}")
                    else:
                        st.info("MACD data not available")
            
            with tab2:
                st.write("**Stock Information**")
                st.write(f"Ticker: **{result['ticker']}**")
                st.write(f"Symbol: **{result['symbol']}**")
                st.write(f"Name: **{result['name']}**")
                st.write(f"Sector: **{result['sector']}**")
                st.write(f"Exchange: **{result['exchange']}**")
                st.write(f"Currency: **{result['currency']}**")
            
            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Stochastic**")
                    stoch = result['indicators']['stochastic']
                    if stoch['k'] is not None:
                        stoch_status = "🔴 Overbought" if stoch['k'] > 80 else "🟢 Oversold" if stoch['k'] < 20 else "⚪ Neutral"
                        st.write(f"%K: {stoch['k']:.1f}")
                        st.write(f"%D: {stoch['d']:.1f}")
                        st.write(f"Status: {stoch_status}")
                    else:
                        st.info("Stochastic data not available")
                
                with col2:
                    st.write("**Price vs Bands**")
                    current = result['current_price']
                    bb = result['indicators']['bollinger_bands']
                    if bb['upper'] is not None:
                        bb_position = (current - bb['lower']) / (bb['upper'] - bb['lower']) * 100
                        st.write(f"BB Position: {bb_position:.1f}%")
                        if bb_position > 80:
                            st.write("📍 Near upper band (Overbought)")
                        elif bb_position < 20:
                            st.write("📍 Near lower band (Oversold)")
                        else:
                            st.write("📍 In middle range (Balanced)")
            
            with tab4:
                st.info("SMC (Smart Money Concepts) analysis not available for stocks yet")
        
        else:
            st.info("👈 Select a stock and click 'Analyze' to start")

def summary_page():
    st.title("💡 Market Summary")
    
    # Initialize auto-refresh
    if 'last_summary_update' not in st.session_state:
        st.session_state.last_summary_update = 0
    if 'summary_auto_refresh' not in st.session_state:
        st.session_state.summary_auto_refresh = True
    
    # Top controls
    col1, col2, col3 = st.columns([3, 1, 1])
    with col2:
        st.session_state.summary_auto_refresh = st.checkbox("🔄 Auto-Refresh", value=st.session_state.summary_auto_refresh)
    with col3:
        if st.button("🔃 Refresh", use_container_width=True):
            st.session_state.last_summary_update = 0
            st.rerun()
    
    try:
        summary_data = make_request("/crypto/summary")
        
        if summary_data:
            # Update timestamp
            st.session_state.last_summary_update = time.time()
            
            # Display prices in cards
            st.subheader("📊 Current Prices")
            
            price_cols = st.columns(3)
            cryptocurrencies = ['BTC', 'ETH', 'SOL']
            emojis = {'BTC': '₿', 'ETH': 'Ξ', 'SOL': '◎'}
            colors = {'BTC': '🟠', 'ETH': '⚫', 'SOL': '🟣'}
            
            for idx, crypto in enumerate(cryptocurrencies):
                with price_cols[idx]:
                    if crypto in summary_data.get('prices', {}):
                        price_info = summary_data['prices'][crypto]
                        
                        if 'error' not in price_info:
                            price = price_info.get('price', 'N/A')
                            change_pct = price_info.get('change_pct_24h', 'N/A')
                            
                            # Determine delta color
                            if isinstance(change_pct, str):
                                try:
                                    pct_val = float(change_pct.replace('%', ''))
                                    delta_str = f"{change_pct}"
                                except:
                                    delta_str = change_pct
                            else:
                                delta_str = f"{change_pct:.2f}%"
                            
                            # Display metric
                            st.metric(
                                label=f"{emojis.get(crypto, '')} {crypto}",
                                value=f"${price:,.2f}" if isinstance(price, (int, float)) else price,
                                delta=delta_str
                            )
                        else:
                            st.warning(f"⚠️ {crypto}: Error fetching data")
                    else:
                        st.warning(f"⚠️ {crypto}: No data available")
            
            # Display Fear & Greed Index
            st.markdown("---")
            st.subheader("😨 Fear & Greed Index")
            
            if summary_data.get('fear_greed') and 'error' not in summary_data['fear_greed']:
                fng = summary_data['fear_greed']
                value = int(fng.get('value', 0))
                classification = fng.get('value_classification', fng.get('classification', 'N/A'))
                
                # Determine sentiment and color based on value ranges
                if value >= 75:
                    sentiment = "🟩 Extreme Greed"
                    sentiment_color = "#00c851"
                    bg_color = "#e8f5e9"
                    emoji = "🤑"
                    description = "Market euphoria - Exercise caution"
                elif value >= 55:
                    sentiment = "🟢 Greed"
                    sentiment_color = "#7cb342"
                    bg_color = "#f1f8e9"
                    emoji = "😊"
                    description = "Strong market sentiment"
                elif value >= 45:
                    sentiment = "⚪ Neutral"
                    sentiment_color = "#ffb74d"
                    bg_color = "#fff3e0"
                    emoji = "😐"
                    description = "Market in balance"
                elif value >= 25:
                    sentiment = "🟠 Fear"
                    sentiment_color = "#ff7043"
                    bg_color = "#ffebee"
                    emoji = "😨"
                    description = "Market uncertainty"
                else:
                    sentiment = "🔴 Extreme Fear"
                    sentiment_color = "#d32f2f"
                    bg_color = "#ffcdd2"
                    emoji = "😱"
                    description = "Panic conditions - Potential opportunities"
                
                # Create gauge chart with Plotly
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Market Sentiment"},
                    delta={'reference': 50, 'suffix': " vs Neutral"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': sentiment_color},
                        'steps': [
                            {'range': [0, 25], 'color': "#ffcdd2"},
                            {'range': [25, 45], 'color': "#ffebee"},
                            {'range': [45, 55], 'color': "#fff3e0"},
                            {'range': [55, 75], 'color': "#f1f8e9"},
                            {'range': [75, 100], 'color': "#e8f5e9"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    },
                    number={'suffix': "/100"}
                ))
                
                fig.update_layout(
                    height=350,
                    font={'size': 12},
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                
                # Display main gauge
                st.plotly_chart(fig, use_container_width=True)
                
                # Display sentiment card with better styling
                st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, {sentiment_color}20, {sentiment_color}10);
                    border-left: 5px solid {sentiment_color};
                    border-radius: 8px;
                    padding: 20px;
                    margin: 15px 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                '>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <h3 style='margin: 0; color: {sentiment_color};'>{sentiment}</h3>
                            <p style='margin: 8px 0 0 0; color: #666; font-size: 14px;'>{description}</p>
                        </div>
                        <div style='font-size: 3em; text-align: center;'>{emoji}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Statistics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Current Score", f"{value}/100", 
                             delta=f"{value - 50}" if value != 50 else "Neutral",
                             delta_color="off" if value == 50 else "inverse")
                
                with col2:
                    distance_extreme = min(abs(value - 0), abs(100 - value))
                    st.metric("📏 Distance to Extreme", distance_extreme, 
                             help="How far from extreme fear (0) or extreme greed (100)")
                
                with col3:
                    st.caption(f"**Classification**\n{classification}")
                
                with col4:
                    st.caption(f"**Last Update**\n{fng.get('timestamp', 'N/A')}")
                
            else:
                st.error("❌ Fear & Greed Index data unavailable")
            
            # Footer with timestamp
            st.markdown("---")
            
            col_time1, col_time2, col_time3 = st.columns([1, 1, 1])
            with col_time1:
                st.caption(f"📍 Last updated: {datetime.now().strftime('%H:%M:%S')}")
            with col_time2:
                if st.session_state.summary_auto_refresh:
                    st.caption("✅ Auto-refresh enabled")
                else:
                    st.caption("⏸️ Auto-refresh disabled")
            with col_time3:
                st.caption(f"🕐 Next refresh in ~60s")
            
            # Add auto-refresh script
            if st.session_state.summary_auto_refresh:
                st.info("💡 Data auto-refreshes every 60 seconds. Click 'Refresh' for immediate update.")
                # JavaScript for auto-refresh
                st.markdown("""
                <script>
                    setTimeout(() => {
                        window.location.reload();
                    }, 60000);
                </script>
                """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"❌ Error fetching summary: {str(e)}")
        st.info("Please try again or check your internet connection.")


# AI Chat Page
def ai_chat_page():
    """AI Chat interface with automatic model selection"""
    st.title("🤖 Tanya AI")
    st.write("Tanya pertanyaan tentang crypto, trading, atau topik umum lainnya. AI akan memilih model terbaik untuk menjawab.")
    
    # Initialize session state for AI chat
    if 'current_conversation' not in st.session_state:
        st.session_state.current_conversation = str(uuid.uuid4())
    if 'ai_messages' not in st.session_state:
        st.session_state.ai_messages = []
    
    # Conversation management tabs
    tab1, tab2 = st.tabs(["💬 Chat", "📋 Conversations"])
    
    # TAB 1: Chat
    with tab1:
        # Main chat area
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Conversation ID:** `{st.session_state.current_conversation[:8]}...`")
        with col2:
            st.caption("Models: 🤖 Deepseek | 📊 Plutus")
        
        st.markdown("---")
        
        # Display chat history
        chat_container = st.container()
        
        with chat_container:
            try:
                # Load conversation history
                response = requests.get(
                    f"{API_BASE_URL}/ai/conversation/{st.session_state.current_conversation}",
                    headers={"Authorization": f"Bearer {st.session_state.token}"}
                )
                
                if response.status_code == 200:
                    messages = response.json().get('messages', [])
                    
                    for msg in messages:
                        if msg['type'] == 'user':
                            st.markdown(f"""
                            <div style='background: #1a1f2e; padding: 12px; border-radius: 8px; margin: 8px 0;'>
                                <div style='color: #00d4ff; font-weight: bold;'>👤 You</div>
                                <div style='color: #ffffff; margin-top: 8px;'>{msg['content']}</div>
                                <div style='color: #666; font-size: 12px; margin-top: 4px;'>{msg['timestamp']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            model = msg.get('model', 'unknown')
                            category = msg.get('category', 'general')
                            
                            # Model indicator
                            if 'plutus' in model.lower():
                                model_badge = "📊 Plutus (Finance)"
                            else:
                                model_badge = "🤖 Deepseek (General)"
                            
                            st.markdown(f"""
                            <div style='background: #0f1318; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 4px solid #00d4ff;'>
                                <div style='color: #00d4ff; font-weight: bold;'>🤖 AI Assistant</div>
                                <div style='font-size: 12px; color: #888; margin: 4px 0;'>{model_badge} • {category.upper()}</div>
                                <div style='color: #ffffff; margin-top: 8px;'>{msg['content']}</div>
                                <div style='color: #666; font-size: 12px; margin-top: 4px;'>{msg['timestamp']}</div>
                            </div>
                            """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"⚠️ Error loading chat history: {str(e)}")
        
        st.markdown("---")
        
        # Initialize clear state
        if 'should_clear_input' not in st.session_state:
            st.session_state.should_clear_input = False
        
        # Input area - Resizeable text area
        user_input = st.text_area(
            "💬 Your message:",
            placeholder="Tanya tentang crypto, trading, finance, atau topik lainnya...\n\nContoh:\n• 'BTC break 45000, RSI 72. Signal?'\n• 'Apa perbedaan spot vs futures?'\n• 'Gimana cara install Python?'",
            height=120,
            key="ai_input",
            max_chars=5000,
            value="" if st.session_state.should_clear_input else st.session_state.get("ai_input", "")
        )
        
        # Reset clear flag after rendering
        if st.session_state.should_clear_input:
            st.session_state.should_clear_input = False
        
        # Character count and buttons
        col_char, col_buttons = st.columns([3, 3])
        
        with col_char:
            char_count = len(user_input) if user_input else 0
            st.caption(f"📝 {char_count}/5000 characters")
        
        with col_buttons:
            col_send1, col_clear = st.columns(2)
            
            with col_send1:
                send_button = st.button("📤 Send", use_container_width=True, type="primary")
            
            with col_clear:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.should_clear_input = True
                    st.rerun()
        
        st.markdown("---")
        
        # Send message
        if send_button and user_input:
            with st.spinner("🤔 Thinking..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/ai/chat",
                        json={
                            "message": user_input,
                            "conversation_id": st.session_state.current_conversation
                        },
                        headers={"Authorization": f"Bearer {st.session_state.token}"},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        ai_response = data.get('response', 'No response')
                        model_used = data.get('model_used', 'unknown')
                        category = data.get('question_category', 'general')
                        
                        # Show response
                        if 'plutus' in model_used.lower():
                            model_badge = "📊 Plutus (Finance)"
                        else:
                            model_badge = "🤖 Deepseek (General)"
                        
                        st.success("✅ Response received!")
                        st.markdown(f"""
                        <div style='background: #0f1318; padding: 12px; border-radius: 8px; border-left: 4px solid #00d4ff;'>
                            <div style='color: #00d4ff; font-weight: bold;'>🤖 AI Assistant</div>
                            <div style='font-size: 12px; color: #888; margin: 4px 0;'>{model_badge} • {category.upper()}</div>
                            <div style='color: #ffffff; margin-top: 8px;'>{ai_response}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {response.status_code}")
                        st.write(response.text)
                
                except requests.exceptions.Timeout:
                    st.error("❌ Request timeout. Model might be processing a complex query. Try again.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Make sure Ollama is running and models are installed:")
                    st.code("ollama pull deepseek-coder:6.7b\nollama pull plutus")
        
        # Help section
        with st.expander("ℹ️ How it works"):
            st.markdown("""
            **Model Selection:**
            - 📊 **Plutus**: For trading, finance, crypto, technical analysis questions
            - 🤖 **Deepseek**: For general questions, coding, and other topics
            
            **Conversation History:**
            - Each conversation is saved automatically
            - You can switch between conversations anytime
            - Delete conversations to free up space
            
            **Tips:**
            - Be specific with your trading questions
            - Include charts, prices, or indicators for better analysis
            - The AI remembers your previous messages in the conversation
            """)
    
    # TAB 2: Conversations
    with tab2:
        st.subheader("💬 Manage Conversations")
        
        col1, col2 = st.columns([2, 2])
        with col1:
            if st.button("➕ New Conversation", use_container_width=True, key="new_conv_tab"):
                st.session_state.current_conversation = str(uuid.uuid4())
                st.session_state.ai_messages = []
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh List", use_container_width=True, key="refresh_conv"):
                st.rerun()
        
        st.markdown("---")
        
        # Load conversation history
        try:
            response = requests.get(
                f"{API_BASE_URL}/ai/conversations",
                headers={"Authorization": f"Bearer {st.session_state.token}"}
            )
            
            if response.status_code == 200:
                conversations = response.json().get('conversations', [])
                
                if conversations:
                    st.write(f"**Total Conversations: {len(conversations)}**")
                    st.markdown("---")
                    
                    for idx, conv in enumerate(conversations):
                        conv_id = conv['id']
                        last_msg = conv['last_message']
                        
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            if st.button(
                                f"💬 {last_msg[:40]}...",
                                key=f"conv_tab_{conv_id}",
                                use_container_width=True,
                                help=f"ID: {conv_id}"
                            ):
                                st.session_state.current_conversation = conv_id
                                st.rerun()
                        
                        with col2:
                            st.caption(f"📅 {last_msg[-10:]}")
                        
                        with col3:
                            if st.button("🗑️", key=f"del_conv_tab_{conv_id}", help="Delete conversation"):
                                requests.delete(
                                    f"{API_BASE_URL}/ai/conversation/{conv_id}",
                                    headers={"Authorization": f"Bearer {st.session_state.token}"}
                                )
                                st.rerun()
                        
                        st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)
                else:
                    st.info("📭 No conversations yet. Start a new one!")
            else:
                st.error(f"❌ Error loading conversations: {response.status_code}")
        
        except Exception as e:
            st.error(f"⚠️ Error loading conversation history: {str(e)}")


# Sidebar Navigation
def sidebar():
    with st.sidebar:
        st.title("📈 Crypto Scanner Pro")
        st.write(f"Welcome, **{st.session_state.username}**!")
        # Theme selector
        theme_choice = st.radio("Theme", ("Dark", "Light"), index=0 if st.session_state.theme == 'dark' else 1)
        new_theme = theme_choice.lower()
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            render_theme_css(new_theme)
            safe_rerun()
        st.markdown("---")
        
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.page = 'dashboard'
            st.rerun()
        
        if st.button("🔍 Single Scan", use_container_width=True):
            st.session_state.page = 'scan'
            st.rerun()
        
        if st.button("⏱️ Multi-Timeframe", use_container_width=True):
            st.session_state.page = 'multi_timeframe'
            st.rerun()
        
        if st.button("👁️ Watchlist", use_container_width=True):
            st.session_state.page = 'watchlist'
            st.rerun()
        
        if st.button("📊 Analytics", use_container_width=True):
            st.session_state.page = 'analytics'
            st.rerun()
        
        if st.button("📜 History", use_container_width=True):
            st.session_state.page = 'history'
            st.rerun()
        
        if st.button("💡 Summary", use_container_width=True):
            st.session_state.page = 'summary'
            st.rerun()
        
        if st.button("🤖 Tanya AI", use_container_width=True):
            st.session_state.page = 'ai_chat'
            st.rerun()
        
        st.markdown("---")
        st.subheader("🇮🇩 Indonesia Stocks")
        
        if st.button("📈 Stock Analysis", use_container_width=True):
            st.session_state.page = 'indonesia_stocks'
            st.rerun()
        
        st.markdown("---")
        
        st.subheader("ℹ️ Info")
        st.write("**Features:**")
        st.write("• 100+ Cryptocurrencies")
        st.write("• DataSectors API (Professional)")
        st.write("• Up to 5000 candles")
        st.write("• 7 Timeframes")
        st.write("• 16+ Indicators")
        st.write("• SMI Indicator")
        st.write("• Pattern Recognition")
        st.write("• Auto-Scan Watchlist")
        st.write("• Performance Analytics")
        st.write("• PDF/Excel Export")
        
        st.markdown("---")
        st.caption("📊 **Data Source:**")
        st.caption("✅ DataSectors API")
        st.caption("✅ Professional market data")
        st.caption("✅ 100+ cryptocurrencies")
        st.caption("✅ Up to 5000 candles")
        st.caption("✅ Real-time & historical data")
        st.caption("✅ Real OHLCV data")
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.page = 'login'
            if 'scan_result' in st.session_state:
                del st.session_state.scan_result
            if 'mtf_result' in st.session_state:
                del st.session_state.mtf_result
            st.rerun()

# Main App Logic
def main():
    if not st.session_state.token:
        if st.session_state.page == 'register':
            register_page()
        else:
            login_page()
    else:
        sidebar()
        
        if st.session_state.page == 'dashboard':
            dashboard_page()
        elif st.session_state.page == 'scan':
            scan_page()
        elif st.session_state.page == 'multi_timeframe':
            multi_timeframe_page()
        elif st.session_state.page == 'watchlist':
            watchlist_page()
        elif st.session_state.page == 'analytics':
            analytics_page()
        elif st.session_state.page == 'history':
            history_page()
        elif st.session_state.page == 'summary':
            summary_page()
        elif st.session_state.page == 'ai_chat':
            ai_chat_page()
        elif st.session_state.page == 'indonesia_stocks':
            indonesia_stocks_page()

if __name__ == "__main__":
    main()
