"""
Trading Decision System - Streamlit Frontend
============================================
Modern, clean UI for Indonesian crypto and stock trading decisions.

Features:
- User authentication with session management
- Real-time trading signals
- Multi-timeframe scanner
- Interactive charts
- Trading history
- AI-powered insights
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from typing import Dict, Optional

# ============================================
# CONFIGURATION
# ============================================

API_BASE_URL = "http://localhost:2401/api"
DATASECTORS_API_URL = "http://148.230.96.135:5672"

# Page Configuration
st.set_page_config(
    page_title="Trading Decision System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #1E88E5;
        --success: #00E676;
        --danger: #FF5252;
        --warning: #FFC107;
        --dark: #1A1A2E;
        --light: #F8F9FA;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        color: white;
        margin: 10px 0;
    }
    
    .signal-card {
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 15px 0;
        transition: transform 0.3s ease;
    }
    
    .signal-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    
    .buy-signal {
        background: linear-gradient(135deg, #00E676 0%, #00C853 100%);
        color: white;
    }
    
    .sell-signal {
        background: linear-gradient(135deg, #FF5252 0%, #D32F2F 100%);
        color: white;
    }
    
    .hold-signal {
        background: linear-gradient(135deg, #90A4AE 0%, #607D8B 100%);
        color: white;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 12px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1A2E 0%, #16213E 100%);
    }
    
    section[data-testid="stSidebar"] .css-1d391kg {
        color: white;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
    }
    
    /* Metric boxes */
    .css-1xarl3l {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE MANAGEMENT
# ============================================

def init_session_state():
    """Initialize session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'token' not in st.session_state:
        st.session_state.token = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Dashboard'

init_session_state()

# ============================================
# API FUNCTIONS
# ============================================

def api_request(endpoint: str, method: str = "GET", data: Dict = None, auth: bool = True) -> Optional[Dict]:
    """Make API request with error handling"""
    url = f"{API_BASE_URL}{endpoint}"
    headers = {}
    
    if auth and st.session_state.token:
        headers['Authorization'] = f"Bearer {st.session_state.token}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=data, timeout=30)
        elif method == "POST":
            headers['Content-Type'] = 'application/json'
            response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            st.session_state.authenticated = False
            st.error("Session expired. Please login again.")
            return None
        else:
            st.error(f"API Error: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def call_datasectors_api(endpoint: str, params: Dict = None) -> Optional[Dict]:
    """Call DataSectors API directly"""
    url = f"{DATASECTORS_API_URL}{endpoint}"
    headers = {
        "X-API-Key": "sangahli",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

# ============================================
# AUTHENTICATION PAGES
# ============================================

def show_login_page():
    """Display login page"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="main-header">
            <h1 style="text-align: center; margin: 0;">📈 Trading Decision System</h1>
            <p style="text-align: center; margin: 10px 0 0 0; opacity: 0.9;">
                Intelligent Trading Signals for Indonesian Markets
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab1:
            st.markdown("### Welcome Back!")
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit = st.form_submit_button("Login", use_container_width=True)
                
                if submit:
                    if username and password:
                        result = api_request(
                            "/auth/login",
                            method="POST",
                            data={"username": username, "password": password},
                            auth=False
                        )
                        
                        if result:
                            st.session_state.authenticated = True
                            st.session_state.user = result['user']
                            st.session_state.token = result['access_token']
                            st.success("✅ Login successful!")
                            st.rerun()
                    else:
                        st.error("Please fill in all fields")
        
        with tab2:
            st.markdown("### Create Account")
            with st.form("register_form"):
                reg_username = st.text_input("Username", placeholder="Choose a username")
                reg_email = st.text_input("Email", placeholder="your.email@example.com")
                reg_password = st.text_input("Password", type="password", placeholder="Create a password")
                reg_confirm = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                register = st.form_submit_button("Register", use_container_width=True)
                
                if register:
                    if not all([reg_username, reg_email, reg_password, reg_confirm]):
                        st.error("Please fill in all fields")
                    elif reg_password != reg_confirm:
                        st.error("Passwords do not match")
                    elif len(reg_password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        result = api_request(
                            "/auth/register",
                            method="POST",
                            data={
                                "username": reg_username,
                                "email": reg_email,
                                "password": reg_password
                            },
                            auth=False
                        )
                        
                        if result:
                            st.session_state.authenticated = True
                            st.session_state.user = result['user']
                            st.session_state.token = result['access_token']
                            st.success("✅ Registration successful!")
                            st.rerun()

# ============================================
# HELPER FUNCTIONS
# ============================================

def create_indicator_chart(candles, indicators, symbol, timeframe):
    """Create a 3-panel chart with price and indicators"""
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f'{symbol} - {timeframe}', 'RSI', 'MACD')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=list(range(len(candles))),
            open=[c['open'] for c in candles],
            high=[c['high'] for c in candles],
            low=[c['low'] for c in candles],
            close=[c['close'] for c in candles],
            name='Price',
            increasing_line_color='#00E676',
            decreasing_line_color='#FF5252'
        ),
        row=1, col=1
    )
    
    # Moving Average
    ma_data = indicators.get('ma', [])
    if ma_data:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(ma_data))),
                y=ma_data,
                name='MA(20)',
                line=dict(color='#FF9800', width=2)
            ),
            row=1, col=1
        )
    
    # RSI
    rsi_data = indicators.get('rsi', [])
    if rsi_data:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(rsi_data))),
                y=rsi_data,
                name='RSI',
                line=dict(color='#9C27B0', width=2)
            ),
            row=2, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="#FF5252", row=2, col=1, opacity=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color="#00E676", row=2, col=1, opacity=0.5)
        fig.add_hline(y=50, line_dash="dot", line_color="#9E9E9E", row=2, col=1, opacity=0.3)
    
    # MACD
    macd_indicator = indicators.get('macd', {})
    if macd_indicator:
        macd_line = macd_indicator.get('macd', [])
        signal_line = macd_indicator.get('signal', [])
        histogram = macd_indicator.get('histogram', [])
        
        if macd_line:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(macd_line))),
                    y=macd_line,
                    name='MACD',
                    line=dict(color='#2196F3', width=2)
                ),
                row=3, col=1
            )
        
        if signal_line:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(signal_line))),
                    y=signal_line,
                    name='Signal',
                    line=dict(color='#FF5252', width=2)
                ),
                row=3, col=1
            )
        
        if histogram:
            colors = ['#00E676' if h > 0 else '#FF5252' for h in histogram]
            fig.add_trace(
                go.Bar(
                    x=list(range(len(histogram))),
                    y=histogram,
                    name='Histogram',
                    marker_color=colors,
                    opacity=0.5
                ),
                row=3, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=700,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        hovermode='x unified',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig

# ============================================
# DASHBOARD PAGE
# ============================================

def show_dashboard():
    """Display main dashboard with history recap"""
    st.markdown(f"""
    <div class="main-header">
        <h1>📊 Dashboard</h1>
        <p>Welcome back, <strong>{st.session_state.user['username']}</strong>! 
        Role: <strong>{st.session_state.user['role'].upper()}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================
    # MARKET DATA SECTION (Real-time insights)
    # ============================================
    st.markdown("### 🔴 Market Insights (Real-time)")
    
    market_col1, market_col2, market_col3, market_col4 = st.columns(4)
    
    # Strong Trend Cryptos
    with market_col1:
        with st.spinner("📈 Loading strong trends..."):
            try:
                trend_result = call_datasectors_api("/crypto/strong-trend")
                if trend_result and trend_result.get('data'):
                    strong_trends = trend_result['data']
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 20px; border-radius: 12px; color: white; height: 100%;">
                        <h4 style="margin: 0 0 15px 0; font-size: 16px;">📈 Strong Trends</h4>
                        <div style="font-size: 32px; font-weight: bold; margin-bottom: 10px;">
                            {count}
                        </div>
                        <p style="margin: 0; font-size: 13px; opacity: 0.9;">
                            Coins with strong trend indicators
                        </p>
                    </div>
                    """.format(count=len(strong_trends) if isinstance(strong_trends, list) else 0), unsafe_allow_html=True)
                else:
                    st.metric("Strong Trends", "—", help="Data not available")
            except Exception as e:
                st.metric("Strong Trends", "—", help=f"Error: {str(e)}")
    
    # Trending Coins
    with market_col2:
        with st.spinner("🔥 Loading trending..."):
            try:
                trending_result = call_datasectors_api("/crypto/trending")
                if trending_result and trending_result.get('data'):
                    trending = trending_result['data']
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                padding: 20px; border-radius: 12px; color: white; height: 100%;">
                        <h4 style="margin: 0 0 15px 0; font-size: 16px;">🔥 Trending</h4>
                        <div style="font-size: 32px; font-weight: bold; margin-bottom: 10px;">
                            {count}
                        </div>
                        <p style="margin: 0; font-size: 13px; opacity: 0.9;">
                            Coins gaining momentum
                        </p>
                    </div>
                    """.format(count=len(trending) if isinstance(trending, list) else 0), unsafe_allow_html=True)
                else:
                    st.metric("Trending", "—", help="Data not available")
            except Exception as e:
                st.metric("Trending", "—", help=f"Error: {str(e)}")
    
    # Orderbook Walls
    with market_col3:
        with st.spinner("🧱 Loading walls..."):
            try:
                walls_result = call_datasectors_api("/crypto/walls")
                if walls_result and walls_result.get('data'):
                    walls = walls_result['data']
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                                padding: 20px; border-radius: 12px; color: white; height: 100%;">
                        <h4 style="margin: 0 0 15px 0; font-size: 16px;">🧱 Walls Detected</h4>
                        <div style="font-size: 32px; font-weight: bold; margin-bottom: 10px;">
                            {count}
                        </div>
                        <p style="margin: 0; font-size: 13px; opacity: 0.9;">
                            Significant orderbook walls
                        </p>
                    </div>
                    """.format(count=len(walls) if isinstance(walls, list) else 0), unsafe_allow_html=True)
                else:
                    st.metric("Walls", "—", help="Data not available")
            except Exception as e:
                st.metric("Walls", "—", help=f"Error: {str(e)}")
    
    # Orderbook Imbalance
    with market_col4:
        with st.spinner("⚖️ Loading imbalance..."):
            try:
                imbalance_result = call_datasectors_api("/crypto/orderbook-imbalance")
                if imbalance_result and imbalance_result.get('data'):
                    imbalance = imbalance_result['data']
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                                padding: 20px; border-radius: 12px; color: white; height: 100%;">
                        <h4 style="margin: 0 0 15px 0; font-size: 16px;">⚖️ Imbalance</h4>
                        <div style="font-size: 32px; font-weight: bold; margin-bottom: 10px;">
                            {count}
                        </div>
                        <p style="margin: 0; font-size: 13px; opacity: 0.9;">
                            Coins with imbalanced orderbooks
                        </p>
                    </div>
                    """.format(count=len(imbalance) if isinstance(imbalance, list) else 0), unsafe_allow_html=True)
                else:
                    st.metric("Imbalance", "—", help="Data not available")
            except Exception as e:
                st.metric("Imbalance", "—", help=f"Error: {str(e)}")
    
    st.markdown("---")
    
    # ============================================
    # DETAILED MARKET DATA
    # ============================================
    tabs = st.tabs(["📈 Strong Trends", "🔥 Trending Coins", "🧱 Orderbook Walls", "⚖️ Imbalance"])
    
    # Tab 1: Strong Trends
    with tabs[0]:
        with st.spinner("Loading strong trend data..."):
            try:
                trend_result = call_datasectors_api("/crypto/strong-trend")
                if trend_result and trend_result.get('data'):
                    trends_df = pd.DataFrame(trend_result['data'])
                    
                    st.markdown("#### Strong Trend Cryptocurrencies")
                    st.dataframe(
                        trends_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            col: st.column_config.TextColumn(width='auto') 
                            for col in trends_df.columns
                        }
                    )
                    
                    # Stats
                    if len(trends_df) > 0:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Strong Trends", len(trends_df))
                        with col2:
                            st.info(f"Updated regularly for optimal trading opportunities")
                else:
                    st.info("No strong trend data available at this moment")
            except Exception as e:
                st.error(f"Error loading strong trends: {str(e)}")
    
    # Tab 2: Trending Coins
    with tabs[1]:
        with st.spinner("Loading trending coins..."):
            try:
                trending_result = call_datasectors_api("/crypto/trending")
                if trending_result and trending_result.get('data'):
                    trending_df = pd.DataFrame(trending_result['data'])
                    
                    st.markdown("#### Currently Trending Coins")
                    st.dataframe(
                        trending_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            col: st.column_config.TextColumn(width='auto') 
                            for col in trending_df.columns
                        }
                    )
                    
                    if len(trending_df) > 0:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Trending", len(trending_df))
                        with col2:
                            st.info(f"Coins gaining momentum in the market")
                else:
                    st.info("No trending coins at this moment")
            except Exception as e:
                st.error(f"Error loading trending coins: {str(e)}")
    
    # Tab 3: Orderbook Walls
    with tabs[2]:
        with st.spinner("Loading orderbook walls..."):
            try:
                walls_result = call_datasectors_api("/crypto/walls")
                if walls_result and walls_result.get('data'):
                    walls_df = pd.DataFrame(walls_result['data'])
                    
                    st.markdown("#### Detected Orderbook Walls")
                    st.dataframe(
                        walls_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            col: st.column_config.TextColumn(width='auto') 
                            for col in walls_df.columns
                        }
                    )
                    
                    if len(walls_df) > 0:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Walls Detected", len(walls_df))
                        with col2:
                            st.info(f"Significant support/resistance zones")
                else:
                    st.info("No orderbook walls detected at this moment")
            except Exception as e:
                st.error(f"Error loading orderbook walls: {str(e)}")
    
    # Tab 4: Orderbook Imbalance
    with tabs[3]:
        with st.spinner("Loading orderbook imbalance..."):
            try:
                imbalance_result = call_datasectors_api("/crypto/orderbook-imbalance")
                if imbalance_result and imbalance_result.get('data'):
                    imbalance_df = pd.DataFrame(imbalance_result['data'])
                    
                    st.markdown("#### Orderbook Imbalance Analysis")
                    st.dataframe(
                        imbalance_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            col: st.column_config.TextColumn(width='auto') 
                            for col in imbalance_df.columns
                        }
                    )
                    
                    if len(imbalance_df) > 0:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Imbalance", len(imbalance_df))
                        with col2:
                            st.info(f"Buy/Sell pressure analysis")
                else:
                    st.info("No orderbook imbalance data at this moment")
            except Exception as e:
                st.error(f"Error loading orderbook imbalance: {str(e)}")
    
    st.markdown("---")
    
    # Fetch history
    result = api_request("/history")
    
    if result and result.get('history'):
        history_data = result['history']
        df = pd.DataFrame(history_data)
        
        st.markdown("### 📈 Trading History Summary")
        
        # Statistics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Signals", len(df))
        
        with col2:
            buy_signals = len(df[df['action'].str.contains('BUY', case=False, na=False)])
            st.metric("Buy Signals", buy_signals)
        
        with col3:
            sell_signals = len(df[df['action'].str.contains('SELL', case=False, na=False)])
            st.metric("Sell Signals", sell_signals)
        
        with col4:
            hold_signals = len(df[df['action'].str.contains('HOLD', case=False, na=False)])
            st.metric("Hold Signals", hold_signals)
        
        st.markdown("---")
        
        # Recent signals table
        st.markdown("### 📋 Recent Signals")
        
        # Format the dataframe for display
        display_df = df.copy()
        
        # Format timestamp
        if 'timestamp' in display_df.columns:
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Format price
        if 'price' in display_df.columns:
            display_df['price'] = display_df['price'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
        
        # Select columns to display
        display_columns = ['timestamp', 'symbol', 'action', 'price']
        available_columns = [col for col in display_columns if col in display_df.columns]
        
        st.dataframe(
            display_df[available_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                'timestamp': st.column_config.TextColumn('Date & Time', width='medium'),
                'symbol': st.column_config.TextColumn('Symbol', width='medium'),
                'action': st.column_config.TextColumn('Signal', width='medium'),
                'price': st.column_config.TextColumn('Price', width='medium')
            }
        )
        
        # Action distribution chart
        st.markdown("### 📊 Signal Distribution")
        
        action_counts = df['action'].value_counts()
        
        import plotly.express as px
        fig = px.pie(
            values=action_counts.values,
            names=action_counts.index,
            title='Trading Signals Distribution',
            color_discrete_sequence=['#00E676', '#FF5252', '#90A4AE', '#FFC107']
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("📭 No trading history yet. Start analyzing symbols to build your history!")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; border-radius: 15px; color: white; margin-top: 20px;">
            <h3 style="margin: 0;">🚀 Get Started</h3>
            <p style="margin: 10px 0;">
                Use the <strong>Multi-Timeframe Analysis</strong> feature to analyze symbols 
                and generate trading signals. Your analysis history will appear here.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# API STATUS PAGE
# ============================================

def show_api_status():
    """Display API key quotas and statistics"""
    st.markdown("""
    <h1 style="color: white; margin-bottom: 30px;">⚙️ API Status Dashboard</h1>
    """, unsafe_allow_html=True)
    
    # Fetch API stats
    stats_result = api_request("/datasectors/stats")
    
    if stats_result:
        api_stats = stats_result
        
        # Summary metrics
        st.subheader("🔑 API Key Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Keys", api_stats.get('total_keys', 0))
        
        with col2:
            current_key = api_stats.get('current_key_number', 0)
            st.metric("Current Key", f"#{current_key}")
        
        with col3:
            failed_count = api_stats.get('failed_keys_count', 0)
            st.metric("Failed Keys", failed_count)
        
        with col4:
            status = "✅ Healthy" if failed_count == 0 else "⚠️ Warning"
            st.metric("Status", status)
        
        st.markdown("---")
        
        # Daily Request Usage Summary
        st.subheader("📊 Daily API Request Usage")
        
        keys = api_stats.get('keys', [])
        
        if keys:
            # Calculate total requests across all keys
            total_requests = sum(key.get('requests', 0) for key in keys)
            total_errors = sum(key.get('errors', 0) for key in keys)
            
            col_usage1, col_usage2, col_usage3, col_usage4 = st.columns(4)
            
            with col_usage1:
                st.metric("📈 Total Requests", total_requests)
            
            with col_usage2:
                st.metric("❌ Total Errors", total_errors)
            
            with col_usage3:
                success_count = total_requests - total_errors
                st.metric("✅ Successful", success_count)
            
            with col_usage4:
                success_rate = (success_count / total_requests * 100) if total_requests > 0 else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            # Request breakdown by key
            st.markdown("#### 🔑 Request Breakdown by Key")
            
            breakdown_data = []
            for key_info in keys:
                breakdown_data.append({
                    'Key': f"Key #{key_info.get('key_number')}",
                    'Requests': key_info.get('requests', 0),
                    'Errors': key_info.get('errors', 0),
                    'Success': key_info.get('requests', 0) - key_info.get('errors', 0),
                    'Status': '🟢 Current' if key_info.get('is_current') else '🔴 Failed' if key_info.get('is_failed') else '⚪ Standby',
                    'Success Rate': key_info.get('success_rate', 'N/A')
                })
            
            breakdown_df = pd.DataFrame(breakdown_data)
            st.dataframe(
                breakdown_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Key': st.column_config.TextColumn('API Key', width='small'),
                    'Requests': st.column_config.NumberColumn('Requests', width='small'),
                    'Errors': st.column_config.NumberColumn('Errors', width='small'),
                    'Success': st.column_config.NumberColumn('Success', width='small'),
                    'Status': st.column_config.TextColumn('Status', width='small'),
                    'Success Rate': st.column_config.TextColumn('Success Rate', width='small')
                }
            )
            
            st.markdown("---")
            
            # Daily usage chart
            if total_requests > 0:
                st.markdown("#### 📈 Daily Usage Trend")
                
                # Create pie chart for request distribution
                request_counts = [key.get('requests', 0) for key in keys]
                key_labels = [f"Key #{key.get('key_number')}" for key in keys]
                
                import plotly.express as px
                fig = px.pie(
                    values=request_counts,
                    names=key_labels,
                    title='API Request Distribution Across Keys',
                    color_discrete_sequence=['#667eea', '#764ba2', '#00E676', '#FF5252']
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Keys detailed breakdown
        st.subheader("📊 API Keys Details")
        
        if keys:
            for key_info in keys:
                # Key card
                col1, col2, col3 = st.columns([1, 3, 2])
                
                with col1:
                    # Status badge
                    if key_info.get('is_current'):
                        st.markdown("🟢 **CURRENT**")
                    elif key_info.get('is_failed'):
                        st.markdown("🔴 **FAILED**")
                    else:
                        st.markdown("⚪ **STANDBY**")
                
                with col2:
                    # Key info
                    st.markdown(f"**Key #{key_info.get('key_number')}**: `{key_info.get('key_preview', 'N/A')}`")
                    
                    # Stats
                    sub_col1, sub_col2, sub_col3 = st.columns(3)
                    with sub_col1:
                        st.write(f"📡 **Requests**: {key_info.get('requests', 0)}")
                    with sub_col2:
                        st.write(f"❌ **Errors**: {key_info.get('errors', 0)}")
                    with sub_col3:
                        st.write(f"📈 **Success**: {key_info.get('success_rate', 'N/A')}")
                
                with col3:
                    last_used = key_info.get('last_used')
                    if last_used:
                        st.write(f"Last used: `{last_used[:19]}`")
                    else:
                        st.write("Never used")
                
                st.divider()
        else:
            st.info("No API keys found")
        
        # Refresh button
        col_refresh = st.columns(1)[0]
        with col_refresh:
            st.info("💡 **Tip**: API stats update automatically when you perform analysis. Click refresh to manually update.")
            if st.button("🔄 Refresh API Stats", use_container_width=True):
                st.rerun()
    else:
        st.error("❌ Failed to fetch API statistics. Please ensure the backend is running.")

# ============================================
# SCANNER PAGE (NEW - Single Timeframe)
# ============================================

def show_scanner():
    """Display single timeframe market scanner"""
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Market Scanner</h1>
        <p>Analyze a symbol with detailed chart and indicators</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol = st.text_input("Symbol", value="BTCUSDT", placeholder="e.g., BTCUSDT, IDX:BBCA",
                               help="For crypto: BTCUSDT, ETHUSDT | For stocks: IDX:BBCA")
    with col2:
        market = st.selectbox("Market", ["crypto", "stock"])
    with col3:
        timeframe_options = {
            "1 Minute": "1m",
            "5 Minutes": "5m",
            "15 Minutes": "15m",
            "30 Minutes": "30m",
            "1 Hour": "1h",
            "4 Hours": "4h",
            "Daily": "D",
            "Weekly": "W",
            "Monthly": "M"
        }
        selected_tf = st.selectbox("Timeframe", list(timeframe_options.keys()), index=6)  # Default to Daily
        timeframe = timeframe_options[selected_tf]
    
    if st.button("🔬 Analyze with Chart", use_container_width=True, type="primary"):
        with st.spinner("Analyzing and generating chart..."):
            result = api_request(
                "/trading/signal",
                method="POST",
                data={
                    "symbol": symbol,
                    "market": market,
                    "timeframe": timeframe,
                    "indicators": ["ma", "rsi", "macd", "stoch", "bb", "atr", "volume"],
                    "max_candles": 1000
                }
            )
            
            if result:
                signal = result['signal']
                indicators = result['indicators']
                chart_data = result.get('chart_data', {})
                
                # ============================================
                # LAYER FILTERING RESULTS
                # ============================================
                st.markdown("### 🔍 Filter Analysis (4-Layer System)")
                
                layers = signal.get('layers', {})
                
                layer_cols = st.columns(4)
                
                for idx, layer_num in enumerate([1, 2, 3, 4]):
                    with layer_cols[idx]:
                        layer_key = f'layer{layer_num}'
                        layer_data = layers.get(layer_key, {})
                        
                        layer_name = layer_data.get('name', f'Layer {layer_num}')
                        passed = layer_data.get('passed', True)
                        reason = layer_data.get('reason', '')
                        
                        # Color based on pass/fail
                        color = "#00E676" if passed else "#FF5252"
                        status = "✅ PASS" if passed else "❌ REJECT"
                        
                        st.markdown(f"""
                        <div style="background: {color}; padding: 15px; border-radius: 10px; color: white; margin: 10px 0;">
                            <h4 style="margin: 0 0 8px 0; font-size: 14px; font-weight: bold;">{layer_name}</h4>
                            <p style="margin: 0 0 8px 0; font-weight: bold; font-size: 13px;">{status}</p>
                            <p style="margin: 0; font-size: 11px; line-height: 1.4; opacity: 0.95;">{reason}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Overall filter result
                overall_passed = layers.get('filter_chain_passed', True)
                filter_status = "✅ Passed All Filters" if overall_passed else "⚠️ Filtered (Some Layers Failed)"
                filter_color = "#00E676" if overall_passed else "#FF9800"
                
                st.markdown(f"""
                <div style="background: {filter_color}; padding: 15px; border-radius: 10px; color: white; margin: 15px 0;">
                    <h3 style="margin: 0;">{filter_status}</h3>
                    <p style="margin: 5px 0 0 0; font-size: 14px;">This signal has gone through all quality filters - proceed with caution on layer failures.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Detailed Layer Analysis
                st.markdown("### 📊 Detailed Layer Analysis")
                
                for layer_num in [1, 2, 3, 4]:
                    layer_key = f'layer{layer_num}'
                    layer_data = layers.get(layer_key, {})
                    
                    layer_name = layer_data.get('name', f'Layer {layer_num}')
                    passed = layer_data.get('passed', True)
                    reason = layer_data.get('reason', '')
                    
                    status_emoji = "✅" if passed else "❌"
                    status_text = "PASS" if passed else "REJECT"
                    
                    with st.expander(f"{status_emoji} **Layer {layer_num}: {layer_name}** — {status_text}", expanded=False):
                        st.markdown(f"**Status:** {status_emoji} {status_text}")
                        st.markdown(f"**Details:** {reason}")
                        
                        # Show layer-specific data
                        if layer_num == 1:
                            st.markdown("""
**Criteria:** Price structure analysis
- Trend direction (from MA)
- Market range size  
- Volatility percentage
- Support & Resistance levels

**Result:**
""")
                            # Display each field directly
                            trend = layer_data.get('trend', 'N/A')
                            range_pct = layer_data.get('range_percent', 'N/A')
                            vol_pct = layer_data.get('volatility_percent', 'N/A')
                            support = layer_data.get('support', 'N/A')
                            resistance = layer_data.get('resistance', 'N/A')
                            
                            st.write(f"🔸 **Trend:** {str(trend).upper()}")
                            st.write(f"📊 **Range:** {range_pct:.2f}%" if isinstance(range_pct, (int, float)) else f"📊 **Range:** {range_pct}")
                            st.write(f"📈 **Volatility:** {vol_pct:.2f}%" if isinstance(vol_pct, (int, float)) else f"📈 **Volatility:** {vol_pct}")
                            st.write(f"🛡️ **Support:** ${support:,.2f}" if isinstance(support, (int, float)) else f"🛡️ **Support:** {support}")
                            st.write(f"⚡ **Resistance:** ${resistance:,.2f}" if isinstance(resistance, (int, float)) else f"⚡ **Resistance:** {resistance}")
                        
                        elif layer_num == 2:
                            st.markdown("""
**Criteria:** Volume confirmation
- Volume ratio >= 0.8 (good volume)
- OR CMF > 0.1 (strong buying)

**Result:**
""")
                            vwap = layer_data.get('vwap', 'N/A')
                            cmf = layer_data.get('cmf', 'N/A')
                            cmf_str = layer_data.get('cmf_strength', 'N/A')
                            vol_ratio = layer_data.get('volume_ratio', 'N/A')
                            price_vs_vwap = layer_data.get('price_vs_vwap', 'N/A')
                            vwap_dist = layer_data.get('vwap_distance', 'N/A')
                            
                            st.write(f"💰 **VWAP:** ${vwap:,.2f}" if isinstance(vwap, (int, float)) else f"💰 **VWAP:** {vwap}")
                            st.write(f"🔄 **CMF:** {cmf:.3f} ({cmf_str})" if isinstance(cmf, (int, float)) else f"🔄 **CMF:** {cmf} ({cmf_str})")
                            vol_status = "✅" if (isinstance(vol_ratio, (int, float)) and vol_ratio >= 0.8) else "⚠️"
                            st.write(f"{vol_status} **Volume Ratio:** {vol_ratio:.2f}x" if isinstance(vol_ratio, (int, float)) else f"{vol_status} **Volume Ratio:** {vol_ratio}")
                            st.write(f"📍 **Position:** {price_vs_vwap} ({vwap_dist:.2f}%)" if isinstance(vwap_dist, (int, float)) else f"📍 **Position:** {price_vs_vwap}")
                        
                        elif layer_num == 3:
                            st.markdown("""
**Criteria:** Strong trend pre-filter
- Symbol must be in strong trend list
- OR trend strength > 1%

**Result:**
""")
                            is_trending = layer_data.get('is_trending', False)
                            trend_strength = layer_data.get('trend_strength', 'N/A')
                            
                            trend_status = "✅ YES" if is_trending else "❌ NO"
                            st.write(f"🚀 **Is Trending:** {trend_status}")
                            st.write(f"💪 **Trend Strength:** {trend_strength:.2f}%" if isinstance(trend_strength, (int, float)) else f"💪 **Trend Strength:** {trend_strength}")
                        
                        elif layer_num == 4:
                            st.markdown("""
**Criteria:** Confirmation (only if layers 2-3 pass)
- Must have buy walls OR sell walls
- Validates whale positioning

**Result:**
""")
                            if layer_data.get('reason', '').startswith('Skipped'):
                                st.info("⚠️ **Skipped:** Layer 4 was skipped because an earlier filter (Layer 2 or 3) rejected the symbol")
                            else:
                                buy_walls = layer_data.get('buy_walls', 'N/A')
                                sell_walls = layer_data.get('sell_walls', 'N/A')
                                
                                st.write(f"🟢 **Buy Walls:** {buy_walls}")
                                st.write(f"🔴 **Sell Walls:** {sell_walls}")
                
                st.markdown("---")
                
                # Debug: Show raw layer data (optional expander)
                with st.expander("🔧 Debug - Raw Layer Data"):
                    st.json(layers)
                
                st.markdown("---")
                
                # Signal Card
                signal_class = "buy-signal" if "BUY" in signal['action'] else "sell-signal" if "SELL" in signal['action'] else "hold-signal"
                
                st.markdown(f"""
                <div class="signal-card {signal_class}">
                    <h1 style="margin: 0; text-align: center;">{signal['action']}</h1>
                    <p style="font-size: 28px; margin: 15px 0; text-align: center;">
                        <strong>{symbol}</strong> @ ${signal['price']:,.4f}
                    </p>
                    <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 10px; margin-top: 15px;">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                            <div>
                                <p style="margin: 5px 0; font-size: 14px; opacity: 0.9;">Confidence</p>
                                <p style="margin: 0; font-size: 24px; font-weight: bold;">{signal['confidence']:.1f}%</p>
                            </div>
                            <div>
                                <p style="margin: 5px 0; font-size: 14px; opacity: 0.9;">RSI</p>
                                <p style="margin: 0; font-size: 24px; font-weight: bold;">{signal['rsi']:.1f}</p>
                            </div>
                        </div>
                        <p style="margin: 15px 0 5px 0; font-size: 14px; opacity: 0.9;">Recommendation:</p>
                        <p style="margin: 0; font-size: 16px; font-weight: 600;">{signal['recommendation']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Indicators Summary
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Moving Average", f"${signal['ma']:,.4f}", 
                             delta=f"{((signal['price']-signal['ma'])/signal['ma']*100):.2f}%")
                with col_b:
                    rsi_delta = "Overbought" if signal['rsi'] > 70 else "Oversold" if signal['rsi'] < 30 else "Neutral"
                    st.metric("RSI", f"{signal['rsi']:.1f}", delta=rsi_delta)
                with col_c:
                    macd_status = "Bullish" if signal['macd_histogram'] > 0 else "Bearish"
                    st.metric("MACD", f"{signal['macd_histogram']:.6f}", delta=macd_status)
                with col_d:
                    st.metric("Score Diff", signal.get('score_difference', signal['long_score'] - signal['short_score']))
                
                st.markdown("---")
                
                # Support & Resistance
                col_sr1, col_sr2, col_sr3, col_sr4 = st.columns(4)
                with col_sr1:
                    st.metric("Support", f"${signal.get('support', signal['price']):,.2f}")
                with col_sr2:
                    st.metric("Resistance", f"${signal.get('resistance', signal['price']):,.2f}")
                with col_sr3:
                    atr_val = signal.get('atr', 0)
                    st.metric("ATR", f"${atr_val:,.2f}")
                with col_sr4:
                    atr_pct = signal.get('atr_percent', 0)
                    st.metric("Volatility", f"{atr_pct:.2f}%")
                
                st.markdown("---")
                
                # Chart Section
                st.markdown("### 📊 Price Chart with Indicators")
                
                candles = chart_data.get('candles', [])
                
                if candles:
                    # Create chart using the helper function
                    fig = create_indicator_chart(candles, indicators, symbol, timeframe)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Chart data not available. Showing analysis only.")
                
                st.markdown("---")
                
                # Analysis Reasons
                st.markdown("### 📋 Detailed Analysis")
                
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("#### 🟢 Long (Buy) Reasons")
                    if signal['long_reasons']:
                        for reason in signal['long_reasons']:
                            st.markdown(f"- {reason}")
                    else:
                        st.info("No strong bullish signals detected")
                
                with col_right:
                    st.markdown("#### 🔴 Short (Sell) Reasons")
                    if signal['short_reasons']:
                        for reason in signal['short_reasons']:
                            st.markdown(f"- {reason}")
                    else:
                        st.info("No strong bearish signals detected")
                
                # Active recommendation
                st.markdown("---")
                st.markdown("### 💡 Current Recommendation")
                
                recommendation_color = "#00E676" if "BUY" in signal['action'] else "#FF5252" if "SELL" in signal['action'] else "#90A4AE"
                
                st.markdown(f"""
                <div style="background: {recommendation_color}; padding: 20px; border-radius: 10px; color: white;">
                    <h3 style="margin: 0 0 10px 0;">{signal['action']}</h3>
                    <p style="margin: 0; font-size: 16px; line-height: 1.6;">
                        {signal['recommendation']}
                    </p>
                    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.3);">
                        <p style="margin: 5px 0; font-weight: bold;">Key Reasons:</p>
                        <ul style="margin: 10px 0; padding-left: 20px;">
                            {"".join([f"<li>{reason}</li>" for reason in signal['active_reasons']])}
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Failed to fetch analysis. Please check if the symbol is correct and try again.")

# ============================================
# MULTI-TIMEFRAME SCANNER
# ============================================

def create_indicator_chart(candles, indicators, symbol, timeframe):
    """Create a 3-panel chart with price and indicators"""
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f'{symbol} - {timeframe}', 'RSI', 'MACD')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=list(range(len(candles))),
            open=[c['open'] for c in candles],
            high=[c['high'] for c in candles],
            low=[c['low'] for c in candles],
            close=[c['close'] for c in candles],
            name='Price',
            increasing_line_color='#00E676',
            decreasing_line_color='#FF5252'
        ),
        row=1, col=1
    )
    
    # Moving Average
    ma_data = indicators.get('ma', [])
    if ma_data:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(ma_data))),
                y=ma_data,
                name='MA(20)',
                line=dict(color='#FF9800', width=2)
            ),
            row=1, col=1
        )
    
    # RSI
    rsi_data = indicators.get('rsi', [])
    if rsi_data:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(rsi_data))),
                y=rsi_data,
                name='RSI',
                line=dict(color='#9C27B0', width=2)
            ),
            row=2, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="#FF5252", row=2, col=1, opacity=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color="#00E676", row=2, col=1, opacity=0.5)
        fig.add_hline(y=50, line_dash="dot", line_color="#9E9E9E", row=2, col=1, opacity=0.3)
    
    # MACD
    macd_indicator = indicators.get('macd', {})
    if macd_indicator:
        macd_line = macd_indicator.get('macd', [])
        signal_line = macd_indicator.get('signal', [])
        histogram = macd_indicator.get('histogram', [])
        
        if macd_line:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(macd_line))),
                    y=macd_line,
                    name='MACD',
                    line=dict(color='#2196F3', width=2)
                ),
                row=3, col=1
            )
        
        if signal_line:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(signal_line))),
                    y=signal_line,
                    name='Signal',
                    line=dict(color='#FF5252', width=2)
                ),
                row=3, col=1
            )
        
        if histogram:
            colors = ['#00E676' if h > 0 else '#FF5252' for h in histogram]
            fig.add_trace(
                go.Bar(
                    x=list(range(len(histogram))),
                    y=histogram,
                    name='Histogram',
                    marker_color=colors,
                    opacity=0.5
                ),
                row=3, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=700,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        hovermode='x unified',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig

# ============================================
# MULTI-TIMEFRAME SCANNER PAGE
# ============================================

def show_mtf_scanner():
    """Display multi-timeframe scanner"""
    st.markdown("""
    <div class="main-header">
        <h1>⏰ Multi-Timeframe Analysis</h1>
        <p>Analyze signals across multiple timeframes with detailed reasons</p>
    </div>
    """, unsafe_allow_html=True)
    
    symbol = st.text_input("Symbol", value="IDX:BBCA", key="mtf_symbol")
    market = st.selectbox("Market", ["stock", "crypto"], key="mtf_market")
    
    if st.button("🔬 Analyze All Timeframes", use_container_width=True, type="primary"):
        # Updated timeframes for better multi-timeframe analysis
        timeframes = ["5m", "15m", "1h", "4h", "D", "W"]
        timeframe_names = {
            "5m": "5 Minutes", 
            "15m": "15 Minutes", 
            "1h": "1 Hour", 
            "4h": "4 Hours", 
            "D": "Daily", 
            "W": "Weekly"
        }
        results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, tf in enumerate(timeframes):
            status_text.text(f"Analyzing {timeframe_names[tf]} timeframe...")
            
            result = api_request(
                "/trading/signal",
                method="POST",
                data={
                    "symbol": symbol,
                    "market": market,
                    "timeframe": tf,
                    "indicators": ["ma", "rsi", "macd", "stoch", "bb", "atr", "volume"],
                    "max_candles": 1000
                }
            )
            
            if result:
                results[tf] = result['signal']
            
            progress_bar.progress((idx + 1) / len(timeframes))
        
        status_text.text("✅ Analysis complete!")
        
        st.markdown("---")
        
        # Display results in cards
        st.markdown("### 📊 Multi-Timeframe Signals")
        
        cols = st.columns(len(timeframes))
        
        for idx, tf in enumerate(timeframes):
            with cols[idx]:
                if tf in results:
                    signal = results[tf]
                    action = signal['action']
                    
                    signal_class = "buy-signal" if "BUY" in action else "sell-signal" if "SELL" in action else "hold-signal"
                    
                    st.markdown(f"""
                    <div class="signal-card {signal_class}">
                        <h4 style="margin: 0; text-align: center;">{timeframe_names[tf]}</h4>
                        <h2 style="margin: 10px 0; text-align: center;">{action}</h2>
                        <p style="text-align: center; margin: 5px 0; font-size: 14px;">
                            {signal['confidence']:.0f}% confidence
                        </p>
                        <hr style="border-color: rgba(255,255,255,0.3); margin: 10px 0;">
                        <p style="margin: 5px 0; font-size: 12px;">Price: ${signal['price']:,.2f}</p>
                        <p style="margin: 5px 0; font-size: 12px;">Support: ${signal.get('support', signal['price']):,.2f}</p>
                        <p style="margin: 5px 0; font-size: 12px;">Resistance: ${signal.get('resistance', signal['price']):,.2f}</p>
                        <p style="margin: 5px 0; font-size: 12px;">RSI: {signal['rsi']:.1f} | ATR: {signal.get('atr_percent', 0):.2f}%</p>
                        <p style="margin: 5px 0; font-size: 12px;">Score: {signal['long_score']}/{signal['short_score']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed analysis for each timeframe
        st.markdown("### 📋 Detailed Analysis by Timeframe")
        
        for tf in timeframes:
            if tf in results:
                signal = results[tf]
                
                with st.expander(f"**{timeframe_names[tf]}** - {signal['action']} ({signal['confidence']:.0f}% confidence)", expanded=True):
                    
                    # Show filter results if available
                    layers = signal.get('layers', {})
                    if layers:
                        st.markdown("**🔍 Filter Status:**")
                        layer_cols = st.columns(4)
                        for idx, layer_num in enumerate([1, 2, 3, 4]):
                            with layer_cols[idx]:
                                layer_key = f'layer{layer_num}'
                                layer_data = layers.get(layer_key, {})
                                if layer_data:
                                    passed = layer_data.get('passed', True)
                                    status = "✅" if passed else "❌"
                                    st.write(f"{status} L{layer_num}: {layer_data.get('name', 'Unknown')}")
                        st.markdown("---")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Price", f"${signal['price']:,.2f}")
                    with col2:
                        st.metric("RSI", f"{signal['rsi']:.1f}")
                    with col3:
                        st.metric("Confidence", f"{signal['confidence']:.0f}%")
                    with col4:
                        st.metric("Score Diff", signal.get('score_difference', abs(signal['long_score'] - signal['short_score'])))
                    
                    # Support/Resistance/ATR
                    col_sr1, col_sr2, col_sr3, col_sr4 = st.columns(4)
                    with col_sr1:
                        st.metric("Support", f"${signal.get('support', signal['price']):,.2f}")
                    with col_sr2:
                        st.metric("Resistance", f"${signal.get('resistance', signal['price']):,.2f}")
                    with col_sr3:
                        st.metric("ATR", f"${signal.get('atr', 0):,.2f}")
                    with col_sr4:
                        st.metric("Volatility", f"{signal.get('atr_percent', 0):.2f}%")
                    
                    st.markdown("---")
                    
                    # Reasons side by side
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("#### 🟢 Long Reasons")
                        if signal.get('long_reasons'):
                            for reason in signal['long_reasons']:
                                st.markdown(f"✓ {reason}")
                        else:
                            st.info("No bullish signals")
                    
                    with col_b:
                        st.markdown("#### 🔴 Short Reasons")
                        if signal.get('short_reasons'):
                            for reason in signal['short_reasons']:
                                st.markdown(f"✓ {reason}")
                        else:
                            st.info("No bearish signals")
                    
                    # Recommendation
                    st.markdown("---")
                    recommendation_color = "#00E676" if "BUY" in signal['action'] else "#FF5252" if "SELL" in signal['action'] else "#90A4AE"
                    
                    st.markdown(f"""
                    <div style="background: {recommendation_color}; padding: 15px; border-radius: 8px; color: white;">
                        <strong>Recommendation:</strong> {signal['recommendation']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Overall consensus
        st.markdown("---")
        st.markdown("### 🎯 Overall Consensus")
        
        buy_count = sum(1 for tf in timeframes if tf in results and "BUY" in results[tf]['action'])
        sell_count = sum(1 for tf in timeframes if tf in results and "SELL" in results[tf]['action'])
        hold_count = sum(1 for tf in timeframes if tf in results and "HOLD" in results[tf]['action'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Buy Signals", f"{buy_count}/{len(timeframes)}")
        with col2:
            st.metric("Sell Signals", f"{sell_count}/{len(timeframes)}")
        with col3:
            st.metric("Hold Signals", f"{hold_count}/{len(timeframes)}")
        
        # Final recommendation
        if buy_count > sell_count and buy_count > hold_count:
            consensus = "BULLISH"
            consensus_color = "#00E676"
            message = "Majority of timeframes showing bullish signals. Consider long positions."
        elif sell_count > buy_count and sell_count > hold_count:
            consensus = "BEARISH"
            consensus_color = "#FF5252"
            message = "Majority of timeframes showing bearish signals. Consider short positions or exit longs."
        else:
            consensus = "MIXED"
            consensus_color = "#90A4AE"
            message = "Mixed signals across timeframes. Wait for clearer alignment before entering positions."
        
        st.markdown(f"""
        <div style="background: {consensus_color}; padding: 25px; border-radius: 15px; color: white; text-align: center; margin-top: 20px;">
            <h2 style="margin: 0 0 15px 0;">Multi-Timeframe Consensus: {consensus}</h2>
            <p style="margin: 0; font-size: 18px; line-height: 1.6;">
                {message}
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# HISTORY PAGE
# ============================================

def show_history():
    """Display trading history"""
    st.markdown("""
    <div class="main-header">
        <h1>📜 Trading History</h1>
        <p>View your past trading signals and decisions</p>
    </div>
    """, unsafe_allow_html=True)
    
    result = api_request("/history")
    
    if result and result.get('history'):
        df = pd.DataFrame(result['history'])
        
        # Display as table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Trades", len(df))
        with col2:
            buys = len(df[df['action'].str.contains('BUY', case=False)])
            st.metric("Buy Signals", buys)
        with col3:
            sells = len(df[df['action'].str.contains('SELL', case=False)])
            st.metric("Sell Signals", sells)
    else:
        st.info("No trading history yet. Start analyzing symbols to build your history!")

# ============================================
# ASK AI PAGE
# ============================================

def show_ask_ai():
    """Display AI assistant page"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Ask AI Trading Assistant</h1>
        <p>Get AI-powered insights and recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 💡 How can I help you today?
    
    Ask me anything about:
    - Market analysis and trends
    - Trading strategies
    - Risk management
    - Technical indicators
    - Symbol recommendations
    """)
    
    question = st.text_area("Your Question", placeholder="e.g., What's the best strategy for volatile markets?", height=100)
    
    if st.button("🚀 Ask AI", use_container_width=True):
        if question:
            with st.spinner("AI is thinking..."):
                # Simulated AI response - in production, integrate with LLM API
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 10px; color: white; margin: 20px 0;">
                    <h4>🤖 AI Response:</h4>
                    <p>Based on current market conditions and your question about volatile markets, 
                    here are my recommendations:</p>
                    <ul>
                        <li><strong>Use wider stop losses</strong> - In volatile markets, price can swing rapidly</li>
                        <li><strong>Reduce position sizes</strong> - Manage risk by trading smaller amounts</li>
                        <li><strong>Focus on RSI</strong> - Look for extreme overbought/oversold conditions</li>
                        <li><strong>Wait for confirmation</strong> - Don't rush into trades, wait for clear signals</li>
                        <li><strong>Use multiple timeframes</strong> - Get better context for your entries</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter a question")

# ============================================
# SIDEBAR NAVIGATION
# ============================================

def show_sidebar():
    """Display sidebar navigation"""
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.1); 
                    border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">👤 {st.session_state.user['username']}</h2>
            <p style="color: #ccc; margin: 5px 0;">{st.session_state.user['email']}</p>
            <span style="background: #667eea; padding: 5px 15px; border-radius: 20px; 
                         color: white; font-size: 12px; font-weight: bold;">
                {st.session_state.user['role'].upper()}
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🧭 Navigation")
        
        pages = {
            "📊 Dashboard": "Dashboard",
            "🔍 Scanner": "Scanner",
            "⏰ Multi-Timeframe": "MTF Scanner",
            "📜 History": "History",
            "⚙️ API Status": "API Status",
            "🤖 Ask AI": "Ask AI"
        }
        
        for icon_name, page in pages.items():
            if st.button(icon_name, use_container_width=True, key=f"nav_{page}"):
                st.session_state.current_page = page
                st.rerun()
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.token = None
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #ccc; font-size: 12px;">
            <p>Trading Decision System v1.0</p>
            <p>© 2025 All Rights Reserved</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application logic"""
    if not st.session_state.authenticated:
        show_login_page()
    else:
        show_sidebar()
        
        # Route to selected page
        page = st.session_state.current_page
        
        if page == "Dashboard":
            show_dashboard()
        elif page == "Scanner":
            show_scanner()
        elif page == "MTF Scanner":
            show_mtf_scanner()
        elif page == "History":
            show_history()
        elif page == "API Status":
            show_api_status()
        elif page == "Ask AI":
            show_ask_ai()

if __name__ == "__main__":
    main()