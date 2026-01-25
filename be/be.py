# main.py - Enhanced FastAPI Backend
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import sqlite3
import bcrypt
import jwt
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import talib
import json
import time
import yfinance as yf
import uuid
import re

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434/api"
OLLAMA_DEEPSEEK_MODEL = "deepseek-coder:6.7b"
OLLAMA_PLUTUS_MODEL = "plutus"

# Custom JSON encoder for pandas and numpy types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super().default(obj)

def safe_json_dumps(obj):
    """Safely dump object to JSON string"""
    return json.dumps(obj, cls=CustomJSONEncoder)

app = FastAPI(title="Crypto Scanner API Enhanced")
security = HTTPBearer()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440

# Alpha Vantage Configuration (Free tier: 5 calls/min, 500/day)
ALPHA_VANTAGE_API_KEY = "demo"  # Replace with your key from https://www.alphavantage.co
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# Polygon.io Configuration (Free tier: 5 API calls/min)
POLYGON_API_KEY = "YOUR_POLYGON_API_KEY"  # Get from https://polygon.io
POLYGON_BASE_URL = "https://api.polygon.io/v1"

# Rate limiting for APIs
LAST_API_CALL = {'alpha_vantage': 0, 'polygon': 0}

# Database Init
def init_db():
    conn = sqlite3.connect('crypto_scanner.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  role TEXT DEFAULT 'user',
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS scan_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  ticker TEXT NOT NULL,
                  timeframe TEXT NOT NULL,
                  signal TEXT,
                  confidence REAL,
                  price REAL,
                  analysis TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS multi_timeframe_scans
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  ticker TEXT NOT NULL,
                  timeframes TEXT NOT NULL,
                  overall_signal TEXT,
                  analysis TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS watchlist
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  ticker TEXT NOT NULL,
                  timeframe TEXT NOT NULL,
                  alert_price REAL,
                  added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  last_scan TEXT,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS trade_records
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  ticker TEXT NOT NULL,
                  entry_price REAL NOT NULL,
                  exit_price REAL,
                  signal TEXT NOT NULL,
                  quantity REAL,
                  profit_loss REAL,
                  status TEXT DEFAULT 'open',
                  entry_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  exit_date TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS ai_chat_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  conversation_id TEXT NOT NULL,
                  user_message TEXT NOT NULL,
                  ai_response TEXT NOT NULL,
                  model_used TEXT NOT NULL,
                  question_category TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()

init_db()

# Models
class UserRegister(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class AIMessageRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class AIMessageResponse(BaseModel):
    response: str
    model_used: str
    question_category: str
    conversation_id: str

class ScanRequest(BaseModel):
    ticker: str
    timeframe: str

class MultiTimeframeScanRequest(BaseModel):
    ticker: str
    timeframes: List[str]

class WatchlistItem(BaseModel):
    ticker: str
    timeframe: str
    alert_price: Optional[float] = None

class WatchlistDelete(BaseModel):
    ticker: str

# Auth Functions
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

# AI Chat Functions
def classify_question(message: str) -> str:
    """
    Classify question into category to determine which model to use
    Returns: 'trading', 'finance', or 'general'
    """
    message_lower = message.lower()
    
    # Trading & Finance keywords
    trading_keywords = ['buy', 'sell', 'trade', 'trading', 'short', 'long', 'chart', 'candle', 'support', 'resistance',
                       'price target', 'entry', 'exit', 'stop loss', 'take profit', 'rsi', 'macd', 'bollinger',
                       'fibonacci', 'trend', 'bullish', 'bearish', 'pump', 'dump', 'volume', 'liquidity', 'order',
                       'crypto', 'bitcoin', 'ethereum', 'btc', 'eth', 'altcoin', 'defi', 'nft', 'token', 'coin',
                       'hodl', 'dca', 'leverage', 'margin', 'futures', 'options', 'volatility', 'momentum']
    
    # Count matching keywords
    matching_count = sum(1 for keyword in trading_keywords if keyword in message_lower)
    
    if matching_count >= 2:
        return 'trading'
    elif matching_count == 1:
        return 'finance'
    else:
        return 'general'

def call_ollama_api(model: str, messages: List[Dict], stream: bool = False) -> Optional[str]:
    """Call Ollama API with streaming support"""
    try:
        url = f"{OLLAMA_BASE_URL}/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        if stream:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if 'message' in chunk:
                        full_response += chunk['message'].get('content', '')
            return full_response
        else:
            data = response.json()
            return data.get('message', {}).get('content', '')
    except Exception as e:
        print(f"Error calling Ollama: {str(e)}")
        return None

def get_ai_response(message: str, user_id: int, conversation_history: List[Dict]) -> tuple:
    """
    Get AI response with automatic model selection
    Returns: (response_text, model_used, question_category)
    """
    # Classify the question
    category = classify_question(message)
    
    # Select model based on category
    if category in ['trading', 'finance']:
        selected_model = OLLAMA_PLUTUS_MODEL
        system_prompt = """You are a professional cryptocurrency and finance advisor. 
        Provide technical analysis, trading insights, and financial advice. 
        Be precise, analytical, and consider market conditions."""
    else:
        selected_model = OLLAMA_DEEPSEEK_MODEL
        system_prompt = """You are a helpful AI assistant. 
        Provide clear, concise, and accurate answers to questions. 
        Be friendly and informative."""
    
    # Build messages with conversation history
    messages = []
    
    # Add system prompt
    messages.append({"role": "system", "content": system_prompt})
    
    # Add conversation history (last 5 messages for context)
    for msg in conversation_history[-10:]:
        messages.append({"role": msg.get('role', 'user'), "content": msg.get('content', '')})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    # Call Ollama
    response = call_ollama_api(selected_model, messages)
    
    if response:
        return response, selected_model, category
    else:
        return "Sorry, I couldn't generate a response. Please try again.", selected_model, category

def save_chat_history(user_id: int, conversation_id: str, user_message: str, 
                     ai_response: str, model_used: str, category: str):
    """Save chat history to database"""
    try:
        conn = sqlite3.connect('crypto_scanner.db')
        c = conn.cursor()
        c.execute('''INSERT INTO ai_chat_history 
                    (user_id, conversation_id, user_message, ai_response, model_used, question_category)
                    VALUES (?, ?, ?, ?, ?, ?)''',
                 (user_id, conversation_id, user_message, ai_response, model_used, category))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving chat history: {str(e)}")

def get_conversation_history(user_id: int, conversation_id: str, limit: int = 10) -> List[Dict]:
    """Get conversation history from database"""
    try:
        conn = sqlite3.connect('crypto_scanner.db')
        c = conn.cursor()
        c.execute('''SELECT user_message, ai_response FROM ai_chat_history 
                    WHERE user_id = ? AND conversation_id = ?
                    ORDER BY created_at DESC LIMIT ?''',
                 (user_id, conversation_id, limit))
        rows = c.fetchall()
        conn.close()
        
        # Convert to messages format (in reverse order)
        messages = []
        for user_msg, ai_resp in reversed(rows):
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": ai_resp})
        
        return messages
    except Exception as e:
        print(f"Error getting conversation history: {str(e)}")
        return []

# CryptoCompare API Functions (Free, No API Key Needed!)
def rate_limit_check(api_name: str, min_interval: float = 0.2):
    """Check rate limit (minimum 0.2s between calls = 5 calls/sec)"""
    global LAST_API_CALL
    current_time = time.time()
    last_call = LAST_API_CALL.get(api_name, 0)
    elapsed = current_time - last_call
    
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    
    LAST_API_CALL[api_name] = time.time()

def fetch_alpha_vantage_intraday(symbol: str, interval: str = "5min", limit: int = 1000) -> Optional[pd.DataFrame]:
    """
    Fetch intraday OHLC data dari Alpha Vantage
    Supports: 1min, 5min, 15min, 30min, 60min
    Returns: DataFrame dengan columns [timestamp, open, high, low, close, volume]
    """
    try:
        rate_limit_check('alpha_vantage', min_interval=0.2)
        
        # Normalize interval
        valid_intraday = ['1min', '5min', '15min', '30min', '60min']
        if interval not in valid_intraday:
            interval = '5min'
        
        # Format symbol untuk Alpha Vantage
        ticker = symbol.split('.')[0]  # Remove .JK suffix jika ada
        
        params = {
            'function': f'TIME_SERIES_INTRADAY',
            'symbol': ticker,
            'interval': interval,
            'apikey': ALPHA_VANTAGE_API_KEY,
            'outputsize': 'full'  # Get max data (up to 1000 candles)
        }
        
        print(f"🔄 Fetching {ticker} intraday {interval} from Alpha Vantage...")
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=15)
        
        if response.status_code != 200:
            print(f"❌ Alpha Vantage HTTP {response.status_code}: {response.text[:200]}")
            return None
        
        data = response.json()
        
        # Check untuk error atau API call limit
        if 'Error Message' in data:
            print(f"❌ Alpha Vantage Error: {data['Error Message']}")
            return None
        
        if 'Note' in data:  # Rate limit
            print(f"⚠️ Alpha Vantage Rate Limit: {data['Note']}")
            return None
        
        # Extract time series
        time_series_key = None
        for key in data.keys():
            if 'Time Series' in key:
                time_series_key = key
                break
        
        if not time_series_key or not data[time_series_key]:
            print(f"❌ No data available for {ticker}")
            return None
        
        ts = data[time_series_key]
        
        # Convert ke DataFrame
        df_data = []
        for timestamp, ohlc in ts.items():
            df_data.append({
                'timestamp': pd.to_datetime(timestamp),
                'open': float(ohlc.get('1. open', 0)),
                'high': float(ohlc.get('2. high', 0)),
                'low': float(ohlc.get('3. low', 0)),
                'close': float(ohlc.get('4. close', 0)),
                'volume': float(ohlc.get('5. volume', 0))
            })
        
        if not df_data:
            print(f"❌ No valid OHLC data for {ticker}")
            return None
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ Fetched {len(df)} {interval} candles from Alpha Vantage ({ticker})")
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
    except requests.exceptions.Timeout:
        print(f"❌ Alpha Vantage timeout")
        return None
    except Exception as e:
        print(f"❌ Alpha Vantage exception: {e}")
        return None

def fetch_alpha_vantage_daily(symbol: str, limit: int = 1000) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLC data dari Alpha Vantage
    Returns: DataFrame dengan columns [timestamp, open, high, low, close, volume]
    """
    try:
        rate_limit_check('alpha_vantage', min_interval=0.2)
        
        ticker = symbol.split('.')[0]
        
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': ticker,
            'apikey': ALPHA_VANTAGE_API_KEY,
            'outputsize': 'full'  # Dapat up to 20 years historical data
        }
        
        print(f"🔄 Fetching {ticker} daily from Alpha Vantage...")
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=15)
        
        if response.status_code != 200:
            print(f"❌ Alpha Vantage HTTP {response.status_code}")
            return None
        
        data = response.json()
        
        if 'Error Message' in data or 'Note' in data:
            print(f"❌ Alpha Vantage Error: {data.get('Error Message', data.get('Note'))}")
            return None
        
        ts = data.get('Time Series (Daily)', {})
        if not ts:
            print(f"❌ No daily data for {ticker}")
            return None
        
        df_data = []
        for date, ohlc in ts.items():
            df_data.append({
                'timestamp': pd.to_datetime(date),
                'open': float(ohlc.get('1. open', 0)),
                'high': float(ohlc.get('2. high', 0)),
                'low': float(ohlc.get('3. low', 0)),
                'close': float(ohlc.get('4. close', 0)),
                'volume': float(ohlc.get('5. volume', 0))
            })
        
        if not df_data:
            return None
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Limit to requested amount
        df = df.tail(limit)
        
        print(f"✅ Fetched {len(df)} daily candles from Alpha Vantage ({ticker})")
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        print(f"❌ Alpha Vantage daily exception: {e}")
        return None

def fetch_polygon_intraday(symbol: str, timespan: str = "minute", multiplier: int = 5, limit: int = 1000) -> Optional[pd.DataFrame]:
    """
    Fetch intraday OHLC dari Polygon.io (Fallback)
    Supports: minute (1min), 5min, 15min, 30min, hour
    """
    try:
        rate_limit_check('polygon', min_interval=0.2)
        
        # Normalize timespan
        timespan_map = {
            '1min': ('minute', 1),
            '5min': ('minute', 5),
            '15min': ('minute', 15),
            '30min': ('minute', 30),
            '1h': ('hour', 1),
            'hour': ('hour', 1),
            '1d': ('day', 1),
        }
        
        if timespan in timespan_map:
            timespan_norm, multiplier_val = timespan_map[timespan]
        else:
            timespan_norm, multiplier_val = 'minute', 5
        
        ticker = symbol.split('.')[0]
        
        # Polygon endpoint: /v1/open-close/{stocksTicker}/{date}
        # Untuk intraday: /v1/bars/{stocksTicker}
        
        # Use Aggs API untuk aggregate bars
        from_date = (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        to_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        url = f"{POLYGON_BASE_URL}/bars/{ticker}"
        
        params = {
            'timespan': timespan_norm,
            'multiplier': multiplier_val,
            'from': from_date,
            'to': to_date,
            'sort': 'asc',
            'limit': 1000,
            'apiKey': POLYGON_API_KEY
        }
        
        print(f"🔄 Fetching {ticker} {timespan} from Polygon.io...")
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code != 200:
            print(f"❌ Polygon.io HTTP {response.status_code}")
            return None
        
        data = response.json()
        
        if 'results' not in data or not data['results']:
            print(f"❌ No data from Polygon.io for {ticker}")
            return None
        
        df_data = []
        for bar in data['results']:
            df_data.append({
                'timestamp': pd.to_datetime(bar['t'], unit='ms'),
                'open': float(bar.get('o', 0)),
                'high': float(bar.get('h', 0)),
                'low': float(bar.get('l', 0)),
                'close': float(bar.get('c', 0)),
                'volume': float(bar.get('v', 0))
            })
        
        if not df_data:
            return None
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ Fetched {len(df)} {timespan} candles from Polygon.io ({ticker})")
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        print(f"❌ Polygon.io exception: {e}")
        return None

def fetch_polygon_daily(symbol: str, limit: int = 1000) -> Optional[pd.DataFrame]:
    """Fetch daily OHLC dari Polygon.io (Fallback)"""
    try:
        rate_limit_check('polygon', min_interval=0.2)
        
        ticker = symbol.split('.')[0]
        from_date = (pd.Timestamp.now() - pd.Timedelta(days=365*5)).strftime('%Y-%m-%d')  # 5 years
        to_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        url = f"{POLYGON_BASE_URL}/bars/{ticker}"
        
        params = {
            'timespan': 'day',
            'from': from_date,
            'to': to_date,
            'sort': 'asc',
            'limit': limit,
            'apiKey': POLYGON_API_KEY
        }
        
        print(f"🔄 Fetching {ticker} daily from Polygon.io...")
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code != 200:
            print(f"❌ Polygon.io HTTP {response.status_code}")
            return None
        
        data = response.json()
        
        if 'results' not in data or not data['results']:
            print(f"❌ No daily data from Polygon.io")
            return None
        
        df_data = []
        for bar in data['results']:
            df_data.append({
                'timestamp': pd.to_datetime(bar['t'], unit='ms'),
                'open': float(bar.get('o', 0)),
                'high': float(bar.get('h', 0)),
                'low': float(bar.get('l', 0)),
                'close': float(bar.get('c', 0)),
                'volume': float(bar.get('v', 0))
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ Fetched {len(df)} daily candles from Polygon.io ({ticker})")
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        print(f"❌ Polygon.io daily exception: {e}")
        return None

def get_binance_klines(symbol: str, interval: str, limit: int = 1000) -> Optional[pd.DataFrame]:
    """
    Fetch cryptocurrency OHLCV data from CryptoCompare API (Free, No Auth Required)
    Used for crypto trading pairs like BTCUSDT, ETHUSDT, etc.
    
    Parameters:
    - symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
    - interval: Timeframe ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w')
    - limit: Number of candles to fetch (max 2000 from CryptoCompare)
    
    Returns: DataFrame with columns [timestamp, open, high, low, close, volume]
    """
    
    # Extract base asset from symbol (BTCUSDT -> BTC)
    base_asset = symbol.replace("USDT", "").replace("BUSD", "").replace("USDC", "")
    
    # Interval mapping untuk CryptoCompare API
    interval_map = {
        "1m": ("histominute", 1),
        "5m": ("histominute", 5),
        "15m": ("histominute", 15),
        "30m": ("histominute", 30),
        "1h": ("histohour", 1),
        "4h": ("histohour", 4),
        "1d": ("histoday", 1),
        "1w": ("histoday", 7)
    }
    
    endpoint, aggregate = interval_map.get(interval, ("histohour", 1))
    
    try:
        # CryptoCompare API (Free, No API Key Required!)
        url = f"https://min-api.cryptocompare.com/data/v2/{endpoint}"
        params = {
            "fsym": base_asset,
            "tsym": "USD",
            "limit": min(limit, 2000),  # CryptoCompare max is 2000
            "aggregate": aggregate
        }
        
        print(f"🔄 Fetching {base_asset} {interval} from CryptoCompare...")
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('Response') == 'Success' and 'Data' in result and 'Data' in result['Data']:
                data = result['Data']['Data']
                
                df_data = []
                for candle in data:
                    df_data.append({
                        'timestamp': pd.to_datetime(candle['time'], unit='s'),
                        'open': float(candle['open']),
                        'high': float(candle['high']),
                        'low': float(candle['low']),
                        'close': float(candle['close']),
                        'volume': float(candle['volumeto'])  # Volume in quote currency
                    })
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    
                    print(f"✅ Fetched {len(df)} {interval} candles from CryptoCompare ({base_asset})")
                    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            else:
                error_msg = result.get('Message', 'Unknown error')
                print(f"❌ CryptoCompare API error: {error_msg}")
                return None
        
        elif response.status_code == 429:
            print(f"⚠️ CryptoCompare rate limit hit - waiting...")
            time.sleep(5)
            return None
        else:
            print(f"❌ CryptoCompare HTTP error: {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"❌ CryptoCompare timeout")
        return None
    except Exception as e:
        print(f"❌ CryptoCompare exception: {e}")
        return None

def fetch_stock_data_yfinance(symbol: str, interval: str = '1d', limit: int = 1000) -> Optional[pd.DataFrame]:
    """
    Fallback: Fetch Indonesia stock data dari yfinance
    Gratis, no API key needed, reliable untuk stocks
    """
    try:
        ticker_str = f"{symbol}.JK"  # Add .JK suffix untuk Indonesia stocks
        
        # Map intervals ke yfinance format
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '1d': '1d',
            '1w': '1wk',
            '1mo': '1mo'
        }
        
        yf_interval = interval_map.get(interval, '1d')
        
        # Determine period based on interval
        if yf_interval in ['1m', '5m', '15m', '30m', '1h']:
            period = '5d'  # 5 hari untuk intraday
        elif yf_interval == '1d':
            period = '1y'  # 1 tahun untuk daily
        else:
            period = '5y'  # 5 tahun untuk weekly/monthly
        
        print(f"🔄 Fetching {symbol} {interval} from yfinance (free fallback)...")
        ticker = yf.Ticker(ticker_str)
        hist = ticker.history(period=period, interval=yf_interval)
        
        if hist is None or len(hist) == 0:
            print(f"❌ No data from yfinance for {symbol}")
            return None
        
        print(f"   Raw data shape: {hist.shape}, columns: {list(hist.columns)}")
        
        # Reset index
        hist = hist.reset_index()
        
        # Debug: print first few rows
        print(f"   First row: {hist.iloc[0].to_dict() if len(hist) > 0 else 'empty'}")
        
        # Standardize column names (handle different cases)
        hist.columns = hist.columns.str.lower()
        
        # Map different column names
        column_mapping = {
            'date': 'timestamp',
            'datetime': 'timestamp',
        }
        hist = hist.rename(columns=column_mapping)
        
        # Keep only OHLCV columns (case insensitive)
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in required_cols if col in hist.columns]
        
        if len(available_cols) < 5:
            print(f"❌ Missing required columns. Available: {list(hist.columns)}")
            return None
        
        df = hist[available_cols].copy()
        
        # Ensure column names are correct
        if 'timestamp' not in df.columns and len(df.columns) > 0:
            df.columns = required_cols[:len(df.columns)]
        
        # Remove rows dengan NaN dalam OHLC (tapi keep volume NaN)
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Fill volume NaN dengan 0
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
        
        # Convert to proper types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any remaining NaN rows
        df = df.dropna()
        
        if len(df) == 0:
            print(f"❌ No valid OHLC data after cleaning")
            return None
        
        print(f"✅ Fetched {len(df)} {interval} candles from yfinance ({symbol})")
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        import traceback
        print(f"❌ yfinance exception: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return None

def fetch_stock_data(symbol: str, interval: str = '1d', limit: int = 1000) -> Optional[pd.DataFrame]:
    """
    Fetch OHLC data dengan triple fallback logic:
    1. Coba Alpha Vantage (memerlukan valid API key)
    2. Jika gagal → Polygon.io (memerlukan valid API key)
    3. Jika gagal → yfinance (GRATIS, no API key needed) ✅
    4. Jika semua gagal → Return None
    """
    print(f"📊 Fetching {symbol} with interval {interval}...")
    
    # Determine jika daily atau intraday
    is_daily = interval in ['1d', 'daily']
    
    if is_daily:
        # Daily: coba Alpha Vantage terlebih dahulu
        df = fetch_alpha_vantage_daily(symbol, limit)
        if df is not None and len(df) > 0:
            return df
        
        # Fallback ke Polygon.io
        print(f"⚠️ Alpha Vantage failed, trying Polygon.io...")
        df = fetch_polygon_daily(symbol, limit)
        if df is not None and len(df) > 0:
            return df
        
        # Last fallback: yfinance (GRATIS, reliable)
        print(f"⚠️ Polygon.io failed, trying yfinance (free)...")
        df = fetch_stock_data_yfinance(symbol, interval, limit)
        if df is not None and len(df) > 0:
            return df
    else:
        # Intraday: coba Alpha Vantage terlebih dahulu
        df = fetch_alpha_vantage_intraday(symbol, interval, limit)
        if df is not None and len(df) > 0:
            return df
        
        # Fallback ke Polygon.io
        print(f"⚠️ Alpha Vantage failed, trying Polygon.io...")
        df = fetch_polygon_intraday(symbol, interval, limit)
        if df is not None and len(df) > 0:
            return df
        
        # Last fallback: yfinance
        print(f"⚠️ Polygon.io failed, trying yfinance (free)...")
        df = fetch_stock_data_yfinance(symbol, interval, limit)
        if df is not None and len(df) > 0:
            return df
    
    print(f"❌ Failed to fetch data for {symbol} from all APIs")
    return None

def get_available_symbols():
    """Get popular cryptocurrencies (predefined list)"""
    
    # Top 100+ cryptocurrencies by market cap
    popular_coins = [
        "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "MATIC",
        "LINK", "UNI", "ATOM", "LTC", "ETC", "XLM", "ALGO", "VET", "ICP", "FIL",
        "TRX", "NEAR", "APT", "ARB", "OP", "SHIB", "DAI", "WBTC", "TON", "BCH",
        "LEO", "OKB", "HBAR", "QNT", "CRO", "LDO", "MNT", "IMX", "AAVE", "GRT",
        "INJ", "MKR", "RUNE", "STX", "SNX", "FTM", "SAND", "MANA", "AXS", "CHZ",
        "THETA", "EOS", "FLOW", "EGLD", "XTZ", "BSV", "KAVA", "ZEC", "DASH",
        "WAVES", "NEO", "IOTA", "KCS", "BAT", "ZIL", "ENJ", "COMP", "YFI",
        "UMA", "ZRX", "BAL", "CRV", "SUSHI", "1INCH", "LUNA", "FTT"
    ]
    
    symbols = []
    for coin in popular_coins:
        symbols.append({
            "symbol": f"{coin}USDT",
            "exchange": "CryptoCompare",
            "status": "TRADING",
            "baseAsset": coin,
            "quoteAsset": "USDT",
            "name": coin
        })
    
    print(f"✅ Loaded {len(symbols)} popular cryptocurrencies")
    return symbols

def get_symbol_info(symbol: str):
    """Get current price info from CryptoCompare"""
    
    base_asset = symbol.replace("USDT", "").replace("BUSD", "")
    
    try:
        url = "https://min-api.cryptocompare.com/data/pricemultifull"
        params = {
            "fsyms": base_asset,
            "tsyms": "USD"
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'RAW' in data and base_asset in data['RAW'] and 'USD' in data['RAW'][base_asset]:
                info = data['RAW'][base_asset]['USD']
                return {
                    "symbol": symbol,
                    "price": info.get('PRICE'),
                    "change_24h": info.get('CHANGEPCT24HOUR'),
                    "volume_24h": info.get('VOLUME24HOURTO'),
                    "source": "CryptoCompare"
                }
    except Exception as e:
        print(f"❌ CryptoCompare price error: {e}")
    
    return None

# Technical Analysis Functions
def calculate_mas(df):
    """Calculate Moving Averages"""
    df['MA12'] = talib.SMA(df['close'], timeperiod=12)
    df['MA26'] = talib.SMA(df['close'], timeperiod=26)
    df['MA50'] = talib.SMA(df['close'], timeperiod=50)
    df['MA200'] = talib.SMA(df['close'], timeperiod=200)
    df['EMA9'] = talib.EMA(df['close'], timeperiod=9)
    df['EMA21'] = talib.EMA(df['close'], timeperiod=21)
    return df

def calculate_bollinger_bands(df):
    """Calculate Bollinger Bands"""
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
    df['BB_upper'] = upper
    df['BB_middle'] = middle
    df['BB_lower'] = lower
    return df

def calculate_rsi(df):
    """Calculate RSI"""
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    return df

def calculate_macd(df):
    """Calculate MACD"""
    macd, signal, hist = talib.MACD(df['close'])
    df['MACD'] = macd
    df['MACD_signal'] = signal
    df['MACD_hist'] = hist
    return df

def calculate_stochastic(df):
    """Calculate Stochastic Oscillator"""
    slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'])
    df['STOCH_K'] = slowk
    df['STOCH_D'] = slowd
    return df

def calculate_smi(df, length_k=10, length_d=3, length_ema=3):
    """
    Calculate Stochastic Momentum Index (SMI)
    Based on TradingView Pine Script implementation
    
    Parameters:
    - length_k: %K Length (default 10)
    - length_d: %D Length (default 3)
    - length_ema: EMA Length (default 3)
    """
    
    def ema_ema(series, length):
        """Double EMA calculation"""
        ema1 = series.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema2
    
    # Calculate highest high and lowest low over length_k period
    highest_high = df['high'].rolling(window=length_k).max()
    lowest_low = df['low'].rolling(window=length_k).min()
    
    # Calculate range
    highest_lowest_range = highest_high - lowest_low
    
    # Calculate relative range
    relative_range = df['close'] - (highest_high + lowest_low) / 2
    
    # Calculate SMI using double EMA
    smi_numerator = ema_ema(relative_range, length_d)
    smi_denominator = ema_ema(highest_lowest_range, length_d)
    
    # Avoid division by zero
    smi = 200 * (smi_numerator / smi_denominator.replace(0, np.nan))
    
    # Calculate SMI-based EMA
    smi_ema = smi.ewm(span=length_ema, adjust=False).mean()
    
    df['SMI'] = smi
    df['SMI_EMA'] = smi_ema
    
    return df

# ============================================================================
# SMART MONEY CONCEPTS (SMC) FUNCTIONS
# ============================================================================

def detect_swing_structure(df, size=50):
    """
    Detect swing structure (pivots) - simplified but more reliable
    Returns recent significant highs and lows
    """
    if len(df) < size:
        return [], []
    
    recent_df = df.tail(size)
    swing_highs = []
    swing_lows = []
    
    # Find local highs and lows in recent data
    highs = recent_df['high'].values
    lows = recent_df['low'].values
    
    # Detect peaks (highs surrounded by lower values)
    for i in range(1, len(highs)-1):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            idx = len(df) - len(highs) + i
            swing_highs.append((idx, float(highs[i])))
    
    # Detect troughs (lows surrounded by higher values)
    for i in range(1, len(lows)-1):
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            idx = len(df) - len(lows) + i
            swing_lows.append((idx, float(lows[i])))
    
    return swing_highs[-3:], swing_lows[-3:]  # Return last 3 swings

def detect_break_of_structure(df, swing_highs, swing_lows, size=5):
    """
    Detect Break of Structure (BOS) - when price breaks previous swing level
    """
    structures = {
        'bullish_bos': [],
        'bearish_bos': [],
        'bullish_choch': [],
        'bearish_choch': []
    }
    
    if len(df) < size or (not swing_highs and not swing_lows):
        return structures
    
    current_price = df['close'].iloc[-1]
    recent_close = df['close'].tail(5)
    
    # Bullish BOS: Price breaks above previous swing high
    if swing_highs:
        prev_high = max([h[1] for h in swing_highs])
        if current_price > prev_high and recent_close.min() < prev_high:
            structures['bullish_bos'].append({
                'level': float(prev_high),
                'index': len(df) - 1,
                'strength': round((current_price - prev_high) / prev_high * 100, 2)
            })
    
    # Bearish BOS: Price breaks below previous swing low
    if swing_lows:
        prev_low = min([l[1] for l in swing_lows])
        if current_price < prev_low and recent_close.max() > prev_low:
            structures['bearish_bos'].append({
                'level': float(prev_low),
                'index': len(df) - 1,
                'strength': round((prev_low - current_price) / prev_low * 100, 2)
            })
    
    return structures

def detect_order_blocks(df, swing_highs, swing_lows, size=5):
    """
    Detect order blocks - areas of strong price action used for liquidity
    
    Returns list of order blocks with high/low levels
    """
    order_blocks = {
        'bullish': [],
        'bearish': []
    }
    
    if len(df) < size or not swing_highs or not swing_lows:
        return order_blocks
    
    # Analyze candles between swings for order blocks
    for i in range(size, len(df)):
        # Look for bullish order blocks (strong upward candles)
        body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
        total_size = df['high'].iloc[i] - df['low'].iloc[i]
        
        if total_size > 0:
            body_ratio = body_size / total_size
            
            # Bullish candle with large body (80%+ of total size)
            if df['close'].iloc[i] > df['open'].iloc[i] and body_ratio > 0.8:
                order_blocks['bullish'].append({
                    'high': float(df['high'].iloc[i]),
                    'low': float(df['low'].iloc[i]),
                    'time': df['timestamp'].iloc[i],
                    'index': i,
                    'strength': body_ratio
                })
            
            # Bearish candle with large body (80%+ of total size)
            if df['close'].iloc[i] < df['open'].iloc[i] and body_ratio > 0.8:
                order_blocks['bearish'].append({
                    'high': float(df['high'].iloc[i]),
                    'low': float(df['low'].iloc[i]),
                    'time': df['timestamp'].iloc[i],
                    'index': i,
                    'strength': body_ratio
                })
    
    # Keep only recent order blocks
    order_blocks['bullish'] = order_blocks['bullish'][-5:]
    order_blocks['bearish'] = order_blocks['bearish'][-5:]
    
    return order_blocks

def detect_fair_value_gaps(df, threshold_percent=0.1):
    """
    Detect Fair Value Gaps (FVG) - gaps in price action
    Bullish FVG: Low of candle 2 > High of candle 0
    Bearish FVG: High of candle 2 < Low of candle 0
    """
    fvgs = {
        'bullish': [],
        'bearish': []
    }
    
    if len(df) < 3:
        return fvgs
    
    # Simple gap detection - check recent candles
    for i in range(2, min(len(df), 100)):  # Check last 100 candles
        candle_0_high = df['high'].iloc[i-2]
        candle_0_low = df['low'].iloc[i-2]
        candle_2_high = df['high'].iloc[i]
        candle_2_low = df['low'].iloc[i]
        
        # Bullish FVG: gap between candle 0 and candle 2
        if candle_2_low > candle_0_high:
            gap_size = candle_2_low - candle_0_high
            gap_percent = (gap_size / candle_0_high) * 100
            if gap_percent > threshold_percent:
                fvgs['bullish'].append(float(candle_0_high))
        
        # Bearish FVG: gap between candle 0 and candle 2
        if candle_2_high < candle_0_low:
            gap_size = candle_0_low - candle_2_high
            gap_percent = (gap_size / candle_0_low) * 100
            if gap_percent > threshold_percent:
                fvgs['bearish'].append(float(candle_0_low))
    
    # Remove duplicates and limit to recent
    fvgs['bullish'] = list(set(fvgs['bullish']))[-3:]
    fvgs['bearish'] = list(set(fvgs['bearish']))[-3:]
    
    return fvgs

def detect_equal_highs_lows(df, length=5, threshold=0.02):
    """
    Detect Equal Highs and Lows - price levels that match multiple times
    """
    equal_levels = {
        'equal_highs': [],
        'equal_lows': []
    }
    
    if len(df) < length:
        return equal_levels
    
    # Get recent highs and lows
    recent_highs = df['high'].tail(length * 2).values
    recent_lows = df['low'].tail(length * 2).values
    
    # Check for equal highs
    current_high = df['high'].iloc[-1]
    for h in recent_highs[:-1]:
        if abs(h - current_high) / current_high < threshold:
            equal_levels['equal_highs'].append(float(current_high))
            break
    
    # Check for equal lows
    current_low = df['low'].iloc[-1]
    for l in recent_lows[:-1]:
        if abs(l - current_low) / current_low < threshold:
            equal_levels['equal_lows'].append(float(current_low))
            break
    
    return equal_levels

def detect_premium_discount_zones(df):
    """
    Detect Premium and Discount Zones based on swing structure
    
    Premium Zone: Above the last swing high
    Discount Zone: Below the last swing low
    Equilibrium: Between the last swing high and low
    """
    if len(df) < 50:
        return {}
    
    # Find recent swing high and low
    recent_high = df['high'].tail(50).max()
    recent_low = df['low'].tail(50).min()
    
    equilibrium = (recent_high + recent_low) / 2
    current_price = df['close'].iloc[-1]
    
    zone_type = 'equilibrium'
    if current_price > equilibrium:
        zone_type = 'premium'
    elif current_price < equilibrium:
        zone_type = 'discount'
    
    return {
        'premium_top': float(recent_high),
        'premium_bottom': float(equilibrium),
        'equilibrium': float(equilibrium),
        'discount_top': float(equilibrium),
        'discount_bottom': float(recent_low),
        'current_zone': zone_type,
        'current_price': float(current_price)
    }

def calculate_smart_money_concepts(df):
    """
    Calculate all Smart Money Concepts indicators
    """
    swing_size = 50  # Swing detection window
    
    # Detect structures
    swing_highs, swing_lows = detect_swing_structure(df, size=swing_size)
    structures = detect_break_of_structure(df, swing_highs, swing_lows)
    order_blocks = detect_order_blocks(df, swing_highs, swing_lows)
    fvgs = detect_fair_value_gaps(df)
    equal_levels = detect_equal_highs_lows(df, length=3)
    zones = detect_premium_discount_zones(df)
    
    # Calculate SMC strength score (0-100)
    smc_signals = 0
    
    # BOS signals
    if structures['bullish_bos']:
        smc_signals += 2
    if structures['bearish_bos']:
        smc_signals -= 2
    
    # Order block signals
    if order_blocks['bullish'] and len(order_blocks['bullish']) > 0:
        smc_signals += 1
    if order_blocks['bearish'] and len(order_blocks['bearish']) > 0:
        smc_signals -= 1
    
    # FVG signals
    if fvgs['bullish']:
        smc_signals += 1
    if fvgs['bearish']:
        smc_signals -= 1
    
    # Zone signals
    if zones.get('current_zone') == 'discount':
        smc_signals += 1
    elif zones.get('current_zone') == 'premium':
        smc_signals -= 1
    
    smc_strength = max(0, min(100, 50 + (smc_signals * 12.5)))
    
    return {
        'swing_structure': {
            'swing_highs': swing_highs,
            'swing_lows': swing_lows
        },
        'break_of_structure': structures,
        'order_blocks': order_blocks,
        'fair_value_gaps': fvgs,
        'equal_levels': equal_levels,
        'premium_discount_zones': zones,
        'smc_signal_strength': round(smc_strength, 2),
        'smc_bias': 'BULLISH' if smc_signals > 0 else ('BEARISH' if smc_signals < 0 else 'NEUTRAL'),
        'smc_signal_count': smc_signals
    }

def calculate_atr(df):
    """Calculate Average True Range"""
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    return df

def calculate_adx(df):
    """Calculate Average Directional Index"""
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['PLUS_DI'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['MINUS_DI'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    return df

def calculate_ichimoku(df):
    """Calculate Ichimoku Cloud"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = df['high'].rolling(window=9).max()
    period9_low = df['low'].rolling(window=9).min()
    df['tenkan_sen'] = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = df['high'].rolling(window=26).max()
    period26_low = df['low'].rolling(window=26).min()
    df['kijun_sen'] = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = df['high'].rolling(window=52).max()
    period52_low = df['low'].rolling(window=52).min()
    df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
    
    # Chikou Span (Lagging Span): Close plotted 26 periods in the past
    df['chikou_span'] = df['close'].shift(-26)
    
    return df

def calculate_volume_indicators(df):
    """Calculate volume-based indicators"""
    df['OBV'] = talib.OBV(df['close'], df['volume'])
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    return df

# Pattern Recognition Functions
def detect_head_and_shoulders(df, window=20):
    """Detect Head and Shoulders pattern"""
    highs = df['high'].values
    lows = df['low'].values
    
    if len(df) < window * 3:
        return None, None
    
    # Find peaks (potential shoulders and head)
    peaks_idx = argrelextrema(highs, np.greater, order=window)[0]
    
    if len(peaks_idx) < 3:
        return None, None
    
    # Check last 3 peaks for H&S pattern
    recent_peaks = peaks_idx[-3:]
    peak_prices = [highs[i] for i in recent_peaks]
    
    # Head and Shoulders: left shoulder < head > right shoulder
    if len(peak_prices) >= 3:
        left_shoulder = peak_prices[0]
        head = peak_prices[1]
        right_shoulder = peak_prices[2]
        
        tolerance = 0.02  # 2% tolerance
        
        # Classic H&S
        if (head > left_shoulder * (1 + tolerance) and 
            head > right_shoulder * (1 + tolerance) and
            abs(left_shoulder - right_shoulder) < left_shoulder * tolerance):
            return "Head and Shoulders", "BEARISH"
        
        # Inverse H&S
        troughs_idx = argrelextrema(lows, np.less, order=window)[0]
        if len(troughs_idx) >= 3:
            recent_troughs = troughs_idx[-3:]
            trough_prices = [lows[i] for i in recent_troughs]
            
            left_s = trough_prices[0]
            head_inv = trough_prices[1]
            right_s = trough_prices[2]
            
            if (head_inv < left_s * (1 - tolerance) and 
                head_inv < right_s * (1 - tolerance) and
                abs(left_s - right_s) < left_s * tolerance):
                return "Inverse Head and Shoulders", "BULLISH"
    
    return None, None

def detect_triangles(df, window=20):
    """Detect Triangle patterns (Ascending, Descending, Symmetrical)"""
    if len(df) < window * 2:
        return None, None
    
    recent_df = df.tail(window * 2)
    highs = recent_df['high'].values
    lows = recent_df['low'].values
    
    # Find trend lines
    x = np.arange(len(recent_df))
    
    # Upper trend line (resistance)
    high_peaks_idx = argrelextrema(highs, np.greater, order=5)[0]
    if len(high_peaks_idx) >= 2:
        high_slope = np.polyfit(high_peaks_idx, highs[high_peaks_idx], 1)[0]
    else:
        high_slope = 0
    
    # Lower trend line (support)
    low_troughs_idx = argrelextrema(lows, np.less, order=5)[0]
    if len(low_troughs_idx) >= 2:
        low_slope = np.polyfit(low_troughs_idx, lows[low_troughs_idx], 1)[0]
    else:
        low_slope = 0
    
    # Classify triangle
    slope_threshold = 0.001
    
    if abs(high_slope) < slope_threshold and low_slope > slope_threshold:
        return "Ascending Triangle", "BULLISH"
    elif high_slope < -slope_threshold and abs(low_slope) < slope_threshold:
        return "Descending Triangle", "BEARISH"
    elif abs(high_slope - low_slope) < slope_threshold and high_slope < 0 and low_slope > 0:
        return "Symmetrical Triangle", "NEUTRAL"
    
    return None, None

def detect_double_top_bottom(df, window=15):
    """Detect Double Top and Double Bottom patterns"""
    if len(df) < window * 3:
        return None, None
    
    highs = df['high'].values
    lows = df['low'].values
    
    # Double Top
    peaks_idx = argrelextrema(highs, np.greater, order=window)[0]
    if len(peaks_idx) >= 2:
        last_two_peaks = peaks_idx[-2:]
        peak_prices = [highs[i] for i in last_two_peaks]
        
        tolerance = 0.02
        if abs(peak_prices[0] - peak_prices[1]) < peak_prices[0] * tolerance:
            return "Double Top", "BEARISH"
    
    # Double Bottom
    troughs_idx = argrelextrema(lows, np.less, order=window)[0]
    if len(troughs_idx) >= 2:
        last_two_troughs = troughs_idx[-2:]
        trough_prices = [lows[i] for i in last_two_troughs]
        
        tolerance = 0.02
        if abs(trough_prices[0] - trough_prices[1]) < trough_prices[0] * tolerance:
            return "Double Bottom", "BULLISH"
    
    return None, None

def detect_wedge_patterns(df, window=20):
    """Detect Rising and Falling Wedge patterns"""
    if len(df) < window * 2:
        return None, None
    
    recent_df = df.tail(window * 2)
    highs = recent_df['high'].values
    lows = recent_df['low'].values
    
    # Calculate slopes
    high_peaks_idx = argrelextrema(highs, np.greater, order=5)[0]
    low_troughs_idx = argrelextrema(lows, np.less, order=5)[0]
    
    if len(high_peaks_idx) >= 2 and len(low_troughs_idx) >= 2:
        high_slope = np.polyfit(high_peaks_idx, highs[high_peaks_idx], 1)[0]
        low_slope = np.polyfit(low_troughs_idx, lows[low_troughs_idx], 1)[0]
        
        # Rising Wedge: both slopes positive, converging
        if high_slope > 0 and low_slope > 0 and high_slope < low_slope:
            return "Rising Wedge", "BEARISH"
        
        # Falling Wedge: both slopes negative, converging
        if high_slope < 0 and low_slope < 0 and high_slope > low_slope:
            return "Falling Wedge", "BULLISH"
    
    return None, None

def detect_all_patterns(df):
    """Detect all chart patterns"""
    patterns = []
    
    # Head and Shoulders
    hs_pattern, hs_signal = detect_head_and_shoulders(df)
    if hs_pattern:
        patterns.append({"pattern": hs_pattern, "signal": hs_signal, "strength": "HIGH"})
    
    # Triangles
    triangle_pattern, triangle_signal = detect_triangles(df)
    if triangle_pattern:
        patterns.append({"pattern": triangle_pattern, "signal": triangle_signal, "strength": "MEDIUM"})
    
    # Double Top/Bottom
    double_pattern, double_signal = detect_double_top_bottom(df)
    if double_pattern:
        patterns.append({"pattern": double_pattern, "signal": double_signal, "strength": "HIGH"})
    
    # Wedges
    wedge_pattern, wedge_signal = detect_wedge_patterns(df)
    if wedge_pattern:
        patterns.append({"pattern": wedge_pattern, "signal": wedge_signal, "strength": "MEDIUM"})
    
    return patterns

def find_support_resistance(df, order=5):
    """Find support and resistance levels"""
    highs = df['high'].values
    lows = df['low'].values
    
    resistance_idx = argrelextrema(highs, np.greater, order=order)[0]
    support_idx = argrelextrema(lows, np.less, order=order)[0]
    
    resistances = df.iloc[resistance_idx]['high'].values
    supports = df.iloc[support_idx]['low'].values
    
    recent_resistance = sorted(resistances[-10:], reverse=True)[:3] if len(resistances) > 0 else []
    recent_support = sorted(supports[-10:], reverse=True)[:3] if len(supports) > 0 else []
    
    return recent_support, recent_resistance

def calculate_fibonacci_levels(df):
    """Calculate Fibonacci retracement levels"""
    max_price = df['high'].max()
    min_price = df['low'].min()
    diff = max_price - min_price
    
    levels = {
        'level_0': max_price,
        'level_236': max_price - (diff * 0.236),
        'level_382': max_price - (diff * 0.382),
        'level_500': max_price - (diff * 0.500),
        'level_618': max_price - (diff * 0.618),
        'level_786': max_price - (diff * 0.786),
        'level_100': min_price
    }
    return levels

def detect_trendline(df):
    """Simple trendline detection"""
    x = np.arange(len(df))
    y = df['close'].values
    
    z = np.polyfit(x[-50:], y[-50:], 1)
    slope = z[0]
    
    if slope > 0:
        return "Uptrend"
    elif slope < 0:
        return "Downtrend"
    else:
        return "Sideways"

def analyze_volume(df):
    """Analyze volume patterns"""
    df_copy = df.copy()
    if 'volume_ma' not in df_copy.columns:
        df_copy['volume_ma'] = df_copy['volume'].rolling(window=20).mean()
    
    current_volume = df_copy['volume'].iloc[-1]
    avg_volume = df_copy['volume_ma'].iloc[-1]
    
    if current_volume > avg_volume * 1.5:
        return "High Volume", "strong"
    elif current_volume > avg_volume:
        return "Above Average", "moderate"
    else:
        return "Low Volume", "weak"

def generate_signal(df, smc_data=None):
    """Generate trading signal based on multiple indicators"""
    current_price = df['close'].iloc[-1]
    ma12 = df['MA12'].iloc[-1]
    ma26 = df['MA26'].iloc[-1]
    ma50 = df['MA50'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    macd_signal = df['MACD_signal'].iloc[-1]
    
    bullish_signals = 0
    bearish_signals = 0
    
    # MA crossover analysis
    if ma12 > ma26 > ma50:
        bullish_signals += 2
    elif ma12 < ma26 < ma50:
        bearish_signals += 2
    
    # Price vs MA
    if current_price > ma50:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    # RSI analysis
    if rsi < 30:
        bullish_signals += 2
    elif rsi > 70:
        bearish_signals += 2
    elif 40 < rsi < 60:
        bullish_signals += 1
    
    # MACD analysis
    if macd > macd_signal:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    # Stochastic analysis
    if 'STOCH_K' in df.columns:
        stoch_k = df['STOCH_K'].iloc[-1]
        stoch_d = df['STOCH_D'].iloc[-1]
        if stoch_k < 20 and stoch_k > stoch_d:
            bullish_signals += 1
        elif stoch_k > 80 and stoch_k < stoch_d:
            bearish_signals += 1
    
    # ADX trend strength
    if 'ADX' in df.columns:
        adx = df['ADX'].iloc[-1]
        plus_di = df['PLUS_DI'].iloc[-1]
        minus_di = df['MINUS_DI'].iloc[-1]
        
        if adx > 25:  # Strong trend
            if plus_di > minus_di:
                bullish_signals += 1
            else:
                bearish_signals += 1
    
    # Ichimoku analysis
    if 'tenkan_sen' in df.columns and 'kijun_sen' in df.columns:
        tenkan = df['tenkan_sen'].iloc[-1]
        kijun = df['kijun_sen'].iloc[-1]
        senkou_a = df['senkou_span_a'].iloc[-1]
        senkou_b = df['senkou_span_b'].iloc[-1]
        
        if not pd.isna(tenkan) and not pd.isna(kijun):
            if tenkan > kijun and current_price > senkou_a and current_price > senkou_b:
                bullish_signals += 2
            elif tenkan < kijun and current_price < senkou_a and current_price < senkou_b:
                bearish_signals += 2
    
    # Smart Money Concepts (SMC) analysis
    if smc_data and isinstance(smc_data, dict):
        smc_bias = smc_data.get('smc_bias', 'NEUTRAL')
        smc_strength = smc_data.get('smc_signal_strength', 50)
        
        if smc_bias == 'BULLISH':
            bullish_signals += max(1, int(smc_strength / 30))
        elif smc_bias == 'BEARISH':
            bearish_signals += max(1, int(smc_strength / 30))
    
    # Volume analysis
    volume_status, volume_strength = analyze_volume(df)
    if volume_strength == "strong":
        if bullish_signals > bearish_signals:
            bullish_signals += 1
        else:
            bearish_signals += 1
    
    # Determine signal
    total_signals = bullish_signals + bearish_signals
    confidence = max(bullish_signals, bearish_signals) / total_signals * 100 if total_signals > 0 else 50
    
    if bullish_signals > bearish_signals + 1:
        signal = "LONG"
    elif bearish_signals > bullish_signals + 1:
        signal = "SHORT"
    else:
        signal = "NEUTRAL"
    
    return signal, confidence, bullish_signals, bearish_signals

def safe_float(value):
    """Safely convert value to float, handling NaN"""
    if pd.isna(value):
        return None
    return float(value)

def perform_full_analysis(df):
    """Perform complete technical analysis"""
    df = calculate_mas(df)
    df = calculate_bollinger_bands(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_stochastic(df)
    df = calculate_smi(df)  # Add SMI
    df = calculate_atr(df)
    df = calculate_adx(df)
    df = calculate_ichimoku(df)
    df = calculate_volume_indicators(df)
    
    # Calculate Smart Money Concepts (stored separately)
    smc_data = calculate_smart_money_concepts(df)
    
    return df, smc_data

# API Endpoints
@app.post("/api/auth/register")
async def register(user: UserRegister):
    conn = sqlite3.connect('crypto_scanner.db')
    c = conn.cursor()
    
    hashed = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
    
    try:
        c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                  (user.username, user.email, hashed))
        conn.commit()
        return {"message": "User registered successfully"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    finally:
        conn.close()

@app.post("/api/auth/login")
async def login(user: UserLogin):
    conn = sqlite3.connect('crypto_scanner.db')
    c = conn.cursor()
    
    c.execute("SELECT id, username, password, role FROM users WHERE username = ?", (user.username,))
    result = c.fetchone()
    conn.close()
    
    if not result or not bcrypt.checkpw(user.password.encode('utf-8'), result[2]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token({"user_id": result[0], "username": result[1], "role": result[3]})
    return {"access_token": token, "token_type": "bearer", "username": result[1]}

@app.get("/api/symbols")
async def get_symbols(
    search: Optional[str] = None,
    limit: Optional[int] = 500,
    payload: dict = Depends(verify_token)
):
    symbols_list = get_available_symbols()
    
    if search:
        symbols_list = [s for s in symbols_list if search.upper() in s['symbol']]
    
    return {
        "symbols": symbols_list[:limit], 
        "total": len(symbols_list),
        "exchanges": list(set([s['exchange'] for s in symbols_list[:limit]]))
    }

@app.get("/api/symbols/{symbol}/info")
async def get_symbol_details(symbol: str, payload: dict = Depends(verify_token)):
    info = get_symbol_info(symbol)
    if info:
        return info
    raise HTTPException(status_code=404, detail="Symbol not found")

@app.post("/api/scan")
async def scan_ticker(request: ScanRequest, payload: dict = Depends(verify_token)):
    # Fetch 1000 candles from Binance Spot API
    df = get_binance_klines(request.ticker, request.timeframe, limit=1000)
    
    if df is None or len(df) < 200:
        raise HTTPException(status_code=400, detail="Unable to fetch data or insufficient data")
    
    df, smc_data = perform_full_analysis(df)
    
    supports, resistances = find_support_resistance(df)
    fib_levels = calculate_fibonacci_levels(df)
    trend = detect_trendline(df)
    volume_status, volume_strength = analyze_volume(df)
    signal, confidence, bull_signals, bear_signals = generate_signal(df, smc_data)
    patterns = detect_all_patterns(df)
    
    current_price = float(df['close'].iloc[-1])
    
    analysis = {
        "signal": signal,
        "confidence": round(confidence, 2),
        "bullish_signals": bull_signals,
        "bearish_signals": bear_signals,
        "current_price": current_price,
        "trend": trend,
        "total_candles": len(df),  # Show how many candles fetched
        "moving_averages": {
            "MA12": safe_float(df['MA12'].iloc[-1]),
            "MA26": safe_float(df['MA26'].iloc[-1]),
            "MA50": safe_float(df['MA50'].iloc[-1]),
            "MA200": safe_float(df['MA200'].iloc[-1]),
            "EMA9": safe_float(df['EMA9'].iloc[-1]),
            "EMA21": safe_float(df['EMA21'].iloc[-1])
        },
        "rsi": safe_float(df['RSI'].iloc[-1]),
        "macd": {
            "macd": safe_float(df['MACD'].iloc[-1]),
            "signal": safe_float(df['MACD_signal'].iloc[-1]),
            "histogram": safe_float(df['MACD_hist'].iloc[-1])
        },
        "stochastic": {
            "k": safe_float(df['STOCH_K'].iloc[-1]),
            "d": safe_float(df['STOCH_D'].iloc[-1])
        },
        "smi": {
            "smi": safe_float(df['SMI'].iloc[-1]),
            "smi_ema": safe_float(df['SMI_EMA'].iloc[-1])
        },
        "atr": safe_float(df['ATR'].iloc[-1]),
        "adx": {
            "adx": safe_float(df['ADX'].iloc[-1]),
            "plus_di": safe_float(df['PLUS_DI'].iloc[-1]),
            "minus_di": safe_float(df['MINUS_DI'].iloc[-1])
        },
        "ichimoku": {
            "tenkan_sen": safe_float(df['tenkan_sen'].iloc[-1]),
            "kijun_sen": safe_float(df['kijun_sen'].iloc[-1]),
            "senkou_span_a": safe_float(df['senkou_span_a'].iloc[-1]),
            "senkou_span_b": safe_float(df['senkou_span_b'].iloc[-1])
        },
        "bollinger_bands": {
            "upper": safe_float(df['BB_upper'].iloc[-1]),
            "middle": safe_float(df['BB_middle'].iloc[-1]),
            "lower": safe_float(df['BB_lower'].iloc[-1])
        },
        "support_levels": [float(s) for s in supports],
        "resistance_levels": [float(r) for r in resistances],
        "fibonacci": {k: float(v) for k, v in fib_levels.items()},
        "volume": {
            "status": volume_status,
            "strength": volume_strength,
            "obv": safe_float(df['OBV'].iloc[-1])
        },
        "patterns": patterns,
        "smart_money_concepts": {
            "smc_bias": smc_data.get('smc_bias', 'NEUTRAL'),
            "smc_signal_strength": smc_data.get('smc_signal_strength', 50),
            "smc_signal_count": smc_data.get('smc_signal_count', 0),
            "break_of_structure": {
                "bullish_bos": smc_data.get('break_of_structure', {}).get('bullish_bos', []),
                "bearish_bos": smc_data.get('break_of_structure', {}).get('bearish_bos', []),
                "bullish_choch": smc_data.get('break_of_structure', {}).get('bullish_choch', []),
                "bearish_choch": smc_data.get('break_of_structure', {}).get('bearish_choch', [])
            },
            "order_blocks": {
                "bullish": smc_data.get('order_blocks', {}).get('bullish', [])[-3:],
                "bearish": smc_data.get('order_blocks', {}).get('bearish', [])[-3:]
            },
            "fair_value_gaps": {
                "bullish": smc_data.get('fair_value_gaps', {}).get('bullish', [])[-3:],
                "bearish": smc_data.get('fair_value_gaps', {}).get('bearish', [])[-3:]
            },
            "equal_levels": smc_data.get('equal_levels', {}),
            "premium_discount_zones": smc_data.get('premium_discount_zones', {})
        },
        "chart_data": df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(100).to_dict('records')
    }
    
    conn = sqlite3.connect('crypto_scanner.db')
    c = conn.cursor()
    c.execute("""INSERT INTO scan_history 
                 (user_id, ticker, timeframe, signal, confidence, price, analysis) 
                 VALUES (?, ?, ?, ?, ?, ?, ?)""",
              (payload['user_id'], request.ticker, request.timeframe, 
               signal, confidence, current_price, safe_json_dumps(analysis)))
    conn.commit()
    conn.close()
    
    return analysis

@app.post("/api/scan/multi-timeframe")
async def multi_timeframe_scan(request: MultiTimeframeScanRequest, payload: dict = Depends(verify_token)):
    """Scan across multiple timeframes"""
    results = {}
    signals_count = {"LONG": 0, "SHORT": 0, "NEUTRAL": 0}
    total_confidence = 0
    all_smc_data = []
    
    for timeframe in request.timeframes:
        df = get_binance_klines(request.ticker, timeframe, limit=1000)
        
        if df is None or len(df) < 200:
            continue
        
        df, smc_data = perform_full_analysis(df)
        signal, confidence, bull_signals, bear_signals = generate_signal(df, smc_data)
        trend = detect_trendline(df)
        
        results[timeframe] = {
            "signal": signal,
            "confidence": round(confidence, 2),
            "trend": trend,
            "price": float(df['close'].iloc[-1]),
            "rsi": safe_float(df['RSI'].iloc[-1]),
            "macd_histogram": safe_float(df['MACD_hist'].iloc[-1]),
            "adx": safe_float(df['ADX'].iloc[-1]),
            "smc_bias": smc_data.get('smc_bias', 'NEUTRAL'),
            "candles_count": len(df)
        }
        
        signals_count[signal] += 1
        total_confidence += confidence
        all_smc_data.append(smc_data)
    
    # Determine overall signal
    if signals_count["LONG"] > signals_count["SHORT"]:
        overall_signal = "LONG"
    elif signals_count["SHORT"] > signals_count["LONG"]:
        overall_signal = "SHORT"
    else:
        overall_signal = "NEUTRAL"
    
    avg_confidence = total_confidence / len(request.timeframes) if request.timeframes else 0
    
    analysis_data = {
        "ticker": request.ticker,
        "overall_signal": overall_signal,
        "average_confidence": round(avg_confidence, 2),
        "timeframe_analysis": results,
        "signal_distribution": signals_count
    }
    
    # Save to database
    conn = sqlite3.connect('crypto_scanner.db')
    c = conn.cursor()
    c.execute("""INSERT INTO multi_timeframe_scans 
                 (user_id, ticker, timeframes, overall_signal, analysis) 
                 VALUES (?, ?, ?, ?, ?)""",
              (payload['user_id'], request.ticker, ','.join(request.timeframes),
               overall_signal, safe_json_dumps(analysis_data)))
    conn.commit()
    conn.close()
    
    return analysis_data

@app.get("/api/scan/{ticker}/smc")
async def get_smc_analysis(
    ticker: str, 
    timeframe: str = "1h",
    payload: dict = Depends(verify_token)
):
    """Get detailed Smart Money Concepts analysis for a ticker"""
    df = get_binance_klines(ticker, timeframe, limit=1000)
    
    if df is None or len(df) < 200:
        raise HTTPException(status_code=400, detail="Unable to fetch data or insufficient data")
    
    df, smc_data = perform_full_analysis(df)
    
    current_price = float(df['close'].iloc[-1])
    
    smc_analysis = {
        "ticker": ticker,
        "timeframe": timeframe,
        "current_price": current_price,
        "timestamp": df['timestamp'].iloc[-1].isoformat(),
        "smc_bias": smc_data.get('smc_bias', 'NEUTRAL'),
        "smc_signal_strength": smc_data.get('smc_signal_strength', 50),
        "smc_signal_count": smc_data.get('smc_signal_count', 0),
        "swing_structure": {
            "highs": [{"index": h[0], "price": h[1]} for h in smc_data.get('swing_structure', {}).get('swing_highs', [])[-5:]],
            "lows": [{"index": l[0], "price": l[1]} for l in smc_data.get('swing_structure', {}).get('swing_lows', [])[-5:]]
        },
        "break_of_structure": {
            "bullish_bos": smc_data.get('break_of_structure', {}).get('bullish_bos', []),
            "bearish_bos": smc_data.get('break_of_structure', {}).get('bearish_bos', []),
            "bullish_choch": smc_data.get('break_of_structure', {}).get('bullish_choch', []),
            "bearish_choch": smc_data.get('break_of_structure', {}).get('bearish_choch', [])
        },
        "order_blocks": {
            "bullish": smc_data.get('order_blocks', {}).get('bullish', [])[-5:],
            "bearish": smc_data.get('order_blocks', {}).get('bearish', [])[-5:]
        },
        "fair_value_gaps": {
            "bullish": smc_data.get('fair_value_gaps', {}).get('bullish', [])[-5:],
            "bearish": smc_data.get('fair_value_gaps', {}).get('bearish', [])[-5:]
        },
        "equal_levels": {
            "equal_highs": smc_data.get('equal_levels', {}).get('equal_highs', []),
            "equal_lows": smc_data.get('equal_levels', {}).get('equal_lows', [])
        },
        "premium_discount_zones": smc_data.get('premium_discount_zones', {})
    }
    
    return smc_analysis

@app.get("/api/history")
async def get_history(payload: dict = Depends(verify_token)):
    conn = sqlite3.connect('crypto_scanner.db')
    c = conn.cursor()
    c.execute("""SELECT id, ticker, timeframe, signal, confidence, price, created_at 
                 FROM scan_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 50""",
              (payload['user_id'],))
    results = c.fetchall()
    conn.close()
    
    history = []
    for row in results:
        history.append({
            "id": row[0],
            "ticker": row[1],
            "timeframe": row[2],
            "signal": row[3],
            "confidence": row[4],
            "price": row[5],
            "created_at": row[6]
        })
    
    return {"history": history}

@app.get("/api/history/multi-timeframe")
async def get_multi_timeframe_history(payload: dict = Depends(verify_token)):
    conn = sqlite3.connect('crypto_scanner.db')
    c = conn.cursor()
    c.execute("""SELECT id, ticker, timeframes, overall_signal, analysis, created_at 
                 FROM multi_timeframe_scans WHERE user_id = ? ORDER BY created_at DESC LIMIT 20""",
              (payload['user_id'],))
    results = c.fetchall()
    conn.close()
    
    history = []
    for row in results:
        history.append({
            "id": row[0],
            "ticker": row[1],
            "timeframes": row[2],
            "overall_signal": row[3],
            "analysis": json.loads(row[4]) if row[4] else {},
            "created_at": row[5]
        })
    
    return {"history": history}

@app.get("/api/dashboard/stats")
async def get_dashboard_stats(payload: dict = Depends(verify_token)):
    conn = sqlite3.connect('crypto_scanner.db')
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM scan_history WHERE user_id = ?", (payload['user_id'],))
    total_scans = c.fetchone()[0]
    
    c.execute("""SELECT signal, COUNT(*) FROM scan_history 
                 WHERE user_id = ? GROUP BY signal""", (payload['user_id'],))
    signal_dist = dict(c.fetchall())
    
    c.execute("""SELECT ticker, signal, confidence, created_at 
                 FROM scan_history WHERE user_id = ? 
                 ORDER BY created_at DESC LIMIT 5""", (payload['user_id'],))
    recent = c.fetchall()
    
    c.execute("SELECT COUNT(*) FROM multi_timeframe_scans WHERE user_id = ?", (payload['user_id'],))
    total_mtf_scans = c.fetchone()[0]
    
    conn.close()
    
    return {
        "total_scans": total_scans,
        "total_multi_timeframe_scans": total_mtf_scans,
        "signal_distribution": signal_dist,
        "recent_scans": [{"ticker": r[0], "signal": r[1], "confidence": r[2], "date": r[3]} for r in recent]
    }

@app.get("/api/export/scan/{scan_id}")
async def export_scan_data(scan_id: int, payload: dict = Depends(verify_token)):
    """Get scan data for export"""
    conn = sqlite3.connect('crypto_scanner.db')
    c = conn.cursor()
    c.execute("""SELECT ticker, timeframe, signal, confidence, price, analysis, created_at 
                 FROM scan_history WHERE id = ? AND user_id = ?""",
              (scan_id, payload['user_id']))
    result = c.fetchone()
    conn.close()
    
    if not result:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    return {
        "ticker": result[0],
        "timeframe": result[1],
        "signal": result[2],
        "confidence": result[3],
        "price": result[4],
        "analysis": json.loads(result[5]) if result[5] else {},
        "created_at": result[6]
    }

# Watchlist Endpoints
@app.post("/api/watchlist/add")
async def add_to_watchlist(item: WatchlistItem, payload: dict = Depends(verify_token)):
    """Add ticker to watchlist"""
    conn = sqlite3.connect('crypto_scanner.db')
    c = conn.cursor()
    
    try:
        c.execute("""INSERT INTO watchlist (user_id, ticker, timeframe, alert_price) 
                     VALUES (?, ?, ?, ?)""",
                  (payload['user_id'], item.ticker, item.timeframe, item.alert_price))
        conn.commit()
        return {"message": "Added to watchlist", "ticker": item.ticker}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Already in watchlist")
    finally:
        conn.close()

@app.get("/api/watchlist")
async def get_watchlist(payload: dict = Depends(verify_token)):
    """Get user's watchlist"""
    conn = sqlite3.connect('crypto_scanner.db')
    c = conn.cursor()
    c.execute("""SELECT id, ticker, timeframe, alert_price, added_at, last_scan 
                 FROM watchlist WHERE user_id = ? ORDER BY added_at DESC""",
              (payload['user_id'],))
    results = c.fetchall()
    conn.close()
    
    watchlist = []
    for row in results:
        watchlist.append({
            "id": row[0],
            "ticker": row[1],
            "timeframe": row[2],
            "alert_price": row[3],
            "added_at": row[4],
            "last_scan": json.loads(row[5]) if row[5] else None
        })
    
    return {"watchlist": watchlist}

@app.post("/api/watchlist/scan-all")
async def scan_watchlist(payload: dict = Depends(verify_token)):
    """Scan all tickers in watchlist"""
    conn = sqlite3.connect('crypto_scanner.db')
    c = conn.cursor()
    c.execute("SELECT id, ticker, timeframe FROM watchlist WHERE user_id = ?",
              (payload['user_id'],))
    watchlist_items = c.fetchall()
    
    results = []
    
    for wl_id, ticker, timeframe in watchlist_items:
        try:
            df = get_binance_klines(ticker, timeframe, limit=1000)
            
            if df is not None and len(df) >= 200:
                df, smc_data = perform_full_analysis(df)
                signal, confidence, bull_signals, bear_signals = generate_signal(df, smc_data)
                patterns = detect_all_patterns(df)
                
                scan_result = {
                    "ticker": ticker,
                    "timeframe": timeframe,
                    "signal": signal,
                    "confidence": round(confidence, 2),
                    "price": float(df['close'].iloc[-1]),
                    "rsi": safe_float(df['RSI'].iloc[-1]),
                    "patterns": patterns,
                    "candles_analyzed": len(df)
                }
                
                # Update last scan
                c.execute("UPDATE watchlist SET last_scan = ? WHERE id = ?",
                         (safe_json_dumps(scan_result), wl_id))
                
                results.append(scan_result)
        except Exception as e:
            print(f"Error scanning {ticker}: {e}")
            continue
    
    conn.commit()
    conn.close()
    
    return {"results": results, "total_scanned": len(results)}

@app.delete("/api/watchlist/{ticker}")
async def remove_from_watchlist(ticker: str, payload: dict = Depends(verify_token)):
    """Remove ticker from watchlist"""
    conn = sqlite3.connect('crypto_scanner.db')
    c = conn.cursor()
    c.execute("DELETE FROM watchlist WHERE user_id = ? AND ticker = ?",
              (payload['user_id'], ticker))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Ticker not in watchlist")
    
    return {"message": "Removed from watchlist", "ticker": ticker}

# Performance Analytics Endpoints
@app.get("/api/analytics/overview")
async def get_analytics_overview(payload: dict = Depends(verify_token)):
    """Get performance analytics overview"""
    conn = sqlite3.connect('crypto_scanner.db')
    c = conn.cursor()
    
    # Total scans by signal
    c.execute("""SELECT signal, COUNT(*), AVG(confidence) 
                 FROM scan_history WHERE user_id = ? 
                 GROUP BY signal""", (payload['user_id'],))
    signal_stats = c.fetchall()
    
    # Scans over time (last 30 days)
    c.execute("""SELECT DATE(created_at) as date, COUNT(*) 
                 FROM scan_history WHERE user_id = ? 
                 AND created_at >= datetime('now', '-30 days')
                 GROUP BY DATE(created_at) ORDER BY date""",
              (payload['user_id'],))
    scans_timeline = c.fetchall()
    
    # Most scanned tickers
    c.execute("""SELECT ticker, COUNT(*) as scan_count 
                 FROM scan_history WHERE user_id = ? 
                 GROUP BY ticker ORDER BY scan_count DESC LIMIT 10""",
              (payload['user_id'],))
    top_tickers = c.fetchall()
    
    # Pattern distribution
    c.execute("""SELECT analysis FROM scan_history WHERE user_id = ? 
                 AND analysis IS NOT NULL""", (payload['user_id'],))
    all_analyses = c.fetchall()
    
    pattern_counts = {}
    for (analysis_str,) in all_analyses:
        try:
            analysis = json.loads(analysis_str)
            if 'patterns' in analysis:
                for pattern in analysis['patterns']:
                    pattern_name = pattern['pattern']
                    pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
        except:
            continue
    
    # Trade performance (if trades recorded)
    c.execute("""SELECT signal, AVG(profit_loss), COUNT(*) 
                 FROM trade_records WHERE user_id = ? AND status = 'closed'
                 GROUP BY signal""", (payload['user_id'],))
    trade_performance = c.fetchall()
    
    conn.close()
    
    return {
        "signal_distribution": [{"signal": s[0], "count": s[1], "avg_confidence": s[2]} 
                                for s in signal_stats],
        "scans_timeline": [{"date": t[0], "count": t[1]} for t in scans_timeline],
        "top_tickers": [{"ticker": t[0], "scans": t[1]} for t in top_tickers],
        "pattern_distribution": pattern_counts,
        "trade_performance": [{"signal": t[0], "avg_profit_loss": t[1], "trades": t[2]} 
                              for t in trade_performance]
    }

@app.get("/api/analytics/accuracy")
async def get_signal_accuracy(payload: dict = Depends(verify_token)):
    """Calculate signal accuracy based on trade outcomes"""
    conn = sqlite3.connect('crypto_scanner.db')
    c = conn.cursor()
    
    c.execute("""SELECT signal, 
                 SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
                 COUNT(*) as total,
                 AVG(profit_loss) as avg_pnl
                 FROM trade_records 
                 WHERE user_id = ? AND status = 'closed'
                 GROUP BY signal""", (payload['user_id'],))
    
    results = c.fetchall()
    conn.close()
    
    accuracy_data = []
    for signal, wins, total, avg_pnl in results:
        accuracy = (wins / total * 100) if total > 0 else 0
        accuracy_data.append({
            "signal": signal,
            "accuracy": round(accuracy, 2),
            "wins": wins,
            "total_trades": total,
            "avg_profit_loss": round(avg_pnl, 2) if avg_pnl else 0
        })
    
    return {"accuracy": accuracy_data}

# Trade Recording (for performance tracking)
@app.post("/api/trades/record")
async def record_trade(
    ticker: str,
    signal: str,
    entry_price: float,
    quantity: float = 1.0,
    payload: dict = Depends(verify_token)
):
    """Record a trade entry"""
    conn = sqlite3.connect('crypto_scanner.db')
    c = conn.cursor()
    c.execute("""INSERT INTO trade_records 
                 (user_id, ticker, signal, entry_price, quantity, status) 
                 VALUES (?, ?, ?, ?, ?, 'open')""",
              (payload['user_id'], ticker, signal, entry_price, quantity))
    trade_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return {"message": "Trade recorded", "trade_id": trade_id}

@app.put("/api/trades/close/{trade_id}")
async def close_trade(
    trade_id: int,
    exit_price: float,
    payload: dict = Depends(verify_token)
):
    """Close a trade and calculate P&L"""
    conn = sqlite3.connect('crypto_scanner.db')
    c = conn.cursor()
    
    c.execute("""SELECT entry_price, quantity, signal FROM trade_records 
                 WHERE id = ? AND user_id = ? AND status = 'open'""",
              (trade_id, payload['user_id']))
    result = c.fetchone()
    
    if not result:
        conn.close()
        raise HTTPException(status_code=404, detail="Trade not found or already closed")
    
    entry_price, quantity, signal = result
    
    # Calculate P&L
    if signal == "LONG":
        profit_loss = (exit_price - entry_price) * quantity
    else:  # SHORT
        profit_loss = (entry_price - exit_price) * quantity
    
    c.execute("""UPDATE trade_records 
                 SET exit_price = ?, profit_loss = ?, status = 'closed', 
                     exit_date = CURRENT_TIMESTAMP 
                 WHERE id = ?""",
              (exit_price, profit_loss, trade_id))
    conn.commit()
    conn.close()
    
    return {
        "message": "Trade closed",
        "profit_loss": round(profit_loss, 2),
        "profit_loss_percent": round((profit_loss / (entry_price * quantity)) * 100, 2)
    }

# ============================================================
# INDONESIA STOCKS ENDPOINTS
# ============================================================

# Daftar saham Indonesia yang tersedia
INDONESIA_STOCKS = {
    'BBCA': {'name': 'Bank Central Asia', 'sector': 'Banking'},
    'BBRI': {'name': 'Bank Rakyat Indonesia', 'sector': 'Banking'},
    'BMRI': {'name': 'Bank Mandiri', 'sector': 'Banking'},
    'BBNI': {'name': 'Bank Negara Indonesia', 'sector': 'Banking'},
    'ASII': {'name': 'Astra International', 'sector': 'Automotive'},
    'TLKM': {'name': 'Telekomunikasi Indonesia', 'sector': 'Telecom'},
    'UNVR': {'name': 'Unilever Indonesia', 'sector': 'Consumer Goods'},
    'INDF': {'name': 'Indofood', 'sector': 'Food & Beverage'},
    'ICBP': {'name': 'Indofood CBP', 'sector': 'Food & Beverage'},
    'SMGR': {'name': 'Semen Gresik', 'sector': 'Construction Materials'},
    'INTP': {'name': 'Indocement', 'sector': 'Construction Materials'},
    'PGAS': {'name': 'Perusahaan Gas Negara', 'sector': 'Energy'},
    'LPKR': {'name': 'Lippo Karawaci', 'sector': 'Real Estate'},
    'MNCN': {'name': 'Media Nusantara Citra', 'sector': 'Media'},
    'JSMR': {'name': 'Jasa Marga', 'sector': 'Infrastructure'},
    'ADRO': {'name': 'Adaro Energy', 'sector': 'Energy'},
    'ITMG': {'name': 'Indo Tambangraya Megah', 'sector': 'Mining'},
    'GSMF': {'name': 'Gajah Tunggal', 'sector': 'Automotive'},
    'CPRO': {'name': 'Central Protein Prime', 'sector': 'Food & Beverage'},
}

def fetch_indonesia_stock_data(ticker: str, interval: str = '1d') -> Dict:
    """
    Fetch Indonesia stock OHLCV data dari Alpha Vantage atau Polygon.io
    Supports: 1m, 5m, 15m, 30m, 1h, 1d
    """
    try:
        # Validate ticker
        ticker_upper = ticker.upper()
        if ticker_upper not in INDONESIA_STOCKS:
            return {'error': f'Invalid ticker: {ticker}. Supported: {", ".join(INDONESIA_STOCKS.keys())}'}
        
        # Map user intervals ke format API
        interval_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1h', 'hourly': '1h',
            '1d': '1d', 'daily': '1d',
            '1w': '1w', 'weekly': '1w',
            '1mo': '1mo', 'monthly': '1mo'
        }
        
        api_interval = interval_map.get(interval, '1d')
        
        # Fetch data dengan fallback
        df = fetch_stock_data(ticker_upper, api_interval, limit=1000)
        
        if df is None or len(df) == 0:
            return {'error': f'No data available for {ticker} with interval {interval}. Try a different interval or check market hours.'}
        
        # Convert DataFrame to list of dicts
        chart_data = []
        for _, row in df.iterrows():
            chart_data.append({
                'timestamp': row['timestamp'].isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume'])
            })
        
        return {
            'ticker': ticker_upper,
            'symbol': f"{ticker_upper}.JK",
            'name': INDONESIA_STOCKS.get(ticker_upper, {}).get('name', 'Unknown'),
            'sector': INDONESIA_STOCKS.get(ticker_upper, {}).get('sector', 'Unknown'),
            'exchange': 'IDX',
            'currency': 'IDR',
            'interval': interval,
            'data': chart_data,
            'count': len(chart_data),
            'source': 'Alpha Vantage + Polygon.io Fallback'
        }
        
    except Exception as e:
        import traceback
        return {'error': f'Failed to fetch data: {str(e)}', 'details': traceback.format_exc()}

def analyze_indonesia_stock(ticker: str, interval: str = '1d') -> Dict:
    """
    Analyze Indonesia stock dengan technical indicators
    """
    try:
        # Fetch data
        data_result = fetch_indonesia_stock_data(ticker, interval)
        
        if 'error' in data_result:
            return data_result
        
        # Convert ke DataFrame
        df = pd.DataFrame(data_result['data'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Validate data (minimum 20 candles)
        if len(df) < 20:
            return {'error': f'Insufficient data: only {len(df)} candles available. Need at least 20.'}
        
        # Clean data: remove NaN dalam OHLC
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        df['volume'] = df['volume'].fillna(0)
        
        if len(df) < 20:
            return {'error': f'Insufficient valid data after cleaning: {len(df)} candles. Need at least 20.'}
        
        # Extract arrays dengan proper casting
        close_prices = df['close'].values.astype(float)
        high_prices = df['high'].values.astype(float)
        low_prices = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)
        
        data_len = len(close_prices)
        print(f"   Analyzing {data_len} candles for {ticker}")
        
        # Calculate indicators dengan proper error handling
        # RSI
        rsi_val = None
        try:
            if data_len >= 14:
                rsi = talib.RSI(close_prices, timeperiod=14)
                rsi_val = float(rsi[-1]) if not np.isnan(rsi[-1]) else None
                print(f"   RSI: {rsi_val}")
        except Exception as e:
            print(f"   RSI calculation error: {e}")
        
        # MACD
        macd_val, signal_val, hist_val = None, None, None
        try:
            if data_len >= 26:
                macd, signal, histogram = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
                macd_val = float(macd[-1]) if not np.isnan(macd[-1]) else None
                signal_val = float(signal[-1]) if not np.isnan(signal[-1]) else None
                hist_val = float(histogram[-1]) if not np.isnan(histogram[-1]) else None
                print(f"   MACD: {macd_val}, Signal: {signal_val}, Hist: {hist_val}")
        except Exception as e:
            print(f"   MACD calculation error: {e}")
        
        # Moving Averages
        ma12, ma26, ma50, ma200 = None, None, None, None
        ema9, ema21 = None, None
        try:
            if data_len >= 12:
                ma12 = float(talib.SMA(close_prices, timeperiod=12)[-1])
            if data_len >= 26:
                ma26 = float(talib.SMA(close_prices, timeperiod=26)[-1])
            if data_len >= 50:
                ma50 = float(talib.SMA(close_prices, timeperiod=50)[-1])
            if data_len >= 200:
                ma200 = float(talib.SMA(close_prices, timeperiod=200)[-1])
            if data_len >= 9:
                ema9 = float(talib.EMA(close_prices, timeperiod=9)[-1])
            if data_len >= 21:
                ema21 = float(talib.EMA(close_prices, timeperiod=21)[-1])
            print(f"   MA12: {ma12}, MA26: {ma26}, MA50: {ma50}")
        except Exception as e:
            print(f"   MA calculation error: {e}")
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = None, None, None
        try:
            if data_len >= 20:
                bb_u, bb_m, bb_l = talib.BBANDS(close_prices, timeperiod=20)
                bb_upper = float(bb_u[-1]) if not np.isnan(bb_u[-1]) else None
                bb_middle = float(bb_m[-1]) if not np.isnan(bb_m[-1]) else None
                bb_lower = float(bb_l[-1]) if not np.isnan(bb_l[-1]) else None
                print(f"   BB: upper={bb_upper}, middle={bb_middle}, lower={bb_lower}")
        except Exception as e:
            print(f"   Bollinger Bands calculation error: {e}")
        
        # Stochastic
        stoch_k_val, stoch_d_val = None, None
        try:
            if data_len >= 14:
                stoch_k, stoch_d = talib.STOCH(high_prices, low_prices, close_prices)
                stoch_k_val = float(stoch_k[-1]) if not np.isnan(stoch_k[-1]) else None
                stoch_d_val = float(stoch_d[-1]) if not np.isnan(stoch_d[-1]) else None
                print(f"   Stochastic K: {stoch_k_val}, D: {stoch_d_val}")
        except Exception as e:
            print(f"   Stochastic calculation error: {e}")
        
        # Current price
        current_price = float(close_prices[-1])
        
        # Generate signal
        bullish_count = 0
        bearish_count = 0
        
        # MA trend
        if ma12 is not None and ma26 is not None:
            if current_price > ma12 and current_price > ma26:
                bullish_count += 2
            elif current_price < ma12 and current_price < ma26:
                bearish_count += 2
        
        # RSI
        if rsi_val is not None:
            if rsi_val < 30:
                bullish_count += 1
            elif rsi_val > 70:
                bearish_count += 1
        
        # MACD
        if hist_val is not None:
            if hist_val > 0:
                bullish_count += 1
            else:
                bearish_count += 1
        
        # Stochastic
        if stoch_k_val is not None:
            if stoch_k_val < 20:
                bullish_count += 1
            elif stoch_k_val > 80:
                bearish_count += 1
        
        # Bollinger Bands
        if bb_lower is not None and bb_upper is not None:
            if current_price < bb_lower:
                bullish_count += 1
            elif current_price > bb_upper:
                bearish_count += 1
        
        # Determine signal
        total_signals = bullish_count + bearish_count
        if total_signals == 0:
            signal = 'NEUTRAL'
            confidence = 50
        else:
            if bullish_count > bearish_count:
                signal = 'LONG'
                confidence = (bullish_count / total_signals) * 100
            elif bearish_count > bullish_count:
                signal = 'SHORT'
                confidence = (bearish_count / total_signals) * 100
            else:
                signal = 'NEUTRAL'
                confidence = 50
        
        # Determine trend
        trend = 'Unknown'
        if ma50 is not None and ma200 is not None:
            trend = 'Uptrend' if ma50 > ma200 else 'Downtrend'
        elif ma12 is not None and ma26 is not None:
            trend = 'Uptrend' if ma12 > ma26 else 'Downtrend'
        
        return {
            'ticker': ticker.upper(),
            'symbol': f"{ticker.upper()}.JK",
            'name': INDONESIA_STOCKS.get(ticker.upper(), {}).get('name', 'Unknown'),
            'sector': INDONESIA_STOCKS.get(ticker.upper(), {}).get('sector', 'Unknown'),
            'current_price': current_price,
            'currency': 'IDR',
            'exchange': 'IDX',
            'signal': signal,
            'confidence': round(confidence, 2),
            'trend': trend,
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count,
            'candles_analyzed': len(df),
            'interval': interval,
            'indicators': {
                'rsi': rsi_val,
                'macd': {
                    'macd': macd_val,
                    'signal': signal_val,
                    'histogram': hist_val
                },
                'moving_averages': {
                    'MA12': ma12,
                    'MA26': ma26,
                    'MA50': ma50,
                    'MA200': ma200,
                    'EMA9': ema9,
                    'EMA21': ema21,
                },
                'bollinger_bands': {
                    'upper': bb_upper,
                    'middle': bb_middle,
                    'lower': bb_lower,
                },
                'stochastic': {
                    'k': stoch_k_val,
                    'd': stoch_d_val
                }
            },
            'chart_data': data_result['data']
        }
        
    except Exception as e:
        import traceback
        print(f"❌ Analysis exception: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return {'error': f'Analysis failed: {str(e)}', 'details': traceback.format_exc()}

@app.get("/api/stocks/indonesia/list")
async def list_indonesia_stocks():
    """Get list of available Indonesia stocks"""
    stocks = []
    for ticker, info in INDONESIA_STOCKS.items():
        stocks.append({
            'ticker': ticker,
            'symbol': f"{ticker}.JK",
            'name': info['name'],
            'sector': info['sector']
        })
    
    return {
        'total': len(stocks),
        'stocks': stocks
    }

@app.get("/api/stocks/indonesia/{ticker}/analyze")
async def analyze_indonesia_stock_endpoint(
    ticker: str,
    interval: str = '1d'
):
    """Analyze Indonesia stock using Alpha Vantage API"""
    if ticker.upper() not in INDONESIA_STOCKS:
        raise HTTPException(status_code=404, detail=f"Stock {ticker} not found. Supported: {', '.join(INDONESIA_STOCKS.keys())}")
    
    result = analyze_indonesia_stock(ticker, interval)
    
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    
    return result

@app.get("/api/stocks/indonesia/{ticker}/data")
async def get_indonesia_stock_data_endpoint(
    ticker: str,
    interval: str = '1d'
):
    """Get Indonesia stock OHLCV data using Alpha Vantage + Polygon.io Fallback"""
    if ticker.upper() not in INDONESIA_STOCKS:
        raise HTTPException(status_code=404, detail=f"Stock {ticker} not found")
    
    result = fetch_indonesia_stock_data(ticker, interval)
    
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    
    return result

@app.get("/api/crypto/summary")
async def get_crypto_summary():
    """Get current prices for BTC, ETH, SOL and Fear & Greed Index"""
    try:
        result = {
            "prices": {},
            "fear_greed": None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Fetch current prices from CryptoCompare
        url = "https://min-api.cryptocompare.com/data/pricemultifull"
        
        for symbol in ['BTC', 'ETH', 'SOL']:
            try:
                params = {
                    "fsyms": symbol,
                    "tsyms": "USD"
                }
                response = requests.get(url, params=params, timeout=5)
                response.raise_for_status()
                data = response.json()
                
                if 'DISPLAY' in data and symbol in data['DISPLAY']:
                    price_data = data['DISPLAY'][symbol]['USD']
                    result['prices'][symbol] = {
                        "price": float(price_data['PRICE'].replace('$', '').replace(',', '')),
                        "change_24h": price_data.get('CHANGE24H', 'N/A'),
                        "change_pct_24h": price_data.get('CHANGEPCT24H', 'N/A'),
                        "market_cap": price_data.get('MKTCAP', 'N/A')
                    }
            except Exception as e:
                print(f"Error fetching {symbol} price: {str(e)}")
                result['prices'][symbol] = {"error": str(e)}
        
        # Fetch Fear & Greed Index
        try:
            fng_url = "https://api.alternative.me/fng/"
            fng_response = requests.get(fng_url, timeout=5)
            fng_response.raise_for_status()
            fng_data = fng_response.json()
            
            if 'data' in fng_data and len(fng_data['data']) > 0:
                fng_entry = fng_data['data'][0]
                result['fear_greed'] = {
                    "value": int(fng_entry['value']),
                    "classification": fng_entry['value_classification'],
                    "timestamp": fng_entry['timestamp']
                }
        except Exception as e:
            print(f"Error fetching Fear & Greed Index: {str(e)}")
            result['fear_greed'] = {"error": str(e)}
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching summary: {str(e)}")

# AI Chat Endpoints
@app.post("/api/ai/chat")
async def chat_with_ai(request: AIMessageRequest, token_data = Depends(verify_token)):
    """Send message to AI and get response with automatic model selection"""
    try:
        user_id = token_data.get('user_id')
        
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Get conversation history
        conversation_history = get_conversation_history(user_id, conversation_id)
        
        # Get AI response
        response, model_used, category = get_ai_response(request.message, user_id, conversation_history)
        
        # Save to database
        save_chat_history(user_id, conversation_id, request.message, response, model_used, category)
        
        return AIMessageResponse(
            response=response,
            model_used=model_used,
            question_category=category,
            conversation_id=conversation_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/ai/conversation/{conversation_id}")
async def get_conversation(conversation_id: str, token_data = Depends(verify_token)):
    """Get conversation history"""
    try:
        user_id = token_data.get('user_id')
        
        conn = sqlite3.connect('crypto_scanner.db')
        c = conn.cursor()
        c.execute('''SELECT user_message, ai_response, model_used, question_category, created_at
                    FROM ai_chat_history 
                    WHERE user_id = ? AND conversation_id = ?
                    ORDER BY created_at ASC''',
                 (user_id, conversation_id))
        rows = c.fetchall()
        conn.close()
        
        messages = []
        for user_msg, ai_resp, model, category, timestamp in rows:
            messages.append({
                "type": "user",
                "content": user_msg,
                "timestamp": timestamp
            })
            messages.append({
                "type": "assistant",
                "content": ai_resp,
                "model": model,
                "category": category,
                "timestamp": timestamp
            })
        
        return {
            "conversation_id": conversation_id,
            "messages": messages
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/ai/conversations")
async def list_conversations(token_data = Depends(verify_token)):
    """Get list of all conversations for user"""
    try:
        user_id = token_data.get('user_id')
        
        conn = sqlite3.connect('crypto_scanner.db')
        c = conn.cursor()
        c.execute('''SELECT conversation_id, MAX(created_at) as last_time, ai_response
                    FROM ai_chat_history 
                    WHERE user_id = ?
                    GROUP BY conversation_id
                    ORDER BY last_time DESC''',
                 (user_id,))
        rows = c.fetchall()
        conn.close()
        
        conversations = []
        for row in rows:
            if row:  # Ensure row exists
                conv_id = row[0]
                last_msg = row[2] if len(row) > 2 else ""
                conversations.append({
                    "id": conv_id,
                    "last_message": last_msg
                })
        
        return {"conversations": conversations}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.delete("/api/ai/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str, token_data = Depends(verify_token)):
    """Delete a conversation"""
    try:
        user_id = token_data.get('user_id')
        
        conn = sqlite3.connect('crypto_scanner.db')
        c = conn.cursor()
        c.execute('DELETE FROM ai_chat_history WHERE user_id = ? AND conversation_id = ?',
                 (user_id, conversation_id))
        conn.commit()
        conn.close()
        
        return {"status": "conversation deleted"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2401)