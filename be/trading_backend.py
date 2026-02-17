"""
Trading Decision System - FastAPI Backend
==========================================
Main backend server handling authentication, data processing, and API integration.

Features:
- User authentication with JWT tokens
- Role-Based Access Control (RBAC)
- SQLite database for user management
- Integration with DataSectors API
- Technical indicator calculations
- Trading signal generation
"""

from fastapi import FastAPI, HTTPException, Depends, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import sqlite3
import hashlib
import secrets
import jwt
import httpx
from enum import Enum
import pandas as pd
import numpy as np
import json
import logging

# ============================================
# LOGGING CONFIGURATION
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

# DataSectors API Configuration
DATASECTORS_API_URL = "http://148.230.96.135:5672"
DATASECTORS_API_KEY = "sangahli"  # X-API-Key header value

# ============================================
# FASTAPI INITIALIZATION
# ============================================

app = FastAPI(
    title="Trading Decision System API",
    description="Backend API for Indonesian Crypto & Stock Trading Decision System",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# ============================================
# DATABASE SETUP
# ============================================

def init_db():
    """Initialize SQLite database with users table"""
    conn = sqlite3.connect('trading_system.db')
    cursor = conn.cursor()
    
    # Users table with RBAC
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # Trading history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trading_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            price REAL,
            quantity REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            indicators TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Watchlist table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            market TEXT NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            UNIQUE(user_id, symbol)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# ============================================
# PYDANTIC MODELS
# ============================================

class UserRole(str, Enum):
    ADMIN = "admin"
    PREMIUM = "premium"
    USER = "user"

class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    
    @validator('username')
    def username_validator(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        return v
    
    @validator('password')
    def password_validator(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters')
        return v

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: Dict[str, Any]

class TradingSignalRequest(BaseModel):
    symbol: str
    market: str  # 'crypto' or 'stock'
    timeframe: str = 'D'  # '1m', '5m', '15m', '30m', '1h', '4h', 'D', 'W', 'M'
    indicators: List[str] = ['ma', 'rsi', 'macd', 'stoch', 'bb', 'atr', 'volume']
    max_candles: int = 1000  # Request max 1000 candles for better analysis
    
    @validator('timeframe')
    def validate_timeframe(cls, v):
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', 'D', 'W', 'M']
        if v not in valid_timeframes:
            raise ValueError(f'Invalid timeframe. Must be one of: {valid_timeframes}')
        return v

class ScannerRequest(BaseModel):
    market: str
    timeframe: str = 'D'  # '1m', '5m', '15m', '30m', '1h', '4h', 'D', 'W', 'M'
    min_volume: Optional[float] = None
    indicators: List[str] = ['ma', 'rsi', 'macd', 'stoch', 'bb', 'atr', 'volume']
    
    @validator('timeframe')
    def validate_timeframe(cls, v):
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', 'D', 'W', 'M']
        if v not in valid_timeframes:
            raise ValueError(f'Invalid timeframe. Must be one of: {valid_timeframes}')
        return v

# ============================================
# UTILITY FUNCTIONS
# ============================================

def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return hash_password(plain_password) == hashed_password

def create_access_token(data: dict) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> Dict:
    """Decode JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Get current authenticated user from token"""
    token = credentials.credentials
    payload = decode_token(token)
    
    conn = sqlite3.connect('trading_system.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, email, role FROM users WHERE id = ? AND is_active = 1', 
                   (payload['user_id'],))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return {
        'id': user[0],
        'username': user[1],
        'email': user[2],
        'role': user[3]
    }

async def call_datasectors_api(endpoint: str, params: Dict = None) -> Dict:
    """Call DataSectors API with authentication and detailed logging"""
    headers = {
        "X-API-Key": DATASECTORS_API_KEY,
        "Content-Type": "application/json"
    }
    
    url = f"{DATASECTORS_API_URL}{endpoint}"
    
    # Log the request
    logger.info(f"📡 API Call: {endpoint}")
    logger.debug(f"   URL: {url}")
    logger.debug(f"   Params: {params}")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, params=params, headers=headers)
            
            # Log response status
            logger.info(f"   ✅ Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                if not isinstance(data, dict):
                    logger.error(f"   ❌ Invalid response type: {type(data)}")
                    raise HTTPException(status_code=500, detail="Invalid API response format")
                
                # Log success with data preview
                if 'data' in data:
                    data_len = len(data['data']) if isinstance(data['data'], list) else 'N/A'
                    logger.info(f"   📊 Data received: {data_len} items")
                
                return data
            else:
                logger.error(f"   ❌ API Error {response.status_code}: {response.text[:200]}")
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"DataSectors API Error: {response.text[:200]}"
                )
                
        except httpx.TimeoutException:
            logger.error(f"   ⏱️ Timeout calling {endpoint}")
            raise HTTPException(status_code=504, detail="API request timeout")
        except httpx.HTTPError as e:
            logger.error(f"   ❌ HTTP Error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"API Connection Error: {str(e)}")
        except Exception as e:
            logger.error(f"   ❌ Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# ============================================
# AUTHENTICATION ENDPOINTS
# ============================================

@app.post("/api/auth/register", response_model=Token, tags=["Authentication"])
async def register(user: UserRegister):
    """
    Register a new user
    
    - **username**: Unique username (min 3 characters)
    - **email**: Valid email address
    - **password**: Password (min 6 characters)
    """
    conn = sqlite3.connect('trading_system.db')
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', 
                   (user.username, user.email))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Username or email already exists")
    
    # Create user
    password_hash = hash_password(user.password)
    cursor.execute('''
        INSERT INTO users (username, email, password_hash, role)
        VALUES (?, ?, ?, ?)
    ''', (user.username, user.email, password_hash, UserRole.USER))
    
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    # Create token
    access_token = create_access_token({"user_id": user_id})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "username": user.username,
            "email": user.email,
            "role": UserRole.USER
        }
    }

@app.post("/api/auth/login", response_model=Token, tags=["Authentication"])
async def login(user: UserLogin):
    """
    Login user and return JWT token
    
    - **username**: User's username
    - **password**: User's password
    """
    conn = sqlite3.connect('trading_system.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, username, email, password_hash, role FROM users WHERE username = ? AND is_active = 1',
                   (user.username,))
    db_user = cursor.fetchone()
    
    if not db_user or not verify_password(user.password, db_user[3]):
        conn.close()
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Update last login
    cursor.execute('UPDATE users SET last_login = ? WHERE id = ?', 
                   (datetime.now(), db_user[0]))
    conn.commit()
    conn.close()
    
    # Create token
    access_token = create_access_token({"user_id": db_user[0]})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": db_user[0],
            "username": db_user[1],
            "email": db_user[2],
            "role": db_user[4]
        }
    }

@app.get("/api/auth/me", tags=["Authentication"])
async def get_me(current_user: Dict = Depends(get_current_user)):
    """Get current user information"""
    return current_user

# ============================================
# TECHNICAL INDICATORS
# ============================================

class TechnicalIndicators:
    """Calculate technical indicators for trading signals"""
    
    @staticmethod
    def calculate_ma(prices: List[float], period: int = 20) -> List[float]:
        """Calculate Moving Average"""
        df = pd.DataFrame({'close': prices})
        ma = df['close'].rolling(window=period).mean()
        # Convert to native Python types and handle NaN
        return [float(x) if not pd.isna(x) else None for x in ma.tolist()]
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
        """Calculate Relative Strength Index"""
        df = pd.DataFrame({'close': prices})
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        # Convert to native Python types and handle NaN
        return [float(x) if not pd.isna(x) else None for x in rsi.tolist()]
    
    @staticmethod
    def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD"""
        df = pd.DataFrame({'close': prices})
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return {
            'macd': [float(x) if not pd.isna(x) else None for x in macd.tolist()],
            'signal': [float(x) if not pd.isna(x) else None for x in signal_line.tolist()],
            'histogram': [float(x) if not pd.isna(x) else None for x in histogram.tolist()]
        }
    
    @staticmethod
    def calculate_stochastic_rsi(prices: List[float], period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Dict:
        """Calculate Stochastic RSI"""
        rsi = TechnicalIndicators.calculate_rsi(prices, period)
        rsi_series = pd.Series([x for x in rsi if x is not None])
        
        min_rsi = rsi_series.rolling(window=period).min()
        max_rsi = rsi_series.rolling(window=period).max()
        
        stoch_rsi = (rsi_series - min_rsi) / (max_rsi - min_rsi) * 100
        k = stoch_rsi.rolling(window=smooth_k).mean()
        d = k.rolling(window=smooth_d).mean()
        
        return {
            'k': [float(x) if not pd.isna(x) else None for x in k.tolist()],
            'd': [float(x) if not pd.isna(x) else None for x in d.tolist()]
        }
    
    @staticmethod
    def identify_order_blocks(highs: List[float], lows: List[float], closes: List[float]) -> List[Dict]:
        """Identify Order Blocks (institutional buying/selling zones)"""
        order_blocks = []
        
        for i in range(3, len(closes) - 1):
            # Bullish Order Block: Strong up move after consolidation
            if closes[i] > closes[i-1] and closes[i] > closes[i-2]:
                pct_change = (closes[i] - closes[i-1]) / closes[i-1]
                if pct_change > 0.02:  # 2% move
                    order_blocks.append({
                        'type': 'bullish',
                        'index': int(i),
                        'low': float(lows[i-1]),
                        'high': float(highs[i-1]),
                        'strength': 'strong' if pct_change > 0.05 else 'medium'
                    })
            
            # Bearish Order Block: Strong down move after consolidation
            if closes[i] < closes[i-1] and closes[i] < closes[i-2]:
                pct_change = (closes[i-1] - closes[i]) / closes[i-1]
                if pct_change > 0.02:  # 2% move
                    order_blocks.append({
                        'type': 'bearish',
                        'index': int(i),
                        'low': float(lows[i-1]),
                        'high': float(highs[i-1]),
                        'strength': 'strong' if pct_change > 0.05 else 'medium'
                    })
        
        return order_blocks[-10:]  # Return last 10 order blocks
    
    @staticmethod
    def calculate_bollinger_bands(closes: List[float], period: int = 20, std_dev: float = 2) -> Dict:
        """
        Calculate Bollinger Bands for volatility analysis
        
        Returns:
            - upper_band: Upper band (buy signal when price touches)
            - middle_band: SMA
            - lower_band: Lower band (sell signal when price touches)
            - bb_width: Band width for volatility measure
            - bb_position: Where price sits in bands (0-1, 0=lower, 1=upper)
        """
        df = pd.DataFrame({'close': closes})
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        # Calculate BB Width (volatility indicator)
        bb_width = (upper_band - lower_band) / sma * 100
        
        # Calculate BB Position (0=lower, 1=upper)
        bb_position = (df['close'] - lower_band) / (upper_band - lower_band)
        
        return {
            'upper': [float(x) if not pd.isna(x) else None for x in upper_band.tolist()],
            'middle': [float(x) if not pd.isna(x) else None for x in sma.tolist()],
            'lower': [float(x) if not pd.isna(x) else None for x in lower_band.tolist()],
            'width': [float(x) if not pd.isna(x) else None for x in bb_width.tolist()],
            'position': [float(x) if not pd.isna(x) else None for x in bb_position.tolist()]
        }
    
    @staticmethod
    def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
        """
        Calculate Average True Range for volatility and risk assessment
        
        ATR > high: High volatility (wider stops needed)
        ATR < low: Low volatility (tighter stops ok)
        """
        df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
        
        # Calculate True Range
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        atr = df['tr'].rolling(window=period).mean()
        
        return [float(x) if not pd.isna(x) else None for x in atr.tolist()]
    
    @staticmethod
    def calculate_stochastic(highs: List[float], lows: List[float], closes: List[float], 
                           period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Dict:
        """
        Calculate Stochastic Oscillator (different from Stochastic RSI)
        
        %K: Current momentum
        %D: Signal line (SMA of %K)
        
        Rules:
        - %K > 80: Overbought (sell signal)
        - %K < 20: Oversold (buy signal)
        - Crossover of %K above %D: Bullish
        - Crossover of %K below %D: Bearish
        """
        df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
        
        lowest_low = df['low'].rolling(window=period).min()
        highest_high = df['high'].rolling(window=period).max()
        
        k_percent = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=smooth_d).mean()
        
        # Smooth K
        k_smooth = k_percent.rolling(window=smooth_k).mean()
        
        return {
            'k': [float(x) if not pd.isna(x) else None for x in k_smooth.tolist()],
            'd': [float(x) if not pd.isna(x) else None for x in d_percent.tolist()]
        }
    
    @staticmethod
    def calculate_support_resistance(highs: List[float], lows: List[float], period: int = 20) -> Dict:
        """
        Identify Support and Resistance levels based on recent price action
        
        Support: Price level where buying interest emerges (stops downtrend)
        Resistance: Price level where selling interest emerges (stops uptrend)
        """
        recent_highs = highs[-period:]
        recent_lows = lows[-period:]
        
        resistance = float(max(recent_highs))
        support = float(min(recent_lows))
        
        current_price = float(highs[-1])
        
        # Calculate distance to levels (percentage)
        resistance_distance = ((resistance - current_price) / current_price * 100) if current_price > 0 else 0
        support_distance = ((current_price - support) / current_price * 100) if current_price > 0 else 0
        
        return {
            'resistance': resistance,
            'support': support,
            'current_price': current_price,
            'resistance_distance': float(resistance_distance),
            'support_distance': float(support_distance),
            'price_position': 'near_resistance' if resistance_distance < 2 else 'near_support' if support_distance < 2 else 'mid_range'
        }
    
    @staticmethod
    def calculate_volume_sma(volumes: List[float], period: int = 20) -> Dict:
        """
        Calculate Volume Moving Average for volume analysis
        
        Returns volume trend and strength
        """
        df = pd.DataFrame({'volume': volumes})
        volume_sma = df['volume'].rolling(window=period).mean()
        
        current_volume = float(volumes[-1])
        avg_volume = float(volume_sma.iloc[-1]) if not pd.isna(volume_sma.iloc[-1]) else current_volume
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Volume trend
        if volume_ratio > 1.5:
            volume_trend = 'spike'  # Unusual volume
        elif volume_ratio > 1.2:
            volume_trend = 'above_average'
        elif volume_ratio > 0.8:
            volume_trend = 'average'
        else:
            volume_trend = 'below_average'
        
        return {
            'current': float(current_volume),
            'average': float(avg_volume),
            'ratio': float(volume_ratio),
            'trend': volume_trend,
            'sma': [float(x) if not pd.isna(x) else None for x in volume_sma.tolist()]
        }
    
    @staticmethod
    def smart_money_concept(highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> Dict:
        """Analyze Smart Money Concepts"""
        df = pd.DataFrame({
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        # Break of Structure (BOS)
        recent_high = float(df['high'].tail(20).max())
        recent_low = float(df['low'].tail(20).min())
        current_price = float(closes[-1])
        
        # Volume analysis
        avg_volume = float(df['volume'].tail(20).mean())
        current_volume = float(volumes[-1])
        volume_spike = bool(current_volume > avg_volume * 1.5)
        
        # Market Structure
        if current_price > recent_high * 0.95:
            market_structure = 'bullish'
        elif current_price < recent_low * 1.05:
            market_structure = 'bearish'
        else:
            market_structure = 'neutral'
        
        return {
            'market_structure': market_structure,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'volume_spike': volume_spike,
            'volume_ratio': float(current_volume / avg_volume if avg_volume > 0 else 0)
        }

# ============================================
# LAYERED FILTERING SYSTEM (CHEAP FIRST → EXPENSIVE LATER)
# ============================================

class LayerFilterResult:
    """Result from each filter layer"""
    def __init__(self, passed: bool, reason: str = ""):
        self.passed = passed
        self.reason = reason

async def layer1_price_structure(data: Dict, closes: List[float], current_price: float, 
                                   ma: float, indicators: Dict, symbol: str, market: str) -> Dict:
    """
    LAYER 1: Price & Structure (WAJIB - Cheap first)
    - Trend direction (EMA slope)
    - Market regime (range vs trend)
    - ATR / volatility
    - Support/Resistance levels
    
    Hit endpoint: /api/chart/price (already have it)
    """
    logger.info(f"LAYER 1 [Price & Structure] for {symbol} ({market})")
    
    # Already have price data, just analyze structure
    recent_high = max(closes[-20:]) if len(closes) >= 20 else max(closes)
    recent_low = min(closes[-20:]) if len(closes) >= 20 else min(closes)
    
    # Trend direction
    ma_trend = "up" if current_price > ma else "down"
    
    # Market regime (range vs trend)
    range_size = recent_high - recent_low
    range_percent = (range_size / current_price * 100) if current_price > 0 else 0
    
    # ATR / volatility
    atr_list = indicators.get('atr', [])
    atr_val = float([x for x in atr_list if x is not None][-1]) if atr_list and any(x is not None for x in atr_list) else 0
    atr_percent = (atr_val / current_price * 100) if current_price > 0 else 0
    
    # Support/Resistance
    sr = indicators.get('support_resistance', {})
    resistance = sr.get('resistance', recent_high)
    support = sr.get('support', recent_low)
    
    layer1_data = {
        'layer': 1,
        'name': 'Price & Structure',
        'passed': True,  # Layer 1 always passes (filtering happens in Layer 2)
        'trend': ma_trend,
        'range_percent': range_percent,
        'volatility_percent': atr_percent,
        'support': support,
        'resistance': resistance,
        'recent_high': recent_high,
        'recent_low': recent_low,
        'reason': f"Trend: {ma_trend}, Range: {range_percent:.1f}%, Vol: {atr_percent:.2f}%"
    }
    
    logger.info(f"  ✓ Layer 1 passed: {layer1_data['reason']}")
    return layer1_data

async def layer2_volume_confirmation(current_price: float, closes: List[float], 
                                     volumes: List[float], symbol: str, market: str) -> Dict:
    """
    LAYER 2: Volume Confirmation (WAJIB - Cheap)
    
    Hit endpoints:
    - /api/indicator/volume/vwap (anchor institusional)
    - /api/indicator/volume/cmf (partisipasi)
    
    Filter rule: Must have volume + VWAP confirmation
    Majority of pair should drop here if volume weak
    """
    logger.info(f"LAYER 2 [Volume Confirmation] for {symbol} ({market})")
    
    try:
        # Get VWAP from DataSectors API
        vwap_response = await call_datasectors_api(
            "/api/indicator/volume/vwap",
            {
                "symbol": symbol,
                "timeframe": "D",  # Daily for consistency
                "range": 50
            }
        )
        
        vwap = None
        if vwap_response and isinstance(vwap_response, dict) and 'data' in vwap_response:
            vwap_data = vwap_response['data']
            if isinstance(vwap_data, list) and len(vwap_data) > 0:
                vwap = float(vwap_data[-1]) if vwap_data[-1] is not None else current_price
        
        if not vwap:
            vwap = current_price
        
        # Get CMF from DataSectors API
        cmf_response = await call_datasectors_api(
            "/api/indicator/volume/cmf",
            {
                "symbol": symbol,
                "timeframe": "D",
                "range": 50,
                "period": 20
            }
        )
        
        cmf = None
        cmf_strength = "weak"
        if cmf_response and isinstance(cmf_response, dict) and 'data' in cmf_response:
            cmf_data = cmf_response['data']
            if isinstance(cmf_data, list) and len(cmf_data) > 0:
                cmf = float(cmf_data[-1]) if cmf_data[-1] is not None else 0
                
                # CMF interpretation
                if cmf > 0.25:
                    cmf_strength = "strong_buying"
                elif cmf > 0.05:
                    cmf_strength = "moderate_buying"
                elif cmf < -0.25:
                    cmf_strength = "strong_selling"
                elif cmf < -0.05:
                    cmf_strength = "moderate_selling"
                else:
                    cmf_strength = "neutral"
        
        if not cmf:
            cmf = 0
        
        # Volume check
        avg_volume = float(pd.Series(volumes).tail(20).mean()) if len(volumes) >= 20 else float(np.mean(volumes))
        current_volume = float(volumes[-1]) if volumes else 0
        volume_ratio = float(current_volume / avg_volume) if avg_volume > 0 else 1.0
        
        # Price position vs VWAP
        price_vs_vwap = "above_vwap" if current_price > vwap else "below_vwap"
        vwap_distance = abs(current_price - vwap) / current_price * 100
        
        # FILTER RULE: Accept if volume is decent OR CMF shows conviction
        passed = (volume_ratio >= 0.8) or (abs(cmf) > 0.1)
        
        layer2_data = {
            'layer': 2,
            'name': 'Volume Confirmation',
            'passed': passed,
            'vwap': float(vwap),
            'cmf': float(cmf),
            'cmf_strength': cmf_strength,
            'volume_ratio': volume_ratio,
            'price_vs_vwap': price_vs_vwap,
            'vwap_distance': vwap_distance,
            'reason': f"VWAP: ${vwap:.2f} ({price_vs_vwap}, {vwap_distance:.1f}%), CMF: {cmf:.3f} ({cmf_strength}), Vol: {volume_ratio:.1f}x"
        }
        
        if passed:
            logger.info(f"  ✓ Layer 2 passed: {layer2_data['reason']}")
        else:
            logger.warning(f"  ✗ Layer 2 REJECTED: Weak volume ({volume_ratio:.1f}x) and weak CMF ({cmf:.3f})")
        
        return layer2_data
    
    except Exception as e:
        logger.error(f"Layer 2 error: {str(e)}")
        # Fallback - pass with neutral data
        return {
            'layer': 2,
            'name': 'Volume Confirmation',
            'passed': True,
            'vwap': current_price,
            'cmf': 0,
            'cmf_strength': 'unknown',
            'volume_ratio': 1.0,
            'price_vs_vwap': 'neutral',
            'vwap_distance': 0,
            'reason': f"Endpoint error - using fallback"
        }

async def layer3_pre_filter(symbol: str, market: str, closes: List[float]) -> Dict:
    """
    LAYER 3: Strong Trend Pre-filter (FILTER - Expensive but saves cost later)
    
    Hit endpoint: /api/crypto/strong-trend
    
    Purpose:
    - Filter hanya 20 pair dari 300 yang sedang trending
    - Menghemat hit ke endpoint expensive lain
    - Cocok untuk breakout/trend-following
    """
    logger.info(f"LAYER 3 [Pre-filter - Strong Trend] for {symbol} ({market})")
    
    try:
        if market == "crypto":
            # Get strong trend coins
            strong_trend_response = await call_datasectors_api(
                "/api/crypto/strong-trend",
                {}
            )
            
            if strong_trend_response and isinstance(strong_trend_response, dict):
                data = strong_trend_response.get('data', {})
                coins = data.get('coins', [])
                
                # Check if symbol is in strong trend list
                symbol_check = symbol.split(':')[0] if ':' in symbol else symbol
                is_trending = any(coin.get('symbol', '').upper() == symbol_check.upper() for coin in coins)
                
                layer3_data = {
                    'layer': 3,
                    'name': 'Pre-filter (Strong Trend)',
                    'passed': is_trending,
                    'is_trending': is_trending,
                    'reason': f"Symbol {'in' if is_trending else 'not in'} strong trend list"
                }
                
                if is_trending:
                    logger.info(f"  ✓ Layer 3 passed: {symbol} is trending")
                else:
                    logger.warning(f"  ✗ Layer 3 REJECTED: {symbol} not in strong trend")
                
                return layer3_data
        
        # For stocks or if crypto check fails, analyze trend from price
        recent_closes = closes[-20:] if len(closes) >= 20 else closes
        sma = float(np.mean(recent_closes))
        current = float(closes[-1])
        trend_strength = abs(current - sma) / sma * 100
        
        # Pass if trend is > 1%
        passed = trend_strength > 1.0
        
        layer3_data = {
            'layer': 3,
            'name': 'Pre-filter (Strong Trend)',
            'passed': passed,
            'trend_strength': trend_strength,
            'reason': f"Trend strength: {trend_strength:.2f}%"
        }
        
        if passed:
            logger.info(f"  ✓ Layer 3 passed: {layer3_data['reason']}")
        else:
            logger.warning(f"  ✗ Layer 3 REJECTED: {layer3_data['reason']}")
        
        return layer3_data
    
    except Exception as e:
        logger.error(f"Layer 3 error: {str(e)}")
        # Fallback - pass anyway
        return {
            'layer': 3,
            'name': 'Pre-filter (Strong Trend)',
            'passed': True,
            'reason': 'Endpoint error - fallback pass'
        }

async def layer4_confirmation(symbol: str, market: str, current_price: float, 
                            closes: List[float], indicators: Dict) -> Dict:
    """
    LAYER 4: Confirmation (Only if Layer 1-3 passed)
    
    Hit endpoints:
    - /api/crypto/orderbook-imbalance (jangka pendek)
    - /api/crypto/walls (breakout validation)
    
    Purpose:
    - Jangan hit endpoint expensive untuk pair yang sudah gugur
    - Hanya untuk candidate terbaik
    - Validasi breakout atau identify support/resistance
    """
    logger.info(f"LAYER 4 [Confirmation] for {symbol} ({market})")
    
    try:
        if market != "crypto":
            # Stocks don't have these endpoints
            logger.info(f"  ℹ️ Layer 4 skipped: Stock market - using structure confirmation only")
            return {
                'layer': 4,
                'name': 'Confirmation',
                'passed': True,
                'imbalance': None,
                'walls': None,
                'reason': 'Stock market - using structure confirmation'
            }
        
        # Get orderbook imbalance
        imbalance_response = await call_datasectors_api(
            "/api/crypto/orderbook-imbalance",
            {
                "symbol": symbol,
                "limit": 100
            }
        )
        
        imbalance_data = {}
        if imbalance_response and isinstance(imbalance_response, dict) and 'data' in imbalance_response:
            imbalance_data = imbalance_response.get('data', {})
        
        # Get orderbook walls
        walls_response = await call_datasectors_api(
            "/api/crypto/walls",
            {
                "symbol": symbol,
                "limit": 100
            }
        )
        
        walls_data = {}
        if walls_response and isinstance(walls_response, dict) and 'data' in walls_response:
            walls_data = walls_response.get('data', {})
        
        buy_walls = walls_data.get('buy_walls', [])
        sell_walls = walls_data.get('sell_walls', [])
        
        # Confirmation logic
        has_support = len(buy_walls) > 0
        has_resistance = len(sell_walls) > 0
        
        layer4_data = {
            'layer': 4,
            'name': 'Confirmation',
            'passed': has_support or has_resistance,
            'buy_walls': len(buy_walls),
            'sell_walls': len(sell_walls),
            'imbalance': imbalance_data,
            'reason': f"Buy walls: {len(buy_walls)}, Sell walls: {len(sell_walls)}"
        }
        
        if layer4_data['passed']:
            logger.info(f"  ✓ Layer 4 confirmed: {layer4_data['reason']}")
        else:
            logger.warning(f"  ⚠️ Layer 4 inconclusive: {layer4_data['reason']}")
        
        return layer4_data
    
    except Exception as e:
        logger.error(f"Layer 4 error: {str(e)}")
        return {
            'layer': 4,
            'name': 'Confirmation',
            'passed': True,
            'reason': 'Endpoint error - fallback to indicators',
            'imbalance': None,
            'walls': None
        }

# ============================================
# TRADING SIGNAL GENERATION (WITH LAYERING)
# ============================================

def generate_trading_signal(data: Dict, indicators: Dict, closes: List[float] = None, 
                           layers_result: Dict = None) -> Dict:
    """
    Generate trading signal based on multiple indicators
    
    Rules:
    - LONG: MA crossover + RSI < 70 + MACD positive + Volume spike + Bullish structure
    - SHORT: MA crossunder + RSI > 30 + MACD negative + Volume spike + Bearish structure
    
    Args:
        data: Original data dict (fallback only)
        indicators: Calculated technical indicators
        closes: Pre-validated close prices array (preferred over extracting from data)
    """
    # Use provided closes if available, otherwise extract from data
    if closes is None:
        closes = [float(candle['close']) for candle in data['data']]
    
    current_price = float(closes[-1])
    
    # Get indicator values - handle None values
    ma_values = [x for x in indicators.get('ma', []) if x is not None]
    ma = float(ma_values[-1]) if ma_values else current_price
    
    rsi_values = [x for x in indicators.get('rsi', []) if x is not None]
    rsi = float(rsi_values[-1]) if rsi_values else 50.0
    
    macd_data = indicators.get('macd', {})
    macd_hist_values = [x for x in macd_data.get('histogram', []) if x is not None]
    macd_hist = float(macd_hist_values[-1]) if macd_hist_values else 0.0
    
    macd_values = [x for x in macd_data.get('macd', []) if x is not None]
    macd_line = float(macd_values[-1]) if macd_values else 0.0
    
    signal_values = [x for x in macd_data.get('signal', []) if x is not None]
    signal_line = float(signal_values[-1]) if signal_values else 0.0
    
    smc = indicators.get('smart_money_concept', {})
    
    # Scoring system (max ~20 points)
    long_score = 0
    short_score = 0
    long_reasons = []
    short_reasons = []
    
    # ============================================
    # 1. TREND ANALYSIS (Moving Average)
    # ============================================
    if current_price > ma:
        long_score += 2
        long_reasons.append(f"📈 Price (${current_price:.2f}) above MA (${ma:.2f}) - Bullish trend")
    else:
        short_score += 2
        short_reasons.append(f"📉 Price (${current_price:.2f}) below MA (${ma:.2f}) - Bearish trend")
    
    # ============================================
    # 2. MOMENTUM ANALYSIS (RSI)
    # ============================================
    if rsi < 30:
        long_score += 3
        long_reasons.append(f"🔥 RSI oversold at {rsi:.1f} (< 30) - Strong buy opportunity")
    elif rsi < 50:
        long_score += 1
        long_reasons.append(f"⬆️ RSI neutral-low at {rsi:.1f} - Moderate bullish")
    elif rsi > 70:
        short_score += 3
        short_reasons.append(f"❄️ RSI overbought at {rsi:.1f} (> 70) - Strong sell opportunity")
    elif rsi > 50:
        short_score += 1
        short_reasons.append(f"⬇️ RSI neutral-high at {rsi:.1f} - Moderate bearish")
    
    # ============================================
    # 3. MOMENTUM CONFIRMATION (MACD)
    # ============================================
    if macd_hist > 0:
        long_score += 2
        long_reasons.append(f"✅ MACD histogram positive ({macd_hist:.6f}) - Bullish momentum")
        if macd_line > signal_line:
            long_score += 1
            long_reasons.append(f"⬆️ MACD line (${macd_line:.6f}) above signal (${signal_line:.6f}) - Strong buy")
    else:
        short_score += 2
        short_reasons.append(f"❌ MACD histogram negative ({macd_hist:.6f}) - Bearish momentum")
        if macd_line < signal_line:
            short_score += 1
            short_reasons.append(f"⬇️ MACD line (${macd_line:.6f}) below signal (${signal_line:.6f}) - Strong sell")
    
    # ============================================
    # 4. STOCHASTIC ANALYSIS
    # ============================================
    stoch = indicators.get('stochastic', {})
    stoch_k = stoch.get('k', [])
    stoch_k_val = float([x for x in stoch_k if x is not None][-1]) if stoch_k and any(x is not None for x in stoch_k) else 50
    
    if stoch_k_val < 20:
        long_score += 2
        long_reasons.append(f"🔥 Stochastic oversold ({stoch_k_val:.1f} < 20) - Reversal potential")
    elif stoch_k_val > 80:
        short_score += 2
        short_reasons.append(f"❄️ Stochastic overbought ({stoch_k_val:.1f} > 80) - Reversal potential")
    
    # ============================================
    # 5. BOLLINGER BANDS (Volatility & Reversal)
    # ============================================
    bb = indicators.get('bollinger_bands', {})
    bb_position_list = bb.get('position', [])
    bb_position = float([x for x in bb_position_list if x is not None][-1]) if bb_position_list and any(x is not None for x in bb_position_list) else 0.5
    
    if bb_position < 0.1:
        long_score += 2
        long_reasons.append(f"📍 Price near lower BB ({bb_position:.2f}) - Potential bounce/reversal")
    elif bb_position > 0.9:
        short_score += 2
        short_reasons.append(f"📍 Price near upper BB ({bb_position:.2f}) - Potential pullback")
    
    # ============================================
    # 6. SUPPORT & RESISTANCE
    # ============================================
    sr = indicators.get('support_resistance', {})
    resistance = sr.get('resistance', current_price)
    support = sr.get('support', current_price)
    sr_position = sr.get('price_position', 'mid_range')
    
    if sr_position == 'near_support':
        long_score += 2
        long_reasons.append(f"🛡️ Price near support level (${support:.2f}) - Buying interest expected")
    elif sr_position == 'near_resistance':
        short_score += 2
        short_reasons.append(f"🚫 Price near resistance (${resistance:.2f}) - Selling pressure expected")
    
    # ============================================
    # 7. VOLUME ANALYSIS
    # ============================================
    vol_analysis = indicators.get('volume_analysis', {})
    volume_trend = vol_analysis.get('trend', 'average')
    volume_ratio = vol_analysis.get('ratio', 1.0)
    
    if volume_trend == 'spike':
        if long_score > short_score:
            long_score += 2
            long_reasons.append(f"📊 Volume spike ({volume_ratio:.1f}x) with bullish signals - Strong buying pressure")
        else:
            short_score += 2
            short_reasons.append(f"📊 Volume spike ({volume_ratio:.1f}x) with bearish signals - Strong selling pressure")
    elif volume_trend == 'above_average':
        if long_score > short_score:
            long_score += 1
            long_reasons.append(f"📈 Above average volume ({volume_ratio:.1f}x) - Confirms uptrend")
        else:
            short_score += 1
            short_reasons.append(f"📉 Above average volume ({volume_ratio:.1f}x) - Confirms downtrend")
    elif volume_trend == 'below_average':
        if long_score > short_score:
            long_reasons.append(f"⚠️ Below average volume - Weak uptrend confirmation")
        else:
            short_reasons.append(f"⚠️ Below average volume - Weak downtrend confirmation")
    
    # ============================================
    # 8. SMART MONEY CONCEPTS
    # ============================================
    smc = indicators.get('smart_money_concept', {})
    market_structure = smc.get('market_structure', 'neutral')
    
    if market_structure == 'bullish':
        long_score += 2
        long_reasons.append(f"🚀 Bullish structure - Higher highs and higher lows")
    elif market_structure == 'bearish':
        short_score += 2
        short_reasons.append(f"📉 Bearish structure - Lower highs and lower lows")
    
    # ============================================
    # 9. ATR (Volatility Assessment)
    # ============================================
    atr_list = indicators.get('atr', [])
    atr_val = float([x for x in atr_list if x is not None][-1]) if atr_list and any(x is not None for x in atr_list) else 0
    atr_percent = (atr_val / current_price * 100) if current_price > 0 else 0
    
    if atr_percent > 3:
        long_reasons.append(f"⚡ High volatility ({atr_percent:.2f}%) - Use wider stops, expect big moves")
        short_reasons.append(f"⚡ High volatility ({atr_percent:.2f}%) - Use wider stops, expect big moves")
    elif atr_percent < 0.5:
        long_reasons.append(f"😴 Low volatility ({atr_percent:.2f}%) - Tighter stops possible")
        short_reasons.append(f"😴 Low volatility ({atr_percent:.2f}%) - Tighter stops possible")
    
    # ============================================
    # DETERMINE FINAL SIGNAL
    # ============================================
    total_score = abs(long_score - short_score)
    
    if long_score >= 10:
        action = "🚀 STRONG BUY"
        color = "green"
        recommendation = f"Strong bullish signal (Score: {long_score}) - Consider entering long position with confidence. Look for pullback to MA for better entry."
        active_reasons = long_reasons
    elif long_score > short_score and long_score >= 6:
        action = "✅ BUY"
        color = "lightgreen"
        recommendation = f"Moderate bullish signal (Score: {long_score}) - Consider buying on confirmation. Wait for support test before entering."
        active_reasons = long_reasons
    elif short_score >= 10:
        action = "🔴 STRONG SELL"
        color = "red"
        recommendation = f"Strong bearish signal (Score: {short_score}) - Consider entering short position or exit longs immediately. Set stops above resistance."
        active_reasons = short_reasons
    elif short_score > long_score and short_score >= 6:
        action = "⚠️ SELL"
        color = "orange"
        recommendation = f"Moderate bearish signal (Score: {short_score}) - Consider selling on confirmation. Watch for resistance rejection."
        active_reasons = short_reasons
    else:
        action = "⏸️ HOLD"
        color = "gray"
        recommendation = "Neutral signal - Mixed indicators suggest waiting for clearer opportunity. No strong directional bias."
        active_reasons = ["Market showing mixed signals", "No clear trend direction", "Best to wait for confirmation"]
    
    # Include layer information if provided
    layers = layers_result if layers_result else {}
    
    return {
        'action': action,
        'color': color,
        'long_score': int(long_score),
        'short_score': int(short_score),
        'score_difference': int(abs(long_score - short_score)),
        'confidence': float(min(100, max(long_score, short_score) / 10 * 100)),
        'recommendation': recommendation,
        'long_reasons': long_reasons,
        'short_reasons': short_reasons,
        'active_reasons': active_reasons,
        'price': current_price,
        'support': sr.get('support', current_price),
        'resistance': sr.get('resistance', current_price),
        'rsi': rsi,
        'ma': ma,
        'atr': atr_val,
        'atr_percent': atr_percent,
        'macd_histogram': macd_hist,
        'macd_line': macd_line,
        'signal_line': signal_line,
        # Layer filtering information
        'layers': {
            'layer1': layers_result.get('layer1', {}) if layers_result else {},
            'layer2': layers_result.get('layer2', {}) if layers_result else {},
            'layer3': layers_result.get('layer3', {}) if layers_result else {},
            'layer4': layers_result.get('layer4', {}) if layers_result else {},
            'filter_chain_passed': all(layers_result.get(f'layer{i}', {}).get('passed', True) for i in range(1, 5)) if layers_result else False
        }
    }

# ============================================
# MARKET DATA ENDPOINTS
# ============================================

@app.post("/api/trading/signal", tags=["Trading"])
async def get_trading_signal(
    request: TradingSignalRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get comprehensive trading signal for a symbol
    
    Analyzes multiple technical indicators and generates buy/sell/hold signal
    """
    try:
        # Determine if crypto or stock
        is_crypto = request.market == 'crypto' or 'USDT' in request.symbol.upper() or 'BTC' in request.symbol.upper()
        
        logger.info(f"🔍 Analyzing {request.symbol} | Market: {request.market} | Timeframe: {request.timeframe}")
        logger.info(f"   Detected as: {'Crypto' if is_crypto else 'Stock'}")
        
        # Fetch price data based on market type
        if is_crypto:
            # Use crypto API for crypto symbols (with or without BINANCE: prefix)
            logger.info(f"   📊 Fetching crypto data for {request.symbol}")
            
            # V2 API only supports these timeframes: 1m, 5m, 15m, 30m, 1h, 4h, 1d (NOT W, M)
            # V1 API supports: 1, 2, 3, 5, 10, 60, 120, 180, 240, D, W, M
            v2_supported_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', 'D']
            use_v2 = request.timeframe in v2_supported_timeframes
            
            if use_v2:
                # Convert timeframe format for V2 crypto API
                timeframe_map_v2 = {
                    '1m': '1m',
                    '5m': '5m',
                    '15m': '15m',
                    '30m': '30m',
                    '1h': '1h',
                    '4h': '4h',
                    'D': '1d'
                }
                
                # V2 API limits: limit must be between -500 and -10
                max_candles_v2 = min(request.max_candles, 500)
                
                params_v2 = {
                    'symbol': request.symbol.upper(),
                    'provider': 'BINANCE',
                    'limit': -max_candles_v2,  # Negative value gets last N candles (max 500)
                    'time_frame': timeframe_map_v2.get(request.timeframe, '1d')
                }
            
            # Try v2 timeseries endpoint only if it supports the timeframe
            data = None
            if use_v2:
                try:
                    logger.info(f"   🔄 Trying V2 API: /api/chart/v2/timeseries (requesting {request.max_candles} candles)")
                    data = await call_datasectors_api('/api/chart/v2/timeseries', params_v2)
                    
                    # Validate V2 response structure
                    if data and 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
                        # Check if candles have required fields
                        first_candle = data['data'][0]
                        required_fields = ['close', 'high', 'low', 'volume']
                        missing = [f for f in required_fields if f not in first_candle]
                        
                        if missing:
                            logger.warning(f"   ⚠️ V2 response missing fields: {missing}")
                            raise Exception(f"Invalid V2 data structure - missing {missing}")
                        
                        logger.info(f"   ✅ V2 API success: {len(data['data'])} candles")
                    else:
                        raise Exception("V2 API returned empty or invalid data")
                except Exception as e:
                    logger.warning(f"   ⚠️ V2 API failed: {str(e)}")
                    data = None
            
            # If V2 failed or not supported, use V1 fallback with retry logic
            if data is None:
                logger.info(f"   🔄 Trying V1 fallback: /api/chart/price")
                
                # V1 API supports: '1', '2', '3', '5', '10', '60', '120', '180', '240', 'D', 'W', 'M'
                # Map user timeframes to V1 compatible formats, with fallback options
                timeframe_fallbacks = {
                    '1m': ['1', '5'],  # Try 1m first, then 5m
                    '5m': ['5', '1'],  # Try 5m first, then 1m
                    '15m': ['5', '10', '30'],  # Try 5m, then 10m, then 30m (no actual 15m support)
                    '30m': ['30', '60'],  # Try 30m first, then 1h
                    '1h': ['60', '120'],  # Try 60m first, then 120m
                    '4h': ['240', '60'],  # Try 240m first, then 1h
                    'D': ['D', '240'],  # Try daily first, then 4h
                    'W': ['W', 'D'],  # Try weekly first, then daily
                    'M': ['M', 'W'],  # Try monthly first, then weekly
                    # Legacy support
                    '60': ['60'],
                    '240': ['240']
                }
                
                # Get fallback timeframes to try
                timeframes_to_try = timeframe_fallbacks.get(request.timeframe, ['D'])
                
                # Try each fallback timeframe
                for v1_timeframe in timeframes_to_try:
                    try:
                        params_fallback = {
                            'symbol': request.symbol.upper(),  # Use without CRYPTO: prefix
                            'timeframe': v1_timeframe,
                            'range': min(request.max_candles, 1000)
                        }
                        
                        logger.info(f"   🔄 Trying V1: timeframe={v1_timeframe} (candles={min(request.max_candles, 1000)})")
                        data = await call_datasectors_api('/api/chart/price', params_fallback)
                        
                        if data and 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
                            logger.info(f"   ✅ V1 success with timeframe {v1_timeframe}: {len(data['data'])} candles")
                            break  # Success! Exit the retry loop
                        else:
                            logger.warning(f"   ⚠️ V1 returned no data for timeframe {v1_timeframe}, trying next...")
                            
                    except Exception as e:
                        logger.warning(f"   ⚠️ V1 failed for timeframe {v1_timeframe}: {str(e)[:100]}")
                        # Continue to next fallback timeframe
                        continue
                
                # Check if we got data after retries
                if not data or not data.get('data') or not isinstance(data.get('data'), list) or len(data.get('data', [])) == 0:
                    logger.error(f"   ❌ All V1 fallback timeframes failed for {request.symbol}")
                    raise HTTPException(
                        status_code=503, 
                        detail=f"Unable to fetch data for {request.symbol} on any supported timeframe. The symbol may not exist or API may be unavailable."
                    )
        else:
            # Use regular stock API with same timeframe fallback logic as V1
            logger.info(f"   📊 Fetching stock data for {request.symbol}")
            
            # Stock API supports: '1', '2', '3', '5', '10', '60', '120', '180', '240', 'D', 'W', 'M', '3M', '6M', '12M'
            # Map user timeframes to stock API compatible formats, with fallback options
            timeframe_fallbacks = {
                '1m': ['1', '5'],  # Try 1m first, then 5m
                '5m': ['5', '1'],  # Try 5m first, then 1m
                '15m': ['5', '10', '30'],  # Try 5m, then 10m, then 30m (no actual 15m support)
                '30m': ['30', '60'],  # Try 30m first, then 1h
                '1h': ['60', '120'],  # Try 60m first, then 120m
                '4h': ['240', '60'],  # Try 240m first, then 1h
                'D': ['D', '240'],  # Try daily first, then 4h
                'W': ['W', 'D'],  # Try weekly first, then daily
                'M': ['M', 'W'],  # Try monthly first, then weekly
                # Legacy support
                '60': ['60'],
                '240': ['240']
            }
            
            # Get fallback timeframes to try
            timeframes_to_try = timeframe_fallbacks.get(request.timeframe, ['D'])
            
            # Try each fallback timeframe
            data = None
            for stock_timeframe in timeframes_to_try:
                try:
                    params = {
                        'symbol': request.symbol,
                        'timeframe': stock_timeframe,
                        'range': min(request.max_candles, 1000)
                    }
                    
                    logger.info(f"   🔄 Trying stock API: timeframe={stock_timeframe} (candles={min(request.max_candles, 1000)})")
                    data = await call_datasectors_api('/api/chart/price', params)
                    
                    if data and 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
                        logger.info(f"   ✅ Stock API success with timeframe {stock_timeframe}: {len(data['data'])} candles")
                        break  # Success! Exit the retry loop
                    else:
                        logger.warning(f"   ⚠️ Stock API returned no data for timeframe {stock_timeframe}, trying next...")
                        
                except Exception as e:
                    logger.warning(f"   ⚠️ Stock API failed for timeframe {stock_timeframe}: {str(e)[:100]}")
                    # Continue to next fallback timeframe
                    continue
            
            # Check if we got data after retries
            if not data or not data.get('data') or not isinstance(data.get('data'), list) or len(data.get('data', [])) == 0:
                logger.error(f"   ❌ All stock API timeframe fallbacks failed for {request.symbol}")
                raise HTTPException(
                    status_code=503, 
                    detail=f"Unable to fetch data for {request.symbol} on any supported timeframe. The symbol may not exist or API may be unavailable."
                )
        
        # Validate final data
        if not data or not data.get('data'):
            logger.error(f"   ❌ No data found for {request.symbol}")
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for symbol {request.symbol}. Check symbol format and try again."
            )
        
        # Extract and validate candles
        candles = data['data']
        
        if not isinstance(candles, list) or len(candles) == 0:
            logger.error(f"   ❌ Invalid candle data: {type(candles)}")
            raise HTTPException(
                status_code=400,
                detail="Invalid candle data format received from API"
            )
        
        # Validate candle structure
        required_fields = ['close', 'high', 'low', 'volume']
        first_candle = candles[0]
        missing_fields = [f for f in required_fields if f not in first_candle]
        
        if missing_fields:
            logger.error(f"   ❌ Candles missing fields: {missing_fields}")
            raise HTTPException(
                status_code=400,
                detail=f"Candle data missing required fields: {missing_fields}"
            )
        
        logger.info(f"   ✅ Successfully extracted {len(candles)} candles")
        
        # Extract OHLCV data with error handling
        try:
            closes = [float(c['close']) for c in candles]
            highs = [float(c['high']) for c in candles]
            lows = [float(c['low']) for c in candles]
            volumes = [float(c['volume']) for c in candles]
            
            logger.info(f"   📈 Price range: ${min(lows):.2f} - ${max(highs):.2f}")
            
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"   ❌ Error parsing candle data: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error parsing candle data: {str(e)}"
            )
        
        # Get timestamps if available
        timestamps = []
        for c in candles:
            if 'time' in c:
                timestamps.append(c['time'])
            elif 'timestamp' in c:
                timestamps.append(c['timestamp'])
            else:
                timestamps.append(None)
        
        # VALIDATE DATA ORDER & INTEGRITY
        # ✅ Ensure candles are in chronological order (oldest first, newest last)
        logger.info(f"   🔍 Validating candle data chronological order...")
        
        # Check if we have timestamps to validate order
        valid_timestamps = [ts for ts in timestamps if ts is not None]
        if len(valid_timestamps) > 1:
            is_ascending = all(valid_timestamps[i] <= valid_timestamps[i+1] for i in range(len(valid_timestamps)-1))
            if not is_ascending:
                logger.warning(f"   ⚠️ Timestamps NOT in ascending order - Data may need reversal!")
                logger.info(f"   🔄 Reversing candles to correct chronological order...")
                # Reverse all arrays to get correct order
                candles = candles[::-1]
                closes = closes[::-1]
                highs = highs[::-1]
                lows = lows[::-1]
                volumes = volumes[::-1]
                timestamps = timestamps[::-1]
                logger.info(f"   ✅ Data reversed to correct order")
        
        # Calculate indicators
        indicators = {}
        
        # Log candle data for debugging
        logger.info(f"   💰 Sample candle data:")
        if len(candles) > 0:
            first_candle = candles[0]
            last_candle = candles[-1]
            logger.info(f"      First: open={first_candle.get('open')}, close={first_candle.get('close')}, time={first_candle.get('time', first_candle.get('timestamp', 'N/A'))}")
            logger.info(f"      Last: open={last_candle.get('open')}, close={last_candle.get('close')}, time={last_candle.get('time', last_candle.get('timestamp', 'N/A'))}")
            logger.info(f"      Total candles: {len(candles)}")
            logger.info(f"      ✅ Current Price (closes[-1]): ${closes[-1]:,.2f}")
        
        if 'ma' in request.indicators:
            indicators['ma'] = TechnicalIndicators.calculate_ma(closes, 20)
        
        if 'rsi' in request.indicators:
            indicators['rsi'] = TechnicalIndicators.calculate_rsi(closes)
        
        if 'macd' in request.indicators:
            indicators['macd'] = TechnicalIndicators.calculate_macd(closes)
        
        if 'stoch_rsi' in request.indicators:
            indicators['stoch_rsi'] = TechnicalIndicators.calculate_stochastic_rsi(closes)
        
        if 'stoch' in request.indicators:
            indicators['stochastic'] = TechnicalIndicators.calculate_stochastic(highs, lows, closes)
        
        if 'bb' in request.indicators:
            indicators['bollinger_bands'] = TechnicalIndicators.calculate_bollinger_bands(closes)
        
        if 'atr' in request.indicators:
            indicators['atr'] = TechnicalIndicators.calculate_atr(highs, lows, closes)
        
        if 'volume' in request.indicators:
            indicators['volume_analysis'] = TechnicalIndicators.calculate_volume_sma(volumes)
        
        # Always calculate support/resistance and order blocks
        indicators['support_resistance'] = TechnicalIndicators.calculate_support_resistance(highs, lows)
        indicators['order_blocks'] = TechnicalIndicators.identify_order_blocks(highs, lows, closes)
        indicators['smart_money_concept'] = TechnicalIndicators.smart_money_concept(highs, lows, closes, volumes)
        
        # ============================================
        # EXECUTE LAYERED FILTERING SYSTEM
        # ============================================
        logger.info(f"🔬 [FILTERING] Starting 4-layer filtering for {request.symbol}")
        
        current_price = float(closes[-1])
        ma_values = [x for x in indicators.get('ma', []) if x is not None]
        ma = float(ma_values[-1]) if ma_values else current_price
        
        # LAYER 1: Price & Structure (Always pass, provides context)
        layer1 = await layer1_price_structure(data, closes, current_price, ma, indicators, request.symbol, request.market)
        
        # LAYER 2: Volume Confirmation (Filter: Cheap but essential)
        layer2 = await layer2_volume_confirmation(current_price, closes, volumes, request.symbol, request.market)
        
        if not layer2.get('passed', False):
            logger.warning(f"❌ FILTERED OUT at Layer 2 (Volume): {layer2.get('reason', 'Weak volume')}")
        
        # LAYER 3: Pre-filter (Strong trend detection)
        layer3 = await layer3_pre_filter(request.symbol, request.market, closes)
        
        if not layer3.get('passed', False):
            logger.warning(f"⚠️  FILTERED OUT at Layer 3 (Pre-filter): {layer3.get('reason', 'Not trending')}")
        
        # LAYER 4: Confirmation (Only if Layer 1-3 passed)
        layer4 = {}
        if layer2.get('passed', False) and layer3.get('passed', False):
            layer4 = await layer4_confirmation(request.symbol, request.market, current_price, closes, indicators)
        else:
            layer4 = {
                'layer': 4,
                'name': 'Confirmation',
                'passed': True,  # Assume pass if not reached due to filter
                'reason': 'Skipped due to earlier filter'
            }
        
        layers_result = {
            'layer1': layer1,
            'layer2': layer2,
            'layer3': layer3,
            'layer4': layer4,
            'overall_passed': layer2.get('passed', True) and layer3.get('passed', True)
        }
        
        logger.info(f"✅ [FILTERING] Complete - Passed all layers: {layers_result['overall_passed']}")
        
        # Generate trading signal with validated closes AND layer information
        signal = generate_trading_signal(data, indicators, closes=closes, layers_result=layers_result)
        
        # Log price data for debugging
        current_price = float(closes[-1])
        logger.info(f"   ✅ FINAL Current Price (from closes[-1]): ${current_price:,.2f}")
        logger.info(f"   📊 Price Range: ${min(lows):,.2f} - ${max(highs):,.2f}")
        logger.info(f"   📈 Signal: {signal['action']} (Confidence: {signal['confidence']:.1f}%)")
        
        # Verify price consistency
        if signal['price'] != current_price:
            logger.warning(f"   ⚠️ Price mismatch: signal['price']={signal['price']:.2f} vs closes[-1]={current_price:.2f}")
        else:
            logger.info(f"   ✅ Price consistency verified: {current_price:,.2f}")
        
        # Validate price data makes sense
        if current_price < 0.01:
            logger.warning(f"   ⚠️ Suspicious low price: ${current_price}")
        if current_price > 1000000:
            logger.warning(f"   ⚠️ Suspicious high price: ${current_price}")
        
        # Save to history
        conn = sqlite3.connect('trading_system.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trading_history (user_id, symbol, action, price, quantity, indicators)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            current_user['id'],
            request.symbol,
            signal['action'],
            signal['price'],
            0,  # quantity placeholder
            json.dumps({
                'rsi': signal['rsi'],
                'ma': signal['ma'],
                'macd': signal['macd_histogram'],
                'confidence': signal['confidence']
            })
        ))
        
        conn.commit()
        conn.close()
        
        return {
            'symbol': request.symbol,
            'market': request.market,
            'timeframe': request.timeframe,
            'signal': signal,
            'indicators': indicators,
            'chart_data': {
                'candles': candles,
                'timestamps': timestamps
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in get_trading_signal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scanner/scan", tags=["Scanner"])
async def scan_market(
    request: ScannerRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Scan market for trading opportunities
    
    Returns top signals based on technical analysis
    """
    # This is a simplified version - in production, you'd scan multiple symbols
    symbols = [
        'IDX:BBCA', 'IDX:BBRI', 'IDX:TLKM', 'IDX:ASII',
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT'
    ] if request.market == 'all' else []
    
    results = []
    
    for symbol in symbols[:5]:  # Limit to 5 for demo
        try:
            signal_request = TradingSignalRequest(
                symbol=symbol,
                market=request.market,
                timeframe=request.timeframe,
                indicators=request.indicators
            )
            
            signal_data = await get_trading_signal(signal_request, current_user)
            results.append(signal_data)
            
        except Exception as e:
            continue
    
    # Sort by confidence
    results.sort(key=lambda x: x['signal']['confidence'], reverse=True)
    
    return {
        'market': request.market,
        'timeframe': request.timeframe,
        'total_scanned': len(symbols),
        'results': results,
        'timestamp': datetime.now().isoformat()
    }

@app.get("/api/history", tags=["History"])
async def get_trading_history(
    current_user: Dict = Depends(get_current_user),
    limit: int = 50
):
    """Get user's trading history"""
    conn = sqlite3.connect('trading_system.db')
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT symbol, action, price, quantity, timestamp, indicators
        FROM trading_history
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (current_user['id'], limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            'symbol': row['symbol'],
            'action': row['action'],
            'price': float(row['price']) if row['price'] else None,
            'quantity': float(row['quantity']) if row['quantity'] else None,
            'timestamp': row['timestamp'],
            'indicators': row['indicators']
        })
    
    return {
        'user_id': current_user['id'],
        'total': len(history),
        'history': history
    }

@app.get("/api/datasectors/stats", tags=["System"])
async def get_datasectors_stats(current_user: Dict = Depends(get_current_user)):
    """
    Get DataSectors API key statistics and quota information
    
    Returns stats for all API keys including request counts, error rates, and success rates
    """
    try:
        logger.info(f"📊 Fetching DataSectors API statistics")
        
        # Call the DataSectors stats endpoint
        stats_data = await call_datasectors_api('/api/datasectors/stats')
        
        if stats_data:
            logger.info(f"   ✅ Got stats: {stats_data.get('total_keys', 0)} keys, Current: #{stats_data.get('current_key_number', 0)}")
            return stats_data
        else:
            logger.error(f"   ❌ No stats data returned")
            raise HTTPException(status_code=500, detail="Failed to fetch API statistics")
            
    except Exception as e:
        logger.error(f"Error fetching DataSectors stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch API statistics: {str(e)}")

# ============================================
# HEALTH CHECK
# ============================================

@app.get("/", tags=["System"])
async def root():
    """API Status"""
    return {
        "name": "Trading Decision System API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2401)