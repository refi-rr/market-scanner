from fastapi import FastAPI, HTTPException, Query, Header, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from typing import Optional, Dict, Any, List
from datetime import datetime
import httpx
from enum import Enum
import logging
from itertools import cycle
import asyncio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="DataSectors API Wrapper with Authentication",
    description="FastAPI wrapper dengan Client API Key Authentication + DataSectors Key Rotation",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# CLIENT API KEY AUTHENTICATION
# ============================================

# Client API Keys (keys untuk mengakses wrapper ini)
CLIENT_API_KEYS = {
    "sangahli": {
        "name": "Default Client",
        "created": "2025-01-31",
        "permissions": ["read", "write"],
        "rate_limit": 1000,  # requests per hour
        "enabled": True
    }
    # Tambahkan client keys lain di sini
}

# Track client usage
client_usage = {}

class ClientAPIKeyAuth:
    """Client API Key Authentication Manager"""
    
    def __init__(self):
        self.api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
    
    async def __call__(self, api_key: str = Depends(APIKeyHeader(name="X-API-Key", auto_error=False))):
        """Validate client API key"""
        
        # Check if API key is provided
        if not api_key:
            logger.warning("Request without API key")
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "Authentication Required",
                    "message": "API key is required. Please provide X-API-Key header.",
                    "example": "X-API-Key: your_api_key_here"
                }
            )
        
        # Validate API key
        if api_key not in CLIENT_API_KEYS:
            logger.warning(f"Invalid API key attempted: {api_key[:10]}...")
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Invalid API Key",
                    "message": "The provided API key is not valid.",
                    "provided_key": f"{api_key[:10]}..." if len(api_key) > 10 else api_key
                }
            )
        
        # Check if key is enabled
        client_info = CLIENT_API_KEYS[api_key]
        if not client_info.get("enabled", True):
            logger.warning(f"Disabled API key attempted: {api_key[:10]}...")
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "API Key Disabled",
                    "message": "This API key has been disabled."
                }
            )
        
        # Track usage
        if api_key not in client_usage:
            client_usage[api_key] = {
                "requests": 0,
                "last_used": None,
                "errors": 0
            }
        
        client_usage[api_key]["requests"] += 1
        client_usage[api_key]["last_used"] = datetime.now().isoformat()
        
        # Log successful authentication
        logger.info(f"Authenticated request from client: {client_info['name']}")
        
        return {
            "api_key": api_key,
            "client": client_info["name"],
            "permissions": client_info["permissions"]
        }

# Initialize auth
client_auth = ClientAPIKeyAuth()

# ============================================
# DATASECTORS API KEY CONFIGURATION
# ============================================

DATASECTORS_BASE_URL = "https://api.datasectors.com"

# Multiple DataSectors API Keys with rotation
DATASECTORS_KEYS = [
    "ds_live_4Im1sd1FgunC4eOnInBMGAYb6Jb2Ed6r",
    "ds_live_UuPNcDEVP68UBgfxqOcyGtpIibtdCDPC",
    "ds_live_RxAAfP-6TYQHh4MYYBNOuXQRfmTpXcFK",
    "ds_live_hcAQO2nGI1pt3EoO1mDF2GbCSE_MFpRh"
]

# API Key Manager (same as v3)
class APIKeyManager:
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.current_index = 0
        self.key_stats = {key: {"requests": 0, "errors": 0, "last_used": None} for key in keys}
        self.failed_keys = set()
        
    def get_current_key(self) -> str:
        return self.keys[self.current_index]
    
    def get_next_key(self) -> Optional[str]:
        for _ in range(len(self.keys)):
            self.current_index = (self.current_index + 1) % len(self.keys)
            key = self.keys[self.current_index]
            
            if key not in self.failed_keys:
                logger.info(f"Switching to DataSectors key #{self.current_index + 1}")
                return key
        
        logger.warning("All DataSectors keys have errors, clearing failed keys...")
        self.failed_keys.clear()
        return self.get_current_key()
    
    def mark_key_used(self, key: str, success: bool = True):
        if key in self.key_stats:
            self.key_stats[key]["requests"] += 1
            self.key_stats[key]["last_used"] = datetime.now().isoformat()
            
            if not success:
                self.key_stats[key]["errors"] += 1
    
    def mark_key_failed(self, key: str):
        self.failed_keys.add(key)
        logger.warning(f"DataSectors key marked as failed: {key[:20]}...")
    
    def get_stats(self) -> Dict[str, Any]:
        stats = []
        for i, key in enumerate(self.keys):
            key_stat = self.key_stats[key].copy()
            key_stat["key_number"] = i + 1
            key_stat["key_preview"] = f"{key[:15]}...{key[-4:]}"
            key_stat["is_current"] = (i == self.current_index)
            key_stat["is_failed"] = key in self.failed_keys
            
            total = key_stat["requests"]
            errors = key_stat["errors"]
            key_stat["success_rate"] = f"{((total - errors) / total * 100):.1f}%" if total > 0 else "N/A"
            
            stats.append(key_stat)
        
        return {
            "total_keys": len(self.keys),
            "current_key_number": self.current_index + 1,
            "failed_keys_count": len(self.failed_keys),
            "keys": stats
        }

key_manager = APIKeyManager(DATASECTORS_KEYS)

# Enums (same as v3)
class TimeFrame(str, Enum):
    m1 = "1"
    m2 = "2"
    m3 = "3"
    m5 = "5"
    m10 = "10"
    h1 = "60"
    h2 = "120"
    h3 = "180"
    h4 = "240"
    daily = "D"
    weekly = "W"
    monthly = "M"
    quarter_3m = "3M"
    quarter_6m = "6M"
    quarter_12m = "12M"

class TimeFrameV2(str, Enum):
    m1 = "1m"
    m5 = "5m"
    m15 = "15m"
    m30 = "30m"
    h1 = "1h"
    h4 = "4h"
    d1 = "1d"

class Market(str, Enum):
    indonesia = "id-id"
    us = "en-us"

class Volatility(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

# Helper function (same as v3)
async def make_request(endpoint: str, params: Dict[str, Any], max_retries: int = 3) -> Dict:
    """Make HTTP request with automatic key rotation"""
    
    for attempt in range(max_retries):
        current_key = key_manager.get_current_key()
        
        headers = {
            "Authorization": current_key,
            "Content-Type": "application/json"
        }
        
        url = f"{DATASECTORS_BASE_URL}{endpoint}"
        
        logger.info(f"[Attempt {attempt + 1}/{max_retries}] {url}")
        logger.debug(f"Using DataSectors key #{key_manager.current_index + 1}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(url, params=params, headers=headers)
                
                logger.info(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    json_response = response.json()
                    key_manager.mark_key_used(current_key, success=True)
                    return json_response
                
                elif response.status_code in [429, 403]:
                    logger.warning(f"Rate limit on DataSectors key #{key_manager.current_index + 1}")
                    key_manager.mark_key_used(current_key, success=False)
                    key_manager.mark_key_failed(current_key)
                    
                    next_key = key_manager.get_next_key()
                    
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying with next DataSectors key...")
                        await asyncio.sleep(0.5)
                        continue
                    else:
                        raise HTTPException(
                            status_code=429,
                            detail={
                                "error": "Rate Limit Exceeded",
                                "message": "All DataSectors API keys have reached their rate limit.",
                                "attempted_keys": max_retries
                            }
                        )
                
                else:
                    logger.error(f"HTTP Error {response.status_code}: {response.text}")
                    key_manager.mark_key_used(current_key, success=False)
                    
                    raise HTTPException(
                        status_code=response.status_code,
                        detail={
                            "error": "DataSectors API Error",
                            "status_code": response.status_code,
                            "message": response.text
                        }
                    )
                
            except httpx.RequestError as e:
                logger.error(f"Request Error: {str(e)}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                else:
                    raise HTTPException(
                        status_code=503,
                        detail={
                            "error": "Service Unavailable",
                            "message": f"Failed to connect to DataSectors API: {str(e)}"
                        }
                    )
    
    raise HTTPException(
        status_code=500,
        detail={
            "error": "Max Retries Exceeded",
            "message": "Failed after maximum retry attempts"
        }
    )

# ============================================
# PUBLIC ENDPOINTS (NO AUTH REQUIRED)
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """Welcome endpoint - No authentication required"""
    return {
        "name": "DataSectors API Wrapper with Authentication",
        "version": "4.0.0",
        "status": "running",
        "authentication": {
            "required": True,
            "method": "API Key in X-API-Key header",
            "example": "X-API-Key: your_api_key_here"
        },
        "features": [
            "Client API Key Authentication",
            "DataSectors Key Rotation",
            "Automatic Fallback",
            "Rate Limit Handling",
            "Multi-Key Support"
        ],
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "client_stats": "/api/client/stats",
            "datasectors_stats": "/api/datasectors/stats",
            "crypto_details": "/api/crypto/details/{ticker}",
            "crypto_trending": "/api/crypto/trending",
            "crypto_heatmap": "/api/crypto/global-heatmap",
            "crypto_walls": "/api/crypto/walls",
            "indicator_mfi": "/api/indicator/volume/mfi",
            "indicator_obv": "/api/indicator/volume/obv",
            "indicator_cmf": "/api/indicator/volume/cmf",
            "indicator_ad": "/api/indicator/volume/ad",
            "indicator_vwap": "/api/indicator/volume/vwap"
        }
    }

@app.get("/health", tags=["System"])
async def health_check():
    """Health check - No authentication required"""
    ds_stats = key_manager.get_stats()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0.0",
        "datasectors": {
            "total_keys": ds_stats["total_keys"],
            "current_key": ds_stats["current_key_number"],
            "failed_keys": ds_stats["failed_keys_count"]
        },
        "clients": {
            "total_registered": len(CLIENT_API_KEYS),
            "total_active": sum(1 for k, v in CLIENT_API_KEYS.items() if v.get("enabled", True))
        }
    }

# ============================================
# AUTHENTICATED ENDPOINTS
# ============================================

@app.get("/api/client/stats", 
         tags=["Client Management"],
         dependencies=[Depends(client_auth)])
async def get_client_stats(auth: dict = Depends(client_auth)):
    """Get client usage statistics - Requires authentication"""
    
    api_key = auth["api_key"]
    client_info = CLIENT_API_KEYS[api_key]
    usage = client_usage.get(api_key, {"requests": 0, "last_used": None, "errors": 0})
    
    return {
        "client": client_info["name"],
        "api_key_preview": f"{api_key[:5]}...{api_key[-3:]}",
        "created": client_info["created"],
        "permissions": client_info["permissions"],
        "rate_limit": client_info["rate_limit"],
        "usage": usage,
        "status": "active" if client_info["enabled"] else "disabled"
    }

@app.get("/api/datasectors/stats",
         tags=["System"],
         dependencies=[Depends(client_auth)])
async def get_datasectors_stats():
    """Get DataSectors key statistics - Requires authentication"""
    return JSONResponse(content=key_manager.get_stats())

@app.post("/api/datasectors/stats/reset",
          tags=["System"],
          dependencies=[Depends(client_auth)])
async def reset_datasectors_stats():
    """Reset DataSectors statistics - Requires authentication"""
    key_manager.failed_keys.clear()
    for key in key_manager.key_stats:
        key_manager.key_stats[key] = {"requests": 0, "errors": 0, "last_used": None}
    
    return {
        "status": "success",
        "message": "DataSectors statistics reset successfully",
        "timestamp": datetime.now().isoformat()
    }

# ============================================
# DATA ENDPOINTS (ALL REQUIRE AUTHENTICATION)
# ============================================

@app.get("/api/search/market", 
         tags=["Search"],
         dependencies=[Depends(client_auth)])
async def search_market(query: str = Query(..., min_length=1, max_length=100, example="BBCA")):
    """Search markets - Requires authentication"""
    try:
        params = {"query": query}
        result = await make_request("/api/search/market", params)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search_market: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chart/price",
         tags=["Chart Data"],
         dependencies=[Depends(client_auth)])
async def get_chart_price(
    symbol: str = Query(..., example="IDX:BBCA"),
    timeframe: TimeFrame = Query(TimeFrame.daily),
    range: int = Query(100, ge=1, le=5000),
    timezone: Optional[str] = Query(None, example="Asia/Jakarta")
):
    """Get OHLCV price data - Requires authentication"""
    try:
        params = {"symbol": symbol, "timeframe": timeframe.value, "range": range}
        if timezone:
            params["timezone"] = timezone
        result = await make_request("/api/chart/price", params)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_chart_price: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chart/range",
         tags=["Chart Data"],
         dependencies=[Depends(client_auth)])
async def get_chart_range(
    symbol: str = Query(..., example="IDX:BBCA"),
    timeframe: TimeFrame = Query(TimeFrame.daily),
    start_date: str = Query(..., alias="from", example="2025-01-01"),
    end_date: str = Query(..., alias="to", example="2025-01-31"),
    timezone: Optional[str] = Query(None, example="Asia/Jakarta")
):
    """Get chart data within date range - Requires authentication"""
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    try:
        params = {"symbol": symbol, "timeframe": timeframe.value, "from": start_date, "to": end_date}
        if timezone:
            params["timezone"] = timezone
        result = await make_request("/api/chart/range", params)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_chart_range: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stocks/v2/search",
         tags=["Stock Info"],
         dependencies=[Depends(client_auth)])
async def search_stock(
    symbol: str = Query(..., min_length=1, max_length=10, example="TLKM"),
    market: Market = Query(Market.indonesia)
):
    """Search stock info (BETA) - Requires authentication"""
    try:
        params = {"symbol": symbol, "market": market.value}
        result = await make_request("/api/stocks/v2/search", params)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search_stock: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chart/v2/data-provider",
         tags=["Chart Data V2"],
         dependencies=[Depends(client_auth)])
async def get_data_provider(query: str = Query(..., min_length=1, example="XAUUSD")):
    """Get data provider filters - Requires authentication"""
    try:
        params = {"query": query}
        result = await make_request("/api/chart/v2/data-provider", params)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_data_provider: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chart/v2/timeseries",
         tags=["Chart Data V2"],
         dependencies=[Depends(client_auth)])
async def get_timeseries(
    symbol: str = Query(..., min_length=1, example="XAUUSD"),
    provider: str = Query(..., min_length=1, example="OANDA"),
    limit: int = Query(-100, example=-100),
    time_frame: TimeFrameV2 = Query(TimeFrameV2.h1, alias="time_frame")
):
    """Get time series candles - Requires authentication"""
    try:
        params = {"symbol": symbol, "provider": provider, "limit": limit, "time_frame": time_frame.value}
        result = await make_request("/api/chart/v2/timeseries", params)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_timeseries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/calendar",
         tags=["Economic Calendar"],
         dependencies=[Depends(client_auth)])
async def get_economic_calendar(
    countryCode: Optional[str] = Query(None, example="US"),
    volatility: Optional[Volatility] = Query(None),
    startDate: Optional[str] = Query(None, example="2025-01-01"),
    endDate: Optional[str] = Query(None, example="2025-12-31"),
    limit: Optional[int] = Query(50, ge=1, le=500),
    timezone: Optional[str] = Query("GMT+0", example="GMT+7")
):
    """Get economic calendar - Requires authentication"""
    try:
        params = {}
        if countryCode:
            params["countryCode"] = countryCode
        if volatility:
            params["volatility"] = volatility.value
        if startDate:
            params["startDate"] = startDate
        if endDate:
            params["endDate"] = endDate
        if limit:
            params["limit"] = limit
        if timezone:
            params["timezone"] = timezone
        result = await make_request("/api/calendar", params)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_economic_calendar: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/calendar/history/this-week",
         tags=["Economic Calendar"],
         dependencies=[Depends(client_auth)])
async def get_calendar_this_week(
    countryCode: Optional[str] = Query(None, example="US"),
    volatility: Optional[Volatility] = Query(None),
    limit: Optional[int] = Query(50, ge=1, le=500),
    timezone: Optional[str] = Query("GMT+0", example="GMT+7")
):
    """Get calendar this week - Requires authentication"""
    try:
        params = {}
        if countryCode:
            params["countryCode"] = countryCode
        if volatility:
            params["volatility"] = volatility.value
        if limit:
            params["limit"] = limit
        if timezone:
            params["timezone"] = timezone
        result = await make_request("/api/calendar/history/this-week", params)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_calendar_this_week: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/crypto/strong-trend",
         tags=["Crypto Analytics"],
         dependencies=[Depends(client_auth)])
async def get_crypto_strong_trend():
    """Get crypto strong trend - Requires authentication"""
    try:
        result = await make_request("/api/crypto/strong-trend", {})
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_crypto_strong_trend: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/crypto/orderbook-imbalance",
         tags=["Crypto Analytics"],
         dependencies=[Depends(client_auth)])
async def get_orderbook_imbalance(
    symbol: str = Query(..., min_length=1, example="BTCUSDT"),
    limit: Optional[int] = Query(100, example=100)
):
    """Get orderbook imbalance - Requires authentication"""
    try:
        params = {"symbol": symbol}
        if limit:
            params["limit"] = limit
        result = await make_request("/api/crypto/orderbook-imbalance", params)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_orderbook_imbalance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/crypto/details/{ticker}",
         tags=["Crypto Analytics"],
         dependencies=[Depends(client_auth)])
async def get_crypto_details(
    ticker: str,
    lang: Optional[str] = Query("en", description="Language code (en, id, etc)")
):
    """
    Get detailed cryptocurrency information for a specific ticker - Requires authentication
    
    **Path Parameter:**
    - ticker: Crypto ticker (e.g., XRP-USDAGR, BTC-USD, ETH-USD)
    
    **Query Parameter:**
    - lang: Language code (default: en)
    
    **Example:**
    ```
    /api/crypto/details/XRP-USDAGR
    /api/crypto/details/BTC-USD?lang=en
    ```
    
    **Response includes:**
    - Detailed coin information
    - Current price and market data
    - Description and links
    - Market statistics
    """
    try:
        params = {}
        if lang:
            params["lang"] = lang
        
        result = await make_request(f"/api/crypto/details/{ticker}", params)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_crypto_details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/crypto/trending",
         tags=["Crypto Analytics"],
         dependencies=[Depends(client_auth)])
async def get_trending_coins():
    """
    Fetch top trending coins, NFTs, and categories - Requires authentication
    
    **Returns:**
    - Top trending cryptocurrencies
    - Trending NFTs
    - Trending categories
    - Real-time trending data
    
    **Use Cases:**
    - Market sentiment analysis
    - Identify hot coins
    - Track trending topics
    - Social trading signals
    
    **Example Response:**
    ```json
    {
      "success": true,
      "data": {
        "coins": [...],
        "nfts": [...],
        "categories": [...]
      }
    }
    ```
    """
    try:
        result = await make_request("/api/crypto/trending", {})
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_trending_coins: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/crypto/global-heatmap",
         tags=["Crypto Analytics"],
         dependencies=[Depends(client_auth)])
async def get_global_heatmap(
    symbol: str = Query(..., min_length=1, description="Crypto symbol (e.g., BTC, ETH, SOL)", example="BTC")
):
    """
    Retrieve position breakdown data by cohort segments (size and PnL) for a cryptocurrency - Requires authentication
    
    **Query Parameter:**
    - symbol: Cryptocurrency symbol (BTC, ETH, SOL, etc.)
    
    **Returns:**
    - Position breakdown by size cohorts
    - PnL (Profit and Loss) segments
    - Whale vs retail distribution
    - Long/Short positioning
    
    **Use Cases:**
    - Understand market positioning
    - Identify whale accumulation/distribution
    - Analyze profit-taking patterns
    - Monitor smart money flows
    
    **Example:**
    ```
    /api/crypto/global-heatmap?symbol=BTC
    /api/crypto/global-heatmap?symbol=ETH
    ```
    """
    try:
        params = {"symbol": symbol}
        result = await make_request("/api/crypto/global-heatmap", params)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_global_heatmap: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/crypto/walls",
         tags=["Crypto Analytics"],
         dependencies=[Depends(client_auth)])
async def detect_orderbook_walls(
    symbol: str = Query(..., min_length=1, description="Crypto symbol (e.g., BTCUSDT, ETHUSDT)", example="BTCUSDT"),
    limit: Optional[int] = Query(100, ge=5, le=1000, description="Orderbook depth: 5, 10, 20, 50, 100, 500, 1000")
):
    """
    Detect large order walls (>3x average) to identify strong support/resistance levels - Requires authentication
    
    **Query Parameters:**
    - symbol: Crypto trading pair (BTCUSDT, ETHUSDT, SOLUSDT)
    - limit: Orderbook depth (5, 10, 20, 50, 100, 500, 1000)
    
    **Wall Detection:**
    - Identifies orders >3x average size
    - Marks as potential support (buy walls) or resistance (sell walls)
    - Indicates whale positioning
    
    **Returns:**
    ```json
    {
      "success": true,
      "data": {
        "symbol": "BTCUSDT",
        "buy_walls": [
          {
            "price": 45000,
            "volume": 500,
            "ratio": 4.5,
            "level": "strong_support"
          }
        ],
        "sell_walls": [
          {
            "price": 48000,
            "volume": 600,
            "ratio": 5.2,
            "level": "strong_resistance"
          }
        ]
      }
    }
    ```
    
    **Use Cases:**
    - Identify key support/resistance levels
    - Detect whale manipulation attempts
    - Find optimal entry/exit points
    - Confirm breakout validity
    - Stop loss placement
    
    **Trading Strategy:**
    1. Strong buy wall = likely support (consider buying near)
    2. Strong sell wall = likely resistance (consider selling near)
    3. Wall removal = potential breakout signal
    4. Multiple walls = very strong level
    
    **Wall Strength:**
    - 3-5x average: Moderate wall
    - 5-10x average: Strong wall
    - >10x average: Very strong wall (whale activity)
    """
    try:
        params = {"symbol": symbol}
        if limit:
            params["limit"] = limit
        
        result = await make_request("/api/crypto/walls", params)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_orderbook_walls: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# VOLUME INDICATOR ENDPOINTS
# ============================================

@app.get("/api/indicator/volume/mfi",
         tags=["Volume Indicators"],
         dependencies=[Depends(client_auth)])
async def get_money_flow_index(
    symbol: str = Query(..., description="Trading symbol in EXCHANGE:SYMBOL format", example="IDX:BBCA"),
    timeframe: TimeFrame = Query(TimeFrame.daily, description="Timeframe"),
    range: int = Query(100, ge=10, le=5000, description="Number of candles to fetch"),
    period: Optional[int] = Query(14, ge=1, le=100, description="MFI period (default: 14)")
):
    """
    Money Flow Index (MFI) - Volume-weighted RSI (0-100) - Requires authentication
    
    **Description:**
    Volume-weighted momentum indicator measuring buying/selling pressure.
    
    **Interpretation:**
    - MFI > 80: Overbought (potential reversal down)
    - MFI < 20: Oversold (potential reversal up)
    - MFI 40-60: Neutral zone
    
    **Parameters:**
    - symbol: Trading symbol (e.g., IDX:BBCA, NASDAQ:AAPL)
    - timeframe: 1, 5, 15, 30, 60 (minutes), D (daily), W (weekly), M (monthly)
    - range: Number of candles (default: 100, max: 5000)
    - period: MFI calculation period (default: 14)
    
    **Trading Signals:**
    - MFI > 80 + Price resistance → SELL signal
    - MFI < 20 + Price support → BUY signal
    - Divergence with price → Reversal warning
    
    **Use Cases:**
    - Identify overbought/oversold conditions
    - Confirm trend strength with volume
    - Spot divergences for reversals
    - Time entries/exits
    """
    try:
        params = {
            "symbol": symbol,
            "timeframe": timeframe.value,
            "range": range
        }
        if period:
            params["period"] = period
        
        result = await make_request("/api/indicator/volume/mfi", params)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_money_flow_index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/indicator/volume/obv",
         tags=["Volume Indicators"],
         dependencies=[Depends(client_auth)])
async def get_on_balance_volume(
    symbol: str = Query(..., description="Trading symbol in EXCHANGE:SYMBOL format", example="IDX:BBCA"),
    timeframe: TimeFrame = Query(TimeFrame.daily, description="Timeframe"),
    range: int = Query(100, ge=10, le=5000, description="Number of candles to fetch")
):
    """
    On Balance Volume (OBV) - Cumulative volume indicator - Requires authentication
    
    **Description:**
    Cumulative volume based on price direction. Rising OBV = accumulation.
    
    **Formula:**
    - If Close > Previous Close: OBV = OBV + Volume
    - If Close < Previous Close: OBV = OBV - Volume
    - If Close = Previous Close: OBV = OBV
    
    **Interpretation:**
    - Rising OBV + Rising Price = Strong uptrend (accumulation)
    - Falling OBV + Falling Price = Strong downtrend (distribution)
    - OBV divergence with price = Potential reversal
    
    **Trading Signals:**
    - OBV breaks above resistance → Bullish breakout confirmation
    - OBV breaks below support → Bearish breakdown confirmation
    - Price up but OBV flat/down → Weak rally (bearish divergence)
    - Price down but OBV flat/up → Weak decline (bullish divergence)
    
    **Use Cases:**
    - Confirm trend direction with volume
    - Spot accumulation/distribution phases
    - Identify divergences for reversals
    - Validate breakouts
    """
    try:
        params = {
            "symbol": symbol,
            "timeframe": timeframe.value,
            "range": range
        }
        
        result = await make_request("/api/indicator/volume/obv", params)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_on_balance_volume: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/indicator/volume/cmf",
         tags=["Volume Indicators"],
         dependencies=[Depends(client_auth)])
async def get_chaikin_money_flow(
    symbol: str = Query(..., description="Trading symbol in EXCHANGE:SYMBOL format", example="IDX:BBCA"),
    timeframe: TimeFrame = Query(TimeFrame.daily, description="Timeframe"),
    range: int = Query(100, ge=10, le=5000, description="Number of candles to fetch"),
    period: Optional[int] = Query(20, ge=1, le=100, description="CMF period (default: 20)")
):
    """
    Chaikin Money Flow (CMF) - Buying/selling pressure (-1 to 1) - Requires authentication
    
    **Description:**
    Measures buying/selling pressure using volume and price location within the range.
    
    **Interpretation:**
    - CMF > 0: Buying pressure (bullish)
    - CMF < 0: Selling pressure (bearish)
    - CMF > 0.25: Strong buying pressure
    - CMF < -0.25: Strong selling pressure
    
    **Parameters:**
    - symbol: Trading symbol
    - timeframe: Candle timeframe
    - range: Number of candles
    - period: CMF calculation period (default: 20)
    
    **Trading Signals:**
    - CMF crosses above 0 → Buy signal
    - CMF crosses below 0 → Sell signal
    - CMF > 0 during uptrend → Trend confirmation
    - CMF < 0 during downtrend → Trend confirmation
    - Divergence with price → Reversal warning
    
    **Use Cases:**
    - Measure accumulation/distribution
    - Confirm trend strength
    - Identify divergences
    - Time entries with momentum
    """
    try:
        params = {
            "symbol": symbol,
            "timeframe": timeframe.value,
            "range": range
        }
        if period:
            params["period"] = period
        
        result = await make_request("/api/indicator/volume/cmf", params)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_chaikin_money_flow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/indicator/volume/ad",
         tags=["Volume Indicators"],
         dependencies=[Depends(client_auth)])
async def get_accumulation_distribution(
    symbol: str = Query(..., description="Trading symbol in EXCHANGE:SYMBOL format", example="IDX:BBCA"),
    timeframe: TimeFrame = Query(TimeFrame.daily, description="Timeframe"),
    range: int = Query(100, ge=10, le=5000, description="Number of candles to fetch")
):
    """
    Accumulation/Distribution (A/D) - Cumulative indicator using volume and price location - Requires authentication
    
    **Description:**
    Cumulative indicator showing whether a stock is being accumulated or distributed.
    
    **Formula:**
    ```
    Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    Money Flow Volume = Money Flow Multiplier * Volume
    A/D = Previous A/D + Money Flow Volume
    ```
    
    **Interpretation:**
    - Rising A/D = Accumulation (buyers in control)
    - Falling A/D = Distribution (sellers in control)
    - A/D confirms price trend = Strong trend
    - A/D diverges from price = Potential reversal
    
    **Trading Signals:**
    - Price up + A/D up = Strong uptrend ✅
    - Price up + A/D down = Weak rally (distribution) ⚠️
    - Price down + A/D down = Strong downtrend ✅
    - Price down + A/D up = Weak decline (accumulation) ⚠️
    
    **Divergence Patterns:**
    - **Bullish Divergence**: Price makes lower low, A/D makes higher low → Buy
    - **Bearish Divergence**: Price makes higher high, A/D makes lower high → Sell
    
    **Use Cases:**
    - Confirm trend direction
    - Identify smart money accumulation
    - Spot distribution before reversals
    - Validate breakouts with volume
    """
    try:
        params = {
            "symbol": symbol,
            "timeframe": timeframe.value,
            "range": range
        }
        
        result = await make_request("/api/indicator/volume/ad", params)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_accumulation_distribution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/indicator/volume/vwap",
         tags=["Volume Indicators"],
         dependencies=[Depends(client_auth)])
async def get_vwap(
    symbol: str = Query(..., description="Trading symbol in EXCHANGE:SYMBOL format", example="IDX:BBCA"),
    timeframe: TimeFrame = Query(TimeFrame.daily, description="Timeframe"),
    range: int = Query(100, ge=10, le=5000, description="Number of candles to fetch")
):
    """
    Volume Weighted Average Price (VWAP) - Institutional benchmark - Requires authentication
    
    **Description:**
    Average price weighted by volume. Shows the true average price institutions paid.
    
    **Formula:**
    ```
    VWAP = Σ(Price * Volume) / Σ(Volume)
    Typical Price = (High + Low + Close) / 3
    ```
    
    **Interpretation:**
    - Price > VWAP: Bullish (above average price)
    - Price < VWAP: Bearish (below average price)
    - VWAP acts as dynamic support/resistance
    
    **Trading Strategies:**
    
    **1. Mean Reversion:**
    - Price far above VWAP → Sell (expect pullback to VWAP)
    - Price far below VWAP → Buy (expect bounce to VWAP)
    
    **2. Trend Following:**
    - Price consistently above VWAP → Stay long
    - Price consistently below VWAP → Stay short
    - Price crosses above VWAP → Buy signal
    - Price crosses below VWAP → Sell signal
    
    **3. Institutional Level:**
    - Institutions aim to buy below VWAP
    - Institutions aim to sell above VWAP
    - VWAP = fair value reference
    
    **Use Cases:**
    - Benchmark institutional entry
    - Identify fair value zones
    - Dynamic support/resistance
    - Intraday trading pivot
    - Assess trade quality (bought below/above VWAP)
    
    **Best Timeframes:**
    - Intraday: 1m, 5m, 15m
    - Daily: For longer-term reference
    """
    try:
        params = {
            "symbol": symbol,
            "timeframe": timeframe.value,
            "range": range
        }
        
        result = await make_request("/api/indicator/volume/vwap", params)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_vwap: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# INDICATOR LISTING
# ============================================

@app.get("/api/indicator/list",
         tags=["Universal Indicators"],
         dependencies=[Depends(client_auth)])
async def list_indicators():
    """
    List all available technical indicators grouped by category - Requires authentication
    
    **Get comprehensive list of 50+ technical indicators**
    
    Returns all available indicators organized by:
    - Moving Averages
    - Momentum
    - Trend
    - Volatility
    - Volume
    - Other
    
    ## Response Format:
    ```json
    {
      "success": true,
      "count": 50,
      "indicators": [
        {
          "name": "rsi",
          "category": "Momentum"
        },
        {
          "name": "macd",
          "category": "Momentum"
        },
        ...
      ]
    }
    ```
    
    ## Usage:
    Use this endpoint to discover available indicators before calling `/api/indicator/calculate`
    
    ## Example:
    ```bash
    curl -H "X-API-Key: sangahli" \
         "http://localhost:8000/api/indicator/list"
    ```
    
    ## Use Cases:
    - Discover available indicators
    - Build dynamic indicator selector UI
    - Validate indicator names
    - Browse by category
    """
    try:
        result = await make_request("/api/indicator/list", {})
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in list_indicators: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# UNIVERSAL INDICATOR CALCULATOR
# ============================================

# ============================================
# UNIVERSAL INDICATOR CALCULATOR
# ============================================

class IndicatorName(str, Enum):
    """Available technical indicators"""
    # Moving Averages
    sma = "sma"
    ema = "ema"
    dema = "dema"
    tema = "tema"
    wma = "wma"
    hma = "hma"
    vwma = "vwma"
    smma = "smma"
    
    # Momentum
    rsi = "rsi"
    macd = "macd"
    stochastic = "stochastic"
    cci = "cci"
    roc = "roc"
    momentum = "momentum"
    williamsr = "williamsr"
    ao = "ao"
    stochrsi = "stochrsi"
    
    # Trend
    adx = "adx"
    aroon = "aroon"
    supertrend = "supertrend"
    psar = "psar"
    vortex = "vortex"
    
    # Volatility
    atr = "atr"
    bollinger = "bollinger"
    bollingerswidth = "bollingerswidth"
    keltner = "keltner"
    donchian = "donchian"
    stddev = "stddev"
    hw = "hw"
    
    # Volume
    obv = "obv"
    mfi = "mfi"
    cmf = "cmf"
    vwap = "vwap"
    volumeoscillator = "volumeoscillator"
    eom = "eom"
    pvi = "pvi"
    netvolume = "netvolume"
    
    # Other
    zigzag = "zigzag"
    alligator = "alligator"
    fractal = "fractal"
    fisher = "fisher"
    pricechannel = "pricechannel"
    ichimoku = "ichimoku"

@app.get("/api/indicator/calculate",
         tags=["Universal Indicators"],
         dependencies=[Depends(client_auth)])
async def calculate_indicator(
    symbol: str = Query(..., description="Trading symbol in EXCHANGE:SYMBOL format", example="IDX:BBCA"),
    timeframe: TimeFrame = Query(TimeFrame.daily, description="Timeframe"),
    range: int = Query(100, ge=10, le=5000, description="Number of candles to fetch"),
    indicator: IndicatorName = Query(..., description="Indicator name"),
    params: Optional[str] = Query(
        None, 
        description="JSON string of indicator parameters",
        example='{"period":14}'
    )
):
    """
    Calculate any technical indicator with custom parameters - Requires authentication
    
    **Universal Indicator Calculator** - Support 50+ indicators!
    
    ## Quick Examples by Indicator:
    
    ### Moving Averages:
    - **RSI**: `{"period":14}` → Most common: 14
    - **EMA**: `{"period":20}` → Popular: 12, 20, 26, 50, 200
    - **SMA**: `{"period":20}` → Popular: 20, 50, 100, 200
    
    ### Momentum:
    - **RSI**: `{"period":14}` → Standard: 14
    - **MACD**: `{"fastPeriod":12,"slowPeriod":26,"signalPeriod":9}` → Standard settings
    - **Stochastic**: `{"kPeriod":14,"dPeriod":3,"smooth":3}` → Standard: 14,3,3
    - **CCI**: `{"period":20}` → Standard: 20
    - **Williams %R**: `{"period":14}` → Standard: 14
    
    ### Trend:
    - **ADX**: `{"period":14}` → Standard: 14
    - **Aroon**: `{"period":25}` → Standard: 25
    - **SuperTrend**: `{"period":10,"multiplier":3}` → Popular: 10,3
    - **Parabolic SAR**: `{"start":0.02,"increment":0.02,"max":0.2}` → Standard
    
    ### Volatility:
    - **ATR**: `{"period":14}` → Standard: 14
    - **Bollinger Bands**: `{"period":20,"stddev":2}` → Standard: 20,2
    - **Keltner Channel**: `{"period":20,"atrPeriod":10,"multiplier":2}` → Standard
    - **Donchian Channel**: `{"period":20}` → Standard: 20
    
    ### Volume:
    - **MFI**: `{"period":14}` → Standard: 14
    - **CMF**: `{"period":20}` → Standard: 20
    - **OBV**: `{}` → No parameters needed
    - **VWAP**: `{}` → No parameters needed
    
    ## Parameter Quick Reference:
    
    | Indicator | Parameters | Example |
    |-----------|------------|---------|
    | RSI | period | `{"period":14}` |
    | MACD | fastPeriod, slowPeriod, signalPeriod | `{"fastPeriod":12,"slowPeriod":26,"signalPeriod":9}` |
    | Bollinger | period, stddev | `{"period":20,"stddev":2}` |
    | ATR | period | `{"period":14}` |
    | Stochastic | kPeriod, dPeriod, smooth | `{"kPeriod":14,"dPeriod":3,"smooth":3}` |
    | EMA | period | `{"period":20}` |
    | ADX | period | `{"period":14}` |
    | SuperTrend | period, multiplier | `{"period":10,"multiplier":3}` |
    
    ## Available Indicators:
    
    ### Moving Averages (8):
    sma, ema, dema, tema, wma, hma, vwma, smma
    
    ### Momentum (9):
    rsi, macd, stochastic, cci, roc, momentum, williamsr, ao, stochrsi
    
    ### Trend (5):
    adx, aroon, supertrend, psar, vortex
    
    ### Volatility (7):
    atr, bollinger, bollingerswidth, keltner, donchian, stddev, hw
    
    ### Volume (8):
    obv, mfi, cmf, vwap, volumeoscillator, eom, pvi, netvolume
    
    ### Other (6):
    zigzag, alligator, fractal, fisher, pricechannel, ichimoku
    
    ## Response Format:
    ```json
    {
      "success": true,
      "symbol": "IDX:BBCA",
      "timeframe": "D",
      "indicator": "rsi",
      "params": {"period": 14},
      "data": [
        {
          "time": 1,
          "datetime": "2024-01-15",
          "additionalProperty": 65.5
        }
      ],
      "count": 100
    }
    ```
    
    ## Common Use Cases:
    
    1. **Overbought/Oversold**: RSI, Stochastic, Williams %R
    2. **Trend Following**: EMA, MACD, ADX, SuperTrend
    3. **Volatility Trading**: ATR, Bollinger Bands, Keltner
    4. **Volume Confirmation**: OBV, MFI, CMF
    5. **Support/Resistance**: Donchian, Price Channel
    """
    try:
        # Build request params
        request_params = {
            "symbol": symbol,
            "timeframe": timeframe.value,
            "range": range,
            "indicator": indicator.value
        }
        
        # Add custom params if provided
        if params:
            request_params["params"] = params
        
        # Make request to DataSectors
        result = await make_request("/api/indicator/calculate", request_params)
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in calculate_indicator: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "path": str(request.url)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=5672,
        log_level="info"
    )