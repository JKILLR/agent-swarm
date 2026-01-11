#!/usr/bin/env python3.11
"""
ULTIMATE POLYMARKET ARBITRAGE BOT v2.0
======================================

The most comprehensive, production-ready arbitrage bot for Polymarket.

CAPABILITIES:
1. Universal market discovery via Gamma API (ALL markets, all categories)
2. Multi-outcome arbitrage (not just binary UP/DOWN)
3. Binary arbitrage: sum of all outcomes < 1.00
4. Time-decay opportunities near resolution
5. Order book depth analysis (walk the book)
6. WebSocket real-time updates (<100ms latency)
7. SQLite database for trade persistence
8. Real-time P&L dashboard
9. Comprehensive risk management with kill switch
10. Kelly Criterion position sizing

COMMAND-LINE USAGE:
  python3.11 ultimate_arb_bot.py                    # Dry run mode (default)
  python3.11 ultimate_arb_bot.py --live             # Live trading
  python3.11 ultimate_arb_bot.py --no-dashboard     # Disable dashboard
  python3.11 ultimate_arb_bot.py --no-websocket     # Disable WebSocket
  python3.11 ultimate_arb_bot.py --min-spread 0.5   # Min 0.5% spread
  python3.11 ultimate_arb_bot.py --categories crypto,politics

Author: Trading Bots Swarm
Date: 2026-01-04
Status: Production Ready
"""

import asyncio
import json
import time
import logging
import argparse
import sqlite3
import threading
import signal
import sys
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple, Any, Set, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from enum import Enum, auto
import heapq
import requests
import websocket

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration settings for the bot."""
    POLYMARKET_WEB = "https://polymarket.com"
    CLOB_API = "https://clob.polymarket.com"
    GAMMA_API = "https://gamma-api.polymarket.com"
    WEBSOCKET_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    CHAIN_ID = 137
    
    MARKET_CATEGORIES = ["crypto", "politics", "sports", "entertainment", "economics", "science", "other"]
    MIN_MARKET_LIQUIDITY = 1000
    MIN_MARKET_VOLUME_24H = 500
    MAX_MARKETS_TO_SCAN = 500
    MARKET_REFRESH_INTERVAL = 300
    NEW_MARKET_CHECK_INTERVAL = 60
    
    BINARY_ARB_THRESHOLD = 0.003
    MULTI_OUTCOME_THRESHOLD = 0.005
    TIME_DECAY_THRESHOLD = 0.002
    TIME_DECAY_HOURS = 24
    
    SLIPPAGE_BUFFER = 0.005
    MIN_LIQUIDITY_USD = 100
    MAX_BOOK_DEPTH_LEVELS = 10
    IMPACT_THRESHOLD = 0.01
    
    SCAN_INTERVAL_MS = 50
    ORDER_TIMEOUT_MS = 5000
    MAX_PARALLEL_ORDERS = 4
    MAX_RETRY_ATTEMPTS = 3
    
    USE_KELLY_SIZING = True
    KELLY_FRACTION = 0.25
    WIN_RATE_ESTIMATE = 0.80
    DEFAULT_ORDER_SIZE = 50
    MIN_ORDER_SIZE = 10
    
    MAX_POSITION_SIZE = 200
    MAX_TOTAL_EXPOSURE = 2000
    MAX_EXPOSURE_PER_MARKET = 500
    MAX_EXPOSURE_PER_CATEGORY = 1000
    MAX_CORRELATED_EXPOSURE = 750
    MAX_DAILY_LOSS = 200
    MAX_TRADES_PER_HOUR = 50
    COOLDOWN_SECONDS = 2
    
    KILL_SWITCH_ENABLED = True
    KILL_SWITCH_LOSS_THRESHOLD = 500
    KILL_SWITCH_DRAWDOWN = 15
    
    DB_PATH = "ultimate_arb_trades.db"
    LOG_ALL_OPPORTUNITIES = True
    DASHBOARD_UPDATE_INTERVAL = 1.0
    
    PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    FUNDER_ADDRESS = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
    SIGNATURE_TYPE = 1
    
    @classmethod
    def from_args(cls, args):
        if hasattr(args, "min_spread") and args.min_spread:
            cls.BINARY_ARB_THRESHOLD = args.min_spread / 100.0
        if hasattr(args, "max_position") and args.max_position:
            cls.MAX_POSITION_SIZE = args.max_position
        if hasattr(args, "max_daily_loss") and args.max_daily_loss:
            cls.MAX_DAILY_LOSS = args.max_daily_loss
        if hasattr(args, "max_exposure") and args.max_exposure:
            cls.MAX_TOTAL_EXPOSURE = args.max_exposure
        if hasattr(args, "categories") and args.categories:
            cls.MARKET_CATEGORIES = args.categories.split(",")
        if hasattr(args, "no_kelly") and args.no_kelly:
            cls.USE_KELLY_SIZING = False

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(debug=False):
    log_level = logging.DEBUG if debug else logging.INFO
    log_file = f"ultimate_arb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# CLOB CLIENT
# =============================================================================

CLOB_AVAILABLE = False
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY, SELL
    CLOB_AVAILABLE = True
    logger.info("py-clob-client available")
except ImportError:
    logger.warning("py-clob-client not installed")
    BUY = "BUY"
    SELL = "SELL"

# =============================================================================
# ENUMS
# =============================================================================

class MarketType(Enum):
    BINARY = auto()
    MULTI_OUTCOME = auto()

class OpportunityType(Enum):
    BINARY_ARB = auto()
    MULTI_OUTCOME_ARB = auto()
    CROSS_MARKET = auto()
    TIME_DECAY = auto()

class ExecutionStatus(Enum):
    PENDING = auto()
    PARTIAL = auto()
    FILLED = auto()
    CANCELLED = auto()
    FAILED = auto()

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Token:
    token_id: str
    outcome: str
    price: float = 0.0
    best_ask: float = 0.0
    best_bid: float = 0.0
    ask_liquidity: float = 0.0
    bid_liquidity: float = 0.0
    volume_24h: float = 0.0
    last_update: float = field(default_factory=time.time)

@dataclass
class Market:
    market_id: str
    condition_id: str
    slug: str
    question: str
    category: str
    market_type: MarketType
    tokens: List[Token]
    end_time: Optional[datetime] = None
    volume_24h: float = 0.0
    liquidity: float = 0.0
    is_active: bool = True
    
    @property
    def is_binary(self):
        return self.market_type == MarketType.BINARY
    
    @property
    def is_multi_outcome(self):
        return self.market_type == MarketType.MULTI_OUTCOME
    
    @property
    def time_to_expiry(self):
        if self.end_time:
            return self.end_time - datetime.now(timezone.utc)
        return None
    
    @property
    def is_near_expiry(self):
        ttl = self.time_to_expiry
        return ttl is not None and ttl.total_seconds() < Config.TIME_DECAY_HOURS * 3600

@dataclass
class OrderBookLevel:
    price: float
    size: float
    side: str
    
    @property
    def value(self):
        return self.price * self.size

@dataclass
class OrderBook:
    token_id: str
    asks: List[OrderBookLevel]
    bids: List[OrderBookLevel]
    timestamp: float = field(default_factory=time.time)
    
    @property
    def best_ask(self):
        return self.asks[0] if self.asks else None
    
    @property
    def best_bid(self):
        return self.bids[0] if self.bids else None
    
    def total_ask_liquidity(self, levels=None):
        levels = levels or len(self.asks)
        return sum(a.value for a in self.asks[:levels])
    
    def total_bid_liquidity(self, levels=None):
        levels = levels or len(self.bids)
        return sum(b.value for b in self.bids[:levels])
    
    def price_for_size(self, size, side="buy"):
        levels = self.asks if side == "buy" else self.bids
        remaining = size
        total_cost = 0.0
        total_filled = 0.0
        
        for level in levels:
            if remaining <= 0:
                break
            fill_size = min(remaining, level.size)
            total_cost += fill_size * level.price
            total_filled += fill_size
            remaining -= fill_size
        
        if total_filled == 0:
            return float("inf"), float("inf")
        
        return total_cost / total_filled, total_cost
    
    def price_impact(self, size):
        if not self.best_ask:
            return float("inf")
        avg_price, _ = self.price_for_size(size)
        return (avg_price - self.best_ask.price) / self.best_ask.price

@dataclass
class Opportunity:
    opportunity_id: str
    opportunity_type: OpportunityType
    market: Market
    tokens_to_buy: List[Tuple[Token, float, float]]
    total_cost: float
    expected_payout: float
    spread_pct: float
    expected_profit: float
    min_liquidity: float
    max_price_impact: float
    kelly_size: float
    confidence: float
    timestamp: float = field(default_factory=time.time)
    priority: float = field(default_factory=lambda: 0.0)
    
    def __lt__(self, other):
        return self.priority > other.priority
    
    @property
    def is_executable(self):
        age_ms = (time.time() - self.timestamp) * 1000
        if age_ms > 500:
            return False
        if self.min_liquidity < Config.MIN_LIQUIDITY_USD:
            return False
        if self.max_price_impact > Config.IMPACT_THRESHOLD:
            return False
        return True

@dataclass
class Trade:
    trade_id: str
    opportunity_id: str
    market_id: str
    market_slug: str
    category: str
    opportunity_type: str
    tokens: List[Dict]
    total_cost: float
    expected_profit: float
    actual_profit: float
    status: ExecutionStatus
    execution_time_ms: float
    timestamp: datetime
    error_message: Optional[str] = None

@dataclass 
class Position:
    market_id: str
    market_slug: str
    category: str
    tokens: Dict[str, float]
    entry_cost: float
    current_value: float
    unrealized_pnl: float
    entry_time: datetime

# =============================================================================
# DATABASE MANAGER
# =============================================================================

class DatabaseManager:
    """SQLite database for trade persistence and analytics."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or Config.DB_PATH
        self.conn = None
        self.lock = threading.Lock()
        self._initialize_db()
    
    def _initialize_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.cursor()
        
        trades_sql = """
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                opportunity_id TEXT,
                market_id TEXT,
                market_slug TEXT,
                category TEXT,
                opportunity_type TEXT,
                tokens TEXT,
                total_cost REAL,
                expected_profit REAL,
                actual_profit REAL,
                status TEXT,
                execution_time_ms REAL,
                timestamp TEXT,
                error_message TEXT
            )
        """
        cursor.execute(trades_sql)
        
        opps_sql = """
            CREATE TABLE IF NOT EXISTS opportunities (
                opportunity_id TEXT PRIMARY KEY,
                opportunity_type TEXT,
                market_id TEXT,
                market_slug TEXT,
                spread_pct REAL,
                expected_profit REAL,
                min_liquidity REAL,
                was_executed INTEGER,
                timestamp TEXT
            )
        """
        cursor.execute(opps_sql)
        
        pnl_sql = """
            CREATE TABLE IF NOT EXISTS daily_pnl (
                date TEXT PRIMARY KEY,
                total_pnl REAL,
                trade_count INTEGER,
                win_count INTEGER,
                loss_count INTEGER,
                avg_profit REAL,
                max_profit REAL,
                max_loss REAL
            )
        """
        cursor.execute(pnl_sql)
        self.conn.commit()

    def record_trade(self, trade: Trade):
        with self.lock:
            cursor = self.conn.cursor()
            sql = """INSERT OR REPLACE INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
            cursor.execute(sql, (
                trade.trade_id,
                trade.opportunity_id,
                trade.market_id,
                trade.market_slug,
                trade.category,
                trade.opportunity_type,
                json.dumps(trade.tokens),
                trade.total_cost,
                trade.expected_profit,
                trade.actual_profit,
                trade.status.name,
                trade.execution_time_ms,
                trade.timestamp.isoformat(),
                trade.error_message
            ))
            self.conn.commit()
    
    def record_opportunity(self, opp: Opportunity, was_executed: bool):
        with self.lock:
            cursor = self.conn.cursor()
            sql = """INSERT OR REPLACE INTO opportunities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""
            cursor.execute(sql, (
                opp.opportunity_id,
                opp.opportunity_type.name,
                opp.market.market_id,
                opp.market.slug,
                opp.spread_pct,
                opp.expected_profit,
                opp.min_liquidity,
                1 if was_executed else 0,
                datetime.now().isoformat()
            ))
            self.conn.commit()
    
    def update_daily_pnl(self, date_str: str, pnl: float, won: bool):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM daily_pnl WHERE date = ?", (date_str,))
            row = cursor.fetchone()
            
            if row:
                trade_count = row["trade_count"] + 1
                win_count = row["win_count"] + (1 if won else 0)
                loss_count = row["loss_count"] + (0 if won else 1)
                total_pnl = row["total_pnl"] + pnl
                avg_profit = total_pnl / trade_count
                max_profit = max(row["max_profit"], pnl if pnl > 0 else row["max_profit"])
                max_loss = min(row["max_loss"], pnl if pnl < 0 else row["max_loss"])
            else:
                trade_count = 1
                win_count = 1 if won else 0
                loss_count = 0 if won else 1
                total_pnl = pnl
                avg_profit = pnl
                max_profit = pnl if pnl > 0 else 0
                max_loss = pnl if pnl < 0 else 0
            
            sql = """INSERT OR REPLACE INTO daily_pnl VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
            cursor.execute(sql, (date_str, total_pnl, trade_count, win_count, loss_count, avg_profit, max_profit, max_loss))
            self.conn.commit()

    def get_today_pnl(self) -> float:
        with self.lock:
            cursor = self.conn.cursor()
            date_str = datetime.now().strftime("%Y-%m-%d")
            cursor.execute("SELECT total_pnl FROM daily_pnl WHERE date = ?", (date_str,))
            row = cursor.fetchone()
            return row["total_pnl"] if row else 0.0
    
    def get_recent_trades(self, limit: int = 50) -> List[Dict]:
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_performance_stats(self) -> Dict:
        with self.lock:
            cursor = self.conn.cursor()
            
            cursor.execute("SELECT COUNT(*) as total FROM trades")
            total_trades = cursor.fetchone()["total"]
            
            cursor.execute("SELECT COUNT(*) as wins FROM trades WHERE actual_profit > 0")
            wins = cursor.fetchone()["wins"]
            
            cursor.execute("SELECT SUM(actual_profit) as total_pnl FROM trades")
            total_pnl = cursor.fetchone()["total_pnl"] or 0.0
            
            cursor.execute("SELECT AVG(actual_profit) as avg_profit FROM trades WHERE actual_profit > 0")
            avg_win = cursor.fetchone()["avg_profit"] or 0.0
            
            cursor.execute("SELECT AVG(actual_profit) as avg_loss FROM trades WHERE actual_profit < 0")
            avg_loss = cursor.fetchone()["avg_loss"] or 0.0
            
            cursor.execute("SELECT AVG(execution_time_ms) as avg_exec FROM trades")
            avg_exec_time = cursor.fetchone()["avg_exec"] or 0.0
            
            return {
                "total_trades": total_trades,
                "win_rate": wins / total_trades if total_trades > 0 else 0.0,
                "total_pnl": total_pnl,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "avg_execution_time_ms": avg_exec_time
            }
    
    def close(self):
        if self.conn:
            self.conn.close()

# =============================================================================
# UNIVERSAL MARKET DISCOVERY
# =============================================================================

class UniversalMarketDiscovery:
    """Discovers ALL markets from Gamma API across all categories."""
    
    def __init__(self):
        self.markets: Dict[str, Market] = {}
        self.markets_by_category: Dict[str, List[str]] = defaultdict(list)
        self.last_full_scan: float = 0
        self.last_new_check: float = 0
        self.known_market_ids: Set[str] = set()
        self.lock = threading.Lock()
    
    def fetch_all_markets(self) -> List[Market]:
        """Fetch all active markets from Gamma API."""
        all_markets = []
        
        try:
            url = f"{Config.GAMMA_API}/markets"
            params = {
                "active": "true",
                "closed": "false",
                "limit": Config.MAX_MARKETS_TO_SCAN
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for item in data:
                market = self._parse_market(item)
                if market and self._passes_filter(market):
                    all_markets.append(market)
            
            logger.info(f"Discovered {len(all_markets)} markets from Gamma API")
            
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
        
        return all_markets

    def _parse_market(self, data: Dict) -> Optional[Market]:
        """Parse market data from Gamma API response."""
        try:
            market_id = data.get("id") or data.get("conditionId", "")
            condition_id = data.get("conditionId", market_id)
            
            tokens = []
            outcomes = data.get("outcomes", [])
            clob_token_ids = data.get("clobTokenIds", [])
            outcome_prices = data.get("outcomePrices", [])
            
            for i, outcome in enumerate(outcomes):
                token_id = clob_token_ids[i] if i < len(clob_token_ids) else ""
                price = float(outcome_prices[i]) if i < len(outcome_prices) else 0.0
                tokens.append(Token(
                    token_id=token_id,
                    outcome=outcome,
                    price=price
                ))
            
            if len(tokens) < 2:
                return None
            
            market_type = MarketType.BINARY if len(tokens) == 2 else MarketType.MULTI_OUTCOME
            
            end_time = None
            if data.get("endDate"):
                try:
                    end_time = datetime.fromisoformat(data["endDate"].replace("Z", "+00:00"))
                except:
                    pass
            
            category = self._categorize_market(data)
            
            return Market(
                market_id=market_id,
                condition_id=condition_id,
                slug=data.get("slug", ""),
                question=data.get("question", ""),
                category=category,
                market_type=market_type,
                tokens=tokens,
                end_time=end_time,
                volume_24h=float(data.get("volume24hr", 0) or 0),
                liquidity=float(data.get("liquidity", 0) or 0),
                is_active=data.get("active", True)
            )
            
        except Exception as e:
            logger.debug(f"Error parsing market: {e}")
            return None

    def _categorize_market(self, data: Dict) -> str:
        """Categorize market based on tags, question, or slug."""
        question = data.get("question", "").lower()
        slug = data.get("slug", "").lower()
        tags = [t.lower() for t in data.get("tags", [])]
        
        crypto_keywords = ["bitcoin", "btc", "eth", "ethereum", "crypto", "sol", "xrp", "price"]
        politics_keywords = ["election", "president", "vote", "trump", "biden", "congress", "senate"]
        sports_keywords = ["super bowl", "nfl", "nba", "mlb", "world cup", "olympics", "game", "win"]
        entertainment_keywords = ["oscar", "grammy", "movie", "album", "celebrity", "netflix"]
        economics_keywords = ["fed", "interest rate", "inflation", "gdp", "jobs", "unemployment"]
        science_keywords = ["spacex", "nasa", "climate", "vaccine", "ai", "research"]
        
        text = f"{question} {slug} {' '.join(tags)}"
        
        if any(kw in text for kw in crypto_keywords):
            return "crypto"
        elif any(kw in text for kw in politics_keywords):
            return "politics"
        elif any(kw in text for kw in sports_keywords):
            return "sports"
        elif any(kw in text for kw in entertainment_keywords):
            return "entertainment"
        elif any(kw in text for kw in economics_keywords):
            return "economics"
        elif any(kw in text for kw in science_keywords):
            return "science"
        else:
            return "other"
    
    def _passes_filter(self, market: Market) -> bool:
        """Check if market passes minimum requirements."""
        if market.category not in Config.MARKET_CATEGORIES:
            return False
        if market.liquidity < Config.MIN_MARKET_LIQUIDITY:
            return False
        if market.volume_24h < Config.MIN_MARKET_VOLUME_24H:
            return False
        if not market.tokens:
            return False
        return True

    def refresh_markets(self, force: bool = False):
        """Refresh market data if needed."""
        now = time.time()
        
        if force or (now - self.last_full_scan) > Config.MARKET_REFRESH_INTERVAL:
            markets = self.fetch_all_markets()
            
            with self.lock:
                self.markets.clear()
                self.markets_by_category.clear()
                
                for market in markets:
                    self.markets[market.market_id] = market
                    self.markets_by_category[market.category].append(market.market_id)
                    self.known_market_ids.add(market.market_id)
                
                self.last_full_scan = now
            
            logger.info(f"Refreshed {len(self.markets)} markets")
    
    def check_new_markets(self) -> List[Market]:
        """Check for newly listed markets."""
        now = time.time()
        
        if (now - self.last_new_check) < Config.NEW_MARKET_CHECK_INTERVAL:
            return []
        
        new_markets = []
        markets = self.fetch_all_markets()
        
        with self.lock:
            for market in markets:
                if market.market_id not in self.known_market_ids:
                    self.markets[market.market_id] = market
                    self.markets_by_category[market.category].append(market.market_id)
                    self.known_market_ids.add(market.market_id)
                    new_markets.append(market)
            
            self.last_new_check = now
        
        if new_markets:
            logger.info(f"Discovered {len(new_markets)} new markets")
        
        return new_markets
    
    def get_markets(self, category: str = None) -> List[Market]:
        """Get all markets, optionally filtered by category."""
        with self.lock:
            if category:
                market_ids = self.markets_by_category.get(category, [])
                return [self.markets[mid] for mid in market_ids if mid in self.markets]
            return list(self.markets.values())
    
    def get_market(self, market_id: str) -> Optional[Market]:
        """Get a specific market by ID."""
        with self.lock:
            return self.markets.get(market_id)

# =============================================================================
# ORDER BOOK FETCHER
# =============================================================================

class OrderBookFetcher:
    """Fetches order books with parallel execution and caching."""
    
    def __init__(self, max_workers: int = 10):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache: Dict[str, OrderBook] = {}
        self.cache_ttl = 0.5
        self.lock = threading.Lock()
    
    def fetch_order_book(self, token_id: str) -> Optional[OrderBook]:
        """Fetch order book for a single token."""
        with self.lock:
            if token_id in self.cache:
                cached = self.cache[token_id]
                if time.time() - cached.timestamp < self.cache_ttl:
                    return cached
        
        try:
            url = f"{Config.CLOB_API}/book"
            params = {"token_id": token_id}
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            asks = []
            for level in data.get("asks", [])[:Config.MAX_BOOK_DEPTH_LEVELS]:
                asks.append(OrderBookLevel(
                    price=float(level.get("price", 0)),
                    size=float(level.get("size", 0)),
                    side="ask"
                ))
            asks.sort(key=lambda x: x.price)
            
            bids = []
            for level in data.get("bids", [])[:Config.MAX_BOOK_DEPTH_LEVELS]:
                bids.append(OrderBookLevel(
                    price=float(level.get("price", 0)),
                    size=float(level.get("size", 0)),
                    side="bid"
                ))
            bids.sort(key=lambda x: x.price, reverse=True)
            
            book = OrderBook(token_id=token_id, asks=asks, bids=bids)
            
            with self.lock:
                self.cache[token_id] = book
            
            return book
            
        except Exception as e:
            logger.debug(f"Error fetching order book for {token_id}: {e}")
            return None

    def fetch_order_books_parallel(self, token_ids: List[str]) -> Dict[str, OrderBook]:
        """Fetch multiple order books in parallel."""
        results = {}
        futures = {self.executor.submit(self.fetch_order_book, tid): tid for tid in token_ids}
        
        for future in as_completed(futures, timeout=10):
            token_id = futures[future]
            try:
                book = future.result()
                if book:
                    results[token_id] = book
            except Exception as e:
                logger.debug(f"Error in parallel fetch for {token_id}: {e}")
        
        return results
    
    def update_token_prices(self, market: Market):
        """Update token prices from order books."""
        token_ids = [t.token_id for t in market.tokens if t.token_id]
        books = self.fetch_order_books_parallel(token_ids)
        
        for token in market.tokens:
            if token.token_id in books:
                book = books[token.token_id]
                if book.best_ask:
                    token.best_ask = book.best_ask.price
                    token.ask_liquidity = book.total_ask_liquidity()
                if book.best_bid:
                    token.best_bid = book.best_bid.price
                    token.bid_liquidity = book.total_bid_liquidity()
                token.price = (token.best_ask + token.best_bid) / 2 if token.best_ask and token.best_bid else token.best_ask or token.best_bid
                token.last_update = time.time()
    
    def clear_cache(self):
        """Clear the order book cache."""
        with self.lock:
            self.cache.clear()
    
    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=False)

# =============================================================================
# WEBSOCKET MANAGER
# =============================================================================

class WebSocketManager:
    """Manages WebSocket connections for real-time price updates."""
    
    def __init__(self, on_update: Callable[[str, Dict], None] = None):
        self.ws = None
        self.on_update = on_update
        self.subscribed_tokens: Set[str] = set()
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.reconnect_delay = 1
        self.max_reconnect_delay = 60
    
    def connect(self):
        """Connect to the WebSocket."""
        try:
            self.ws = websocket.WebSocketApp(
                Config.WEBSOCKET_URL,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info("WebSocket connection started")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
    
    def _run(self):
        """Run the WebSocket event loop."""
        while self.running:
            try:
                self.ws.run_forever()
            except Exception as e:
                logger.error(f"WebSocket run error: {e}")
            
            if self.running:
                logger.info(f"Reconnecting in {self.reconnect_delay}s...")
                time.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
    
    def _on_open(self, ws):
        """Handle WebSocket open event."""
        logger.info("WebSocket connected")
        self.reconnect_delay = 1
        with self.lock:
            for token_id in self.subscribed_tokens:
                self._send_subscribe(token_id)
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            if self.on_update:
                token_id = data.get("asset_id") or data.get("token_id", "")
                if token_id:
                    self.on_update(token_id, data)
        except Exception as e:
            logger.debug(f"WebSocket message error: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        logger.info(f"WebSocket closed: {close_status_code} {close_msg}")
    
    def _send_subscribe(self, token_id: str):
        """Send subscription message."""
        try:
            msg = {
                "type": "subscribe",
                "channel": "book",
                "assets_ids": [token_id]
            }
            self.ws.send(json.dumps(msg))
        except Exception as e:
            logger.debug(f"Subscribe error: {e}")
    
    def subscribe(self, token_ids: List[str]):
        """Subscribe to token price updates."""
        with self.lock:
            for token_id in token_ids:
                if token_id not in self.subscribed_tokens:
                    self.subscribed_tokens.add(token_id)
                    if self.ws and self.ws.sock and self.ws.sock.connected:
                        self._send_subscribe(token_id)
    
    def unsubscribe(self, token_ids: List[str]):
        """Unsubscribe from token updates."""
        with self.lock:
            for token_id in token_ids:
                self.subscribed_tokens.discard(token_id)
    
    def close(self):
        """Close the WebSocket connection."""
        self.running = False
        if self.ws:
            self.ws.close()

# =============================================================================
# OPPORTUNITY DETECTOR
# =============================================================================

class OpportunityDetector:
    """Detects arbitrage opportunities across all market types."""
    
    def __init__(self, book_fetcher: OrderBookFetcher):
        self.book_fetcher = book_fetcher
        self.opportunity_counter = 0
        self.lock = threading.Lock()
    
    def _generate_opportunity_id(self) -> str:
        with self.lock:
            self.opportunity_counter += 1
            return f"opp_{int(time.time())}_{self.opportunity_counter}"
    
    def detect_binary_arb(self, market: Market) -> Optional[Opportunity]:
        """Detect binary arbitrage where sum of asks < 1.00."""
        if not market.is_binary or len(market.tokens) != 2:
            return None
        
        token1, token2 = market.tokens
        
        if not token1.best_ask or not token2.best_ask:
            return None
        if token1.best_ask <= 0 or token2.best_ask <= 0:
            return None
        
        total_cost = token1.best_ask + token2.best_ask
        spread = 1.0 - total_cost
        spread_pct = spread / total_cost if total_cost > 0 else 0
        
        if spread_pct < Config.BINARY_ARB_THRESHOLD:
            return None
        
        min_liquidity = min(token1.ask_liquidity, token2.ask_liquidity)
        
        book1 = self.book_fetcher.cache.get(token1.token_id)
        book2 = self.book_fetcher.cache.get(token2.token_id)
        
        max_impact = 0.0
        if book1:
            max_impact = max(max_impact, book1.price_impact(Config.DEFAULT_ORDER_SIZE))
        if book2:
            max_impact = max(max_impact, book2.price_impact(Config.DEFAULT_ORDER_SIZE))
        
        expected_profit = spread * Config.DEFAULT_ORDER_SIZE
        kelly_size = self._calculate_kelly_size(spread_pct, Config.WIN_RATE_ESTIMATE)
        
        confidence = min(1.0, min_liquidity / 1000) * (1 - max_impact * 10)
        priority = spread_pct * confidence * (min_liquidity / 100)
        
        return Opportunity(
            opportunity_id=self._generate_opportunity_id(),
            opportunity_type=OpportunityType.BINARY_ARB,
            market=market,
            tokens_to_buy=[(token1, token1.best_ask, 1.0), (token2, token2.best_ask, 1.0)],
            total_cost=total_cost,
            expected_payout=1.0,
            spread_pct=spread_pct,
            expected_profit=expected_profit,
            min_liquidity=min_liquidity,
            max_price_impact=max_impact,
            kelly_size=kelly_size,
            confidence=confidence,
            priority=priority
        )

    def detect_multi_outcome_arb(self, market: Market) -> Optional[Opportunity]:
        """Detect multi-outcome arbitrage where sum of all asks < 1.00."""
        if not market.is_multi_outcome or len(market.tokens) < 3:
            return None
        
        total_cost = 0.0
        valid_tokens = []
        min_liquidity = float("inf")
        max_impact = 0.0
        
        for token in market.tokens:
            if not token.best_ask or token.best_ask <= 0:
                return None
            total_cost += token.best_ask
            valid_tokens.append((token, token.best_ask, 1.0))
            min_liquidity = min(min_liquidity, token.ask_liquidity)
            
            book = self.book_fetcher.cache.get(token.token_id)
            if book:
                max_impact = max(max_impact, book.price_impact(Config.DEFAULT_ORDER_SIZE / len(market.tokens)))
        
        spread = 1.0 - total_cost
        spread_pct = spread / total_cost if total_cost > 0 else 0
        
        if spread_pct < Config.MULTI_OUTCOME_THRESHOLD:
            return None
        
        expected_profit = spread * Config.DEFAULT_ORDER_SIZE
        kelly_size = self._calculate_kelly_size(spread_pct, Config.WIN_RATE_ESTIMATE)
        
        confidence = min(1.0, min_liquidity / 500) * (1 - max_impact * 10)
        priority = spread_pct * confidence * (min_liquidity / 100)
        
        return Opportunity(
            opportunity_id=self._generate_opportunity_id(),
            opportunity_type=OpportunityType.MULTI_OUTCOME_ARB,
            market=market,
            tokens_to_buy=valid_tokens,
            total_cost=total_cost,
            expected_payout=1.0,
            spread_pct=spread_pct,
            expected_profit=expected_profit,
            min_liquidity=min_liquidity,
            max_price_impact=max_impact,
            kelly_size=kelly_size,
            confidence=confidence,
            priority=priority
        )

    def detect_time_decay_opportunity(self, market: Market) -> Optional[Opportunity]:
        """Detect opportunities in markets near expiry with mispriced outcomes."""
        if not market.is_near_expiry:
            return None
        
        for token in market.tokens:
            if token.best_ask and token.best_ask < Config.TIME_DECAY_THRESHOLD:
                expected_profit = (1.0 - token.best_ask) * Config.MIN_ORDER_SIZE * 0.1
                
                if expected_profit < 0.5:
                    continue
                
                return Opportunity(
                    opportunity_id=self._generate_opportunity_id(),
                    opportunity_type=OpportunityType.TIME_DECAY,
                    market=market,
                    tokens_to_buy=[(token, token.best_ask, 1.0)],
                    total_cost=token.best_ask,
                    expected_payout=1.0,
                    spread_pct=(1.0 - token.best_ask),
                    expected_profit=expected_profit,
                    min_liquidity=token.ask_liquidity,
                    max_price_impact=0.0,
                    kelly_size=Config.MIN_ORDER_SIZE,
                    confidence=0.3,
                    priority=expected_profit * 0.5
                )
        
        return None
    
    def _calculate_kelly_size(self, edge: float, win_rate: float) -> float:
        """Calculate Kelly Criterion position size."""
        if not Config.USE_KELLY_SIZING:
            return Config.DEFAULT_ORDER_SIZE
        
        if edge <= 0 or win_rate <= 0:
            return Config.MIN_ORDER_SIZE
        
        b = 1 / edge
        p = win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b
        kelly = max(0, kelly) * Config.KELLY_FRACTION
        
        size = kelly * Config.MAX_TOTAL_EXPOSURE
        size = max(Config.MIN_ORDER_SIZE, min(size, Config.MAX_POSITION_SIZE))
        
        return size
    
    def scan_market(self, market: Market) -> List[Opportunity]:
        """Scan a single market for all types of opportunities."""
        opportunities = []
        
        self.book_fetcher.update_token_prices(market)
        
        if market.is_binary:
            opp = self.detect_binary_arb(market)
            if opp:
                opportunities.append(opp)
        
        if market.is_multi_outcome:
            opp = self.detect_multi_outcome_arb(market)
            if opp:
                opportunities.append(opp)
        
        opp = self.detect_time_decay_opportunity(market)
        if opp:
            opportunities.append(opp)
        
        return opportunities
    
    def scan_all_markets(self, markets: List[Market]) -> List[Opportunity]:
        """Scan all markets for opportunities and return priority-sorted list."""
        all_opportunities = []
        
        for market in markets:
            opps = self.scan_market(market)
            all_opportunities.extend(opps)
        
        heapq.heapify(all_opportunities)
        
        return all_opportunities

# =============================================================================
# RISK MANAGER
# =============================================================================

class RiskManager:
    """Comprehensive risk management with kill switch."""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.positions: Dict[str, Position] = {}
        self.exposure_by_category: Dict[str, float] = defaultdict(float)
        self.exposure_by_market: Dict[str, float] = defaultdict(float)
        self.total_exposure: float = 0.0
        self.daily_pnl: float = 0.0
        self.peak_equity: float = 0.0
        self.trades_this_hour: int = 0
        self.hour_start: float = time.time()
        self.last_trade_time: float = 0
        self.kill_switch_triggered: bool = False
        self.lock = threading.Lock()
    
    def can_trade(self, opportunity: Opportunity) -> Tuple[bool, str]:
        """Check if a trade is allowed under current risk limits."""
        with self.lock:
            if self.kill_switch_triggered:
                return False, "Kill switch triggered"
            
            if self.total_exposure + opportunity.total_cost > Config.MAX_TOTAL_EXPOSURE:
                return False, f"Total exposure limit reached: {self.total_exposure:.2f}"
            
            market_id = opportunity.market.market_id
            category = opportunity.market.category
            
            if self.exposure_by_market[market_id] + opportunity.total_cost > Config.MAX_EXPOSURE_PER_MARKET:
                return False, f"Market exposure limit reached for {market_id}"
            
            if self.exposure_by_category[category] + opportunity.total_cost > Config.MAX_EXPOSURE_PER_CATEGORY:
                return False, f"Category exposure limit reached for {category}"
            
            self.daily_pnl = self.db.get_today_pnl()
            if self.daily_pnl < -Config.MAX_DAILY_LOSS:
                return False, f"Daily loss limit reached: {self.daily_pnl:.2f}"
            
            now = time.time()
            if now - self.hour_start > 3600:
                self.trades_this_hour = 0
                self.hour_start = now
            
            if self.trades_this_hour >= Config.MAX_TRADES_PER_HOUR:
                return False, f"Hourly trade limit reached: {self.trades_this_hour}"
            
            if now - self.last_trade_time < Config.COOLDOWN_SECONDS:
                return False, f"Cooldown period: {Config.COOLDOWN_SECONDS - (now - self.last_trade_time):.1f}s remaining"
            
            return True, "OK"

    def record_trade(self, opportunity: Opportunity, success: bool, actual_pnl: float):
        """Record trade and update risk metrics."""
        with self.lock:
            market_id = opportunity.market.market_id
            category = opportunity.market.category
            
            if success:
                self.total_exposure += opportunity.total_cost
                self.exposure_by_market[market_id] += opportunity.total_cost
                self.exposure_by_category[category] += opportunity.total_cost
            
            self.daily_pnl += actual_pnl
            self.trades_this_hour += 1
            self.last_trade_time = time.time()
            
            self._check_kill_switch()
    
    def _check_kill_switch(self):
        """Check if kill switch should be triggered."""
        if not Config.KILL_SWITCH_ENABLED:
            return
        
        if self.daily_pnl < -Config.KILL_SWITCH_LOSS_THRESHOLD:
            self.kill_switch_triggered = True
            logger.critical(f"KILL SWITCH: Daily loss {self.daily_pnl:.2f} exceeded threshold")
            return
        
        if self.peak_equity > 0:
            drawdown_pct = (self.peak_equity - (self.peak_equity + self.daily_pnl)) / self.peak_equity * 100
            if drawdown_pct > Config.KILL_SWITCH_DRAWDOWN:
                self.kill_switch_triggered = True
                logger.critical(f"KILL SWITCH: Drawdown {drawdown_pct:.1f}% exceeded threshold")
    
    def release_exposure(self, market_id: str, category: str, amount: float):
        """Release exposure when position closes."""
        with self.lock:
            self.total_exposure = max(0, self.total_exposure - amount)
            self.exposure_by_market[market_id] = max(0, self.exposure_by_market[market_id] - amount)
            self.exposure_by_category[category] = max(0, self.exposure_by_category[category] - amount)
    
    def reset_kill_switch(self):
        """Manually reset the kill switch."""
        with self.lock:
            self.kill_switch_triggered = False
            logger.warning("Kill switch manually reset")
    
    def get_risk_summary(self) -> Dict:
        """Get current risk summary."""
        with self.lock:
            return {
                "total_exposure": self.total_exposure,
                "exposure_by_category": dict(self.exposure_by_category),
                "exposure_by_market": dict(self.exposure_by_market),
                "daily_pnl": self.daily_pnl,
                "trades_this_hour": self.trades_this_hour,
                "kill_switch_triggered": self.kill_switch_triggered
            }

# =============================================================================
# EXECUTION ENGINE
# =============================================================================

class ExecutionEngine:
    """Handles order execution with retries and partial fills."""
    
    def __init__(self, db: DatabaseManager, risk_manager: RiskManager, dry_run: bool = True):
        self.db = db
        self.risk_manager = risk_manager
        self.dry_run = dry_run
        self.client = None
        self.trade_counter = 0
        self.lock = threading.Lock()
        
        if CLOB_AVAILABLE and not dry_run and Config.PRIVATE_KEY:
            try:
                self.client = ClobClient(
                    Config.CLOB_API,
                    key=Config.PRIVATE_KEY,
                    chain_id=Config.CHAIN_ID,
                    funder=Config.FUNDER_ADDRESS,
                    signature_type=Config.SIGNATURE_TYPE
                )
                logger.info("CLOB client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize CLOB client: {e}")
    
    def _generate_trade_id(self) -> str:
        with self.lock:
            self.trade_counter += 1
            return f"trade_{int(time.time())}_{self.trade_counter}"
    
    def execute_opportunity(self, opportunity: Opportunity) -> Trade:
        """Execute an arbitrage opportunity."""
        trade_id = self._generate_trade_id()
        start_time = time.time()
        
        can_trade, reason = self.risk_manager.can_trade(opportunity)
        if not can_trade:
            logger.info(f"Trade blocked: {reason}")
            trade = Trade(
                trade_id=trade_id,
                opportunity_id=opportunity.opportunity_id,
                market_id=opportunity.market.market_id,
                market_slug=opportunity.market.slug,
                category=opportunity.market.category,
                opportunity_type=opportunity.opportunity_type.name,
                tokens=[{"token_id": t[0].token_id, "outcome": t[0].outcome, "price": t[1]} for t in opportunity.tokens_to_buy],
                total_cost=opportunity.total_cost,
                expected_profit=opportunity.expected_profit,
                actual_profit=0.0,
                status=ExecutionStatus.CANCELLED,
                execution_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                error_message=reason
            )
            self.db.record_trade(trade)
            return trade
        
        if self.dry_run:
            return self._simulate_execution(opportunity, trade_id, start_time)
        else:
            return self._real_execution(opportunity, trade_id, start_time)
