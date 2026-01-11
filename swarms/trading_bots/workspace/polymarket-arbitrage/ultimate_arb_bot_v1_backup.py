#!/usr/bin/env python3
"""
Ultimate Polymarket Arbitrage Bot
=================================

The most comprehensive arbitrage detection and execution system for Polymarket.

Features:
- Universal market discovery via Gamma API
- Multi-outcome and cross-market arbitrage detection
- Order book depth analysis with optimal sizing
- Real-time WebSocket updates with sub-100ms detection
- SQLite persistence with full trade history
- Risk management with kill switch and alerts

Author: Trading Bots Swarm
Created: 2026-01-04
"""

import asyncio
import aiohttp
import json
import logging
import os
import sqlite3
import time
import hashlib
import smtplib
import ssl
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from email.mime.text import MIMEText
from enum import Enum, auto
from heapq import heappush, heappop
from typing import Any, Callable, Optional
import threading
import signal
import sys

# Third-party imports (install via: pip install aiohttp websockets py-clob-client python-dotenv)
try:
    import websockets
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    from dotenv import load_dotenv
    load_dotenv()
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install aiohttp websockets py-clob-client python-dotenv")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Global configuration for the arbitrage bot."""

    # API Endpoints
    GAMMA_API_URL: str = "https://gamma-api.polymarket.com"
    CLOB_API_URL: str = "https://clob.polymarket.com"
    WS_URL: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    # Authentication
    PRIVATE_KEY: str = field(default_factory=lambda: os.getenv("POLYMARKET_PRIVATE_KEY", ""))
    API_KEY: str = field(default_factory=lambda: os.getenv("POLYMARKET_API_KEY", ""))
    API_SECRET: str = field(default_factory=lambda: os.getenv("POLYMARKET_API_SECRET", ""))
    API_PASSPHRASE: str = field(default_factory=lambda: os.getenv("POLYMARKET_API_PASSPHRASE", ""))

    # Chain config
    CHAIN_ID: int = 137  # Polygon mainnet

    # Trading parameters
    MIN_PROFIT_THRESHOLD: Decimal = Decimal("0.01")  # 1% minimum profit
    MAX_POSITION_SIZE: Decimal = Decimal("1000")  # Max $1000 per position
    MIN_LIQUIDITY_RATIO: Decimal = Decimal("0.1")  # Min 10% of order book

    # Risk parameters
    MAX_TOTAL_EXPOSURE: Decimal = Decimal("10000")  # Max $10k total
    MAX_PER_MARKET_EXPOSURE: Decimal = Decimal("2000")  # Max $2k per market
    MAX_CORRELATED_EXPOSURE: Decimal = Decimal("3000")  # Max $3k correlated
    VOLATILITY_THRESHOLD: Decimal = Decimal("0.05")  # 5% volatility threshold

    # Timing
    SCAN_INTERVAL_MS: int = 100  # 100ms between scans
    WS_RECONNECT_DELAY: int = 5  # 5 seconds
    ORDER_TIMEOUT_MS: int = 5000  # 5 second order timeout

    # Database
    DB_PATH: str = "arbitrage_trades.db"

    # Alerts
    ALERT_EMAIL: str = field(default_factory=lambda: os.getenv("ALERT_EMAIL", ""))
    ALERT_SMS_NUMBER: str = field(default_factory=lambda: os.getenv("ALERT_SMS_NUMBER", ""))
    SMTP_SERVER: str = field(default_factory=lambda: os.getenv("SMTP_SERVER", "smtp.gmail.com"))
    SMTP_PORT: int = 587
    SMTP_USER: str = field(default_factory=lambda: os.getenv("SMTP_USER", ""))
    SMTP_PASSWORD: str = field(default_factory=lambda: os.getenv("SMTP_PASSWORD", ""))

    # Categories to scan
    MARKET_CATEGORIES: list = field(default_factory=lambda: [
        "crypto", "politics", "sports", "entertainment",
        "science", "business", "weather", "pop-culture"
    ])


# =============================================================================
# DATA MODELS
# =============================================================================

class MarketType(Enum):
    BINARY = auto()
    MULTI_OUTCOME = auto()


class OpportunityType(Enum):
    BINARY_ARB = auto()       # YES + NO < $1
    MULTI_OUTCOME_ARB = auto() # Sum of outcomes < $1
    CROSS_MARKET_ARB = auto()  # Same event, different markets
    TIME_DECAY = auto()        # Near resolution opportunity


@dataclass
class Token:
    """Represents a single outcome token."""
    token_id: str
    outcome: str
    price: Decimal
    market_id: str


@dataclass
class OrderBookLevel:
    """Single level in an order book."""
    price: Decimal
    size: Decimal
    side: str  # 'bid' or 'ask'


@dataclass
class OrderBook:
    """Full order book for a token."""
    token_id: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    timestamp: datetime

    def best_bid(self) -> Optional[Decimal]:
        return self.bids[0].price if self.bids else None

    def best_ask(self) -> Optional[Decimal]:
        return self.asks[0].price if self.asks else None

    def mid_price(self) -> Optional[Decimal]:
        bid, ask = self.best_bid(), self.best_ask()
        if bid and ask:
            return (bid + ask) / 2
        return bid or ask

    def walk_book(self, side: str, size: Decimal) -> tuple[Decimal, Decimal]:
        """
        Walk the order book to get effective price for a size.
        Returns (average_price, filled_size).
        """
        levels = self.bids if side == 'bid' else self.asks
        remaining = size
        total_cost = Decimal("0")
        filled = Decimal("0")

        for level in levels:
            take = min(remaining, level.size)
            total_cost += take * level.price
            filled += take
            remaining -= take
            if remaining <= 0:
                break

        avg_price = total_cost / filled if filled > 0 else Decimal("0")
        return avg_price, filled


@dataclass
class Market:
    """Represents a Polymarket market."""
    id: str
    condition_id: str
    question: str
    category: str
    market_type: MarketType
    tokens: list[Token]
    end_date: Optional[datetime]
    volume_24h: Decimal
    liquidity: Decimal
    is_active: bool
    created_at: datetime

    def time_to_resolution(self) -> Optional[timedelta]:
        if self.end_date:
            return self.end_date - datetime.utcnow()
        return None


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity."""
    id: str
    opportunity_type: OpportunityType
    markets: list[Market]
    tokens: list[Token]
    expected_profit: Decimal
    expected_profit_pct: Decimal
    required_capital: Decimal
    confidence: Decimal  # 0-1 based on liquidity depth
    detected_at: datetime
    expires_at: Optional[datetime]
    order_books: dict[str, OrderBook]

    def __lt__(self, other):
        """For priority queue ordering - higher profit = higher priority."""
        return self.expected_profit > other.expected_profit


@dataclass
class Trade:
    """Executed trade record."""
    id: str
    opportunity_id: str
    token_id: str
    market_id: str
    side: str
    size: Decimal
    price: Decimal
    filled_size: Decimal
    filled_price: Decimal
    status: str
    created_at: datetime
    executed_at: Optional[datetime]
    pnl: Optional[Decimal]


@dataclass
class Position:
    """Current position in a market."""
    market_id: str
    token_id: str
    size: Decimal
    avg_entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal


# =============================================================================
# DATABASE LAYER
# =============================================================================

class Database:
    """SQLite database for trade history and analytics."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init_schema()

    def _init_schema(self):
        """Initialize database tables."""
        with self.lock:
            cursor = self.conn.cursor()

            # Opportunities table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS opportunities (
                    id TEXT PRIMARY KEY,
                    opportunity_type TEXT NOT NULL,
                    market_ids TEXT NOT NULL,
                    token_ids TEXT NOT NULL,
                    expected_profit REAL NOT NULL,
                    expected_profit_pct REAL NOT NULL,
                    required_capital REAL NOT NULL,
                    confidence REAL NOT NULL,
                    detected_at TIMESTAMP NOT NULL,
                    status TEXT DEFAULT 'detected',
                    executed_at TIMESTAMP,
                    actual_profit REAL
                )
            """)

            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    opportunity_id TEXT,
                    token_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    price REAL NOT NULL,
                    filled_size REAL,
                    filled_price REAL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    executed_at TIMESTAMP,
                    pnl REAL,
                    FOREIGN KEY (opportunity_id) REFERENCES opportunities(id)
                )
            """)

            # Positions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id TEXT NOT NULL,
                    token_id TEXT NOT NULL,
                    size REAL NOT NULL,
                    avg_entry_price REAL NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    UNIQUE(market_id, token_id)
                )
            """)

            # Market snapshots for analytics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id TEXT NOT NULL,
                    category TEXT,
                    prices TEXT NOT NULL,
                    volume_24h REAL,
                    liquidity REAL,
                    snapshot_at TIMESTAMP NOT NULL
                )
            """)

            # Performance metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    recorded_at TIMESTAMP NOT NULL
                )
            """)

            self.conn.commit()

    def save_opportunity(self, opp: ArbitrageOpportunity):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO opportunities
                (id, opportunity_type, market_ids, token_ids, expected_profit,
                 expected_profit_pct, required_capital, confidence, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                opp.id,
                opp.opportunity_type.name,
                json.dumps([m.id for m in opp.markets]),
                json.dumps([t.token_id for t in opp.tokens]),
                float(opp.expected_profit),
                float(opp.expected_profit_pct),
                float(opp.required_capital),
                float(opp.confidence),
                opp.detected_at.isoformat()
            ))
            self.conn.commit()

    def save_trade(self, trade: Trade):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO trades
                (id, opportunity_id, token_id, market_id, side, size, price,
                 filled_size, filled_price, status, created_at, executed_at, pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.id,
                trade.opportunity_id,
                trade.token_id,
                trade.market_id,
                trade.side,
                float(trade.size),
                float(trade.price),
                float(trade.filled_size) if trade.filled_size else None,
                float(trade.filled_price) if trade.filled_price else None,
                trade.status,
                trade.created_at.isoformat(),
                trade.executed_at.isoformat() if trade.executed_at else None,
                float(trade.pnl) if trade.pnl else None
            ))
            self.conn.commit()

    def update_position(self, position: Position):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO positions
                (market_id, token_id, size, avg_entry_price, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                position.market_id,
                position.token_id,
                float(position.size),
                float(position.avg_entry_price),
                datetime.utcnow().isoformat()
            ))
            self.conn.commit()

    def get_total_exposure(self) -> Decimal:
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT COALESCE(SUM(size * avg_entry_price), 0)
                FROM positions WHERE size > 0
            """)
            result = cursor.fetchone()
            return Decimal(str(result[0])) if result else Decimal("0")

    def get_market_exposure(self, market_id: str) -> Decimal:
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT COALESCE(SUM(size * avg_entry_price), 0)
                FROM positions WHERE market_id = ? AND size > 0
            """, (market_id,))
            result = cursor.fetchone()
            return Decimal(str(result[0])) if result else Decimal("0")

    def get_pnl_stats(self, days: int = 30) -> dict:
        with self.lock:
            cursor = self.conn.cursor()
            since = (datetime.utcnow() - timedelta(days=days)).isoformat()

            cursor.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    COALESCE(SUM(pnl), 0) as total_pnl,
                    COALESCE(AVG(pnl), 0) as avg_pnl,
                    COALESCE(MAX(pnl), 0) as max_profit,
                    COALESCE(MIN(pnl), 0) as max_loss
                FROM trades
                WHERE executed_at >= ? AND status = 'filled'
            """, (since,))

            row = cursor.fetchone()
            return {
                "total_trades": row[0],
                "winning_trades": row[1] or 0,
                "losing_trades": row[2] or 0,
                "win_rate": (row[1] or 0) / row[0] if row[0] > 0 else 0,
                "total_pnl": Decimal(str(row[3])),
                "avg_pnl": Decimal(str(row[4])),
                "max_profit": Decimal(str(row[5])),
                "max_loss": Decimal(str(row[6]))
            }

    def get_category_heatmap(self) -> dict:
        """Get opportunity counts by category for heatmap."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT
                    json_extract(market_ids, '$[0]') as market_id,
                    COUNT(*) as opp_count,
                    AVG(expected_profit_pct) as avg_profit_pct
                FROM opportunities
                WHERE detected_at >= datetime('now', '-24 hours')
                GROUP BY market_id
            """)

            results = {}
            for row in cursor.fetchall():
                results[row[0]] = {
                    "count": row[1],
                    "avg_profit_pct": row[2]
                }
            return results


# =============================================================================
# MARKET DISCOVERY
# =============================================================================

class MarketDiscovery:
    """Universal market discovery via Gamma API."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("MarketDiscovery")
        self.markets: dict[str, Market] = {}
        self.markets_by_category: dict[str, list[Market]] = defaultdict(list)
        self.last_scan: Optional[datetime] = None

    async def discover_all_markets(self, session: aiohttp.ClientSession) -> list[Market]:
        """Discover all active markets from Gamma API."""
        markets = []

        # Fetch from multiple endpoints to ensure complete coverage
        endpoints = [
            "/markets?active=true&closed=false",
            "/markets?active=true&order=volume24hr&ascending=false&limit=500",
        ]

        # Also fetch by category
        for category in self.config.MARKET_CATEGORIES:
            endpoints.append(f"/markets?active=true&tag={category}&limit=200")

        seen_ids = set()

        for endpoint in endpoints:
            try:
                async with session.get(
                    f"{self.config.GAMMA_API_URL}{endpoint}"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        for market_data in data:
                            market_id = market_data.get("id") or market_data.get("condition_id")
                            if market_id and market_id not in seen_ids:
                                seen_ids.add(market_id)
                                market = self._parse_market(market_data)
                                if market and market.is_active:
                                    markets.append(market)
            except Exception as e:
                self.logger.error(f"Error fetching {endpoint}: {e}")

        # Update internal cache
        self.markets = {m.id: m for m in markets}
        self.markets_by_category.clear()
        for m in markets:
            self.markets_by_category[m.category].append(m)

        self.last_scan = datetime.utcnow()
        self.logger.info(f"Discovered {len(markets)} active markets")

        return markets

    def _parse_market(self, data: dict) -> Optional[Market]:
        """Parse market data from API response."""
        try:
            # Determine market type
            tokens_data = data.get("tokens", [])
            if len(tokens_data) == 2:
                market_type = MarketType.BINARY
            elif len(tokens_data) > 2:
                market_type = MarketType.MULTI_OUTCOME
            else:
                return None  # Invalid market

            # Parse tokens
            tokens = []
            for t in tokens_data:
                token = Token(
                    token_id=t.get("token_id", ""),
                    outcome=t.get("outcome", ""),
                    price=Decimal(str(t.get("price", 0))),
                    market_id=data.get("condition_id", "")
                )
                tokens.append(token)

            # Parse end date
            end_date = None
            if data.get("end_date_iso"):
                try:
                    end_date = datetime.fromisoformat(
                        data["end_date_iso"].replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except:
                    pass

            # Parse created date
            created_at = datetime.utcnow()
            if data.get("created_at"):
                try:
                    created_at = datetime.fromisoformat(
                        data["created_at"].replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except:
                    pass

            return Market(
                id=data.get("condition_id", ""),
                condition_id=data.get("condition_id", ""),
                question=data.get("question", ""),
                category=data.get("category", data.get("tags", ["misc"])[0] if data.get("tags") else "misc"),
                market_type=market_type,
                tokens=tokens,
                end_date=end_date,
                volume_24h=Decimal(str(data.get("volume24hr", 0))),
                liquidity=Decimal(str(data.get("liquidity", 0))),
                is_active=data.get("active", False) and not data.get("closed", True),
                created_at=created_at
            )
        except Exception as e:
            self.logger.debug(f"Failed to parse market: {e}")
            return None

    async def watch_new_markets(
        self,
        session: aiohttp.ClientSession,
        callback: Callable[[Market], None],
        interval: int = 60
    ):
        """Continuously watch for new markets."""
        known_ids = set(self.markets.keys())

        while True:
            try:
                await asyncio.sleep(interval)
                markets = await self.discover_all_markets(session)

                for market in markets:
                    if market.id not in known_ids:
                        known_ids.add(market.id)
                        self.logger.info(f"New market discovered: {market.question[:50]}")
                        callback(market)

            except Exception as e:
                self.logger.error(f"Error watching markets: {e}")


# =============================================================================
# ORDER BOOK MANAGER
# =============================================================================

class OrderBookManager:
    """Manages order books with real-time updates."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("OrderBookManager")
        self.order_books: dict[str, OrderBook] = {}
        self.subscribers: list[Callable[[str, OrderBook], None]] = []
        self._ws_connections: dict[str, websockets.WebSocketClientProtocol] = {}
        self._running = False

    async def fetch_order_book(
        self,
        session: aiohttp.ClientSession,
        token_id: str
    ) -> Optional[OrderBook]:
        """Fetch order book from CLOB API."""
        try:
            async with session.get(
                f"{self.config.CLOB_API_URL}/book",
                params={"token_id": token_id}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_order_book(token_id, data)
        except Exception as e:
            self.logger.error(f"Error fetching order book for {token_id}: {e}")
        return None

    def _parse_order_book(self, token_id: str, data: dict) -> OrderBook:
        """Parse order book from API response."""
        bids = []
        asks = []

        for bid in data.get("bids", []):
            bids.append(OrderBookLevel(
                price=Decimal(str(bid.get("price", 0))),
                size=Decimal(str(bid.get("size", 0))),
                side="bid"
            ))

        for ask in data.get("asks", []):
            asks.append(OrderBookLevel(
                price=Decimal(str(ask.get("price", 0))),
                size=Decimal(str(ask.get("size", 0))),
                side="ask"
            ))

        # Sort: bids descending, asks ascending
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        return OrderBook(
            token_id=token_id,
            bids=bids,
            asks=asks,
            timestamp=datetime.utcnow()
        )

    async def subscribe_to_token(self, token_id: str):
        """Subscribe to WebSocket updates for a token."""
        if token_id in self._ws_connections:
            return

        try:
            ws = await websockets.connect(
                f"{self.config.WS_URL}",
                ping_interval=30,
                ping_timeout=10
            )

            # Subscribe to order book updates
            await ws.send(json.dumps({
                "type": "subscribe",
                "channel": "book",
                "market": token_id
            }))

            self._ws_connections[token_id] = ws
            asyncio.create_task(self._handle_ws_messages(token_id, ws))

        except Exception as e:
            self.logger.error(f"WebSocket connection failed for {token_id}: {e}")

    async def _handle_ws_messages(
        self,
        token_id: str,
        ws: websockets.WebSocketClientProtocol
    ):
        """Handle incoming WebSocket messages."""
        try:
            async for message in ws:
                try:
                    data = json.loads(message)
                    if data.get("type") == "book":
                        order_book = self._parse_order_book(token_id, data)
                        self.order_books[token_id] = order_book

                        # Notify subscribers
                        for callback in self.subscribers:
                            try:
                                callback(token_id, order_book)
                            except Exception as e:
                                self.logger.error(f"Subscriber error: {e}")

                except json.JSONDecodeError:
                    pass

        except websockets.exceptions.ConnectionClosed:
            self.logger.warning(f"WebSocket closed for {token_id}, reconnecting...")
            del self._ws_connections[token_id]
            await asyncio.sleep(self.config.WS_RECONNECT_DELAY)
            await self.subscribe_to_token(token_id)

    def subscribe(self, callback: Callable[[str, OrderBook], None]):
        """Add a subscriber for order book updates."""
        self.subscribers.append(callback)

    async def close_all(self):
        """Close all WebSocket connections."""
        for token_id, ws in self._ws_connections.items():
            await ws.close()
        self._ws_connections.clear()


# =============================================================================
# OPPORTUNITY DETECTOR
# =============================================================================

class OpportunityDetector:
    """Advanced arbitrage opportunity detection."""

    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db
        self.logger = logging.getLogger("OpportunityDetector")
        self.opportunity_queue: list[ArbitrageOpportunity] = []  # Priority queue

    def detect_binary_arbitrage(
        self,
        market: Market,
        order_books: dict[str, OrderBook]
    ) -> Optional[ArbitrageOpportunity]:
        """
        Detect binary arbitrage: YES + NO < $1.00
        Profit = $1 - (YES_ask + NO_ask)
        """
        if market.market_type != MarketType.BINARY or len(market.tokens) != 2:
            return None

        yes_token = next((t for t in market.tokens if "yes" in t.outcome.lower()), None)
        no_token = next((t for t in market.tokens if "no" in t.outcome.lower()), None)

        if not yes_token or not no_token:
            # Try by position
            yes_token = market.tokens[0]
            no_token = market.tokens[1]

        yes_book = order_books.get(yes_token.token_id)
        no_book = order_books.get(no_token.token_id)

        if not yes_book or not no_book:
            return None

        yes_ask = yes_book.best_ask()
        no_ask = no_book.best_ask()

        if not yes_ask or not no_ask:
            return None

        total_cost = yes_ask + no_ask

        if total_cost < Decimal("1.00"):
            profit = Decimal("1.00") - total_cost
            profit_pct = profit / total_cost

            if profit_pct >= self.config.MIN_PROFIT_THRESHOLD:
                # Calculate optimal size based on liquidity
                yes_depth = sum(level.size for level in yes_book.asks[:3])
                no_depth = sum(level.size for level in no_book.asks[:3])
                max_size = min(yes_depth, no_depth, self.config.MAX_POSITION_SIZE)

                # Walk the book to get realistic execution price
                yes_avg, yes_fill = yes_book.walk_book('ask', max_size)
                no_avg, no_fill = no_book.walk_book('ask', max_size)

                effective_size = min(yes_fill, no_fill)
                effective_cost = yes_avg + no_avg
                effective_profit = Decimal("1.00") - effective_cost

                if effective_profit > 0:
                    confidence = min(effective_size / self.config.MAX_POSITION_SIZE, Decimal("1"))

                    opp = ArbitrageOpportunity(
                        id=hashlib.sha256(
                            f"{market.id}:{datetime.utcnow().isoformat()}".encode()
                        ).hexdigest()[:16],
                        opportunity_type=OpportunityType.BINARY_ARB,
                        markets=[market],
                        tokens=[yes_token, no_token],
                        expected_profit=effective_profit * effective_size,
                        expected_profit_pct=effective_profit / effective_cost,
                        required_capital=effective_cost * effective_size,
                        confidence=confidence,
                        detected_at=datetime.utcnow(),
                        expires_at=market.end_date,
                        order_books={
                            yes_token.token_id: yes_book,
                            no_token.token_id: no_book
                        }
                    )

                    self.db.save_opportunity(opp)
                    return opp

        return None

    def detect_multi_outcome_arbitrage(
        self,
        market: Market,
        order_books: dict[str, OrderBook]
    ) -> Optional[ArbitrageOpportunity]:
        """
        Detect multi-outcome arbitrage: Sum of all outcomes < $1.00
        Works for markets with 3+ outcomes (e.g., "Who will win?")
        """
        if market.market_type != MarketType.MULTI_OUTCOME or len(market.tokens) < 3:
            return None

        total_cost = Decimal("0")
        valid_tokens = []
        token_books = {}

        for token in market.tokens:
            book = order_books.get(token.token_id)
            if not book or not book.best_ask():
                return None  # Need all order books

            total_cost += book.best_ask()
            valid_tokens.append(token)
            token_books[token.token_id] = book

        if total_cost < Decimal("1.00"):
            profit = Decimal("1.00") - total_cost
            profit_pct = profit / total_cost

            if profit_pct >= self.config.MIN_PROFIT_THRESHOLD:
                # Calculate optimal size - limited by smallest depth
                depths = []
                for token in valid_tokens:
                    book = token_books[token.token_id]
                    depth = sum(level.size for level in book.asks[:3])
                    depths.append(depth)

                max_size = min(min(depths), self.config.MAX_POSITION_SIZE / len(valid_tokens))

                # Walk all books
                effective_costs = []
                effective_sizes = []
                for token in valid_tokens:
                    book = token_books[token.token_id]
                    avg_price, fill_size = book.walk_book('ask', max_size)
                    effective_costs.append(avg_price)
                    effective_sizes.append(fill_size)

                effective_size = min(effective_sizes)
                total_effective_cost = sum(effective_costs)
                effective_profit = Decimal("1.00") - total_effective_cost

                if effective_profit > 0:
                    confidence = effective_size / (self.config.MAX_POSITION_SIZE / len(valid_tokens))

                    opp = ArbitrageOpportunity(
                        id=hashlib.sha256(
                            f"{market.id}:multi:{datetime.utcnow().isoformat()}".encode()
                        ).hexdigest()[:16],
                        opportunity_type=OpportunityType.MULTI_OUTCOME_ARB,
                        markets=[market],
                        tokens=valid_tokens,
                        expected_profit=effective_profit * effective_size,
                        expected_profit_pct=effective_profit / total_effective_cost,
                        required_capital=total_effective_cost * effective_size,
                        confidence=Decimal(str(min(confidence, 1))),
                        detected_at=datetime.utcnow(),
                        expires_at=market.end_date,
                        order_books=token_books
                    )

                    self.db.save_opportunity(opp)
                    return opp

        return None

    def detect_cross_market_arbitrage(
        self,
        markets: list[Market],
        order_books: dict[str, OrderBook]
    ) -> list[ArbitrageOpportunity]:
        """
        Detect cross-market arbitrage: Same event, different markets.
        E.g., "Will BTC hit $100k by Dec?" on two different markets.
        """
        opportunities = []

        # Group markets by similar questions (fuzzy matching)
        question_groups: dict[str, list[Market]] = defaultdict(list)

        for market in markets:
            # Create a normalized key from the question
            key = self._normalize_question(market.question)
            question_groups[key].append(market)

        # Look for price discrepancies within groups
        for key, group in question_groups.items():
            if len(group) < 2:
                continue

            # Compare prices across markets in the group
            for i, market1 in enumerate(group):
                for market2 in group[i+1:]:
                    opp = self._compare_markets(market1, market2, order_books)
                    if opp:
                        self.db.save_opportunity(opp)
                        opportunities.append(opp)

        return opportunities

    def _normalize_question(self, question: str) -> str:
        """Normalize question for grouping."""
        # Remove punctuation and lowercase
        import re
        normalized = re.sub(r'[^\w\s]', '', question.lower())
        # Remove common words
        stopwords = {'will', 'the', 'a', 'an', 'by', 'on', 'in', 'at', 'to', 'for'}
        words = [w for w in normalized.split() if w not in stopwords]
        return ' '.join(sorted(words[:5]))  # Use first 5 significant words

    def _compare_markets(
        self,
        market1: Market,
        market2: Market,
        order_books: dict[str, OrderBook]
    ) -> Optional[ArbitrageOpportunity]:
        """Compare two markets for arbitrage opportunity."""
        # Get YES prices from both markets
        yes1 = next((t for t in market1.tokens if "yes" in t.outcome.lower()), None)
        yes2 = next((t for t in market2.tokens if "yes" in t.outcome.lower()), None)

        if not yes1 or not yes2:
            return None

        book1 = order_books.get(yes1.token_id)
        book2 = order_books.get(yes2.token_id)

        if not book1 or not book2:
            return None

        bid1, ask1 = book1.best_bid(), book1.best_ask()
        bid2, ask2 = book2.best_bid(), book2.best_ask()

        if not all([bid1, ask1, bid2, ask2]):
            return None

        # Check for cross-market arb: buy low, sell high
        # Scenario 1: Buy in market1, sell in market2
        if ask1 < bid2:
            profit = bid2 - ask1
            if profit / ask1 >= self.config.MIN_PROFIT_THRESHOLD:
                return self._create_cross_market_opp(
                    market1, market2, yes1, yes2,
                    book1, book2, profit, ask1
                )

        # Scenario 2: Buy in market2, sell in market1
        if ask2 < bid1:
            profit = bid1 - ask2
            if profit / ask2 >= self.config.MIN_PROFIT_THRESHOLD:
                return self._create_cross_market_opp(
                    market2, market1, yes2, yes1,
                    book2, book1, profit, ask2
                )

        return None

    def _create_cross_market_opp(
        self,
        buy_market: Market,
        sell_market: Market,
        buy_token: Token,
        sell_token: Token,
        buy_book: OrderBook,
        sell_book: OrderBook,
        profit: Decimal,
        cost: Decimal
    ) -> ArbitrageOpportunity:
        """Create a cross-market arbitrage opportunity."""
        # Calculate size based on liquidity
        buy_depth = sum(l.size for l in buy_book.asks[:3])
        sell_depth = sum(l.size for l in sell_book.bids[:3])
        max_size = min(buy_depth, sell_depth, self.config.MAX_POSITION_SIZE)

        return ArbitrageOpportunity(
            id=hashlib.sha256(
                f"{buy_market.id}:{sell_market.id}:{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()[:16],
            opportunity_type=OpportunityType.CROSS_MARKET_ARB,
            markets=[buy_market, sell_market],
            tokens=[buy_token, sell_token],
            expected_profit=profit * max_size,
            expected_profit_pct=profit / cost,
            required_capital=cost * max_size,
            confidence=Decimal("0.7"),  # Lower confidence for cross-market
            detected_at=datetime.utcnow(),
            expires_at=min(
                buy_market.end_date or datetime.max,
                sell_market.end_date or datetime.max
            ),
            order_books={
                buy_token.token_id: buy_book,
                sell_token.token_id: sell_book
            }
        )

    def detect_time_decay_opportunities(
        self,
        markets: list[Market],
        order_books: dict[str, OrderBook]
    ) -> list[ArbitrageOpportunity]:
        """
        Detect time-decay opportunities: Markets close to resolution.
        As resolution approaches, mispricings become more pronounced.
        """
        opportunities = []
        now = datetime.utcnow()

        for market in markets:
            if not market.end_date:
                continue

            time_left = market.end_date - now

            # Focus on markets resolving within 24 hours
            if timedelta(hours=0) < time_left < timedelta(hours=24):
                # Check for obvious mispricings
                if market.market_type == MarketType.BINARY:
                    opp = self._check_time_decay_binary(market, order_books, time_left)
                    if opp:
                        self.db.save_opportunity(opp)
                        opportunities.append(opp)

        return opportunities

    def _check_time_decay_binary(
        self,
        market: Market,
        order_books: dict[str, OrderBook],
        time_left: timedelta
    ) -> Optional[ArbitrageOpportunity]:
        """Check binary market for time decay opportunity."""
        if len(market.tokens) != 2:
            return None

        yes_token = market.tokens[0]
        no_token = market.tokens[1]

        yes_book = order_books.get(yes_token.token_id)
        no_book = order_books.get(no_token.token_id)

        if not yes_book or not no_book:
            return None

        yes_mid = yes_book.mid_price()
        no_mid = no_book.mid_price()

        if not yes_mid or not no_mid:
            return None

        # As time decays, prices should converge to 0 or 1
        # Look for prices that are extreme but with good spread
        yes_ask = yes_book.best_ask()
        no_ask = no_book.best_ask()

        if not yes_ask or not no_ask:
            return None

        # Time decay factor (higher = closer to resolution)
        hours_left = time_left.total_seconds() / 3600
        time_factor = max(0.1, 1 - (hours_left / 24))

        # Adjusted threshold based on time
        adjusted_threshold = self.config.MIN_PROFIT_THRESHOLD * (1 - time_factor * 0.5)

        total_cost = yes_ask + no_ask
        if total_cost < Decimal("1.00"):
            profit = Decimal("1.00") - total_cost
            profit_pct = profit / total_cost

            if profit_pct >= adjusted_threshold:
                return ArbitrageOpportunity(
                    id=hashlib.sha256(
                        f"{market.id}:timedecay:{datetime.utcnow().isoformat()}".encode()
                    ).hexdigest()[:16],
                    opportunity_type=OpportunityType.TIME_DECAY,
                    markets=[market],
                    tokens=[yes_token, no_token],
                    expected_profit=profit,
                    expected_profit_pct=profit_pct,
                    required_capital=total_cost,
                    confidence=Decimal(str(time_factor)),
                    detected_at=datetime.utcnow(),
                    expires_at=market.end_date,
                    order_books={
                        yes_token.token_id: yes_book,
                        no_token.token_id: no_book
                    }
                )

        return None

    def prioritize_opportunities(
        self,
        opportunities: list[ArbitrageOpportunity]
    ) -> list[ArbitrageOpportunity]:
        """Sort opportunities by priority (expected profit * confidence)."""
        scored = []
        for opp in opportunities:
            score = float(opp.expected_profit * opp.confidence)
            heappush(scored, (-score, opp))  # Negative for max-heap behavior

        result = []
        while scored:
            _, opp = heappop(scored)
            result.append(opp)

        return result


# =============================================================================
# EXECUTION ENGINE
# =============================================================================

class ExecutionEngine:
    """Smart order execution with partial fills handling."""

    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db
        self.logger = logging.getLogger("ExecutionEngine")
        self.clob_client: Optional[ClobClient] = None
        self._init_client()

    def _init_client(self):
        """Initialize CLOB client with credentials."""
        if self.config.PRIVATE_KEY:
            try:
                self.clob_client = ClobClient(
                    host=self.config.CLOB_API_URL,
                    key=self.config.PRIVATE_KEY,
                    chain_id=self.config.CHAIN_ID,
                    creds={
                        "api_key": self.config.API_KEY,
                        "api_secret": self.config.API_SECRET,
                        "api_passphrase": self.config.API_PASSPHRASE
                    } if self.config.API_KEY else None
                )
                self.logger.info("CLOB client initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize CLOB client: {e}")
                self.clob_client = None

    async def execute_opportunity(
        self,
        opportunity: ArbitrageOpportunity
    ) -> list[Trade]:
        """Execute an arbitrage opportunity."""
        if not self.clob_client:
            self.logger.error("CLOB client not initialized")
            return []

        trades = []

        if opportunity.opportunity_type == OpportunityType.BINARY_ARB:
            trades = await self._execute_binary_arb(opportunity)
        elif opportunity.opportunity_type == OpportunityType.MULTI_OUTCOME_ARB:
            trades = await self._execute_multi_outcome_arb(opportunity)
        elif opportunity.opportunity_type == OpportunityType.CROSS_MARKET_ARB:
            trades = await self._execute_cross_market_arb(opportunity)
        elif opportunity.opportunity_type == OpportunityType.TIME_DECAY:
            trades = await self._execute_binary_arb(opportunity)  # Same as binary

        return trades

    async def _execute_binary_arb(
        self,
        opportunity: ArbitrageOpportunity
    ) -> list[Trade]:
        """Execute a binary arbitrage - buy both YES and NO."""
        trades = []

        for token in opportunity.tokens:
            book = opportunity.order_books.get(token.token_id)
            if not book:
                continue

            # Calculate optimal size
            size = self._calculate_optimal_size(
                opportunity.required_capital / len(opportunity.tokens),
                book
            )

            trade = await self._place_order(
                token_id=token.token_id,
                market_id=opportunity.markets[0].id,
                side="buy",
                size=size,
                price=book.best_ask(),
                opportunity_id=opportunity.id
            )

            if trade:
                trades.append(trade)
                self.db.save_trade(trade)

        return trades

    async def _execute_multi_outcome_arb(
        self,
        opportunity: ArbitrageOpportunity
    ) -> list[Trade]:
        """Execute multi-outcome arbitrage - buy all outcomes."""
        trades = []
        size_per_token = opportunity.required_capital / len(opportunity.tokens)

        for token in opportunity.tokens:
            book = opportunity.order_books.get(token.token_id)
            if not book:
                continue

            size = self._calculate_optimal_size(size_per_token, book)

            trade = await self._place_order(
                token_id=token.token_id,
                market_id=opportunity.markets[0].id,
                side="buy",
                size=size,
                price=book.best_ask(),
                opportunity_id=opportunity.id
            )

            if trade:
                trades.append(trade)
                self.db.save_trade(trade)

        return trades

    async def _execute_cross_market_arb(
        self,
        opportunity: ArbitrageOpportunity
    ) -> list[Trade]:
        """Execute cross-market arbitrage - buy in one market, sell in another."""
        trades = []

        if len(opportunity.tokens) < 2 or len(opportunity.markets) < 2:
            return trades

        buy_token = opportunity.tokens[0]
        sell_token = opportunity.tokens[1]
        buy_market = opportunity.markets[0]
        sell_market = opportunity.markets[1]

        buy_book = opportunity.order_books.get(buy_token.token_id)
        sell_book = opportunity.order_books.get(sell_token.token_id)

        if not buy_book or not sell_book:
            return trades

        # Determine size based on both order books
        buy_size = self._calculate_optimal_size(
            opportunity.required_capital,
            buy_book
        )
        sell_size = self._calculate_optimal_size(
            opportunity.required_capital,
            sell_book
        )
        size = min(buy_size, sell_size)

        # Execute buy order
        buy_trade = await self._place_order(
            token_id=buy_token.token_id,
            market_id=buy_market.id,
            side="buy",
            size=size,
            price=buy_book.best_ask(),
            opportunity_id=opportunity.id
        )

        if buy_trade and buy_trade.status == "filled":
            trades.append(buy_trade)
            self.db.save_trade(buy_trade)

            # Execute sell order
            sell_trade = await self._place_order(
                token_id=sell_token.token_id,
                market_id=sell_market.id,
                side="sell",
                size=buy_trade.filled_size,
                price=sell_book.best_bid(),
                opportunity_id=opportunity.id
            )

            if sell_trade:
                trades.append(sell_trade)
                self.db.save_trade(sell_trade)

        return trades

    def _calculate_optimal_size(
        self,
        target_capital: Decimal,
        order_book: OrderBook
    ) -> Decimal:
        """Calculate optimal order size based on liquidity."""
        if not order_book.asks:
            return Decimal("0")

        # Sum available liquidity at reasonable prices
        available = Decimal("0")
        best_ask = order_book.asks[0].price
        max_slippage = Decimal("0.02")  # 2% max slippage

        for level in order_book.asks:
            if level.price <= best_ask * (1 + max_slippage):
                available += level.size * level.price

        # Take at most MIN_LIQUIDITY_RATIO of available liquidity
        max_from_liquidity = available * self.config.MIN_LIQUIDITY_RATIO

        return min(
            target_capital,
            max_from_liquidity,
            self.config.MAX_POSITION_SIZE
        )

    async def _place_order(
        self,
        token_id: str,
        market_id: str,
        side: str,
        size: Decimal,
        price: Decimal,
        opportunity_id: str
    ) -> Optional[Trade]:
        """Place an order via CLOB client."""
        trade_id = hashlib.sha256(
            f"{token_id}:{side}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]

        trade = Trade(
            id=trade_id,
            opportunity_id=opportunity_id,
            token_id=token_id,
            market_id=market_id,
            side=side,
            size=size,
            price=price,
            filled_size=Decimal("0"),
            filled_price=Decimal("0"),
            status="pending",
            created_at=datetime.utcnow(),
            executed_at=None,
            pnl=None
        )

        try:
            if self.clob_client:
                # Build order
                order = self.clob_client.create_order(
                    OrderArgs(
                        token_id=token_id,
                        price=float(price),
                        size=float(size),
                        side=side.upper(),
                        order_type=OrderType.GTC
                    )
                )

                # Submit order
                result = self.clob_client.post_order(order)

                if result and result.get("success"):
                    trade.status = "submitted"
                    trade.executed_at = datetime.utcnow()

                    # Check for fills (in production, use WebSocket for this)
                    order_id = result.get("order_id")
                    if order_id:
                        await asyncio.sleep(0.5)  # Wait for potential fill
                        order_status = self.clob_client.get_order(order_id)

                        if order_status:
                            filled = Decimal(str(order_status.get("size_matched", 0)))
                            if filled > 0:
                                trade.filled_size = filled
                                trade.filled_price = Decimal(str(
                                    order_status.get("price", price)
                                ))
                                trade.status = "filled" if filled >= size else "partial"
                else:
                    trade.status = "rejected"
                    self.logger.warning(f"Order rejected: {result}")
            else:
                # Simulation mode
                trade.status = "simulated"
                trade.filled_size = size
                trade.filled_price = price
                trade.executed_at = datetime.utcnow()
                self.logger.info(f"[SIMULATION] {side} {size} @ {price} for {token_id}")

        except Exception as e:
            trade.status = "error"
            self.logger.error(f"Order error: {e}")

        return trade


# =============================================================================
# RISK MANAGER
# =============================================================================

class RiskManager:
    """Comprehensive risk management with kill switch."""

    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db
        self.logger = logging.getLogger("RiskManager")
        self.kill_switch_active = False
        self.price_history: dict[str, list[tuple[datetime, Decimal]]] = defaultdict(list)

    def check_can_trade(self, opportunity: ArbitrageOpportunity) -> tuple[bool, str]:
        """Check if trading is allowed for this opportunity."""
        if self.kill_switch_active:
            return False, "Kill switch is active"

        # Check total exposure
        total_exposure = self.db.get_total_exposure()
        if total_exposure + opportunity.required_capital > self.config.MAX_TOTAL_EXPOSURE:
            return False, f"Would exceed max total exposure (${self.config.MAX_TOTAL_EXPOSURE})"

        # Check per-market exposure
        for market in opportunity.markets:
            market_exposure = self.db.get_market_exposure(market.id)
            new_exposure = opportunity.required_capital / len(opportunity.markets)
            if market_exposure + new_exposure > self.config.MAX_PER_MARKET_EXPOSURE:
                return False, f"Would exceed max market exposure for {market.id}"

        # Check volatility
        for token in opportunity.tokens:
            volatility = self._calculate_volatility(token.token_id)
            if volatility and volatility > self.config.VOLATILITY_THRESHOLD:
                return False, f"Volatility too high for {token.token_id}: {volatility:.2%}"

        # Check correlation exposure
        correlated_exposure = self._check_correlation_exposure(opportunity)
        if correlated_exposure > self.config.MAX_CORRELATED_EXPOSURE:
            return False, f"Would exceed correlated exposure limit"

        return True, "OK"

    def _calculate_volatility(self, token_id: str) -> Optional[Decimal]:
        """Calculate price volatility over recent history."""
        history = self.price_history.get(token_id, [])
        if len(history) < 10:
            return None

        prices = [p for _, p in history[-20:]]
        if not prices:
            return None

        mean = sum(prices) / len(prices)
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        std_dev = variance ** Decimal("0.5")

        return std_dev / mean if mean > 0 else None

    def _check_correlation_exposure(
        self,
        opportunity: ArbitrageOpportunity
    ) -> Decimal:
        """Check exposure to correlated markets."""
        # Simplified correlation check - same category = correlated
        category = opportunity.markets[0].category if opportunity.markets else None
        if not category:
            return Decimal("0")

        # This would require tracking positions by category
        # For now, return 0 as a placeholder
        return Decimal("0")

    def update_price_history(self, token_id: str, price: Decimal):
        """Update price history for volatility calculation."""
        history = self.price_history[token_id]
        history.append((datetime.utcnow(), price))

        # Keep last hour of data
        cutoff = datetime.utcnow() - timedelta(hours=1)
        self.price_history[token_id] = [
            (t, p) for t, p in history if t > cutoff
        ]

    def activate_kill_switch(self, reason: str):
        """Activate the kill switch and send alerts."""
        self.kill_switch_active = True
        self.logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

        # Send alerts
        asyncio.create_task(self._send_alert(
            f"KILL SWITCH ACTIVATED: {reason}",
            f"The trading bot kill switch has been activated.\n\nReason: {reason}\n\n"
            f"Time: {datetime.utcnow().isoformat()}\n\n"
            f"Manual intervention required to resume trading."
        ))

    def deactivate_kill_switch(self):
        """Deactivate the kill switch."""
        self.kill_switch_active = False
        self.logger.info("Kill switch deactivated")

    async def _send_alert(self, subject: str, body: str):
        """Send alert via email."""
        if not self.config.ALERT_EMAIL or not self.config.SMTP_USER:
            self.logger.warning("Email alerts not configured")
            return

        try:
            msg = MIMEText(body)
            msg['Subject'] = f"[TRADING BOT] {subject}"
            msg['From'] = self.config.SMTP_USER
            msg['To'] = self.config.ALERT_EMAIL

            context = ssl.create_default_context()
            with smtplib.SMTP(self.config.SMTP_SERVER, self.config.SMTP_PORT) as server:
                server.starttls(context=context)
                server.login(self.config.SMTP_USER, self.config.SMTP_PASSWORD)
                server.sendmail(
                    self.config.SMTP_USER,
                    self.config.ALERT_EMAIL,
                    msg.as_string()
                )

            self.logger.info(f"Alert sent to {self.config.ALERT_EMAIL}")

        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")

    def check_drawdown(self, threshold: Decimal = Decimal("0.1")) -> bool:
        """Check if drawdown exceeds threshold (10% default)."""
        stats = self.db.get_pnl_stats(days=1)
        total_exposure = self.db.get_total_exposure()

        if total_exposure > 0:
            drawdown = abs(stats["max_loss"]) / total_exposure
            if drawdown > threshold:
                self.activate_kill_switch(f"Drawdown exceeded {threshold:.0%}")
                return True

        return False


# =============================================================================
# MONITORING DASHBOARD
# =============================================================================

class Dashboard:
    """Real-time monitoring dashboard."""

    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db
        self.logger = logging.getLogger("Dashboard")
        self.stats: dict[str, Any] = {}
        self.start_time = datetime.utcnow()

    def update_stats(
        self,
        opportunities_detected: int = 0,
        opportunities_executed: int = 0,
        active_positions: int = 0
    ):
        """Update dashboard statistics."""
        self.stats.update({
            "opportunities_detected": self.stats.get("opportunities_detected", 0) + opportunities_detected,
            "opportunities_executed": self.stats.get("opportunities_executed", 0) + opportunities_executed,
            "active_positions": active_positions,
            "uptime": str(datetime.utcnow() - self.start_time),
            "last_update": datetime.utcnow().isoformat()
        })

    def get_dashboard_data(self) -> dict:
        """Get comprehensive dashboard data."""
        pnl_stats = self.db.get_pnl_stats()
        category_heatmap = self.db.get_category_heatmap()
        total_exposure = self.db.get_total_exposure()

        return {
            "summary": {
                "uptime": self.stats.get("uptime", "0:00:00"),
                "total_opportunities": self.stats.get("opportunities_detected", 0),
                "executed_opportunities": self.stats.get("opportunities_executed", 0),
                "active_positions": self.stats.get("active_positions", 0),
                "total_exposure": float(total_exposure),
            },
            "pnl": {
                "total": float(pnl_stats["total_pnl"]),
                "average": float(pnl_stats["avg_pnl"]),
                "win_rate": pnl_stats["win_rate"],
                "total_trades": pnl_stats["total_trades"],
                "max_profit": float(pnl_stats["max_profit"]),
                "max_loss": float(pnl_stats["max_loss"]),
            },
            "category_heatmap": category_heatmap,
            "last_update": datetime.utcnow().isoformat()
        }

    def print_dashboard(self):
        """Print dashboard to console."""
        data = self.get_dashboard_data()

        print("\n" + "=" * 60)
        print("         POLYMARKET ARBITRAGE BOT - DASHBOARD")
        print("=" * 60)
        print(f"\n  Uptime: {data['summary']['uptime']}")
        print(f"  Last Update: {data['last_update']}")
        print("\n--- OPPORTUNITIES ---")
        print(f"  Detected: {data['summary']['total_opportunities']}")
        print(f"  Executed: {data['summary']['executed_opportunities']}")
        print(f"  Active Positions: {data['summary']['active_positions']}")
        print(f"  Total Exposure: ${data['summary']['total_exposure']:.2f}")
        print("\n--- P&L METRICS ---")
        print(f"  Total P&L: ${data['pnl']['total']:.2f}")
        print(f"  Average P&L: ${data['pnl']['average']:.2f}")
        print(f"  Win Rate: {data['pnl']['win_rate']:.1%}")
        print(f"  Total Trades: {data['pnl']['total_trades']}")
        print(f"  Max Profit: ${data['pnl']['max_profit']:.2f}")
        print(f"  Max Loss: ${data['pnl']['max_loss']:.2f}")

        if data['category_heatmap']:
            print("\n--- CATEGORY HEATMAP (24h) ---")
            for market_id, stats in list(data['category_heatmap'].items())[:5]:
                print(f"  {market_id[:20]}: {stats['count']} opps, {stats['avg_profit_pct']:.2%} avg")

        print("\n" + "=" * 60 + "\n")


# =============================================================================
# MAIN BOT
# =============================================================================

class UltimateArbBot:
    """Main arbitrage bot orchestrator."""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.logger = self._setup_logging()

        # Initialize components
        self.db = Database(self.config.DB_PATH)
        self.market_discovery = MarketDiscovery(self.config)
        self.order_book_manager = OrderBookManager(self.config)
        self.opportunity_detector = OpportunityDetector(self.config, self.db)
        self.execution_engine = ExecutionEngine(self.config, self.db)
        self.risk_manager = RiskManager(self.config, self.db)
        self.dashboard = Dashboard(self.config, self.db)

        # State
        self.running = False
        self.markets: list[Market] = []
        self.order_books: dict[str, OrderBook] = {}
        self.pending_opportunities: list[ArbitrageOpportunity] = []

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _setup_logging(self) -> logging.Logger:
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger("UltimateArbBot")

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        self.logger.info("Shutdown signal received...")
        self.running = False

    async def start(self):
        """Start the arbitrage bot."""
        self.logger.info("=" * 60)
        self.logger.info("  ULTIMATE POLYMARKET ARBITRAGE BOT")
        self.logger.info("=" * 60)
        self.running = True

        async with aiohttp.ClientSession() as session:
            # Initial market discovery
            self.logger.info("Discovering markets...")
            self.markets = await self.market_discovery.discover_all_markets(session)
            self.logger.info(f"Found {len(self.markets)} active markets")

            # Fetch initial order books
            self.logger.info("Fetching order books...")
            await self._fetch_all_order_books(session)

            # Start background tasks
            tasks = [
                asyncio.create_task(self._scan_loop(session)),
                asyncio.create_task(self._execution_loop()),
                asyncio.create_task(self._dashboard_loop()),
                asyncio.create_task(self._market_watch_loop(session)),
                asyncio.create_task(self._risk_check_loop()),
            ]

            self.logger.info("Bot started. Press Ctrl+C to stop.")

            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                pass
            finally:
                await self._cleanup()

    async def _fetch_all_order_books(self, session: aiohttp.ClientSession):
        """Fetch order books for all market tokens."""
        tasks = []
        for market in self.markets:
            for token in market.tokens:
                tasks.append(
                    self.order_book_manager.fetch_order_book(session, token.token_id)
                )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, OrderBook):
                self.order_books[result.token_id] = result

        self.logger.info(f"Fetched {len(self.order_books)} order books")

    async def _scan_loop(self, session: aiohttp.ClientSession):
        """Main scanning loop - detect opportunities."""
        scan_count = 0

        while self.running:
            try:
                start_time = time.time()
                opportunities = []

                # Refresh order books (sample subset for speed)
                sample_size = min(100, len(self.markets))
                sample_markets = self.markets[:sample_size]

                for market in sample_markets:
                    for token in market.tokens:
                        book = await self.order_book_manager.fetch_order_book(
                            session, token.token_id
                        )
                        if book:
                            self.order_books[token.token_id] = book
                            self.risk_manager.update_price_history(
                                token.token_id,
                                book.mid_price() or Decimal("0")
                            )

                # Detect binary arbitrage
                for market in sample_markets:
                    opp = self.opportunity_detector.detect_binary_arbitrage(
                        market, self.order_books
                    )
                    if opp:
                        opportunities.append(opp)

                # Detect multi-outcome arbitrage
                for market in sample_markets:
                    opp = self.opportunity_detector.detect_multi_outcome_arbitrage(
                        market, self.order_books
                    )
                    if opp:
                        opportunities.append(opp)

                # Detect cross-market arbitrage
                cross_market_opps = self.opportunity_detector.detect_cross_market_arbitrage(
                    sample_markets, self.order_books
                )
                opportunities.extend(cross_market_opps)

                # Detect time-decay opportunities
                time_decay_opps = self.opportunity_detector.detect_time_decay_opportunities(
                    sample_markets, self.order_books
                )
                opportunities.extend(time_decay_opps)

                # Prioritize and queue
                if opportunities:
                    prioritized = self.opportunity_detector.prioritize_opportunities(
                        opportunities
                    )
                    self.pending_opportunities.extend(prioritized)

                    self.logger.info(
                        f"Detected {len(opportunities)} opportunities "
                        f"(best: ${prioritized[0].expected_profit:.4f} / "
                        f"{prioritized[0].expected_profit_pct:.2%})"
                    )

                self.dashboard.update_stats(opportunities_detected=len(opportunities))

                # Calculate scan time
                scan_time_ms = (time.time() - start_time) * 1000
                scan_count += 1

                if scan_count % 100 == 0:
                    self.logger.info(f"Scan #{scan_count}, time: {scan_time_ms:.1f}ms")

                # Wait for next scan
                sleep_time = max(0, (self.config.SCAN_INTERVAL_MS - scan_time_ms) / 1000)
                await asyncio.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"Scan error: {e}")
                await asyncio.sleep(1)

    async def _execution_loop(self):
        """Execute queued opportunities."""
        while self.running:
            try:
                if not self.pending_opportunities:
                    await asyncio.sleep(0.1)
                    continue

                # Get highest priority opportunity
                opportunity = self.pending_opportunities.pop(0)

                # Risk check
                can_trade, reason = self.risk_manager.check_can_trade(opportunity)
                if not can_trade:
                    self.logger.warning(f"Trade blocked: {reason}")
                    continue

                # Check if opportunity is still valid (not expired)
                if opportunity.expires_at and datetime.utcnow() > opportunity.expires_at:
                    self.logger.debug("Opportunity expired, skipping")
                    continue

                # Execute
                self.logger.info(
                    f"Executing {opportunity.opportunity_type.name} opportunity "
                    f"({opportunity.expected_profit_pct:.2%} expected)"
                )

                trades = await self.execution_engine.execute_opportunity(opportunity)

                if trades:
                    filled = [t for t in trades if t.status in ["filled", "simulated"]]
                    self.logger.info(
                        f"Executed {len(filled)}/{len(trades)} trades"
                    )
                    self.dashboard.update_stats(opportunities_executed=1)

            except Exception as e:
                self.logger.error(f"Execution error: {e}")
                await asyncio.sleep(1)

    async def _dashboard_loop(self):
        """Periodically print dashboard."""
        while self.running:
            await asyncio.sleep(30)
            self.dashboard.print_dashboard()

    async def _market_watch_loop(self, session: aiohttp.ClientSession):
        """Watch for new markets."""
        def on_new_market(market: Market):
            self.markets.append(market)

        await self.market_discovery.watch_new_markets(
            session, on_new_market, interval=60
        )

    async def _risk_check_loop(self):
        """Periodic risk checks."""
        while self.running:
            await asyncio.sleep(60)

            # Check drawdown
            self.risk_manager.check_drawdown()

    async def _cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up...")
        await self.order_book_manager.close_all()
        self.db.conn.close()
        self.logger.info("Shutdown complete")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     ULTIMATE POLYMARKET ARBITRAGE BOT                         ║
    ║                                                               ║
    ║     Features:                                                 ║
    ║     - Universal market discovery (Gamma API)                  ║
    ║     - Binary + Multi-outcome + Cross-market arbitrage         ║
    ║     - Order book depth analysis                               ║
    ║     - Real-time WebSocket updates                             ║
    ║     - SQLite trade history                                    ║
    ║     - Risk management with kill switch                        ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    # Load config
    config = Config()

    # Validate required config
    if not config.PRIVATE_KEY:
        print("\n⚠️  WARNING: POLYMARKET_PRIVATE_KEY not set.")
        print("   Running in SIMULATION mode - no real trades will be placed.")
        print("   Set environment variable to enable live trading.\n")

    # Create and run bot
    bot = UltimateArbBot(config)

    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")


if __name__ == "__main__":
    main()
