#\!/usr/bin/env python3.11
"""
ULTIMATE POLYMARKET ARBITRAGE BOT
==================================

The most comprehensive, production-ready arbitrage bot for Polymarket.
Implements universal market discovery, advanced opportunity detection,
WebSocket real-time updates, and comprehensive risk management.

Author: Trading Bots Swarm
Date: 2026-01-04
Version: 2.0.0 (Ultimate Edition)
"""

import asyncio
import aiohttp
import json
import time
import logging
import argparse
import sqlite3
import threading
import signal
import sys
import os
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple, Any, Set, Callable
from dataclasses import dataclass, field, asdict
from decimal import Decimal, ROUND_DOWN
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from queue import PriorityQueue
from enum import Enum, auto
import heapq
import requests
import websocket


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Centralized configuration with environment variable support"""
    
    # API Endpoints
    POLYMARKET_WEB = "https://polymarket.com"
    CLOB_API = "https://clob.polymarket.com"
    GAMMA_API = "https://gamma-api.polymarket.com"
    WEBSOCKET_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    CHAIN_ID = 137
    
    # Market Discovery
    SCAN_ALL_MARKETS = True
    MARKET_CATEGORIES = ["crypto", "politics", "sports", "entertainment", "economics", "science", "other"]
    MIN_MARKET_LIQUIDITY = 1000
    MIN_MARKET_VOLUME_24H = 500
    MAX_MARKETS_TO_SCAN = 500
    MARKET_REFRESH_INTERVAL = 300
    NEW_MARKET_CHECK_INTERVAL = 60
    
    # Opportunity Detection Thresholds
    BINARY_ARB_THRESHOLD = 0.003
    MULTI_OUTCOME_THRESHOLD = 0.005
    CROSS_MARKET_THRESHOLD = 0.008
    TIME_DECAY_THRESHOLD = 0.002
    TIME_DECAY_HOURS = 24
    
    # Order Book Analysis
    SLIPPAGE_BUFFER = 0.005
    MIN_LIQUIDITY_USD = 100
    MAX_BOOK_DEPTH_LEVELS = 10
    IMPACT_THRESHOLD = 0.01
    
    # Execution
    SCAN_INTERVAL_MS = 50
    ORDER_TIMEOUT_MS = 5000
    MAX_PARALLEL_ORDERS = 4
    PARTIAL_FILL_RETRY_DELAY = 100
    MAX_RETRY_ATTEMPTS = 3
    
    # Position Sizing (Kelly Criterion)
    USE_KELLY_SIZING = True
    KELLY_FRACTION = 0.25
    WIN_RATE_ESTIMATE = 0.80
    DEFAULT_ORDER_SIZE = 50
    MIN_ORDER_SIZE = 10
    
    # Risk Management
    MAX_POSITION_SIZE = 200
    MAX_TOTAL_EXPOSURE = 2000
    MAX_EXPOSURE_PER_MARKET = 500
    MAX_EXPOSURE_PER_CATEGORY = 1000
    MAX_CORRELATED_EXPOSURE = 750
    MAX_DAILY_LOSS = 200
    MAX_DRAWDOWN_PERCENT = 10
    MAX_TRADES_PER_HOUR = 50
    COOLDOWN_SECONDS = 2
    VOLATILITY_WINDOW_HOURS = 24
    
    # Kill Switch
    KILL_SWITCH_ENABLED = True
    KILL_SWITCH_LOSS_THRESHOLD = 500
    KILL_SWITCH_DRAWDOWN = 15
    
    # Database
    DB_PATH = "ultimate_arb_trades.db"
    LOG_ALL_OPPORTUNITIES = True
    
    # Dashboard
    DASHBOARD_UPDATE_INTERVAL = 1.0
    
    # Credentials (override from environment)
    PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    FUNDER_ADDRESS = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
    SIGNATURE_TYPE = 1
    
    @classmethod
    def from_args(cls, args):
        """Override config from command-line arguments"""
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

