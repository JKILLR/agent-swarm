#!/usr/bin/env python3
"""
SMART MONEY TRACKER - Polymarket Trading Bot

THIS IS DIFFERENT.

Instead of predicting crypto direction from Binance prices,
we FOLLOW what Polymarket traders are doing.

If UP shares are being bought aggressively (price rising fast),
someone with better info/models is betting UP.

We follow them. We don't predict. We react.

Logic:
1. Track UP and DOWN prices every second for first 3 min of window
2. Calculate price velocity (how fast each side is moving)
3. If one side surging → follow it with heavy bias
4. If both stable → small arb or skip
5. Only trade in first 3-4 minutes (before outcome becomes obvious)

Usage:
    python3.11 smart_money.py --live --yolo
"""

import requests
import re
import json
import time
import argparse
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import deque

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'POLYMARKET_WEB': 'https://polymarket.com',
    'CLOB_API': 'https://clob.polymarket.com',
    'CHAIN_ID': 137,
    
    'ASSETS': ['btc', 'eth', 'sol', 'xrp'],
    'TIMEFRAMES': ['15m'],
    
    # Smart money settings
    'PRICE_TRACK_SECONDS': 30,        # Track last 30 seconds of Polymarket prices
    'VELOCITY_THRESHOLD': 0.02,       # 2% price move = signal
    'STRONG_SIGNAL_THRESHOLD': 0.05,  # 5% move = strong signal
    
    # Only trade in first N minutes of window
    'TRADE_WINDOW_MINUTES': 4,
    
    # Spread still required (we're not idiots)
    'MIN_SPREAD_PERCENT': 0.3,
    
    'SCAN_INTERVAL': 0.5,
    'MARKET_REFRESH': 300,
    'REQUEST_TIMEOUT': 2,
    
    # Position sizing
    'BASE_SHARES': 10,                # Minimum per side
    'SIGNAL_SHARES': 20,              # Total when we have signal
    'STRONG_SIGNAL_SHARES': 30,       # Total on strong signal
    'MAX_BIAS': 0.8,                  # Never go more than 80/20
    
    # =========================================================================
    # CREDENTIALS
    # =========================================================================
    'PRIVATE_KEY': '0xb92c6d5ae586a416cd45ecda3d8d7a1bb253777025fe31f863c8dcd9ea7e5bb0',
    'SIGNATURE_TYPE': 1,
    'FUNDER_ADDRESS': '0x1640782e9E71029B78555b9f23478712aC47396E',
}

TIMEFRAME_SECONDS = {'15m': 900, '1h': 3600, '4h': 14400}

# =============================================================================
# CLOB Client imports
# =============================================================================

CLOB_AVAILABLE = False
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY
    CLOB_AVAILABLE = True
except ImportError:
    print("⚠️  py-clob-client not installed")
    BUY = "BUY"

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Market:
    asset: str
    timeframe: str
    slug: str
    question: str
    up_token: str
    down_token: str
    window_start: int  # Epoch timestamp when this window started

@dataclass
class Signal:
    direction: str          # 'up', 'down', 'none'
    strength: str           # 'weak', 'normal', 'strong'
    up_velocity: float      # % change in UP price
    down_velocity: float    # % change in DOWN price
    confidence: float       # 0-1

@dataclass
class SmartOrder:
    market: Market
    up_shares: int
    down_shares: int
    up_price: float
    down_price: float
    signal: Signal
    time_in_window: int     # Seconds since window opened

# =============================================================================
# POLYMARKET PRICE TRACKER
# =============================================================================

class PolymarketTracker:
    """
    Tracks Polymarket UP/DOWN prices over time.
    This is the secret sauce - we watch what the market is doing.
    """
    
    def __init__(self):
        # {market_key: deque of (timestamp, up_price, down_price)}
        self.price_history: Dict[str, deque] = {}
        self.lock = threading.Lock()
    
    def record_prices(self, market: Market, up_price: float, down_price: float):
        """Record current prices for a market."""
        key = f"{market.asset}-{market.timeframe}"
        
        with self.lock:
            if key not in self.price_history:
                self.price_history[key] = deque(maxlen=100)
            
            self.price_history[key].append((time.time(), up_price, down_price))
    
    def get_signal(self, market: Market) -> Signal:
        """
        Analyze price movement to detect smart money flow.
        
        If UP price rising fast → buyers think it's going up
        If DOWN price rising fast → buyers think it's going down
        """
        key = f"{market.asset}-{market.timeframe}"
        
        with self.lock:
            history = list(self.price_history.get(key, []))
        
        if len(history) < 5:
            return Signal('none', 'weak', 0, 0, 0)
        
        # Get prices from N seconds ago
        now = time.time()
        window = CONFIG['PRICE_TRACK_SECONDS']
        
        # Find oldest price in window
        old_up, old_down = None, None
        for ts, up, down in history:
            if now - ts >= window * 0.7:  # At least 70% of window ago
                old_up, old_down = up, down
                break
        
        if old_up is None:
            old_up = history[0][1]
            old_down = history[0][2]
        
        # Current prices
        current_up = history[-1][1]
        current_down = history[-1][2]
        
        # Calculate velocity (% change)
        up_velocity = (current_up - old_up) / max(old_up, 0.01)
        down_velocity = (current_down - old_down) / max(old_down, 0.01)
        
        # Determine signal
        velocity_diff = up_velocity - down_velocity
        
        threshold = CONFIG['VELOCITY_THRESHOLD']
        strong_threshold = CONFIG['STRONG_SIGNAL_THRESHOLD']
        
        if velocity_diff > strong_threshold:
            return Signal('up', 'strong', up_velocity, down_velocity, 0.9)
        elif velocity_diff > threshold:
            return Signal('up', 'normal', up_velocity, down_velocity, 0.7)
        elif velocity_diff < -strong_threshold:
            return Signal('down', 'strong', up_velocity, down_velocity, 0.9)
        elif velocity_diff < -threshold:
            return Signal('down', 'normal', up_velocity, down_velocity, 0.7)
        else:
            return Signal('none', 'weak', up_velocity, down_velocity, 0.3)
    
    def clear_old_data(self, market: Market):
        """Clear history when new window starts."""
        key = f"{market.asset}-{market.timeframe}"
        with self.lock:
            if key in self.price_history:
                self.price_history[key].clear()

# Global tracker
TRACKER = PolymarketTracker()

# =============================================================================
# MARKET DISCOVERY
# =============================================================================

MARKET_CACHE: Dict[str, Market] = {}

def get_epoch(timeframe: str) -> int:
    seconds = TIMEFRAME_SECONDS.get(timeframe, 900)
    return (int(time.time()) // seconds) * seconds

def get_time_in_window(timeframe: str) -> int:
    """Returns seconds since current window started."""
    seconds = TIMEFRAME_SECONDS.get(timeframe, 900)
    epoch = get_epoch(timeframe)
    return int(time.time()) - epoch

def fetch_market(asset: str, timeframe: str) -> Optional[Market]:
    cache_key = f"{asset}-{timeframe}"
    epoch = get_epoch(timeframe)
    slug = f"{asset}-updown-{timeframe}-{epoch}"
    
    if cache_key in MARKET_CACHE:
        cached = MARKET_CACHE[cache_key]
        if cached.slug == slug:
            return cached
        else:
            # New window - clear price history
            TRACKER.clear_old_data(cached)
    
    for offset_mult in [0, 1, -1]:
        offset = offset_mult * TIMEFRAME_SECONDS.get(timeframe, 900)
        test_epoch = epoch + offset
        slug = f"{asset}-updown-{timeframe}-{test_epoch}"
        url = f"{CONFIG['POLYMARKET_WEB']}/event/{slug}"
        
        try:
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=3)
            if resp.status_code != 200:
                continue
            
            match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', resp.text, re.DOTALL)
            if not match:
                continue
            
            data = json.loads(match.group(1))
            queries = data.get('props', {}).get('pageProps', {}).get('dehydratedState', {}).get('queries', [])
            
            for q in queries:
                state_data = q.get('state', {}).get('data')
                if isinstance(state_data, dict) and 'markets' in state_data:
                    for mkt in state_data['markets']:
                        tokens = mkt.get('clobTokenIds', [])
                        if len(tokens) >= 2:
                            market = Market(
                                asset=asset.upper(),
                                timeframe=timeframe,
                                slug=slug,
                                question=mkt.get('question', ''),
                                up_token=tokens[0],
                                down_token=tokens[1],
                                window_start=test_epoch
                            )
                            MARKET_CACHE[cache_key] = market
                            return market
        except:
            continue
    return None

def discover_markets() -> List[Market]:
    markets = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(fetch_market, a, t) for a in CONFIG['ASSETS'] for t in CONFIG['TIMEFRAMES']]
        for f in as_completed(futures):
            try:
                m = f.result()
                if m:
                    markets.append(m)
            except:
                pass
    return markets

# =============================================================================
# PRICE FETCHING
# =============================================================================

def get_prices(market: Market) -> Tuple[Optional[float], Optional[float]]:
    """Fetch current UP and DOWN prices."""
    results = [None, None]
    
    def fetch_up():
        try:
            r = requests.get(f"{CONFIG['CLOB_API']}/midpoint", 
                           params={'token_id': market.up_token}, 
                           timeout=CONFIG['REQUEST_TIMEOUT'])
            if r.ok:
                results[0] = float(r.json().get('mid', 0))
        except:
            pass
    
    def fetch_down():
        try:
            r = requests.get(f"{CONFIG['CLOB_API']}/midpoint", 
                           params={'token_id': market.down_token}, 
                           timeout=CONFIG['REQUEST_TIMEOUT'])
            if r.ok:
                results[1] = float(r.json().get('mid', 0))
        except:
            pass
    
    t1 = threading.Thread(target=fetch_up)
    t2 = threading.Thread(target=fetch_down)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    # Record for tracking
    if results[0] and results[1]:
        TRACKER.record_prices(market, results[0], results[1])
    
    return results[0], results[1]

# =============================================================================
# ORDER CALCULATION
# =============================================================================

def calculate_smart_order(market: Market, up_price: float, down_price: float) -> Optional[SmartOrder]:
    """
    Calculate order based on smart money signal.
    """
    total = up_price + down_price
    spread_pct = (1 - total) * 100
    
    # Need minimum spread
    if spread_pct < CONFIG['MIN_SPREAD_PERCENT']:
        return None
    
    # Check time in window
    time_in_window = get_time_in_window(market.timeframe)
    max_time = CONFIG['TRADE_WINDOW_MINUTES'] * 60
    
    if time_in_window > max_time:
        return None  # Too late in window
    
    # Get signal
    signal = TRACKER.get_signal(market)
    
    # Calculate position size based on signal strength
    if signal.strength == 'strong':
        total_shares = CONFIG['STRONG_SIGNAL_SHARES']
        bias = CONFIG['MAX_BIAS']
    elif signal.strength == 'normal':
        total_shares = CONFIG['SIGNAL_SHARES']
        bias = 0.65
    else:
        # No signal - small arb position or skip
        total_shares = CONFIG['BASE_SHARES'] * 2
        bias = 0.5
    
    # Apply bias based on direction
    if signal.direction == 'up':
        up_shares = int(round(total_shares * bias))
        down_shares = total_shares - up_shares
    elif signal.direction == 'down':
        down_shares = int(round(total_shares * bias))
        up_shares = total_shares - down_shares
    else:
        up_shares = total_shares // 2
        down_shares = total_shares - up_shares
    
    # Minimum 5 shares per side
    up_shares = max(5, up_shares)
    down_shares = max(5, down_shares)
    
    return SmartOrder(
        market=market,
        up_shares=up_shares,
        down_shares=down_shares,
        up_price=up_price,
        down_price=down_price,
        signal=signal,
        time_in_window=time_in_window
    )

# =============================================================================
# EXECUTOR
# =============================================================================

class SmartExecutor:
    def __init__(self):
        self.client = None
        self.initialized = False
        self.lock = threading.Lock()
        self.recent_trades: Dict[str, float] = {}  # Prevent duplicate trades
    
    def initialize(self) -> bool:
        if not CLOB_AVAILABLE or not CONFIG['PRIVATE_KEY']:
            return False
        
        try:
            self.client = ClobClient(
                CONFIG['CLOB_API'],
                key=CONFIG['PRIVATE_KEY'],
                chain_id=CONFIG['CHAIN_ID'],
                signature_type=CONFIG['SIGNATURE_TYPE'],
                funder=CONFIG['FUNDER_ADDRESS'] if CONFIG['FUNDER_ADDRESS'] else None
            )
            self.client.set_api_creds(self.client.create_or_derive_api_creds())
            self.initialized = True
            return True
        except Exception as e:
            print(f"❌ Init failed: {e}")
            return False
    
    def should_trade(self, market: Market) -> bool:
        """Prevent trading same market twice in same window."""
        key = f"{market.slug}"
        last_trade = self.recent_trades.get(key, 0)
        
        # Don't trade same market within 60 seconds
        if time.time() - last_trade < 60:
            return False
        return True
    
    def record_trade(self, market: Market):
        """Record that we traded this market."""
        self.recent_trades[market.slug] = time.time()
    
    def execute(self, order: SmartOrder) -> Dict:
        """Execute the smart order."""
        result = {
            'success': False,
            'error': None,
            'up_filled': 0,
            'down_filled': 0
        }
        
        if not self.initialized:
            result['error'] = "Not initialized"
            return result
        
        if not self.should_trade(order.market):
            result['error'] = "Already traded this window"
            return result
        
        with self.lock:
            errors = []
            
            def buy_up():
                try:
                    args = OrderArgs(
                        price=order.up_price,
                        size=float(order.up_shares),
                        side=BUY,
                        token_id=order.market.up_token
                    )
                    signed = self.client.create_order(args)
                    self.client.post_order(signed, OrderType.GTC)
                    result['up_filled'] = order.up_shares
                except Exception as e:
                    errors.append(f"UP: {e}")
            
            def buy_down():
                try:
                    args = OrderArgs(
                        price=order.down_price,
                        size=float(order.down_shares),
                        side=BUY,
                        token_id=order.market.down_token
                    )
                    signed = self.client.create_order(args)
                    self.client.post_order(signed, OrderType.GTC)
                    result['down_filled'] = order.down_shares
                except Exception as e:
                    errors.append(f"DOWN: {e}")
            
            # Parallel execution
            t1 = threading.Thread(target=buy_up)
            t2 = threading.Thread(target=buy_down)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            
            if errors:
                result['error'] = "; ".join(errors)
            
            if result['up_filled'] > 0 or result['down_filled'] > 0:
                result['success'] = True
                self.record_trade(order.market)
            
            return result

# =============================================================================
# MAIN SCANNER
# =============================================================================

class SmartMoneyScanner:
    def __init__(self, live: bool = False):
        self.live = live
        self.markets: List[Market] = []
        self.last_refresh = 0
        self.executor = SmartExecutor() if live else None
        
        self.stats = {
            'scans': 0,
            'opportunities': 0,
            'signals': 0,
            'trades': 0,
            'skipped_no_signal': 0,
            'skipped_late': 0,
            'up_signals': 0,
            'down_signals': 0,
        }
    
    def refresh_markets(self):
        print("\n🔍 Refreshing markets...")
        self.markets = discover_markets()
        self.last_refresh = time.time()
        print(f"   Found {len(self.markets)} markets")
        for m in self.markets:
            time_in = get_time_in_window(m.timeframe)
            print(f"   • {m.asset} {m.timeframe} ({time_in}s into window)")
        print()
    
    def run(self):
        print("=" * 70)
        print("🧠 SMART MONEY TRACKER")
        print("=" * 70)
        print("Follows Polymarket price momentum, not Binance.")
        print("If traders are buying UP aggressively → we follow.")
        print("=" * 70)
        print(f"Mode: {'🔴 LIVE TRADING' if self.live else '⚪ SIMULATION'}")
        print(f"Trade window: First {CONFIG['TRADE_WINDOW_MINUTES']} minutes only")
        print(f"Velocity threshold: {CONFIG['VELOCITY_THRESHOLD']*100}% / {CONFIG['STRONG_SIGNAL_THRESHOLD']*100}% (strong)")
        print(f"Position sizing: {CONFIG['BASE_SHARES']*2} base / {CONFIG['SIGNAL_SHARES']} signal / {CONFIG['STRONG_SIGNAL_SHARES']} strong")
        print("=" * 70)
        
        if self.live:
            if self.executor and self.executor.initialize():
                print("✅ Trading client ready")
            else:
                print("❌ Failed to initialize - running simulation")
                self.live = False
        
        self.refresh_markets()
        
        if not self.markets:
            print("❌ No markets found")
            return
        
        print("🧠 TRACKING SMART MONEY... (Ctrl+C to stop)\n")
        
        try:
            while True:
                self.stats['scans'] += 1
                
                if time.time() - self.last_refresh > CONFIG['MARKET_REFRESH']:
                    self.refresh_markets()
                
                # Check each market
                for market in self.markets:
                    self.check_market(market)
                
                time.sleep(CONFIG['SCAN_INTERVAL'])
                
        except KeyboardInterrupt:
            self.print_summary()
    
    def check_market(self, market: Market):
        """Check a single market for opportunities."""
        up_price, down_price = get_prices(market)
        
        if not up_price or not down_price:
            return
        
        order = calculate_smart_order(market, up_price, down_price)
        
        if not order:
            return
        
        self.stats['opportunities'] += 1
        
        # Check if we have a signal
        signal = order.signal
        
        ts = datetime.now().strftime('%H:%M:%S')
        time_left = CONFIG['TRADE_WINDOW_MINUTES'] * 60 - order.time_in_window
        
        if signal.direction == 'none':
            # No signal - only print occasionally
            if self.stats['scans'] % 20 == 0:
                print(f"[{ts}] ➡️ {market.asset} | No signal | "
                      f"UP Δ{signal.up_velocity:+.1%} DOWN Δ{signal.down_velocity:+.1%} | "
                      f"{time_left}s left")
            self.stats['skipped_no_signal'] += 1
            return
        
        # We have a signal!
        self.stats['signals'] += 1
        
        arrow = "📈" if signal.direction == "up" else "📉"
        strength = "🔥" if signal.strength == "strong" else "⚡"
        
        if signal.direction == 'up':
            self.stats['up_signals'] += 1
        else:
            self.stats['down_signals'] += 1
        
        spread = (1 - (up_price + down_price)) * 100
        
        print(f"\n[{ts}] {arrow}{strength} {market.asset} | "
              f"Signal: {signal.direction.upper()} ({signal.strength}) | "
              f"UP Δ{signal.up_velocity:+.1%} DOWN Δ{signal.down_velocity:+.1%}")
        print(f"         Prices: UP ${up_price:.3f} + DOWN ${down_price:.3f} | Spread: {spread:.1f}%")
        print(f"         Order: {order.up_shares} UP + {order.down_shares} DOWN | {time_left}s left in window")
        
        if self.live and self.executor:
            result = self.executor.execute(order)
            
            if result['success']:
                self.stats['trades'] += 1
                cost = (order.up_shares * up_price) + (order.down_shares * down_price)
                print(f"         ✅ EXECUTED! Cost: ${cost:.2f}")
            else:
                print(f"         ❌ {result['error']}")
    
    def print_summary(self):
        print("\n\n" + "=" * 70)
        print("📊 SESSION SUMMARY")
        print("=" * 70)
        print(f"Total scans:         {self.stats['scans']}")
        print(f"Opportunities:       {self.stats['opportunities']}")
        print(f"Signals detected:    {self.stats['signals']}")
        print(f"  - UP signals:      {self.stats['up_signals']} 📈")
        print(f"  - DOWN signals:    {self.stats['down_signals']} 📉")
        print(f"Trades executed:     {self.stats['trades']}")
        print(f"Skipped (no signal): {self.stats['skipped_no_signal']}")
        print("=" * 70)

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Smart Money Tracker')
    parser.add_argument('--live', '-l', action='store_true', help='Enable live trading')
    parser.add_argument('--yolo', action='store_true', help='Skip confirmation')
    parser.add_argument('--threshold', '-t', type=float, default=0.3, help='Min spread %%')
    parser.add_argument('--window', '-w', type=int, default=4, help='Trade window (minutes)')
    parser.add_argument('--velocity', '-v', type=float, default=0.02, help='Velocity threshold')
    
    args = parser.parse_args()
    
    CONFIG['MIN_SPREAD_PERCENT'] = args.threshold
    CONFIG['TRADE_WINDOW_MINUTES'] = args.window
    CONFIG['VELOCITY_THRESHOLD'] = args.velocity
    
    if args.live:
        if not CONFIG['PRIVATE_KEY']:
            print("❌ Add PRIVATE_KEY to CONFIG section first")
            return
        
        if not args.yolo:
            print("\n" + "=" * 70)
            print("🧠 SMART MONEY MODE")
            print("=" * 70)
            print("This bot follows Polymarket trader momentum.")
            print("It's still speculative - but based on what the market is doing,")
            print("not what we think crypto will do.")
            print("=" * 70)
            print("\nType 'FOLLOW THE MONEY' to continue: ", end="")
            if input() != 'FOLLOW THE MONEY':
                print("Cancelled.")
                return
    
    scanner = SmartMoneyScanner(live=args.live)
    scanner.run()

if __name__ == '__main__':
    main()
