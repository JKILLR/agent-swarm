#!/usr/bin/env python3
"""
MOMENTUM ARBITRAGE - Polymarket Trading Bot

Weights orders based on real-time crypto price momentum:
- If BTC trending UP → Buy more UP shares, fewer DOWN shares
- If BTC trending DOWN → Buy more DOWN shares, fewer UP shares
- If FLAT → Equal shares (pure arb)

Still requires a spread to exist before trading.

⚠️  WARNING: This is NOT pure arbitrage. You can lose money!

Usage:
    python3.11 momentum_arb.py --live --yolo
    python3.11 momentum_arb.py --live --yolo --aggression 0.8   # More aggressive
    python3.11 momentum_arb.py --live --yolo --aggression 0.5   # More conservative
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
    
    # Momentum settings
    'MOMENTUM_WINDOW_SECONDS': 60,      # Look at last 60 seconds of price data
    'PRICE_CHECK_INTERVAL': 2,          # Check prices every 2 seconds
    'MOMENTUM_THRESHOLD': 0.001,        # 0.1% move = trending (not flat)
    
    # Trading settings
    'MIN_SPREAD_PERCENT': 0.5,          # Still need spread to trade
    'SCAN_INTERVAL': 0.3,
    'MARKET_REFRESH': 300,
    
    'TOTAL_SHARES': 20,                 # Total shares per trade (split between up/down)
    'MIN_HEDGE_PERCENT': 0.2,           # Always keep at least 20% as hedge
    'AGGRESSION': 0.7,                  # How much to weight toward momentum (0.5 = pure arb, 1.0 = all-in)
    
    # =========================================================================
    # CREDENTIALS
    # =========================================================================
    'PRIVATE_KEY': '0xb92c6d5ae586a416cd45ecda3d8d7a1bb253777025fe31f863c8dcd9ea7e5bb0',
    'SIGNATURE_TYPE': 1,
    'FUNDER_ADDRESS': '0x1640782e9E71029B78555b9f23478712aC47396E',
}

TIMEFRAME_SECONDS = {'15m': 900, '1h': 3600, '4h': 14400}

# Crypto price API endpoints
PRICE_APIS = {
    'btc': 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT',
    'eth': 'https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT',
    'sol': 'https://api.binance.com/api/v3/ticker/price?symbol=SOLUSDT',
    'xrp': 'https://api.binance.com/api/v3/ticker/price?symbol=XRPUSDT',
}

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

@dataclass
class Momentum:
    direction: str          # 'up', 'down', 'flat'
    strength: float         # 0.0 to 1.0
    price_change_pct: float # Actual % change
    current_price: float
    
@dataclass
class WeightedOrder:
    market: Market
    up_shares: int
    down_shares: int
    up_price: float
    down_price: float
    momentum: Momentum
    expected_profit_if_correct: float
    expected_loss_if_wrong: float

# =============================================================================
# MOMENTUM TRACKER
# =============================================================================

class MomentumTracker:
    """Tracks real-time crypto prices and calculates momentum."""
    
    def __init__(self):
        # Store recent prices: {asset: deque of (timestamp, price)}
        self.price_history: Dict[str, deque] = {
            asset: deque(maxlen=100) for asset in CONFIG['ASSETS']
        }
        self.current_prices: Dict[str, float] = {}
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
    
    def start(self):
        """Start background price tracking."""
        self.running = True
        self.thread = threading.Thread(target=self._track_prices, daemon=True)
        self.thread.start()
        print("📊 Momentum tracker started")
        time.sleep(3)  # Wait for initial data
    
    def stop(self):
        """Stop tracking."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def _track_prices(self):
        """Background thread to fetch prices."""
        while self.running:
            for asset in CONFIG['ASSETS']:
                try:
                    resp = requests.get(PRICE_APIS[asset], timeout=2)
                    if resp.ok:
                        price = float(resp.json()['price'])
                        with self.lock:
                            self.price_history[asset].append((time.time(), price))
                            self.current_prices[asset] = price
                except:
                    pass
            time.sleep(CONFIG['PRICE_CHECK_INTERVAL'])
    
    def get_momentum(self, asset: str) -> Optional[Momentum]:
        """Calculate momentum for an asset."""
        asset = asset.lower()
        
        with self.lock:
            history = list(self.price_history.get(asset, []))
            current = self.current_prices.get(asset)
        
        if not history or not current or len(history) < 3:
            return None
        
        # Get price from MOMENTUM_WINDOW_SECONDS ago
        now = time.time()
        window = CONFIG['MOMENTUM_WINDOW_SECONDS']
        
        old_price = None
        for ts, price in history:
            if now - ts >= window * 0.8:  # At least 80% of window ago
                old_price = price
                break
        
        if old_price is None:
            old_price = history[0][1]  # Use oldest available
        
        # Calculate change
        price_change_pct = (current - old_price) / old_price
        
        # Determine direction and strength
        threshold = CONFIG['MOMENTUM_THRESHOLD']
        
        if price_change_pct > threshold:
            direction = 'up'
            strength = min(1.0, price_change_pct / (threshold * 5))  # Scale to 0-1
        elif price_change_pct < -threshold:
            direction = 'down'
            strength = min(1.0, abs(price_change_pct) / (threshold * 5))
        else:
            direction = 'flat'
            strength = 0.0
        
        return Momentum(
            direction=direction,
            strength=strength,
            price_change_pct=price_change_pct * 100,
            current_price=current
        )
    
    def get_all_momentum(self) -> Dict[str, Momentum]:
        """Get momentum for all assets."""
        result = {}
        for asset in CONFIG['ASSETS']:
            m = self.get_momentum(asset)
            if m:
                result[asset] = m
        return result

# =============================================================================
# MARKET DISCOVERY
# =============================================================================

MARKET_CACHE: Dict[str, Market] = {}

def get_epoch(timeframe: str) -> int:
    seconds = TIMEFRAME_SECONDS.get(timeframe, 900)
    return (int(time.time()) // seconds) * seconds

def fetch_market(asset: str, timeframe: str) -> Optional[Market]:
    cache_key = f"{asset}-{timeframe}"
    epoch = get_epoch(timeframe)
    slug = f"{asset}-updown-{timeframe}-{epoch}"
    
    if cache_key in MARKET_CACHE:
        cached = MARKET_CACHE[cache_key]
        if cached.slug == slug:
            return cached
    
    for offset_mult in [0, 1, -1]:
        offset = offset_mult * TIMEFRAME_SECONDS.get(timeframe, 900)
        slug = f"{asset}-updown-{timeframe}-{epoch + offset}"
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
                                down_token=tokens[1]
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
# PRICE FETCHING (Polymarket)
# =============================================================================

def get_midpoint(token_id: str) -> Optional[float]:
    try:
        r = requests.get(f"{CONFIG['CLOB_API']}/midpoint", 
                        params={'token_id': token_id}, timeout=2)
        if r.ok:
            mid = r.json().get('mid')
            if mid:
                return float(mid)
    except:
        pass
    return None

def get_best_ask(token_id: str) -> Optional[float]:
    try:
        r = requests.get(f"{CONFIG['CLOB_API']}/book", 
                        params={'token_id': token_id}, timeout=2)
        if r.ok:
            book = r.json()
            asks = book.get('asks', [])
            if asks:
                sorted_asks = sorted(asks, key=lambda x: float(x.get('price', 999)))
                return float(sorted_asks[0].get('price', 0))
    except:
        pass
    return None

def get_prices(market: Market) -> Tuple[Optional[float], Optional[float]]:
    up_price = get_best_ask(market.up_token) or get_midpoint(market.up_token)
    down_price = get_best_ask(market.down_token) or get_midpoint(market.down_token)
    return up_price, down_price

# =============================================================================
# WEIGHTED ORDER CALCULATOR
# =============================================================================

def calculate_weighted_order(market: Market, momentum: Momentum, 
                              up_price: float, down_price: float) -> Optional[WeightedOrder]:
    """Calculate share allocation based on momentum."""
    
    total = up_price + down_price
    spread_pct = (1 - total) * 100
    
    # Need minimum spread
    if spread_pct < CONFIG['MIN_SPREAD_PERCENT']:
        return None
    
    total_shares = CONFIG['TOTAL_SHARES']
    aggression = CONFIG['AGGRESSION']
    min_hedge = CONFIG['MIN_HEDGE_PERCENT']
    
    # Base split is 50/50
    base_up = total_shares / 2
    base_down = total_shares / 2
    
    # Calculate tilt based on momentum
    if momentum.direction == 'up':
        # Tilt toward UP
        tilt = momentum.strength * aggression * (0.5 - min_hedge)  # Max tilt
        up_ratio = 0.5 + tilt
        down_ratio = 0.5 - tilt
    elif momentum.direction == 'down':
        # Tilt toward DOWN
        tilt = momentum.strength * aggression * (0.5 - min_hedge)
        up_ratio = 0.5 - tilt
        down_ratio = 0.5 + tilt
    else:
        # Flat - pure arb
        up_ratio = 0.5
        down_ratio = 0.5
    
    # Ensure minimum hedge
    up_ratio = max(min_hedge, min(1 - min_hedge, up_ratio))
    down_ratio = 1 - up_ratio
    
    up_shares = int(round(total_shares * up_ratio))
    down_shares = int(round(total_shares * down_ratio))
    
    # Ensure we have at least 5 shares on each side (Polymarket minimum $1)
    up_shares = max(5, up_shares)
    down_shares = max(5, down_shares)
    
    # Calculate expected outcomes
    up_cost = up_shares * up_price
    down_cost = down_shares * down_price
    total_cost = up_cost + down_cost
    
    # If UP wins
    profit_if_up = up_shares - total_cost
    # If DOWN wins
    profit_if_down = down_shares - total_cost
    
    # Expected based on momentum direction
    if momentum.direction == 'up':
        expected_profit = profit_if_up
        expected_loss = profit_if_down
    elif momentum.direction == 'down':
        expected_profit = profit_if_down
        expected_loss = profit_if_up
    else:
        expected_profit = min(profit_if_up, profit_if_down)  # Arb profit
        expected_loss = expected_profit  # Same (it's arb)
    
    return WeightedOrder(
        market=market,
        up_shares=up_shares,
        down_shares=down_shares,
        up_price=up_price,
        down_price=down_price,
        momentum=momentum,
        expected_profit_if_correct=expected_profit,
        expected_loss_if_wrong=expected_loss
    )

# =============================================================================
# EXECUTOR
# =============================================================================

class MomentumExecutor:
    def __init__(self):
        self.client = None
        self.initialized = False
        self.lock = threading.Lock()
    
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
    
    def execute(self, order: WeightedOrder) -> Dict:
        """Execute a momentum-weighted order."""
        result = {'success': False, 'error': None, 'up_filled': 0, 'down_filled': 0}
        
        if not self.initialized:
            result['error'] = "Not initialized"
            return result
        
        with self.lock:
            errors = []
            
            # Place UP order
            try:
                up_args = OrderArgs(
                    price=order.up_price,
                    size=float(order.up_shares),
                    side=BUY,
                    token_id=order.market.up_token
                )
                signed = self.client.create_order(up_args)
                self.client.post_order(signed, OrderType.GTC)
                result['up_filled'] = order.up_shares
            except Exception as e:
                errors.append(f"UP: {e}")
            
            # Place DOWN order
            try:
                down_args = OrderArgs(
                    price=order.down_price,
                    size=float(order.down_shares),
                    side=BUY,
                    token_id=order.market.down_token
                )
                signed = self.client.create_order(down_args)
                self.client.post_order(signed, OrderType.GTC)
                result['down_filled'] = order.down_shares
            except Exception as e:
                errors.append(f"DOWN: {e}")
            
            if errors:
                result['error'] = "; ".join(errors)
            
            if result['up_filled'] > 0 or result['down_filled'] > 0:
                result['success'] = True
            
            return result

# =============================================================================
# MAIN SCANNER
# =============================================================================

class MomentumScanner:
    def __init__(self, live: bool = False):
        self.live = live
        self.markets: List[Market] = []
        self.last_refresh = 0
        self.momentum_tracker = MomentumTracker()
        self.executor = MomentumExecutor() if live else None
        
        self.stats = {
            'opportunities': 0,
            'trades': 0,
            'up_biased': 0,
            'down_biased': 0,
            'neutral': 0,
        }
    
    def refresh_markets(self):
        print("\n🔍 Refreshing markets...")
        self.markets = discover_markets()
        self.last_refresh = time.time()
        print(f"   Found {len(self.markets)} markets")
        for m in self.markets:
            print(f"   • {m.asset} {m.timeframe}")
        print()
    
    def run(self):
        print("=" * 70)
        print("🚀 MOMENTUM ARBITRAGE SCANNER")
        print("=" * 70)
        print("⚠️  WARNING: This is NOT pure arbitrage!")
        print("   Weights orders based on price momentum.")
        print("   You CAN lose money if momentum reverses!")
        print("=" * 70)
        print(f"Mode: {'🔴 LIVE TRADING' if self.live else '⚪ SIMULATION'}")
        print(f"Total shares per trade: {CONFIG['TOTAL_SHARES']}")
        print(f"Aggression: {CONFIG['AGGRESSION']} (0.5=arb, 1.0=yolo)")
        print(f"Min hedge: {CONFIG['MIN_HEDGE_PERCENT']*100}%")
        print(f"Momentum window: {CONFIG['MOMENTUM_WINDOW_SECONDS']}s")
        print("=" * 70)
        
        # Start momentum tracker
        self.momentum_tracker.start()
        
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
        
        print("🚀 SCANNING... (Ctrl+C to stop)\n")
        
        try:
            while True:
                if time.time() - self.last_refresh > CONFIG['MARKET_REFRESH']:
                    self.refresh_markets()
                
                # Get current momentum for all assets
                momentum_data = self.momentum_tracker.get_all_momentum()
                
                # Print momentum status
                self.print_momentum_status(momentum_data)
                
                # Check each market
                for market in self.markets:
                    asset_lower = market.asset.lower()
                    momentum = momentum_data.get(asset_lower)
                    
                    if not momentum:
                        continue
                    
                    up_price, down_price = get_prices(market)
                    if not up_price or not down_price:
                        continue
                    
                    order = calculate_weighted_order(market, momentum, up_price, down_price)
                    if order:
                        self.handle_opportunity(order)
                
                time.sleep(CONFIG['SCAN_INTERVAL'])
                
        except KeyboardInterrupt:
            self.momentum_tracker.stop()
            self.print_summary()
    
    def print_momentum_status(self, momentum_data: Dict[str, Momentum]):
        """Print current momentum for all assets."""
        ts = datetime.now().strftime('%H:%M:%S')
        
        parts = []
        for asset in CONFIG['ASSETS']:
            m = momentum_data.get(asset)
            if m:
                arrow = '↑' if m.direction == 'up' else '↓' if m.direction == 'down' else '→'
                color_pct = f"{m.price_change_pct:+.2f}%"
                parts.append(f"{asset.upper()}{arrow}{color_pct}")
        
        status = " | ".join(parts)
        print(f"[{ts}] Momentum: {status}", end='\r')
    
    def handle_opportunity(self, order: WeightedOrder):
        self.stats['opportunities'] += 1
        
        if order.momentum.direction == 'up':
            self.stats['up_biased'] += 1
            direction_icon = "📈"
        elif order.momentum.direction == 'down':
            self.stats['down_biased'] += 1
            direction_icon = "📉"
        else:
            self.stats['neutral'] += 1
            direction_icon = "➡️"
        
        ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        spread_pct = (1 - (order.up_price + order.down_price)) * 100
        
        print(f"\n[{ts}] {direction_icon} {order.market.asset} | "
              f"Momentum: {order.momentum.direction.upper()} ({order.momentum.price_change_pct:+.2f}%)")
        print(f"         Prices: UP ${order.up_price:.3f} + DOWN ${order.down_price:.3f} | Spread: {spread_pct:.1f}%")
        print(f"         Order: {order.up_shares} UP + {order.down_shares} DOWN = {order.up_shares + order.down_shares} total")
        print(f"         If {order.momentum.direction.upper()} wins: ${order.expected_profit_if_correct:.2f}")
        print(f"         If {order.momentum.direction.upper()} loses: ${order.expected_loss_if_wrong:.2f}")
        
        if self.live and self.executor:
            result = self.executor.execute(order)
            
            if result['success']:
                self.stats['trades'] += 1
                print(f"         ✅ EXECUTED! UP: {result['up_filled']} DOWN: {result['down_filled']}")
            else:
                print(f"         ❌ {result['error']}")
    
    def print_summary(self):
        print("\n\n" + "=" * 70)
        print("📊 SESSION SUMMARY")
        print("=" * 70)
        print(f"Opportunities:    {self.stats['opportunities']}")
        print(f"Trades executed:  {self.stats['trades']}")
        print(f"Up-biased:        {self.stats['up_biased']} 📈")
        print(f"Down-biased:      {self.stats['down_biased']} 📉")
        print(f"Neutral (arb):    {self.stats['neutral']} ➡️")
        print("=" * 70)
        print("⚠️  Remember: Check Polymarket for actual P&L!")
        print("=" * 70)

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Momentum Arbitrage Scanner')
    parser.add_argument('--live', '-l', action='store_true', help='Enable live trading')
    parser.add_argument('--yolo', action='store_true', help='Skip confirmation')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Min spread %%')
    parser.add_argument('--shares', '-s', type=int, default=20, help='Total shares per trade')
    parser.add_argument('--aggression', '-a', type=float, default=0.7, 
                        help='Momentum weight (0.5=pure arb, 1.0=all-in on momentum)')
    parser.add_argument('--window', '-w', type=int, default=60,
                        help='Momentum window in seconds')
    
    args = parser.parse_args()
    
    CONFIG['MIN_SPREAD_PERCENT'] = args.threshold
    CONFIG['TOTAL_SHARES'] = max(10, args.shares)
    CONFIG['AGGRESSION'] = max(0.5, min(1.0, args.aggression))
    CONFIG['MOMENTUM_WINDOW_SECONDS'] = args.window
    
    if args.live:
        if not CONFIG['PRIVATE_KEY']:
            print("❌ Add PRIVATE_KEY to CONFIG section first")
            return
        
        if not args.yolo:
            print("\n" + "=" * 70)
            print("⚠️  MOMENTUM TRADING - READ CAREFULLY!")
            print("=" * 70)
            print("This is NOT pure arbitrage. You're betting on momentum continuing.")
            print(f"Aggression: {CONFIG['AGGRESSION']} (higher = more directional risk)")
            print()
            print("If momentum continues → You profit MORE than pure arb")
            print("If momentum reverses → You LOSE money")
            print("=" * 70)
            print("\nType 'I UNDERSTAND THE RISK' to continue: ", end="")
            if input() != 'I UNDERSTAND THE RISK':
                print("Cancelled.")
                return
    
    scanner = MomentumScanner(live=args.live)
    scanner.run()

if __name__ == '__main__':
    main()
