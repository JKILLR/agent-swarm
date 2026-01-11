#!/usr/bin/env python3
"""
FAST MOMENTUM Polymarket Arbitrage Scanner

Same as fast_arb.py but weights orders based on crypto momentum.
Uses exact same scanning logic that was working well.

Usage:
    python3.11 fast_momentum_arb.py --live --yolo
    python3.11 fast_momentum_arb.py --live --yolo --bias 0.6   # 60/40 split
"""

import requests
import re
import json
import time
import argparse
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
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
    
    'MIN_SPREAD_PERCENT': 0.3,
    'SCAN_INTERVAL': 0.1,
    'MARKET_REFRESH': 300,
    'REQUEST_TIMEOUT': 2,
    
    'TOTAL_SHARES': 20,       # Total shares to split between up/down
    'MOMENTUM_BIAS': 0.6,     # 0.5 = equal, 0.6 = 60/40 split, 0.7 = 70/30
    'SKIP_FLAT': False,       # Skip trades when no momentum signal
    
    # =========================================================================
    # CREDENTIALS - FILL THESE IN
    # =========================================================================
    'PRIVATE_KEY': '0xb92c6d5ae586a416cd45ecda3d8d7a1bb253777025fe31f863c8dcd9ea7e5bb0',
    'SIGNATURE_TYPE': 1,
    'FUNDER_ADDRESS': '0x1640782e9E71029B78555b9f23478712aC47396E',
}

TIMEFRAME_SECONDS = {'15m': 900, '1h': 3600, '4h': 14400}

# Binance price endpoints
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
# MOMENTUM TRACKER (lightweight)
# =============================================================================

class MomentumTracker:
    """Simple momentum tracker - stores last 2 prices per asset."""
    
    def __init__(self):
        self.prices: Dict[str, deque] = {a: deque(maxlen=10) for a in CONFIG['ASSETS']}
        self.lock = threading.Lock()
    
    def update(self, asset: str, price: float):
        with self.lock:
            self.prices[asset.lower()].append((time.time(), price))
    
    def get_direction(self, asset: str) -> str:
        """Returns 'up', 'down', or 'flat'"""
        with self.lock:
            history = list(self.prices.get(asset.lower(), []))
        
        if len(history) < 2:
            return 'flat'
        
        old_price = history[0][1]
        new_price = history[-1][1]
        
        pct_change = (new_price - old_price) / old_price
        
        if pct_change > 0.0005:  # 0.05% threshold
            return 'up'
        elif pct_change < -0.0005:
            return 'down'
        return 'flat'
    
    def fetch_all(self):
        """Fetch current prices from Binance."""
        for asset in CONFIG['ASSETS']:
            try:
                resp = requests.get(PRICE_APIS[asset], timeout=1)
                if resp.ok:
                    price = float(resp.json()['price'])
                    self.update(asset, price)
            except:
                pass

# Global tracker
MOMENTUM = MomentumTracker()

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
class Opportunity:
    market: Market
    up_price: float
    down_price: float
    total: float
    spread_pct: float

# =============================================================================
# MARKET DISCOVERY (same as fast_arb)
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
# PRICE FETCHING (same as fast_arb)
# =============================================================================

def get_prices_fast(up_token: str, down_token: str) -> tuple:
    results = [None, None]
    
    def fetch_up():
        try:
            r = requests.get(f"{CONFIG['CLOB_API']}/midpoint", 
                           params={'token_id': up_token}, 
                           timeout=CONFIG['REQUEST_TIMEOUT'])
            if r.ok:
                results[0] = float(r.json().get('mid', 0))
        except:
            pass
    
    def fetch_down():
        try:
            r = requests.get(f"{CONFIG['CLOB_API']}/midpoint", 
                           params={'token_id': down_token}, 
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
    
    return results[0], results[1]

def check_market_fast(market: Market) -> Optional[Opportunity]:
    up_price, down_price = get_prices_fast(market.up_token, market.down_token)
    
    if up_price is None or down_price is None:
        return None
    
    total = up_price + down_price
    spread_pct = (1 - total) * 100
    
    if spread_pct >= CONFIG['MIN_SPREAD_PERCENT']:
        return Opportunity(
            market=market,
            up_price=up_price,
            down_price=down_price,
            total=total,
            spread_pct=spread_pct
        )
    return None

# =============================================================================
# WEIGHTED SHARE CALCULATOR
# =============================================================================

def calculate_shares(asset: str) -> tuple:
    """Calculate up/down share split based on momentum."""
    direction = MOMENTUM.get_direction(asset)
    total = CONFIG['TOTAL_SHARES']
    bias = CONFIG['MOMENTUM_BIAS']
    
    if direction == 'up':
        up_shares = int(round(total * bias))
        down_shares = total - up_shares
    elif direction == 'down':
        down_shares = int(round(total * bias))
        up_shares = total - down_shares
    else:
        up_shares = total // 2
        down_shares = total - up_shares
    
    # Ensure minimum 5 shares each side (for Polymarket $1 minimum)
    up_shares = max(5, up_shares)
    down_shares = max(5, down_shares)
    
    return up_shares, down_shares, direction

# =============================================================================
# FAST EXECUTION WITH MOMENTUM
# =============================================================================

class FastMomentumExecutor:
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
    
    def execute(self, opp: Opportunity) -> Dict[str, Any]:
        """Execute with momentum-weighted shares."""
        result = {
            'success': False, 
            'error': None, 
            'up_shares': 0,
            'down_shares': 0,
            'direction': 'flat',
            'cost': 0
        }
        
        if not self.initialized:
            result['error'] = "Not initialized"
            return result
        
        with self.lock:
            # Get momentum-weighted shares
            up_shares, down_shares, direction = calculate_shares(opp.market.asset)
            result['up_shares'] = up_shares
            result['down_shares'] = down_shares
            result['direction'] = direction
            
            up_result = None
            down_result = None
            errors = []
            
            def buy_up():
                nonlocal up_result
                try:
                    order_args = OrderArgs(
                        price=opp.up_price,
                        size=float(up_shares),
                        side=BUY,
                        token_id=opp.market.up_token
                    )
                    signed_order = self.client.create_order(order_args)
                    up_result = self.client.post_order(signed_order, OrderType.GTC)
                except Exception as e:
                    errors.append(f"UP: {e}")
            
            def buy_down():
                nonlocal down_result
                try:
                    order_args = OrderArgs(
                        price=opp.down_price,
                        size=float(down_shares),
                        side=BUY,
                        token_id=opp.market.down_token
                    )
                    signed_order = self.client.create_order(order_args)
                    down_result = self.client.post_order(signed_order, OrderType.GTC)
                except Exception as e:
                    errors.append(f"DOWN: {e}")
            
            # Execute BOTH legs in parallel
            t1 = threading.Thread(target=buy_up)
            t2 = threading.Thread(target=buy_down)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            
            if errors:
                result['error'] = "; ".join(errors)
                if up_result or down_result:
                    result['error'] += " ⚠️ PARTIAL FILL!"
                return result
            
            result['success'] = True
            result['cost'] = (up_shares * opp.up_price) + (down_shares * opp.down_price)
            return result

# =============================================================================
# MAIN SCANNER
# =============================================================================

class FastMomentumScanner:
    def __init__(self, live: bool = False):
        self.live = live
        self.markets: List[Market] = []
        self.last_refresh = 0
        self.last_momentum_update = 0
        self.executor = FastMomentumExecutor() if live else None
        
        self.stats = {
            'scans': 0,
            'opportunities': 0,
            'attempts': 0,
            'successes': 0,
            'failures': 0,
            'skipped': 0,
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
        print("⚡ FAST MOMENTUM ARBITRAGE SCANNER")
        print("=" * 70)
        print("Same scanning as fast_arb + momentum-weighted orders")
        print("=" * 70)
        print(f"Mode: {'🔴 LIVE TRADING' if self.live else '⚪ SIMULATION'}")
        print(f"Threshold: {CONFIG['MIN_SPREAD_PERCENT']}% spread")
        print(f"Total shares: {CONFIG['TOTAL_SHARES']} (split by momentum)")
        print(f"Momentum bias: {CONFIG['MOMENTUM_BIAS']} (0.5=equal, 0.6=60/40)")
        print(f"Skip flat: {'YES - only trade with momentum signal' if CONFIG['SKIP_FLAT'] else 'NO - trade all opportunities'}")
        print("=" * 70)
        
        if self.live:
            if self.executor and self.executor.initialize():
                print("✅ Trading client ready")
            else:
                print("❌ Failed to initialize - running simulation")
                self.live = False
        
        # Initial momentum fetch
        print("📊 Fetching initial prices...")
        for _ in range(3):
            MOMENTUM.fetch_all()
            time.sleep(0.5)
        
        self.refresh_markets()
        
        if not self.markets:
            print("❌ No markets found")
            return
        
        print("⚡ SCANNING... (Ctrl+C to stop)\n")
        
        try:
            while True:
                self.stats['scans'] += 1
                
                # Update momentum every 2 seconds
                if time.time() - self.last_momentum_update > 2:
                    MOMENTUM.fetch_all()
                    self.last_momentum_update = time.time()
                
                if time.time() - self.last_refresh > CONFIG['MARKET_REFRESH']:
                    self.refresh_markets()
                
                with ThreadPoolExecutor(max_workers=4) as ex:
                    futures = {ex.submit(check_market_fast, m): m for m in self.markets}
                    
                    for future in as_completed(futures):
                        try:
                            opp = future.result()
                            if opp:
                                self.handle_opportunity(opp)
                        except:
                            pass
                
                time.sleep(CONFIG['SCAN_INTERVAL'])
                
        except KeyboardInterrupt:
            self.print_summary()
    
    def handle_opportunity(self, opp: Opportunity):
        self.stats['opportunities'] += 1
        ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        
        # Pre-calculate what we'd do
        up_shares, down_shares, direction = calculate_shares(opp.market.asset)
        
        # Skip flat trades if enabled
        if CONFIG['SKIP_FLAT'] and direction == 'flat':
            print(f"[{ts}] ➡️ {opp.market.asset} | Spread: {opp.spread_pct:.1f}% | SKIPPED (no momentum)")
            self.stats['skipped'] += 1
            return
        
        arrow = "📈" if direction == "up" else "📉" if direction == "down" else "➡️"
        
        print(f"[{ts}] {arrow} {opp.market.asset} | "
              f"UP ${opp.up_price:.3f} + DOWN ${opp.down_price:.3f} = ${opp.total:.3f} | "
              f"Spread: {opp.spread_pct:.1f}% | "
              f"Split: {up_shares}/{down_shares}")
        
        if self.live and self.executor:
            self.stats['attempts'] += 1
            result = self.executor.execute(opp)
            
            if result['success']:
                self.stats['successes'] += 1
                if result['direction'] == 'up':
                    self.stats['up_biased'] += 1
                elif result['direction'] == 'down':
                    self.stats['down_biased'] += 1
                else:
                    self.stats['neutral'] += 1
                print(f"         ✅ EXECUTED! {result['up_shares']} UP + {result['down_shares']} DOWN = ${result['cost']:.2f}")
            else:
                self.stats['failures'] += 1
                print(f"         ❌ {result['error']}")
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("📊 SESSION SUMMARY")
        print("=" * 70)
        print(f"Opportunities:    {self.stats['opportunities']}")
        print(f"Skipped (flat):   {self.stats['skipped']} ➡️")
        print(f"Trade attempts:   {self.stats['attempts']}")
        print(f"Successful:       {self.stats['successes']}")
        print(f"Failed:           {self.stats['failures']}")
        print(f"Up-biased:        {self.stats['up_biased']} 📈")
        print(f"Down-biased:      {self.stats['down_biased']} 📉")
        print("=" * 70)

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='FAST MOMENTUM Arbitrage Scanner')
    parser.add_argument('--live', '-l', action='store_true', help='Enable live trading')
    parser.add_argument('--yolo', action='store_true', help='Skip confirmation')
    parser.add_argument('--threshold', '-t', type=float, default=0.3, help='Min spread %%')
    parser.add_argument('--shares', '-s', type=int, default=20, help='Total shares per trade')
    parser.add_argument('--bias', '-b', type=float, default=0.6, help='Momentum bias (0.5-0.8)')
    parser.add_argument('--skip-flat', action='store_true', help='Only trade when momentum signal exists')
    
    args = parser.parse_args()
    
    CONFIG['MIN_SPREAD_PERCENT'] = args.threshold
    CONFIG['TOTAL_SHARES'] = max(10, args.shares)
    CONFIG['MOMENTUM_BIAS'] = max(0.5, min(0.8, args.bias))
    CONFIG['SKIP_FLAT'] = args.skip_flat
    
    if args.live:
        if not CONFIG['PRIVATE_KEY']:
            print("❌ Add PRIVATE_KEY to CONFIG section first")
            return
        
        if not args.yolo:
            print(f"\n⚠️  LIVE TRADING with {CONFIG['MOMENTUM_BIAS']*100:.0f}/{100-CONFIG['MOMENTUM_BIAS']*100:.0f} momentum split")
            print("Type 'I ACCEPT THE RISK' to continue: ", end="")
            if input() != 'I ACCEPT THE RISK':
                print("Cancelled.")
                return
    
    scanner = FastMomentumScanner(live=args.live)
    scanner.run()

if __name__ == '__main__':
    main()
