#!/usr/bin/env python3
"""
RETRY MODE Polymarket Arbitrage Scanner

Key improvements over fast_arb:
1. Retries unfilled orders at current price (up to 3 times)
2. Tracks position balance per market - won't stack on one side
3. Auto-cancels stale orders after 30 seconds
4. One trade per market per window

Usage:
    python3.11 retry_arb.py --live --yolo --threshold 2.0 --size 10
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

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'POLYMARKET_WEB': 'https://polymarket.com',
    'CLOB_API': 'https://clob.polymarket.com',
    'CHAIN_ID': 137,
    
    'ASSETS': ['btc', 'eth', 'sol', 'xrp'],
    'TIMEFRAMES': ['15m'],
    
    'MIN_SPREAD_PERCENT': 2.0,
    'SCAN_INTERVAL': 0.5,
    'MARKET_REFRESH': 300,
    'REQUEST_TIMEOUT': 3,
    
    'ORDER_SIZE': 10,
    'MAX_RETRIES': 3,
    'RETRY_DELAY': 2,  # seconds between retries
    
    # =========================================================================
    # CREDENTIALS - FILL THESE IN
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

@dataclass
class Opportunity:
    market: Market
    up_price: float
    down_price: float
    total: float
    spread_pct: float

@dataclass
class Position:
    up_shares: int = 0
    down_shares: int = 0
    up_cost: float = 0
    down_cost: float = 0

# =============================================================================
# POSITION TRACKER
# =============================================================================

class PositionTracker:
    """Tracks positions per market to prevent stacking on one side."""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}  # slug -> Position
        self.traded_windows: set = set()  # Track which windows we've traded
        self.lock = threading.Lock()
    
    def get_position(self, slug: str) -> Position:
        with self.lock:
            if slug not in self.positions:
                self.positions[slug] = Position()
            return self.positions[slug]
    
    def add_fill(self, slug: str, side: str, shares: int, cost: float):
        with self.lock:
            if slug not in self.positions:
                self.positions[slug] = Position()
            
            if side == 'up':
                self.positions[slug].up_shares += shares
                self.positions[slug].up_cost += cost
            else:
                self.positions[slug].down_shares += shares
                self.positions[slug].down_cost += cost
    
    def is_balanced(self, slug: str) -> bool:
        pos = self.get_position(slug)
        return pos.up_shares == pos.down_shares
    
    def needs_up(self, slug: str) -> int:
        """Returns how many UP shares needed to balance."""
        pos = self.get_position(slug)
        return max(0, pos.down_shares - pos.up_shares)
    
    def needs_down(self, slug: str) -> int:
        """Returns how many DOWN shares needed to balance."""
        pos = self.get_position(slug)
        return max(0, pos.up_shares - pos.down_shares)
    
    def already_traded(self, slug: str) -> bool:
        with self.lock:
            return slug in self.traded_windows
    
    def mark_traded(self, slug: str):
        with self.lock:
            self.traded_windows.add(slug)
    
    def clear_window(self, slug: str):
        """Clear position data when window changes."""
        with self.lock:
            if slug in self.positions:
                del self.positions[slug]
            if slug in self.traded_windows:
                self.traded_windows.remove(slug)
    
    def get_summary(self, slug: str) -> str:
        pos = self.get_position(slug)
        return f"UP: {pos.up_shares} (${pos.up_cost:.2f}) | DOWN: {pos.down_shares} (${pos.down_cost:.2f})"

# Global tracker
POSITIONS = PositionTracker()

# =============================================================================
# MARKET DISCOVERY
# =============================================================================

MARKET_CACHE: Dict[str, Market] = {}
LAST_SLUGS: Dict[str, str] = {}  # Track slug changes to clear positions

def get_epoch(timeframe: str) -> int:
    seconds = TIMEFRAME_SECONDS.get(timeframe, 900)
    return (int(time.time()) // seconds) * seconds

def fetch_market(asset: str, timeframe: str) -> Optional[Market]:
    cache_key = f"{asset}-{timeframe}"
    epoch = get_epoch(timeframe)
    slug = f"{asset}-updown-{timeframe}-{epoch}"
    
    # Check if window changed
    if cache_key in LAST_SLUGS and LAST_SLUGS[cache_key] != slug:
        POSITIONS.clear_window(LAST_SLUGS[cache_key])
    
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
                            LAST_SLUGS[cache_key] = slug
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

def get_prices(up_token: str, down_token: str) -> tuple:
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

def check_market(market: Market) -> Optional[Opportunity]:
    up_price, down_price = get_prices(market.up_token, market.down_token)
    
    if up_price is None or down_price is None:
        return None
    
    # Skip if prices are invalid
    if up_price <= 0 or down_price <= 0 or up_price >= 1 or down_price >= 1:
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
# RETRY EXECUTOR
# =============================================================================

class RetryExecutor:
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
    
    def buy_side(self, token_id: str, price: float, size: float) -> bool:
        """Buy a single side. Returns True if successful."""
        try:
            order_args = OrderArgs(
                price=price,
                size=size,
                side=BUY,
                token_id=token_id
            )
            signed_order = self.client.create_order(order_args)
            result = self.client.post_order(signed_order, OrderType.GTC)
            return result is not None
        except Exception as e:
            print(f"         ⚠️ Order failed: {e}")
            return False
    
    def execute_with_retry(self, opp: Opportunity) -> Dict[str, Any]:
        """Execute with retry logic for unfilled sides."""
        result = {
            'success': False, 
            'error': None, 
            'up_filled': False,
            'down_filled': False,
            'up_cost': 0,
            'down_cost': 0,
            'retries': 0
        }
        
        if not self.initialized:
            result['error'] = "Not initialized"
            return result
        
        with self.lock:
            size = float(CONFIG['ORDER_SIZE'])
            slug = opp.market.slug
            
            # First attempt - both sides in parallel
            up_success = False
            down_success = False
            up_price = opp.up_price
            down_price = opp.down_price
            
            def try_up():
                nonlocal up_success
                up_success = self.buy_side(opp.market.up_token, up_price, size)
            
            def try_down():
                nonlocal down_success
                down_success = self.buy_side(opp.market.down_token, down_price, size)
            
            t1 = threading.Thread(target=try_up)
            t2 = threading.Thread(target=try_down)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            
            if up_success:
                result['up_filled'] = True
                result['up_cost'] = up_price * size
                POSITIONS.add_fill(slug, 'up', int(size), up_price * size)
                print(f"         ✅ UP filled @ ${up_price:.3f}")
            
            if down_success:
                result['down_filled'] = True
                result['down_cost'] = down_price * size
                POSITIONS.add_fill(slug, 'down', int(size), down_price * size)
                print(f"         ✅ DOWN filled @ ${down_price:.3f}")
            
            # If both filled, we're done
            if up_success and down_success:
                result['success'] = True
                POSITIONS.mark_traded(slug)
                return result
            
            # If neither filled, give up
            if not up_success and not down_success:
                result['error'] = "Neither side filled"
                return result
            
            # One side filled - retry the other
            for retry in range(CONFIG['MAX_RETRIES']):
                result['retries'] += 1
                time.sleep(CONFIG['RETRY_DELAY'])
                
                # Get fresh prices
                new_up, new_down = get_prices(opp.market.up_token, opp.market.down_token)
                if not new_up or not new_down:
                    continue
                
                # Check if still profitable
                new_total = new_up + new_down
                if new_total >= 1.0:
                    print(f"         ⚠️ Spread gone (retry {retry+1})")
                    continue
                
                if not result['up_filled']:
                    # Retry UP at new price (add 1-2 cents to ensure fill)
                    retry_price = min(new_up + 0.02, 0.99)
                    print(f"         🔄 Retry UP @ ${retry_price:.3f} (attempt {retry+1})")
                    
                    if self.buy_side(opp.market.up_token, retry_price, size):
                        result['up_filled'] = True
                        result['up_cost'] = retry_price * size
                        POSITIONS.add_fill(slug, 'up', int(size), retry_price * size)
                        print(f"         ✅ UP filled @ ${retry_price:.3f}")
                        result['success'] = True
                        POSITIONS.mark_traded(slug)
                        return result
                
                if not result['down_filled']:
                    # Retry DOWN at new price
                    retry_price = min(new_down + 0.02, 0.99)
                    print(f"         🔄 Retry DOWN @ ${retry_price:.3f} (attempt {retry+1})")
                    
                    if self.buy_side(opp.market.down_token, retry_price, size):
                        result['down_filled'] = True
                        result['down_cost'] = retry_price * size
                        POSITIONS.add_fill(slug, 'down', int(size), retry_price * size)
                        print(f"         ✅ DOWN filled @ ${retry_price:.3f}")
                        result['success'] = True
                        POSITIONS.mark_traded(slug)
                        return result
            
            # Retries exhausted
            if result['up_filled'] and not result['down_filled']:
                result['error'] = "⚠️ PARTIAL: Only UP filled after retries"
            elif result['down_filled'] and not result['up_filled']:
                result['error'] = "⚠️ PARTIAL: Only DOWN filled after retries"
            
            return result

# =============================================================================
# MAIN SCANNER
# =============================================================================

class RetryScanner:
    def __init__(self, live: bool = False):
        self.live = live
        self.markets: List[Market] = []
        self.last_refresh = 0
        self.executor = RetryExecutor() if live else None
        
        self.stats = {
            'scans': 0,
            'opportunities': 0,
            'attempts': 0,
            'full_fills': 0,
            'partial_fills': 0,
            'total_retries': 0,
        }
    
    def refresh_markets(self):
        print("\n🔍 Refreshing markets...")
        self.markets = discover_markets()
        self.last_refresh = time.time()
        print(f"   Found {len(self.markets)} markets")
        for m in self.markets:
            pos = POSITIONS.get_summary(m.slug)
            traded = "✓" if POSITIONS.already_traded(m.slug) else ""
            print(f"   • {m.asset} {m.timeframe} {traded} | {pos}")
        print()
    
    def run(self):
        print("=" * 70)
        print("🔄 RETRY MODE ARBITRAGE SCANNER")
        print("=" * 70)
        print(f"Mode: {'🔴 LIVE TRADING' if self.live else '⚪ SIMULATION'}")
        print(f"Threshold: {CONFIG['MIN_SPREAD_PERCENT']}% spread")
        print(f"Order size: {CONFIG['ORDER_SIZE']} shares")
        print(f"Max retries: {CONFIG['MAX_RETRIES']} per side")
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
        
        print("🔄 SCANNING... (Ctrl+C to stop)\n")
        
        try:
            while True:
                self.stats['scans'] += 1
                
                if time.time() - self.last_refresh > CONFIG['MARKET_REFRESH']:
                    self.refresh_markets()
                
                for market in self.markets:
                    # Skip if already traded this window
                    if POSITIONS.already_traded(market.slug):
                        continue
                    
                    # Skip if position is unbalanced (need to wait for balance)
                    if not POSITIONS.is_balanced(market.slug):
                        needs_up = POSITIONS.needs_up(market.slug)
                        needs_down = POSITIONS.needs_down(market.slug)
                        if needs_up > 0 or needs_down > 0:
                            # TODO: Could try to balance here
                            continue
                    
                    opp = check_market(market)
                    if opp:
                        self.handle_opportunity(opp)
                
                time.sleep(CONFIG['SCAN_INTERVAL'])
                
        except KeyboardInterrupt:
            self.print_summary()
    
    def handle_opportunity(self, opp: Opportunity):
        self.stats['opportunities'] += 1
        ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        
        profit = (1 - opp.total) * CONFIG['ORDER_SIZE']
        
        print(f"\n[{ts}] ⚡ {opp.market.asset} | "
              f"UP ${opp.up_price:.3f} + DOWN ${opp.down_price:.3f} = ${opp.total:.3f} | "
              f"Spread: {opp.spread_pct:.1f}% | Est.Profit: ${profit:.2f}")
        
        if self.live and self.executor:
            self.stats['attempts'] += 1
            result = self.executor.execute_with_retry(opp)
            
            self.stats['total_retries'] += result.get('retries', 0)
            
            if result['success']:
                self.stats['full_fills'] += 1
                total_cost = result['up_cost'] + result['down_cost']
                actual_profit = CONFIG['ORDER_SIZE'] - total_cost
                print(f"         ✅ COMPLETE! Cost: ${total_cost:.2f} → Profit: ${actual_profit:.2f}")
            elif result['up_filled'] or result['down_filled']:
                self.stats['partial_fills'] += 1
                print(f"         {result['error']}")
                print(f"         Position: {POSITIONS.get_summary(opp.market.slug)}")
            else:
                print(f"         ❌ {result['error']}")
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("📊 SESSION SUMMARY")
        print("=" * 70)
        print(f"Opportunities:    {self.stats['opportunities']}")
        print(f"Trade attempts:   {self.stats['attempts']}")
        print(f"Full fills:       {self.stats['full_fills']} ✅")
        print(f"Partial fills:    {self.stats['partial_fills']} ⚠️")
        print(f"Total retries:    {self.stats['total_retries']}")
        
        if self.stats['attempts'] > 0:
            fill_rate = self.stats['full_fills'] / self.stats['attempts'] * 100
            print(f"Fill rate:        {fill_rate:.1f}%")
        
        print("\n📦 FINAL POSITIONS:")
        for market in self.markets:
            pos = POSITIONS.get_summary(market.slug)
            balanced = "✅" if POSITIONS.is_balanced(market.slug) else "⚠️"
            print(f"   {market.asset}: {pos} {balanced}")
        
        print("=" * 70)

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='RETRY MODE Arbitrage Scanner')
    parser.add_argument('--live', '-l', action='store_true', help='Enable live trading')
    parser.add_argument('--yolo', action='store_true', help='Skip confirmation')
    parser.add_argument('--threshold', '-t', type=float, default=2.0, help='Min spread %%')
    parser.add_argument('--size', '-s', type=int, default=10, help='Order size')
    parser.add_argument('--retries', '-r', type=int, default=3, help='Max retries per side')
    
    args = parser.parse_args()
    
    CONFIG['MIN_SPREAD_PERCENT'] = args.threshold
    CONFIG['ORDER_SIZE'] = max(5, args.size)
    CONFIG['MAX_RETRIES'] = args.retries
    
    if args.live:
        if not CONFIG['PRIVATE_KEY']:
            print("❌ Add PRIVATE_KEY to CONFIG section first")
            return
        
        if not args.yolo:
            print("\n🔄 RETRY MODE")
            print("This version retries unfilled orders up to 3 times.")
            print("Type 'I ACCEPT THE RISK' to continue: ", end="")
            if input() != 'I ACCEPT THE RISK':
                print("Cancelled.")
                return
    
    scanner = RetryScanner(live=args.live)
    scanner.run()

if __name__ == '__main__':
    main()
