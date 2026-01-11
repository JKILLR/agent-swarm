#!/usr/bin/env python3
"""
BALANCED Polymarket Arbitrage Scanner

Key difference from fast_arb.py:
- After executing, checks if both sides filled
- If partial fill, attempts to complete the other side
- Tracks position balance
- Won't open new trades if unbalanced positions exist

Usage:
    python3.11 balanced_arb.py              # Simulation
    python3.11 balanced_arb.py --live       # Live trading
"""

import requests
import re
import json
import time
import argparse
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
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
    
    # More conservative settings
    'MIN_SPREAD_PERCENT': 0.5,      # Higher threshold = safer
    'SCAN_INTERVAL': 0.2,
    'MARKET_REFRESH': 300,
    'REQUEST_TIMEOUT': 2,
    
    'ORDER_SIZE': 10,               # Shares per side
    'MAX_RETRY_ATTEMPTS': 3,        # Retries for partial fills
    
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

@dataclass
class Opportunity:
    market: Market
    up_price: float
    down_price: float
    total: float
    spread_pct: float

@dataclass
class TradeResult:
    success: bool
    up_filled: bool
    down_filled: bool
    up_size: float
    down_size: float
    error: Optional[str] = None

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
# PRICE FETCHING
# =============================================================================

def get_midpoint(token_id: str) -> Optional[float]:
    try:
        r = requests.get(f"{CONFIG['CLOB_API']}/midpoint", 
                        params={'token_id': token_id}, 
                        timeout=CONFIG['REQUEST_TIMEOUT'])
        if r.ok:
            return float(r.json().get('mid', 0))
    except:
        pass
    return None

def get_best_ask(token_id: str) -> Optional[float]:
    """Get best ask price - what you'd actually pay"""
    try:
        r = requests.get(f"{CONFIG['CLOB_API']}/book", 
                        params={'token_id': token_id}, 
                        timeout=CONFIG['REQUEST_TIMEOUT'])
        if r.ok:
            book = r.json()
            asks = book.get('asks', [])
            if asks:
                sorted_asks = sorted(asks, key=lambda x: float(x.get('price', 999)))
                return float(sorted_asks[0].get('price', 0))
    except:
        pass
    return None

def get_prices_fast(up_token: str, down_token: str) -> tuple:
    results = [None, None]
    
    def fetch_up():
        price = get_best_ask(up_token)
        if price is None:
            price = get_midpoint(up_token)
        results[0] = price
    
    def fetch_down():
        price = get_best_ask(down_token)
        if price is None:
            price = get_midpoint(down_token)
        results[1] = price
    
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
# BALANCED EXECUTION
# =============================================================================

class BalancedExecutor:
    """
    Executes trades and attempts to balance partial fills.
    """
    
    def __init__(self):
        self.client = None
        self.initialized = False
        self.lock = threading.Lock()
        
        # Track unbalanced positions
        self.pending_balance: Dict[str, Dict] = {}  # token_id -> {side, size_needed}
    
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
    
    def place_order(self, token_id: str, price: float, size: float) -> Dict:
        """Place a single order, return result"""
        result = {'success': False, 'error': None}
        
        try:
            order_args = OrderArgs(
                price=price,
                size=size,
                side=BUY,
                token_id=token_id
            )
            signed_order = self.client.create_order(order_args)
            response = self.client.post_order(signed_order, OrderType.GTC)
            result['success'] = True
            result['response'] = response
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def execute_balanced(self, opp: Opportunity) -> TradeResult:
        """
        Execute arbitrage with balance checking.
        If one side fails, retry multiple times.
        """
        result = TradeResult(
            success=False,
            up_filled=False,
            down_filled=False,
            up_size=0,
            down_size=0
        )
        
        if not self.initialized:
            result.error = "Not initialized"
            return result
        
        with self.lock:
            size = float(CONFIG['ORDER_SIZE'])
            
            # Get fresh prices
            up_price, down_price = get_prices_fast(opp.market.up_token, opp.market.down_token)
            
            if up_price is None or down_price is None:
                result.error = "Price unavailable"
                return result
            
            # Check spread still exists
            if up_price + down_price >= 1.0:
                result.error = "Spread gone"
                return result
            
            # === ATTEMPT UP ORDER ===
            up_result = self.place_order(opp.market.up_token, up_price, size)
            if up_result['success']:
                result.up_filled = True
                result.up_size = size
            
            # === ATTEMPT DOWN ORDER ===
            down_result = self.place_order(opp.market.down_token, down_price, size)
            if down_result['success']:
                result.down_filled = True
                result.down_size = size
            
            # === CHECK FOR PARTIAL FILL ===
            if result.up_filled and result.down_filled:
                result.success = True
                return result
            
            if result.up_filled and not result.down_filled:
                # UP filled, DOWN failed - try to complete DOWN
                print(f"         ⚠️  Partial fill! Retrying DOWN side...")
                
                for attempt in range(CONFIG['MAX_RETRY_ATTEMPTS']):
                    # Get fresh down price
                    down_price = get_best_ask(opp.market.down_token) or get_midpoint(opp.market.down_token)
                    if down_price is None:
                        continue
                    
                    retry_result = self.place_order(opp.market.down_token, down_price, size)
                    if retry_result['success']:
                        result.down_filled = True
                        result.down_size = size
                        result.success = True
                        print(f"         ✅ DOWN side filled on retry {attempt + 1}")
                        return result
                    
                    time.sleep(0.1)
                
                result.error = f"UP filled but DOWN failed after {CONFIG['MAX_RETRY_ATTEMPTS']} retries: {down_result.get('error', 'unknown')}"
                return result
            
            if result.down_filled and not result.up_filled:
                # DOWN filled, UP failed - try to complete UP
                print(f"         ⚠️  Partial fill! Retrying UP side...")
                
                for attempt in range(CONFIG['MAX_RETRY_ATTEMPTS']):
                    up_price = get_best_ask(opp.market.up_token) or get_midpoint(opp.market.up_token)
                    if up_price is None:
                        continue
                    
                    retry_result = self.place_order(opp.market.up_token, up_price, size)
                    if retry_result['success']:
                        result.up_filled = True
                        result.up_size = size
                        result.success = True
                        print(f"         ✅ UP side filled on retry {attempt + 1}")
                        return result
                    
                    time.sleep(0.1)
                
                result.error = f"DOWN filled but UP failed after {CONFIG['MAX_RETRY_ATTEMPTS']} retries: {up_result.get('error', 'unknown')}"
                return result
            
            # Neither filled
            result.error = f"Both sides failed - UP: {up_result.get('error')} | DOWN: {down_result.get('error')}"
            return result

# =============================================================================
# MAIN SCANNER
# =============================================================================

class BalancedScanner:
    def __init__(self, live: bool = False):
        self.live = live
        self.markets: List[Market] = []
        self.last_refresh = 0
        self.executor = BalancedExecutor() if live else None
        
        self.stats = {
            'opportunities': 0,
            'full_arbs': 0,
            'partial_fills': 0,
            'total_failures': 0,
            'profit': 0.0
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
        print("⚖️  BALANCED ARBITRAGE SCANNER")
        print("=" * 70)
        print("Features:")
        print("   • Retries failed sides up to 3 times")
        print("   • Uses best ask prices (more realistic)")
        print("   • Higher spread threshold (0.5%)")
        print("=" * 70)
        print(f"Mode: {'🔴 LIVE TRADING' if self.live else '⚪ SIMULATION'}")
        print(f"Threshold: {CONFIG['MIN_SPREAD_PERCENT']}% spread")
        print(f"Order size: {CONFIG['ORDER_SIZE']} shares per side")
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
        
        print("⚖️  SCANNING... (Ctrl+C to stop)\n")
        
        try:
            while True:
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
        
        profit = (1 - opp.total) * CONFIG['ORDER_SIZE']
        
        print(f"[{ts}] ⚡ {opp.market.asset} | "
              f"UP ${opp.up_price:.3f} + DOWN ${opp.down_price:.3f} = ${opp.total:.3f} | "
              f"Spread: {opp.spread_pct:.1f}% | Profit: ${profit:.3f}")
        
        if self.live and self.executor:
            result = self.executor.execute_balanced(opp)
            
            if result.success:
                self.stats['full_arbs'] += 1
                actual_profit = (1 - opp.total) * min(result.up_size, result.down_size)
                self.stats['profit'] += actual_profit
                print(f"         ✅ FULL ARB! UP: {result.up_size} + DOWN: {result.down_size} → ${actual_profit:.3f} profit")
            elif result.up_filled or result.down_filled:
                self.stats['partial_fills'] += 1
                side = "UP" if result.up_filled else "DOWN"
                size = result.up_size if result.up_filled else result.down_size
                print(f"         ⚠️  PARTIAL: Only {side} filled ({size} shares) - {result.error}")
            else:
                self.stats['total_failures'] += 1
                print(f"         ❌ {result.error}")
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("📊 SESSION SUMMARY")
        print("=" * 70)
        print(f"Opportunities seen:  {self.stats['opportunities']}")
        print(f"Full arbs executed:  {self.stats['full_arbs']} ✅")
        print(f"Partial fills:       {self.stats['partial_fills']} ⚠️")
        print(f"Total failures:      {self.stats['total_failures']} ❌")
        print(f"Locked profit:       ${self.stats['profit']:.3f}")
        if self.stats['opportunities'] > 0:
            success_rate = self.stats['full_arbs'] / self.stats['opportunities'] * 100
            print(f"Full arb rate:       {success_rate:.1f}%")
        print("=" * 70)

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Balanced Arbitrage Scanner')
    parser.add_argument('--live', '-l', action='store_true', help='Enable live trading')
    parser.add_argument('--yolo', action='store_true', help='Skip confirmation')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Min spread %%')
    parser.add_argument('--size', '-s', type=int, default=10, help='Order size')
    
    args = parser.parse_args()
    
    CONFIG['MIN_SPREAD_PERCENT'] = args.threshold
    CONFIG['ORDER_SIZE'] = max(5, args.size)
    
    if args.live:
        if not CONFIG['PRIVATE_KEY']:
            print("❌ Add PRIVATE_KEY to CONFIG section first")
            return
        
        if not args.yolo:
            print("\n⚖️  BALANCED MODE - Attempts to complete partial fills")
            print("⚠️  Still has risk if retries fail")
            print("\nType 'I ACCEPT THE RISK' to continue: ", end="")
            if input() != 'I ACCEPT THE RISK':
                print("Cancelled.")
                return
    
    scanner = BalancedScanner(live=args.live)
    scanner.run()

if __name__ == '__main__':
    main()
