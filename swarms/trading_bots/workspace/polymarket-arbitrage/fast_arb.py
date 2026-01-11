#!/usr/bin/env python3
"""
FAST MODE Polymarket Arbitrage Scanner

⚠️  WARNING: This version skips safety checks for speed!
    Risk of partial fills and slippage losses.
    Only use with money you can afford to lose.
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
    
    'MIN_SPREAD_PERCENT': 0.3,
    'SCAN_INTERVAL': 0.1,
    'MARKET_REFRESH': 300,
    'REQUEST_TIMEOUT': 2,
    
    'ORDER_SIZE': 5,
    
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
    BUY = "BUY"  # Placeholder

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
# FAST EXECUTION - FIXED OrderArgs syntax
# =============================================================================

class FastExecutor:
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
    
    def execute_fast(self, opp: Opportunity) -> Dict[str, Any]:
        """Execute using correct OrderArgs syntax"""
        result = {'success': False, 'error': None, 'cost': 0, 'profit': 0}
        
        if not self.initialized:
            result['error'] = "Not initialized"
            return result
        
        with self.lock:
            size = float(CONFIG['ORDER_SIZE'])
            up_result = None
            down_result = None
            errors = []
            
            def buy_up():
                nonlocal up_result
                try:
                    # FIXED: Use OrderArgs object
                    order_args = OrderArgs(
                        price=opp.up_price,
                        size=size,
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
                    # FIXED: Use OrderArgs object
                    order_args = OrderArgs(
                        price=opp.down_price,
                        size=size,
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
            result['cost'] = opp.total * size
            result['profit'] = (1 - opp.total) * size
            return result

# =============================================================================
# MAIN SCANNER
# =============================================================================

class FastScanner:
    def __init__(self, live: bool = False):
        self.live = live
        self.markets: List[Market] = []
        self.last_refresh = 0
        self.executor = FastExecutor() if live else None
        
        self.stats = {
            'scans': 0,
            'opportunities': 0,
            'attempts': 0,
            'successes': 0,
            'failures': 0,
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
        print("⚡ FAST MODE ARBITRAGE SCANNER (FIXED)")
        print("=" * 70)
        print("⚠️  WARNING: Safety checks DISABLED for speed!")
        print("=" * 70)
        print(f"Mode: {'🔴 LIVE TRADING' if self.live else '⚪ SIMULATION'}")
        print(f"Threshold: {CONFIG['MIN_SPREAD_PERCENT']}% spread")
        print(f"Order size: {CONFIG['ORDER_SIZE']} shares")
        print("=" * 70)
        
        if self.live:
            if self.executor and self.executor.initialize():
                print("✅ Trading client ready - LIVE ORDERS ENABLED")
            else:
                print("❌ Failed to initialize - running simulation")
                self.live = False
        
        self.refresh_markets()
        
        if not self.markets:
            print("❌ No markets found")
            return
        
        print("⚡ FAST SCANNING... (Ctrl+C to stop)\n")
        
        try:
            while True:
                self.stats['scans'] += 1
                
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
            self.stats['attempts'] += 1
            result = self.executor.execute_fast(opp)
            
            if result['success']:
                self.stats['successes'] += 1
                self.stats['profit'] += result['profit']
                print(f"         ✅ EXECUTED! Cost: ${result['cost']:.2f} → Profit: ${result['profit']:.3f}")
            else:
                self.stats['failures'] += 1
                print(f"         ❌ {result['error']}")
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("📊 SESSION SUMMARY")
        print("=" * 70)
        print(f"Opportunities:    {self.stats['opportunities']}")
        print(f"Trade attempts:   {self.stats['attempts']}")
        print(f"Successful:       {self.stats['successes']}")
        print(f"Failed:           {self.stats['failures']}")
        print(f"Total profit:     ${self.stats['profit']:.3f}")
        print("=" * 70)

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='FAST MODE Arbitrage Scanner')
    parser.add_argument('--live', '-l', action='store_true', help='Enable live trading')
    parser.add_argument('--yolo', action='store_true', help='Skip confirmation')
    parser.add_argument('--threshold', '-t', type=float, default=0.3, help='Min spread %%')
    parser.add_argument('--size', '-s', type=int, default=5, help='Order size')
    
    args = parser.parse_args()
    
    CONFIG['MIN_SPREAD_PERCENT'] = args.threshold
    CONFIG['ORDER_SIZE'] = max(5, args.size)
    
    if args.live:
        if not CONFIG['PRIVATE_KEY']:
            print("❌ Add PRIVATE_KEY to CONFIG section first")
            return
        
        if not args.yolo:
            print("\n⚠️  LIVE TRADING - Type 'I ACCEPT THE RISK' to continue: ", end="")
            if input() != 'I ACCEPT THE RISK':
                print("Cancelled.")
                return
    
    scanner = FastScanner(live=args.live)
    scanner.run()

if __name__ == '__main__':
    main()
