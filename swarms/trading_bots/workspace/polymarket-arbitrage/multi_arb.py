#!/usr/bin/env python3
"""
Polymarket Multi-Market Arbitrage Scanner

Scans ALL crypto up/down markets simultaneously:
- Assets: BTC, ETH, SOL, XRP
- Timeframes: 15min, 1hour, 4hour

Executes arbitrage when UP + DOWN < $1.00

Usage:
    python3.11 multi_arb.py              # Simulation
    python3.11 multi_arb.py --live       # Live trading
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
    # API Endpoints
    'POLYMARKET_WEB': 'https://polymarket.com',
    'CLOB_API': 'https://clob.polymarket.com',
    'CHAIN_ID': 137,
    
    # Markets to scan
    'ASSETS': ['btc', 'eth', 'sol', 'xrp'],
    'TIMEFRAMES': ['15m', '1h', '4h'],
    
    # Scanner settings
    'MIN_SPREAD_PERCENT': 0.5,      # Minimum spread to trigger (0.5%)
    'SCAN_INTERVAL': 0.3,           # Seconds between full scans
    'MARKET_REFRESH': 120,          # Seconds between market discovery
    
    # Trading settings
    'ORDER_SIZE': 5,                # Shares per side
    'MAX_CONCURRENT': 4,            # Parallel price fetches
    
    # =========================================================================
    # CREDENTIALS - FILL THESE IN
    # =========================================================================
    'PRIVATE_KEY': '0xb92c6d5ae586a416cd45ecda3d8d7a1bb253777025fe31f863c8dcd9ea7e5bb0',              # Your private key (0x...)
    'SIGNATURE_TYPE': 1,            # 0=MetaMask, 1=Magic.link
    'FUNDER_ADDRESS': '0x1640782e9E71029B78555b9f23478712aC47396E',           # Your Polymarket address
}

# Epoch intervals for each timeframe
TIMEFRAME_SECONDS = {
    '15m': 900,
    '1h': 3600,
    '4h': 14400,
}

# =============================================================================
# CLOB Client Setup
# =============================================================================

CLOB_AVAILABLE = False
ClobClient = None

try:
    from py_clob_client.client import ClobClient as _ClobClient
    ClobClient = _ClobClient
    CLOB_AVAILABLE = True
except ImportError:
    print("⚠️  py-clob-client not installed - live trading disabled")

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
    potential_profit: float

# =============================================================================
# MARKET DISCOVERY
# =============================================================================

def get_epoch(timeframe: str) -> int:
    """Get current epoch for a timeframe"""
    seconds = TIMEFRAME_SECONDS.get(timeframe, 900)
    return (int(time.time()) // seconds) * seconds

def get_slug(asset: str, timeframe: str, epoch: int) -> str:
    """Generate market slug"""
    return f"{asset}-updown-{timeframe}-{epoch}"

def fetch_market(asset: str, timeframe: str) -> Optional[Market]:
    """Fetch a single market via web scraping"""
    epoch = get_epoch(timeframe)
    
    # Try current, next, and previous epochs
    for offset_mult in [0, 1, -1]:
        offset = offset_mult * TIMEFRAME_SECONDS.get(timeframe, 900)
        slug = get_slug(asset, timeframe, epoch + offset)
        url = f"{CONFIG['POLYMARKET_WEB']}/event/{slug}"
        
        try:
            resp = requests.get(
                url,
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=5
            )
            
            if resp.status_code != 200:
                continue
            
            match = re.search(
                r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                resp.text,
                re.DOTALL
            )
            
            if not match:
                continue
            
            data = json.loads(match.group(1))
            queries = (data.get('props', {})
                          .get('pageProps', {})
                          .get('dehydratedState', {})
                          .get('queries', []))
            
            for q in queries:
                state_data = q.get('state', {}).get('data')
                if isinstance(state_data, dict) and 'markets' in state_data:
                    for mkt in state_data['markets']:
                        tokens = mkt.get('clobTokenIds', [])
                        if len(tokens) >= 2:
                            return Market(
                                asset=asset.upper(),
                                timeframe=timeframe,
                                slug=slug,
                                question=mkt.get('question', ''),
                                up_token=tokens[0],
                                down_token=tokens[1]
                            )
        except:
            continue
    
    return None

def discover_all_markets() -> List[Market]:
    """Discover all active markets in parallel"""
    markets = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for asset in CONFIG['ASSETS']:
            for tf in CONFIG['TIMEFRAMES']:
                future = executor.submit(fetch_market, asset, tf)
                futures[future] = (asset, tf)
        
        for future in as_completed(futures):
            asset, tf = futures[future]
            try:
                market = future.result()
                if market:
                    markets.append(market)
            except:
                pass
    
    return markets

# =============================================================================
# PRICE FETCHING
# =============================================================================

def get_best_ask(token_id: str) -> Optional[float]:
    """Get best ask price"""
    try:
        resp = requests.get(
            f"{CONFIG['CLOB_API']}/book",
            params={'token_id': token_id},
            timeout=3
        )
        if resp.ok:
            book = resp.json()
            asks = book.get('asks', [])
            if asks:
                sorted_asks = sorted(asks, key=lambda x: float(x.get('price', 999)))
                return float(sorted_asks[0].get('price', 0))
    except:
        pass
    return None

def get_midpoint(token_id: str) -> Optional[float]:
    """Get midpoint price"""
    try:
        resp = requests.get(
            f"{CONFIG['CLOB_API']}/midpoint",
            params={'token_id': token_id},
            timeout=3
        )
        if resp.ok:
            return float(resp.json().get('mid', 0))
    except:
        pass
    return None

def check_market(market: Market) -> Optional[Opportunity]:
    """Check a single market for arbitrage opportunity"""
    # Get prices (prefer asks for realistic execution)
    up_price = get_best_ask(market.up_token)
    down_price = get_best_ask(market.down_token)
    
    # Fall back to midpoint
    if up_price is None:
        up_price = get_midpoint(market.up_token)
    if down_price is None:
        down_price = get_midpoint(market.down_token)
    
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
            spread_pct=spread_pct,
            potential_profit=(1 - total) * CONFIG['ORDER_SIZE']
        )
    
    return None

def scan_all_markets(markets: List[Market]) -> List[Opportunity]:
    """Scan all markets in parallel for opportunities"""
    opportunities = []
    
    with ThreadPoolExecutor(max_workers=CONFIG['MAX_CONCURRENT']) as executor:
        futures = {executor.submit(check_market, m): m for m in markets}
        
        for future in as_completed(futures):
            try:
                opp = future.result()
                if opp:
                    opportunities.append(opp)
            except:
                pass
    
    # Sort by spread (best first)
    opportunities.sort(key=lambda x: x.spread_pct, reverse=True)
    return opportunities

# =============================================================================
# TRADE EXECUTION
# =============================================================================

class Executor:
    def __init__(self):
        self.client = None
        self.lock = threading.Lock()
        self.initialized = False
    
    def initialize(self) -> bool:
        if not CLOB_AVAILABLE:
            return False
        if not CONFIG['PRIVATE_KEY']:
            return False
        
        try:
            self.client = ClobClient(
                host=CONFIG['CLOB_API'],
                key=CONFIG['PRIVATE_KEY'],
                chain_id=CONFIG['CHAIN_ID'],
                signature_type=CONFIG['SIGNATURE_TYPE'],
                funder=CONFIG['FUNDER_ADDRESS'] if CONFIG['FUNDER_ADDRESS'] else None
            )
            creds = self.client.derive_api_key()
            self.client.set_api_creds(creds)
            self.initialized = True
            return True
        except Exception as e:
            print(f"❌ Init failed: {e}")
            return False
    
    def execute(self, opp: Opportunity) -> Dict[str, Any]:
        """Execute arbitrage trade"""
        result = {'success': False, 'error': None, 'profit': 0}
        
        if not self.initialized:
            result['error'] = "Not initialized"
            return result
        
        with self.lock:  # Prevent concurrent executions
            try:
                # Re-check prices
                up_price = get_best_ask(opp.market.up_token)
                down_price = get_best_ask(opp.market.down_token)
                
                if up_price is None or down_price is None:
                    result['error'] = "Price unavailable"
                    return result
                
                total = up_price + down_price
                if total >= 1.0:
                    result['error'] = "Opportunity gone"
                    return result
                
                size = CONFIG['ORDER_SIZE']
                
                # Execute both legs
                up_order = self.client.create_order(
                    token_id=opp.market.up_token,
                    price=up_price,
                    size=size,
                    side="BUY"
                )
                self.client.post_order(up_order)
                
                down_order = self.client.create_order(
                    token_id=opp.market.down_token,
                    price=down_price,
                    size=size,
                    side="BUY"
                )
                self.client.post_order(down_order)
                
                result['success'] = True
                result['profit'] = (1 - total) * size
                return result
                
            except Exception as e:
                result['error'] = str(e)
                return result

# =============================================================================
# MAIN SCANNER
# =============================================================================

class MultiScanner:
    def __init__(self, live: bool = False):
        self.live = live
        self.markets: List[Market] = []
        self.last_refresh = 0
        self.executor = Executor() if live else None
        
        self.stats = {
            'scans': 0,
            'opportunities': 0,
            'trades': 0,
            'profit': 0.0
        }
    
    def refresh_markets(self):
        """Refresh all markets"""
        print("🔍 Discovering markets...")
        self.markets = discover_all_markets()
        self.last_refresh = time.time()
        
        print(f"   Found {len(self.markets)} active markets:")
        for m in self.markets:
            print(f"   • {m.asset} {m.timeframe}: {m.question[:50]}...")
        print()
    
    def run(self):
        print("=" * 70)
        print("🚀 POLYMARKET MULTI-MARKET ARBITRAGE SCANNER")
        print("=" * 70)
        print(f"Assets: {', '.join(CONFIG['ASSETS'])}")
        print(f"Timeframes: {', '.join(CONFIG['TIMEFRAMES'])}")
        print(f"Mode: {'🔴 LIVE' if self.live else '⚪ SIM'}")
        print(f"Min spread: {CONFIG['MIN_SPREAD_PERCENT']}%")
        print("=" * 70)
        
        if self.live and self.executor:
            if self.executor.initialize():
                print("✅ Trading client ready")
            else:
                print("❌ Trading disabled - running simulation")
                self.live = False
        
        self.refresh_markets()
        
        if not self.markets:
            print("❌ No markets found. Try again later.")
            return
        
        print("🎯 Scanning... (Ctrl+C to stop)\n")
        
        try:
            while True:
                self.stats['scans'] += 1
                
                # Refresh markets periodically
                if time.time() - self.last_refresh > CONFIG['MARKET_REFRESH']:
                    self.refresh_markets()
                
                # Scan for opportunities
                opps = scan_all_markets(self.markets)
                
                ts = datetime.now().strftime('%H:%M:%S')
                
                if opps:
                    for opp in opps:
                        self.stats['opportunities'] += 1
                        
                        print(f"[{ts}] ✅ {opp.market.asset} {opp.market.timeframe} | "
                              f"UP: ${opp.up_price:.3f} DOWN: ${opp.down_price:.3f} | "
                              f"Total: ${opp.total:.3f} | Spread: {opp.spread_pct:.2f}%")
                        print(f"        💰 Potential: ${opp.potential_profit:.3f} profit")
                        
                        if self.live and self.executor:
                            result = self.executor.execute(opp)
                            if result['success']:
                                self.stats['trades'] += 1
                                self.stats['profit'] += result['profit']
                                print(f"        ✅ EXECUTED! Profit: ${result['profit']:.3f}")
                            else:
                                print(f"        ❌ Failed: {result['error']}")
                else:
                    # Print compact status
                    prices_str = " | ".join([
                        f"{m.asset}:{get_midpoint(m.up_token) or 0:.2f}/{get_midpoint(m.down_token) or 0:.2f}"
                        for m in self.markets[:4]
                    ])
                    print(f"[{ts}] Scanning {len(self.markets)} markets... {prices_str}", end='\r')
                
                time.sleep(CONFIG['SCAN_INTERVAL'])
                
        except KeyboardInterrupt:
            print("\n\n" + "=" * 70)
            print("📊 SESSION SUMMARY")
            print("=" * 70)
            print(f"Scans:         {self.stats['scans']}")
            print(f"Opportunities: {self.stats['opportunities']}")
            print(f"Trades:        {self.stats['trades']}")
            print(f"Total profit:  ${self.stats['profit']:.3f}")
            print("=" * 70)

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-Market Arbitrage Scanner')
    parser.add_argument('--live', '-l', action='store_true', help='Enable live trading')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='Min spread %% (default: 0.5)')
    parser.add_argument('--size', '-s', type=int, default=5,
                        help='Order size (default: 5)')
    
    args = parser.parse_args()
    
    CONFIG['MIN_SPREAD_PERCENT'] = args.threshold
    CONFIG['ORDER_SIZE'] = max(5, args.size)
    
    if args.live and not CONFIG['PRIVATE_KEY']:
        print("❌ Live mode requires PRIVATE_KEY in CONFIG")
        return
    
    scanner = MultiScanner(live=args.live)
    scanner.run()

if __name__ == '__main__':
    main()
