#!/usr/bin/env python3
"""
Polymarket BTC 15-Minute Arbitrage Scanner & Executor

Monitors Bitcoin Up/Down 15-minute prediction markets on Polymarket
and executes arbitrage trades when combined cost of UP + DOWN < $1.00

Usage:
    python3.11 polymarket_btc_arb.py              # Simulation mode
    python3.11 polymarket_btc_arb.py --live       # Live trading
    python3.11 polymarket_btc_arb.py -t 0.99      # Custom threshold (1% spread)
"""

import requests
import re
import json
import time
import argparse
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

# =============================================================================
# CONFIGURATION - EDIT THESE VALUES
# =============================================================================

CONFIG = {
    # API Endpoints (don't change these)
    'GAMMA_API': 'https://gamma-api.polymarket.com',
    'CLOB_API': 'https://clob.polymarket.com',
    'POLYMARKET_WEB': 'https://polymarket.com',
    'CHAIN_ID': 137,  # Polygon mainnet
    
    # Scanner Settings
    'MIN_SPREAD_PERCENT': 0.5,      # Minimum spread to trigger (0.5% = $0.005 profit per share)
    'SCAN_INTERVAL': 0.5,           # Seconds between price checks
    'MARKET_REFRESH': 60,           # Seconds between market discovery
    
    # Trading Settings
    'ORDER_SIZE': 5,                # Shares per side (minimum 5)
    'MAX_SLIPPAGE': 0.02,           # Max slippage tolerance (2 cents)
    
    # =========================================================================
    # EXECUTION SETTINGS - FILL THESE IN FOR LIVE TRADING
    # =========================================================================
    
    # Your private key from Polymarket (starts with 0x)
    # To export: Polymarket → Profile → Settings → Export Private Key
    'PRIVATE_KEY': '0xb92c6d5ae586a416cd45ecda3d8d7a1bb253777025fe31f863c8dcd9ea7e5bb0',  # Example: '0xabcd1234...'
    
    # Signature type:
    #   0 = EOA wallet (MetaMask, hardware wallet)
    #   1 = Magic.link (email login on Polymarket) - MOST COMMON
    #   2 = Gnosis Safe
    'SIGNATURE_TYPE': 1,
    
    # Your Polymarket proxy/deposit address (only needed for Magic.link)
    # To find: Polymarket → Profile → Copy Address (or check Deposit page)
    'FUNDER_ADDRESS': '0x1640782e9E71029B78555b9f23478712aC47396E',  # Example: '0x5678efgh...'
}

# =============================================================================
# Check for optional dependencies
# =============================================================================

CLOB_CLIENT_AVAILABLE = False
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, BookParams
    CLOB_CLIENT_AVAILABLE = True
except ImportError:
    print("⚠️  py-clob-client not installed - live trading disabled")
    print("   To enable: pip install py-clob-client")

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MarketInfo:
    """Current market information"""
    slug: str
    question: str
    up_token: str
    down_token: str
    end_time: str = ""

@dataclass 
class PriceData:
    """Current price data"""
    up_price: float
    down_price: float
    total: float
    spread: float
    spread_pct: float
    is_opportunity: bool

# =============================================================================
# MARKET DISCOVERY (via web scraping)
# =============================================================================

def get_current_epoch() -> int:
    """Get current 15-minute epoch (rounded down to nearest 900 seconds)"""
    return (int(time.time()) // 900) * 900

def get_market_slug(epoch: Optional[int] = None) -> str:
    """Generate market slug for given epoch"""
    if epoch is None:
        epoch = get_current_epoch()
    return f"btc-updown-15m-{epoch}"

def fetch_market_from_web(slug: str) -> Optional[MarketInfo]:
    """
    Fetch market data by scraping Polymarket website.
    The 15-min markets aren't in the API, only on the web.
    """
    url = f"{CONFIG['POLYMARKET_WEB']}/event/{slug}"
    
    try:
        resp = requests.get(
            url,
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=10
        )
        
        if resp.status_code != 200:
            return None
        
        # Extract __NEXT_DATA__ JSON payload
        match = re.search(
            r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
            resp.text,
            re.DOTALL
        )
        
        if not match:
            return None
        
        data = json.loads(match.group(1))
        queries = (data.get('props', {})
                      .get('pageProps', {})
                      .get('dehydratedState', {})
                      .get('queries', []))
        
        for q in queries:
            state_data = q.get('state', {}).get('data')
            if isinstance(state_data, dict) and 'markets' in state_data:
                for market in state_data['markets']:
                    tokens = market.get('clobTokenIds', [])
                    if len(tokens) >= 2:
                        return MarketInfo(
                            slug=slug,
                            question=market.get('question', ''),
                            up_token=tokens[0],
                            down_token=tokens[1],
                            end_time=market.get('endDate', '')
                        )
        return None
        
    except Exception as e:
        print(f"Error fetching market: {e}")
        return None

def discover_current_market() -> Optional[MarketInfo]:
    """Find the currently active 15-minute BTC market"""
    current_epoch = get_current_epoch()
    
    # Try current, next, and previous epochs
    for offset in [0, 900, -900]:
        epoch = current_epoch + offset
        market = fetch_market_from_web(get_market_slug(epoch))
        if market:
            return market
    
    return None

# =============================================================================
# PRICE FETCHING
# =============================================================================

def get_midpoint(token_id: str) -> Optional[float]:
    """Get midpoint price for a token"""
    try:
        resp = requests.get(
            f"{CONFIG['CLOB_API']}/midpoint",
            params={'token_id': token_id},
            timeout=5
        )
        if resp.ok:
            return float(resp.json().get('mid', 0))
    except:
        pass
    return None

def get_best_ask(token_id: str) -> Optional[float]:
    """Get best ask price (what you'd actually pay)"""
    try:
        resp = requests.get(
            f"{CONFIG['CLOB_API']}/book",
            params={'token_id': token_id},
            timeout=5
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

def get_prices(market: MarketInfo, use_asks: bool = True) -> Optional[PriceData]:
    """Get current prices and calculate spread"""
    if use_asks:
        up_price = get_best_ask(market.up_token)
        down_price = get_best_ask(market.down_token)
    else:
        up_price = get_midpoint(market.up_token)
        down_price = get_midpoint(market.down_token)
    
    # Fall back to midpoint if asks unavailable
    if up_price is None:
        up_price = get_midpoint(market.up_token)
    if down_price is None:
        down_price = get_midpoint(market.down_token)
    
    if up_price is None or down_price is None:
        return None
    
    total = up_price + down_price
    spread = 1.0 - total
    spread_pct = spread * 100
    
    return PriceData(
        up_price=up_price,
        down_price=down_price,
        total=total,
        spread=spread,
        spread_pct=spread_pct,
        is_opportunity=(spread_pct >= CONFIG['MIN_SPREAD_PERCENT'])
    )

# =============================================================================
# TRADE EXECUTION
# =============================================================================

class TradeExecutor:
    """Handles live trade execution via Polymarket CLOB API"""
    
    def __init__(self):
        self.client = None
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the CLOB client with credentials"""
        if not CLOB_CLIENT_AVAILABLE:
            print("❌ py-clob-client not installed")
            return False
        
        if not CONFIG['PRIVATE_KEY']:
            print("❌ No PRIVATE_KEY configured")
            return False
        
        try:
            self.client = ClobClient(
                host=CONFIG['CLOB_API'],
                key=CONFIG['PRIVATE_KEY'],
                chain_id=CONFIG['CHAIN_ID'],
                signature_type=CONFIG['SIGNATURE_TYPE'],
                funder=CONFIG['FUNDER_ADDRESS'] if CONFIG['FUNDER_ADDRESS'] else None
            )
            
            # Derive and set API credentials
            creds = self.client.derive_api_key()
            self.client.set_api_creds(creds)
            
            self.initialized = True
            print("✅ Trading client initialized")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            return False
    
    def execute_arbitrage(self, market: MarketInfo, prices: PriceData, size: int) -> Dict:
        """
        Execute arbitrage: buy both UP and DOWN sides.
        Returns result dict with success status and details.
        """
        result = {
            'success': False,
            'up_order': None,
            'down_order': None,
            'total_cost': 0,
            'expected_profit': 0,
            'error': None
        }
        
        if not self.initialized:
            result['error'] = "Client not initialized"
            return result
        
        try:
            # Double-check prices haven't moved
            current_prices = get_prices(market)
            if not current_prices or not current_prices.is_opportunity:
                result['error'] = "Opportunity disappeared"
                return result
            
            # Create and submit UP order
            up_order = self.client.create_order(
                token_id=market.up_token,
                price=current_prices.up_price,
                size=size,
                side="BUY"
            )
            up_result = self.client.post_order(up_order)
            
            # Create and submit DOWN order
            down_order = self.client.create_order(
                token_id=market.down_token,
                price=current_prices.down_price,
                size=size,
                side="BUY"
            )
            down_result = self.client.post_order(down_order)
            
            result['success'] = True
            result['up_order'] = up_result
            result['down_order'] = down_result
            result['total_cost'] = current_prices.total * size
            result['expected_profit'] = current_prices.spread * size
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            return result

# =============================================================================
# MAIN SCANNER
# =============================================================================

class ArbitrageScanner:
    """Main scanner that monitors markets and executes trades"""
    
    def __init__(self, live_mode: bool = False, threshold: float = 0.995):
        self.live_mode = live_mode
        self.threshold = threshold
        self.current_market: Optional[MarketInfo] = None
        self.last_refresh = 0
        self.executor = TradeExecutor() if live_mode else None
        
        # Statistics
        self.stats = {
            'scans': 0,
            'opportunities': 0,
            'trades': 0,
            'invested': 0.0,
            'expected_profit': 0.0
        }
    
    def refresh_market(self) -> bool:
        """Refresh current market data"""
        market = discover_current_market()
        if market:
            if self.current_market is None or market.slug != self.current_market.slug:
                print(f"\n📊 Market: {market.question}")
                print(f"   Slug: {market.slug}")
            self.current_market = market
            self.last_refresh = time.time()
            return True
        return False
    
    def print_status(self, prices: PriceData):
        """Print current price status"""
        icon = "✅ ARB!" if prices.is_opportunity else "❌"
        mode = "🔴 LIVE" if self.live_mode else "⚪ SIM"
        ts = datetime.now().strftime('%H:%M:%S')
        
        print(f"[{ts}] UP: ${prices.up_price:.3f} | "
              f"DOWN: ${prices.down_price:.3f} | "
              f"Total: ${prices.total:.3f} | "
              f"Spread: {prices.spread_pct:+.2f}% {icon} [{mode}]")
        
        if prices.is_opportunity:
            size = CONFIG['ORDER_SIZE']
            cost = prices.total * size
            profit = prices.spread * size
            print(f"   💰 Buy {size} each @ ${cost:.2f} → Payout ${size:.2f} = ${profit:.3f} profit")
    
    def run(self):
        """Main scanning loop"""
        print("=" * 70)
        print("🚀 POLYMARKET BTC 15-MIN ARBITRAGE SCANNER")
        print("=" * 70)
        print(f"Mode: {'🔴 LIVE TRADING' if self.live_mode else '⚪ SIMULATION'}")
        print(f"Threshold: ${self.threshold:.3f} (need {(1-self.threshold)*100:.1f}%+ spread)")
        print(f"Order size: {CONFIG['ORDER_SIZE']} shares per side")
        print("=" * 70)
        
        # Initialize executor if live mode
        if self.live_mode and self.executor:
            if not self.executor.initialize():
                print("\n❌ Failed to initialize trading. Running in simulation mode.")
                self.live_mode = False
        
        # Discover initial market
        print("\n🔍 Discovering current market...")
        if not self.refresh_market():
            print("❌ No active market found. Try during US hours (9am-5pm ET)")
            return
        
        print(f"\n🎯 Scanning... (Ctrl+C to stop)\n")
        
        try:
            while True:
                self.stats['scans'] += 1
                
                # Refresh market periodically
                if time.time() - self.last_refresh > CONFIG['MARKET_REFRESH']:
                    self.refresh_market()
                
                # Get prices
                if self.current_market:
                    prices = get_prices(self.current_market)
                    
                    if prices:
                        self.print_status(prices)
                        
                        # Check for opportunity
                        if prices.is_opportunity:
                            self.stats['opportunities'] += 1
                            
                            # Execute if live mode
                            if self.live_mode and self.executor:
                                result = self.executor.execute_arbitrage(
                                    self.current_market,
                                    prices,
                                    CONFIG['ORDER_SIZE']
                                )
                                
                                if result['success']:
                                    self.stats['trades'] += 1
                                    self.stats['invested'] += result['total_cost']
                                    self.stats['expected_profit'] += result['expected_profit']
                                    print(f"   ✅ TRADED! Cost: ${result['total_cost']:.2f}, "
                                          f"Expected profit: ${result['expected_profit']:.3f}")
                                else:
                                    print(f"   ❌ Trade failed: {result['error']}")
                
                time.sleep(CONFIG['SCAN_INTERVAL'])
                
        except KeyboardInterrupt:
            self.print_summary()
    
    def print_summary(self):
        """Print session summary"""
        print("\n" + "=" * 70)
        print("📊 SESSION SUMMARY")
        print("=" * 70)
        print(f"Total scans:        {self.stats['scans']}")
        print(f"Opportunities:      {self.stats['opportunities']}")
        print(f"Trades executed:    {self.stats['trades']}")
        print(f"Total invested:     ${self.stats['invested']:.2f}")
        print(f"Expected profit:    ${self.stats['expected_profit']:.3f}")
        print("=" * 70)

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Polymarket BTC 15-min Arbitrage Scanner')
    
    parser.add_argument('--live', '-l', action='store_true',
                        help='Enable live trading (requires credentials in CONFIG)')
    
    parser.add_argument('--threshold', '-t', type=float, default=0.995,
                        help='Max combined cost to trigger (default: 0.995 = 0.5%% spread)')
    
    parser.add_argument('--size', '-s', type=int, default=5,
                        help='Shares per side (default: 5, minimum: 5)')
    
    args = parser.parse_args()
    
    # Update config
    CONFIG['ORDER_SIZE'] = max(5, args.size)
    CONFIG['MIN_SPREAD_PERCENT'] = (1 - args.threshold) * 100
    
    # Validate live mode
    if args.live:
        if not CONFIG['PRIVATE_KEY']:
            print("❌ Live mode requires PRIVATE_KEY in CONFIG section")
            print("   Edit this file and add your credentials, or run without --live")
            return
        if not CLOB_CLIENT_AVAILABLE:
            print("❌ Live mode requires py-clob-client")
            print("   Install with: pip install py-clob-client")
            return
    
    # Run scanner
    scanner = ArbitrageScanner(live_mode=args.live, threshold=args.threshold)
    scanner.run()

if __name__ == '__main__':
    main()
