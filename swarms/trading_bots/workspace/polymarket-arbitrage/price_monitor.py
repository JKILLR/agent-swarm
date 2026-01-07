#!/usr/bin/env python3
"""
Polymarket Real-Time Price Monitor
==================================

Connects to Polymarket's WebSocket feeds to detect:
1. Spread dislocations (UP + DOWN < $1)
2. Momentum lag (prediction price vs actual crypto price)

This is the "JaneStreet" momentum lag strategy detector.

SETUP:
    pip install websockets aiohttp requests rich

USAGE:
    python price_monitor.py                    # Monitor all 15-min markets
    python price_monitor.py --market <slug>    # Monitor specific market
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
import argparse

try:
    import websockets
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False
    print("⚠️ Install websockets: pip install websockets")

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("⚠️ Install aiohttp: pip install aiohttp")

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

import requests

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Polymarket WebSocket
    'WS_URL': 'wss://ws-subscriptions-clob.polymarket.com/ws/market',
    
    # Real-time data service
    'RTDS_URL': 'wss://data.polymarket.com/rtds',
    
    # APIs
    'GAMMA_API': 'https://gamma-api.polymarket.com',
    'CLOB_API': 'https://clob.polymarket.com',
    
    # Binance for actual crypto prices
    'BINANCE_WS': 'wss://stream.binance.com:9443/ws',
    
    # Alert thresholds
    'MIN_SPREAD_ALERT': 3.0,        # Alert if spread > 3%
    'MOMENTUM_LAG_THRESHOLD': 2.0,   # Alert if prediction lags by 2%+
    'PRICE_STALE_SECONDS': 30,       # Consider price stale after 30s
}

# Crypto symbols mapping
CRYPTO_MAP = {
    'btc': 'btcusdt',
    'bitcoin': 'btcusdt',
    'eth': 'ethusdt',
    'ethereum': 'ethusdt',
    'sol': 'solusdt',
    'solana': 'solusdt',
    'xrp': 'xrpusdt',
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MarketState:
    """Current state of a prediction market"""
    market_id: str
    question: str
    up_token_id: str
    down_token_id: str
    
    # Current prices
    up_bid: float = 0.0
    up_ask: float = 0.0
    down_bid: float = 0.0
    down_ask: float = 0.0
    
    # Computed
    up_mid: float = 0.0
    down_mid: float = 0.0
    combined_cost: float = 0.0
    spread_percent: float = 0.0
    
    # Timestamps
    last_update: float = 0.0
    
    # Underlying asset
    underlying_symbol: str = ""
    underlying_price: float = 0.0
    underlying_update: float = 0.0
    
    def update_computed(self):
        """Recalculate derived fields"""
        self.up_mid = (self.up_bid + self.up_ask) / 2 if self.up_ask > 0 else self.up_bid
        self.down_mid = (self.down_bid + self.down_ask) / 2 if self.down_ask > 0 else self.down_bid
        self.combined_cost = self.up_mid + self.down_mid
        if self.combined_cost > 0:
            self.spread_percent = (1.0 - self.combined_cost) / self.combined_cost * 100


@dataclass
class Alert:
    """Trading alert"""
    timestamp: datetime
    alert_type: str  # 'SPREAD' or 'MOMENTUM_LAG'
    market: str
    message: str
    opportunity_size: float = 0.0


# =============================================================================
# MARKET DATA FETCHER
# =============================================================================

class MarketDataFetcher:
    """Fetches initial market data from REST APIs"""
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_15min_markets(self) -> List[Dict]:
        """Fetch all active 15-minute crypto markets"""
        markets = []
        offset = 0
        
        while True:
            response = self.session.get(
                f"{CONFIG['GAMMA_API']}/markets",
                params={'limit': 100, 'offset': offset, 'active': 'true'},
                timeout=30
            )
            response.raise_for_status()
            batch = response.json()
            
            if not batch:
                break
            
            # Filter for 15-min crypto markets
            for m in batch:
                q = m.get('question', '').lower()
                if ('15 min' in q or '15min' in q or '15-min' in q) and \
                   any(c in q for c in ['btc', 'bitcoin', 'eth', 'ethereum', 'sol', 'solana', 'xrp']):
                    markets.append(m)
            
            offset += 100
            if len(batch) < 100:
                break
        
        return markets
    
    def get_market_by_slug(self, slug: str) -> Optional[Dict]:
        """Get a specific market by slug"""
        response = self.session.get(
            f"{CONFIG['GAMMA_API']}/markets",
            params={'slug': slug},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data[0] if data else None
    
    def get_prices(self, token_ids: List[str]) -> Dict[str, Dict]:
        """Get current prices for multiple tokens"""
        prices = {}
        for token_id in token_ids:
            try:
                # Get midpoint
                response = self.session.get(
                    f"{CONFIG['CLOB_API']}/midpoint",
                    params={'token_id': token_id},
                    timeout=10
                )
                if response.ok:
                    prices[token_id] = {'mid': float(response.json().get('mid', 0))}
                
                # Get spread
                response = self.session.get(
                    f"{CONFIG['CLOB_API']}/spread",
                    params={'token_id': token_id},
                    timeout=10
                )
                if response.ok:
                    data = response.json()
                    if token_id in prices:
                        prices[token_id].update({
                            'bid': float(data.get('bid', 0)),
                            'ask': float(data.get('ask', 0))
                        })
                
                time.sleep(0.1)
            except Exception:
                pass
        
        return prices


# =============================================================================
# WEBSOCKET PRICE MONITOR
# =============================================================================

class PriceMonitor:
    """Real-time price monitoring via WebSockets"""
    
    def __init__(self):
        self.markets: Dict[str, MarketState] = {}
        self.alerts: List[Alert] = []
        self.running = False
        self.console = Console() if RICH_AVAILABLE else None
        self.fetcher = MarketDataFetcher()
        
        # Callbacks for alerts
        self.on_spread_alert: Optional[Callable] = None
        self.on_lag_alert: Optional[Callable] = None
    
    def add_market(self, market: Dict):
        """Add a market to monitor"""
        tokens = market.get('clobTokenIds', [])
        outcomes = market.get('outcomes', [])
        
        if len(tokens) != 2:
            return
        
        # Determine UP vs DOWN
        outcome_lower = [o.lower() for o in outcomes]
        if 'up' in outcome_lower[0] or 'yes' in outcome_lower[0]:
            up_token, down_token = tokens[0], tokens[1]
        else:
            up_token, down_token = tokens[1], tokens[0]
        
        # Detect underlying crypto
        question = market.get('question', '').lower()
        underlying = ""
        for crypto, symbol in CRYPTO_MAP.items():
            if crypto in question:
                underlying = symbol
                break
        
        state = MarketState(
            market_id=market.get('conditionId', ''),
            question=market.get('question', ''),
            up_token_id=up_token,
            down_token_id=down_token,
            underlying_symbol=underlying
        )
        
        self.markets[state.market_id] = state
    
    async def fetch_initial_prices(self):
        """Fetch initial prices for all markets"""
        all_tokens = []
        for state in self.markets.values():
            all_tokens.extend([state.up_token_id, state.down_token_id])
        
        prices = self.fetcher.get_prices(all_tokens)
        
        for state in self.markets.values():
            up_data = prices.get(state.up_token_id, {})
            down_data = prices.get(state.down_token_id, {})
            
            state.up_mid = up_data.get('mid', 0)
            state.up_bid = up_data.get('bid', state.up_mid)
            state.up_ask = up_data.get('ask', state.up_mid)
            
            state.down_mid = down_data.get('mid', 0)
            state.down_bid = down_data.get('bid', state.down_mid)
            state.down_ask = down_data.get('ask', state.down_mid)
            
            state.update_computed()
            state.last_update = time.time()
    
    def check_alerts(self, state: MarketState):
        """Check if current state triggers any alerts"""
        now = datetime.now(timezone.utc)
        
        # Spread alert
        if state.spread_percent >= CONFIG['MIN_SPREAD_ALERT']:
            alert = Alert(
                timestamp=now,
                alert_type='SPREAD',
                market=state.question[:50],
                message=f"Spread: {state.spread_percent:.2f}% | UP: ${state.up_mid:.3f} + DOWN: ${state.down_mid:.3f} = ${state.combined_cost:.3f}",
                opportunity_size=state.spread_percent
            )
            self.alerts.append(alert)
            
            if self.on_spread_alert:
                self.on_spread_alert(alert, state)
        
        # Momentum lag alert (if we have underlying price)
        if state.underlying_price > 0 and state.last_update > 0:
            # Calculate implied probability from prediction prices
            # UP price near 0.5 = market thinks 50/50
            # If actual price moved significantly, the prediction should adjust
            pass  # TODO: Implement momentum lag detection
    
    def render_dashboard(self) -> Table:
        """Render current state as Rich table"""
        table = Table(title="🔴 LIVE MARKET MONITOR")
        
        table.add_column("Market", width=40)
        table.add_column("UP $", width=8)
        table.add_column("DOWN $", width=8)
        table.add_column("Total", width=8)
        table.add_column("Spread %", width=10)
        table.add_column("Updated", width=10)
        
        for state in sorted(self.markets.values(), key=lambda x: -x.spread_percent):
            age = time.time() - state.last_update if state.last_update > 0 else 999
            age_str = f"{age:.0f}s ago" if age < 60 else f"{age/60:.0f}m ago"
            
            spread_color = "green" if state.spread_percent >= CONFIG['MIN_SPREAD_ALERT'] else "white"
            
            table.add_row(
                state.question[:38] + "..." if len(state.question) > 40 else state.question,
                f"{state.up_mid:.3f}",
                f"{state.down_mid:.3f}",
                f"{state.combined_cost:.3f}",
                f"[{spread_color}]{state.spread_percent:.2f}%[/]",
                age_str
            )
        
        return table
    
    def render_alerts(self) -> Panel:
        """Render recent alerts"""
        recent = self.alerts[-5:] if self.alerts else []
        
        if not recent:
            return Panel("[dim]No alerts yet[/dim]", title="🚨 Alerts")
        
        lines = []
        for alert in reversed(recent):
            emoji = "💰" if alert.alert_type == 'SPREAD' else "⚡"
            lines.append(f"{emoji} [{alert.timestamp.strftime('%H:%M:%S')}] {alert.message}")
        
        return Panel("\n".join(lines), title="🚨 Recent Alerts")
    
    async def run_polling_monitor(self, interval: float = 5.0):
        """Polling-based monitor (fallback when WebSocket unavailable)"""
        self.running = True
        
        print(f"\n📡 Starting polling monitor (interval: {interval}s)")
        print(f"   Monitoring {len(self.markets)} markets")
        print("   Press Ctrl+C to stop\n")
        
        await self.fetch_initial_prices()
        
        try:
            if RICH_AVAILABLE:
                with Live(self.render_dashboard(), refresh_per_second=1) as live:
                    while self.running:
                        await self.fetch_initial_prices()
                        
                        for state in self.markets.values():
                            self.check_alerts(state)
                        
                        live.update(self.render_dashboard())
                        await asyncio.sleep(interval)
            else:
                while self.running:
                    await self.fetch_initial_prices()
                    
                    # Simple text output
                    print(f"\n{'='*60}")
                    print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
                    print(f"{'='*60}")
                    
                    for state in sorted(self.markets.values(), key=lambda x: -x.spread_percent):
                        status = "🟢" if state.spread_percent >= CONFIG['MIN_SPREAD_ALERT'] else "⚪"
                        print(f"{status} {state.spread_percent:5.2f}% | {state.question[:50]}")
                    
                    await asyncio.sleep(interval)
                    
        except KeyboardInterrupt:
            self.running = False
            print("\n\n👋 Monitor stopped")
    
    async def run(self, use_websocket: bool = False, poll_interval: float = 5.0):
        """Main entry point"""
        if use_websocket and WS_AVAILABLE:
            # TODO: Implement WebSocket-based monitoring
            # For now, fall back to polling
            await self.run_polling_monitor(poll_interval)
        else:
            await self.run_polling_monitor(poll_interval)


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Polymarket Real-Time Price Monitor")
    parser.add_argument('--market', '-m', type=str, help='Specific market slug to monitor')
    parser.add_argument('--interval', '-i', type=float, default=10.0, help='Poll interval in seconds')
    parser.add_argument('--alert-threshold', '-a', type=float, default=3.0, help='Alert spread threshold %')
    
    args = parser.parse_args()
    
    CONFIG['MIN_SPREAD_ALERT'] = args.alert_threshold
    
    monitor = PriceMonitor()
    fetcher = MarketDataFetcher()
    
    print("📡 Fetching 15-minute crypto markets...")
    
    if args.market:
        market = fetcher.get_market_by_slug(args.market)
        if market:
            monitor.add_market(market)
        else:
            print(f"❌ Market not found: {args.market}")
            return
    else:
        markets = fetcher.get_15min_markets()
        print(f"✅ Found {len(markets)} markets")
        
        for m in markets:
            monitor.add_market(m)
    
    if not monitor.markets:
        print("❌ No markets to monitor")
        return
    
    print(f"👁️ Monitoring {len(monitor.markets)} markets")
    
    await monitor.run(poll_interval=args.interval)


if __name__ == "__main__":
    asyncio.run(main())
