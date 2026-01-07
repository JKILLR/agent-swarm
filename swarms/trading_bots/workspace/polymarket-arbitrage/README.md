# Polymarket 15-Minute Crypto Arbitrage Scanner

A Python toolkit for finding and executing arbitrage opportunities on Polymarket's 15-minute crypto prediction markets, inspired by the "JaneStreetIndia" / "Account88888" strategy that generated $324K+ profit.

## Strategy Overview

### Delta-Neutral Spread Arbitrage

The core strategy exploits price inefficiencies in binary prediction markets:

```
When:  UP_price + DOWN_price < $1.00
Then:  Buying both sides guarantees profit

Example:
  • UP shares:   $0.48
  • DOWN shares: $0.46
  • Combined:    $0.94
  
  One side MUST win and pay $1.00
  Guaranteed profit: $0.06 per share (6.4% return)
```

### Why This Edge Exists

1. **No professional market makers** - 15-min markets are too new/small for sophisticated players
2. **Retail guesses direction** - Creates spread dislocations when they favor one side
3. **Volatility compression** - During consolidation, both sides drift toward 50¢
4. **Resolution speed** - 15 minutes is too fast for traditional arb bots

### Position Sizing

The visualization from the original posts shows:
- Position cost: 84-96 cents per share pair
- Resolution payout: $1.00
- Daily P&L: $5K-$33K
- Average hold: 8-12 minutes

## Installation

```bash
# Clone or download this repository
cd polymarket-arbitrage

# Install dependencies
pip install -r requirements.txt

# Or install manually:
pip install py-clob-client requests websockets aiohttp rich
```

## Files

| File | Purpose |
|------|---------|
| `polymarket_arb.py` | Main arbitrage scanner |
| `price_monitor.py` | Real-time WebSocket price monitor |
| `execute_trade.py` | Trade execution helper (requires wallet) |
| `requirements.txt` | Python dependencies |

## Quick Start

### 1. Scan for Opportunities

```bash
# Basic scan
python polymarket_arb.py

# With custom threshold (3% minimum spread)
python polymarket_arb.py --min-spread 3.0

# Continuous monitoring
python polymarket_arb.py --watch --interval 30

# JSON output for programmatic use
python polymarket_arb.py --json
```

### 2. Real-Time Price Monitor

```bash
# Monitor all 15-minute markets
python price_monitor.py

# Monitor specific market
python price_monitor.py --market "bitcoin-up-or-down-dec-31-10pm-et"

# Custom alert threshold
python price_monitor.py --alert-threshold 4.0 --interval 5
```

### 3. Execute Trades (Requires Wallet Setup)

```bash
# Set environment variables
export POLY_PRIVATE_KEY="your-private-key"
export POLY_FUNDER_ADDRESS="your-wallet-address"

# Dry run (no actual trades)
python execute_trade.py --opportunity-id <id> --size 100 --dry-run

# Live execution
python execute_trade.py --opportunity-id <id> --size 100
```

## Wallet Setup

### Getting Your Private Key

**Option A: Magic/Email Login (Recommended)**
1. Go to https://reveal.magic.link/polymarket
2. Enter your email
3. Approve the request
4. Copy your private key

**Option B: Browser Wallet (MetaMask)**
1. Export private key from MetaMask
2. Note: Requires setting allowances manually

### Setting Allowances

Before trading, your wallet needs to approve Polymarket's contracts:

```python
from py_clob_client.client import ClobClient

client = ClobClient(
    host="https://clob.polymarket.com",
    key="YOUR_PRIVATE_KEY",
    chain_id=137,
    signature_type=1,
    funder="YOUR_WALLET_ADDRESS"
)

# Set allowances (one-time)
client.set_allowances()
```

## API Reference

### Gamma API (Market Discovery)
- Base URL: `https://gamma-api.polymarket.com`
- No authentication required
- Endpoints:
  - `GET /markets` - List all markets
  - `GET /events` - List events (grouped markets)
  - `GET /markets?slug=<slug>` - Get specific market

### CLOB API (Order Book)
- Base URL: `https://clob.polymarket.com`
- Authentication required for trading
- Endpoints:
  - `GET /midpoint?token_id=<id>` - Get midpoint price
  - `GET /spread?token_id=<id>` - Get bid/ask spread
  - `GET /book?token_id=<id>` - Get full order book
  - `POST /order` - Place order (requires auth)

### WebSocket (Real-Time Data)
- URL: `wss://ws-subscriptions-clob.polymarket.com/ws/market`
- Subscribe to order book updates
- Real-time price feeds

## Configuration

Edit `CONFIG` in `polymarket_arb.py`:

```python
CONFIG = {
    'MIN_SPREAD_PERCENT': 2.0,      # Minimum spread to flag
    'MIN_LIQUIDITY': 1000,          # Minimum market liquidity
    'SCAN_INTERVAL': 30,            # Seconds between scans
    'MAX_MARKETS_TO_SCAN': 500,     # Performance limit
    'REQUEST_DELAY': 0.1,           # Rate limiting
}
```

## Risk Management

### Position Limits
- Never exceed 10% of market liquidity
- Start with small positions ($100-500)
- Scale up as you verify execution

### Slippage
- Spreads can change rapidly
- Your order might not fill at expected price
- Use limit orders, not market orders

### Resolution Risk
- Oracle delays can occur
- Read market rules carefully
- Some markets have edge cases

### Liquidity Risk
- Thin markets = hard to exit
- Check bid depth before entering
- Prefer markets with >$5K liquidity

## Monitoring & Alerts

The scanner can send alerts via:

```python
# Custom alert callback
def my_alert_handler(alert, market_state):
    # Send to Discord, Telegram, email, etc.
    print(f"ALERT: {alert.message}")

monitor.on_spread_alert = my_alert_handler
```

## Performance Optimization

### For Fastest Execution
1. Run on a VPS close to Polymarket's servers
2. Use WebSocket feeds instead of polling
3. Pre-sign orders for instant submission
4. Monitor Polygon gas prices

### Batch Processing
```python
# Get multiple prices in one call
from py_clob_client.clob_types import BookParams

books = client.get_order_books([
    BookParams(token_id=token1),
    BookParams(token_id=token2),
])
```

## Legal & Compliance

- **US Users**: Trading is restricted via ToS (use VPN at your own risk)
- **Taxes**: Profits are taxable income in most jurisdictions
- **Regulation**: Polymarket acquired QCEX (CFTC-licensed) for US re-entry

## Troubleshooting

### "Not enough balance / allowance"
```bash
# Check your USDC balance on Polygon
# Ensure allowances are set
```

### "Order rejected"
- Price may have moved
- Check token ID is correct
- Verify market is still active

### Rate Limiting
- Add delays between API calls
- Use batch endpoints where available

## Resources

- [Polymarket Docs](https://docs.polymarket.com)
- [py-clob-client GitHub](https://github.com/Polymarket/py-clob-client)
- [Gamma API Reference](https://docs.polymarket.com/developers/gamma-markets-api/overview)
- [CLOB API Reference](https://docs.polymarket.com/developers/CLOB/quickstart)

## Disclaimer

This software is for educational purposes only. Trading prediction markets carries significant risk. Past performance (including the JaneStreetIndia account) does not guarantee future results. The edge described may have been arbitraged away since the strategy became public. Always do your own research and never risk more than you can afford to lose.

## License

MIT License - Use at your own risk.
