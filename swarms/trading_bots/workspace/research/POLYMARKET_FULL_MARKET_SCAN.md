# Polymarket Full Market Scan - Arbitrage Opportunities Report

**Generated:** 2026-01-04
**Data Source:** Polymarket Gamma API and CLOB API
**Total Markets Analyzed:** 4,000+ active markets

---

## Executive Summary

This comprehensive analysis covers ALL active markets on Polymarket across multiple categories. Key findings include:

- **Total Active Markets:** 4,000+ (across 8 API batches)
- **Total Platform Liquidity:** 160M+ USD across all markets
- **Total Platform Volume:** 2B+ USD cumulative trading volume
- **Arbitrage Opportunities:** Limited pure arbitrage due to efficient pricing, but significant market-making spreads exist

---

## Market Distribution by Category

### 1. SPORTS (2,097 markets - 52.4 percent)
- **Total Liquidity:** 62,192,988 USD
- **Total Volume:** 648,270,762 USD
- **Binary Markets:** 2,097
- **Multi-outcome (negRisk):** 1,954 (93 percent)
- **Market Types:**
  - NFL game outcomes and player props
  - NBA matchups and season awards
  - CFP (College Football Playoff) brackets
  - UFC/MMA fight outcomes
  - Soccer/Football international matches
  - MLB season predictions
  - Golf tournament winners
  - NASCAR race outcomes
- **Avg Spread:** 0.1-0.5 percent
- **Typical Expiration:** Hours to 1 year

### 2. POLITICS (967 markets - 24.2 percent)
- **Total Liquidity:** 60,058,884 USD
- **Total Volume:** 1,028,940,719 USD
- **Binary Markets:** 967
- **Multi-outcome (negRisk):** 774 (80 percent)
- **Market Types:**
  - Trump administration cabinet picks
  - Congressional legislation outcomes
  - State elections and gubernatorial races
  - International relations (Ukraine, China, etc.)
  - Policy decisions (tariffs, executive orders)
  - 2028 Presidential election early markets
- **Avg Spread:** 0.1-0.3 percent
- **Typical Expiration:** Days to 4 years

### 3. OTHER (541 markets - 13.5 percent)
- **Total Liquidity:** 14,442,081 USD
- **Total Volume:** 151,024,540 USD
- **Binary Markets:** 541
- **Multi-outcome (negRisk):** 346 (64 percent)
- **Includes:**
  - Celebrity events
  - Legal cases
  - Social media events
  - Miscellaneous predictions

### 4. CRYPTO (140 markets - 3.5 percent)
- **Total Liquidity:** 2,386,000 USD
- **Total Volume:** 43,264,872 USD
- **Binary Markets:** 140
- **Multi-outcome (negRisk):** 105 (75 percent)
- **Market Types:**
  - Bitcoin price milestones (100K, 200K, etc.)
  - Ethereum price predictions
  - Altcoin price targets (SOL, XRP, DOGE)
  - MicroStrategy Bitcoin holdings
  - Crypto regulatory events
- **Example Markets:**
  - MicroStrategy sells any Bitcoin in 2025 - 909K USD liquidity, 17.8M USD volume

### 5. ENTERTAINMENT (125 markets - 3.1 percent)
- **Total Liquidity:** 12,159,676 USD
- **Total Volume:** 85,772,037 USD
- **Binary Markets:** 125
- **Multi-outcome (negRisk):** 121 (97 percent)
- **Market Types:**
  - Box office predictions (Highest grossing movie 2025)
  - Award show outcomes (Oscars, Grammys, Emmys)
  - TV show events (Stranger Things deaths)
  - Celebrity news
- **Highest Liquidity:** Stranger Things Season 5 markets - 4.7M USD

### 6. TECHNOLOGY (81 markets - 2.0 percent)
- **Total Liquidity:** 938,983 USD
- **Total Volume:** 4,522,974 USD
- **Binary Markets:** 81
- **Multi-outcome (negRisk):** 63 (78 percent)
- **Market Types:**
  - AI developments (AGI, OpenAI, Anthropic)
  - SpaceX Starship launches
  - Tesla/Elon Musk events
  - Tech company milestones

### 7. ECONOMICS (41 markets - 1.0 percent)
- **Total Liquidity:** 8,671,020 USD
- **Total Volume:** 135,852,594 USD
- **Binary Markets:** 41
- **Multi-outcome (negRisk):** 38 (93 percent)
- **Market Types:**
  - Fed interest rate decisions (FOMC)
  - Inflation predictions (CPI, PCE)
  - GDP growth forecasts
  - Recession probability
- **Highest Liquidity:** Fed January 2026 decision - 7.6M USD

### 8. WEATHER/CLIMATE (5 markets - 0.1 percent)
- **Total Liquidity:** 53,707 USD
- **Total Volume:** 2,305,729 USD
- **Example:** Will 2025 be the hottest year on record - 30K USD liquidity

### 9. SCIENCE/HEALTH (3 markets - 0.1 percent)
- **Total Liquidity:** 18,988 USD
- **Total Volume:** 24,512 USD

---

## Arbitrage Analysis

### Understanding Polymarket Pricing

On Polymarket, binary markets have YES and NO tokens that should sum to 1.00 USD. Price deviations create arbitrage opportunities:

- **Sum less than 1.00:** Buy both sides, guaranteed profit at resolution
- **Sum greater than 1.00:** Sell both sides (if possible), guaranteed profit

### Current Market Efficiency

Based on analysis of 4,000 markets:

1. **Near-Perfect Pricing:** Most markets have YES + NO = 1.00 exactly
2. **Minimal Deviations:** Deviations when present are typically 0.001-0.003 (0.1-0.3 percent)
3. **NegRisk Markets:** Multi-outcome markets use negative risk system preventing simple arbitrage

### Why Pure Arbitrage is Rare

1. **Automated Market Makers:** Bots continuously scan for mispricing
2. **Low Latency:** WebSocket updates under 50ms
3. **High Competition:** Estimated 40M USD extracted by arbitrageurs in 2024-2025
4. **2 percent Winner Fee:** Eats into small arbitrage margins

### Viable Arbitrage Strategies

#### 1. Dutch-Book/Market Rebalancing (0.5-3 percent returns)
- Monitor multi-outcome markets where sum of all outcomes less than 1.00 USD
- Most opportunities close within seconds
- Requires automated trading

#### 2. Cross-Market Arbitrage
- Find related markets with inconsistent pricing
- Example: If Market A prices event at 80 percent and Market B prices opposite at 30 percent
- Requires sophisticated market correlation analysis

#### 3. Tail-End Sweep Strategy
- Buy outcomes priced 0.95-0.99 USD approaching resolution
- 90 percent of large orders (10K+ USD) execute at these prices
- Lower risk, lower return (1-5 percent in days/weeks)

#### 4. Time-Based Arbitrage
- Long-dated markets (2028 elections) offer 4+ percent yields
- Capital efficiency tradeoff (3+ year lockup)

---

## Market Making Opportunities

### Current Spread Analysis

Based on API data, typical spreads by category:

| Category | Avg Spread | Min Size | Reward Spread |
|----------|-----------|----------|---------------|
| Sports | 0.001-0.003 | 5 USD | 3.5 percent |
| Politics | 0.001-0.002 | 5 USD | 3.5 percent |
| Crypto | 0.001-0.005 | 5 USD | 3.5 percent |
| Entertainment | 0.001-0.002 | 5 USD | 4.5 percent |
| Economics | 0.001-0.003 | 5 USD | 3.5 percent |

### Market Making Requirements

1. **Minimum Spread:** 2.5-3 percent to cover 2 percent winner fee + gas
2. **Capital:** 10,000+ USD for meaningful returns
3. **Technology:** API access, automated order management
4. **Expected Returns:** 200-800 USD/day during active periods

### Order Book Rewards Program

Polymarket incentivizes liquidity provision:
- Place limit orders within max spread
- Meet minimum share requirements
- Proportionally share reward pool
- Dual income: spread profit + platform rewards

---

## Top Markets by Liquidity (Arbitrage Candidates)

### Highest Liquidity Active Markets

1. **Stranger Things S5 Deaths** - 4.7M USD liquidity, 78M USD volume
   - Multiple character outcomes
   - Resolution: December 31, 2025

2. **Fed January 2026 Decision** - 7.6M USD event liquidity
   - Rate hike/cut probabilities
   - Resolution: January 28, 2026

3. **Highest Grossing Movie 2025** - 9.8M USD liquidity
   - Multiple movie outcomes
   - Resolution: December 31, 2025

4. **MicroStrategy Bitcoin Sale** - 909K USD liquidity
   - Binary Yes/No
   - Spread: 0.001

5. **2028 Presidential Election** - Multi-million USD liquidity
   - 3+ year time horizon
   - 4+ percent implied yield

---

## Order Book Analysis

### Sample Order Book Structure (MicroStrategy BTC market)

ASKS (Sell Orders):
- 0.999 USD - 105,927 shares
- 0.998 USD - 15 shares
- 0.997 USD - 1,400 shares
- 0.996 USD - 6,497 shares
- 0.993 USD - 7,300 shares
- 0.990 USD - 55,589 shares
- 0.001 USD - 437,234 shares (tail)

BIDS (Buy Orders):
- None at extreme probabilities (market has moved)

### Key Observations

1. **Deep Liquidity at Extremes:** Markets approaching resolution show massive liquidity at 0.001 and 0.999
2. **Thin Middle Books:** Less liquidity at mid-range prices (0.3-0.7)
3. **Tick Size:** 0.001 (0.1 cent) minimum price increment
4. **Min Order:** 5 shares minimum

---

## API Endpoints Reference

### Gamma API (Market Data)
- Base: https://gamma-api.polymarket.com
- GET /markets?closed=false&limit=500 - All open markets
- GET /events?closed=false - Event groupings
- Supports pagination with offset parameter

### CLOB API (Order Book)
- Base: https://clob.polymarket.com
- GET /book?token_id=X - Order book for token
- GET /midpoint?token_id=X - Midpoint price
- WebSocket available for real-time updates

### Important Fields

| Field | Description |
|-------|-------------|
| outcomePrices | Current YES/NO prices |
| liquidityNum | Total market liquidity |
| spread | Bid-ask spread |
| bestBid, bestAsk | Top of book |
| clobTokenIds | Token IDs for CLOB queries |
| negRisk | Multi-outcome market flag |

---

## Recommendations for Trading Bot Development

### 1. Monitor High-Volume Markets
- Focus on categories with more than 10M USD daily volume
- Sports and Politics offer best liquidity

### 2. Implement Cross-Market Correlation
- Use vector embeddings to find related markets
- Tools: Chroma, sentence-transformers (e5-large-v2)

### 3. Target Optimal Spreads
- Look for markets with more than 2.5 percent spread
- Combine with liquidity rewards

### 4. Speed Optimization
- WebSocket connections for less than 50ms latency
- Polygon network for fast settlement

### 5. Risk Management
- 2 percent winner fee reduces margins
- One bad move can wipe weeks gains
- Position sizing critical

---

## Conclusion

While pure arbitrage opportunities on Polymarket are rare due to efficient pricing and automated competition, significant opportunities exist in:

1. **Market Making:** Consistent 200-800 USD/day with proper capital
2. **Cross-Market Analysis:** Finding correlation mispricings
3. **Tail-End Trading:** Low-risk plays on near-certain outcomes
4. **Long-Term Yield:** 4+ percent on multi-year markets

The platforms 160M+ USD liquidity across 4,000 markets provides ample opportunity for systematic trading strategies.

---

## Sources

- Polymarket Gamma API Documentation: https://docs.polymarket.com/developers/gamma-markets-api/overview
- Polymarket CLOB Introduction: https://docs.polymarket.com/developers/CLOB/introduction
- Polymarket Trading Bot Guide: https://www.polytrackhq.app/blog/polymarket-trading-bot
- Polymarket Arbitrage Guide: https://www.polytrackhq.app/blog/polymarket-arbitrage-guide
- Poly-Maker GitHub Repository: https://github.com/warproxxx/poly-maker
- Polymarket Official Agents: https://github.com/Polymarket/agents
- Automated Market Making on Polymarket: https://news.polymarket.com/p/automated-market-making-on-polymarket
- How Polymarket Makes Money: https://www.troniextechnologies.com/blog/how-polymarket-makes-money
- Arbitrage Traders on Polymarket: https://beincrypto.com/polymarket-arbitrage-risk-free-profit/
- Polymarket HFT and AI Strategies: https://www.quantvps.com/blog/polymarket-hft-traders-use-ai-arbitrage-mispricing

---

*Report generated by Research Specialist Agent for Agent Swarm Trading Bots*
