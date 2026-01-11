# Trading Bots - Quick Reference Guide
**Date**: 2026-01-02
**Purpose**: Fast lookup for key information from comprehensive research

---

## 30-Second Overview

**Can We Build Profitable Trading Bots?** Yes, but it's challenging.

**Expected Returns**: 20-30% annual (excellent by industry standards)

**Capital Needed**: $30K-$65K total ($25K-$50K trading + $5K-$15K development)

**Time to Profitability**: 6-12 months development/testing + 6-12 months to consistent profits

**Success Probability**: 60-70% chance of break-even or better with this plan

**Risk Level**: Moderate-to-High (expect 15-20% drawdowns)

---

## Recommended Strategy Portfolio ($50K Capital)

| Strategy | Allocation | Expected Return | Risk Level | Markets |
|----------|-----------|----------------|------------|---------|
| **Momentum Trading** | 40% ($20K) | 30-60% | Medium | BTC, ETH, Major Alts |
| **Mean Reversion** | 30% ($15K) | 20-40% | Medium-High | Mid-cap Altcoins |
| **Cross-Exchange Arbitrage** | 30% ($15K) | 10-25% | Low-Medium | Major Pairs, Multiple Exchanges |

**Portfolio Expected**: 25-35% gross return, 15-25% net after costs

---

## Timeline Cheat Sheet

| Phase | Duration | Cost | Key Activities | Go/No-Go |
|-------|----------|------|----------------|----------|
| **1. Foundation** | 2 months | $1-2K | Infrastructure, backtesting framework, exchange integration | - |
| **2. Strategy Dev** | 2 months | $1-2K | Implement & backtest 3 strategies, optimize parameters | - |
| **3. Paper Trading** | 3 months | $2-3K | Test in live markets without real money | ✓ Decision |
| **4. Small Live** | 1 month | $1K + fees | Trade with $5K (10% capital) | - |
| **5. Scale Up** | 4 months | $3-5K | Gradually increase to 50-100% capital | ✓ Monthly reviews |

**Total Year 1**: $8-13K costs, $50K trading capital

---

## Risk Management Rules (NEVER VIOLATE)

### Position Level
- **Max Risk Per Trade**: 1% of capital ($500 on $50K account)
- **Position Sizing**: Risk% / (Entry × ATR × Multiplier)
- **Stop Loss**: Always set BEFORE entering trade
- **Never**: Move stop loss further away from entry

### Portfolio Level
- **Max Total Exposure**: 50-100% of capital
- **Max Drawdown**: 20% (then pause all trading)
- **Diversification**: Minimum 3 uncorrelated strategies
- **Correlation Limit**: If strategies correlate >0.7, reduce allocation

### System Level
- **Kill Switches**: Auto-halt on max daily loss, consecutive losses, anomalies
- **Circuit Breakers**: Pause on volatility spikes, exchange issues
- **Daily P&L Limit**: -5% halt for review
- **Monitoring**: Daily review required, no exceptions

---

## Top 5 Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Overfitting** | High | Medium | Walk-forward analysis, 6+ months paper trading |
| **Poor Risk Mgmt** | Medium | Critical | 1% per trade, 20% max DD, kill switches |
| **Exchange Hack/Failure** | Low-Medium | High | Use reputable exchanges, withdraw profits regularly, diversify venues |
| **Market Regime Change** | Medium | Medium | Multiple strategies, continuous monitoring, adaptation |
| **Technical Failures** | High→Low | Medium | Comprehensive testing, monitoring, alerts, redundancy |

---

## Technology Stack

**Core**: Python + CCXT + PostgreSQL + TimescaleDB + Docker

**Components**:
- Exchange API: CCXT (unified interface to 100+ exchanges)
- Database: PostgreSQL with TimescaleDB extension
- Backtesting: Custom engine + vectorbt
- Monitoring: Prometheus + Grafana
- Message Queue: RabbitMQ or Kafka
- Hosting: AWS or GCP

**Development Time**: 3-6 months for full system

---

## Strategy Quick Reference

### 1. Cross-Exchange Arbitrage (Recommended First)
**Logic**: Buy on cheap exchange, sell on expensive exchange
**Entry**: Price difference > (fees + transfer cost + min profit)
**Exit**: Complete the arbitrage loop
**Expected**: 10-25% annual, 60-80% win rate, 5-15% max DD
**Capital**: $5K-$50K
**Complexity**: Medium
**Why First**: Simple logic, teaches exchange integration, lower risk

### 2. Cryptocurrency Momentum
**Logic**: Follow strong trends in crypto markets
**Entry**: Price > SMA(50) > SMA(200), RSI < 70, MACD crossover, ADX > 25
**Exit**: 2× ATR stop loss, 3× ATR take profit, trailing stop
**Expected**: 30-60% annual, 42% win rate, 20-30% max DD
**Capital**: $10K-$100K+
**Complexity**: Low-Medium
**Why Good**: Crypto trends strongly, high volatility = high returns

### 3. Mean Reversion (Altcoins)
**Logic**: Buy oversold, sell overbought in ranging markets
**Entry**: Price touches lower Bollinger Band, RSI < 30, volume spike
**Exit**: Return to middle BB, 2× ATR stop, 48-hour time limit
**Expected**: 20-40% annual, 58% win rate, 15-25% max DD
**Capital**: $5K-$50K
**Complexity**: Medium
**Why Good**: Uncorrelated with momentum, frequent opportunities

---

## Testing Checklist

### Backtest Phase
- [ ] Test on minimum 3 years of historical data
- [ ] Include bull, bear, and ranging market regimes
- [ ] Walk-forward analysis (optimize in-sample, test out-of-sample)
- [ ] Include realistic slippage (0.1-0.3% on major pairs)
- [ ] Include all fees (trading, withdrawal)
- [ ] Monte Carlo simulation (1000+ runs)
- [ ] Sharpe ratio >1.0

### Paper Trading Phase (Minimum 3 Months)
- [ ] Positive returns over full period
- [ ] Performance within 30% of backtest
- [ ] Max drawdown within expected range
- [ ] Win rate ± 10% of backtest
- [ ] System uptime >99%
- [ ] No critical bugs in last 30 days

### Live Trading Go-Live Criteria
- [ ] All paper trading checks passed
- [ ] Risk management tested and working
- [ ] Kill switches and circuit breakers verified
- [ ] Monitoring and alerts functioning
- [ ] Team trained on procedures
- [ ] Start with 10% of intended capital ($5K)

---

## Financial Projections (Base Case)

### Year 1 (Validation)
```
Capital:          $50,000
Costs:            -$8,000 (infrastructure)
Gross Return:     +$12,500 (25%)
Net Profit:       $4,500
Net Return:       9%
End Capital:      $54,500
```

### Year 2 (Growth)
```
Capital:          $54,500
Costs:            -$10,000
Gross Return:     +$19,075 (35%)
Net Profit:       $9,075
Net Return:       17%
End Capital:      $63,575
```

### Year 3 (Scale)
```
Capital:          $100,000 (add $36.5K)
Costs:            -$12,000
Gross Return:     +$30,000 (30%)
Net Profit:       $18,000
Net Return:       18%
End Capital:      $118,000
```

**5-Year Target**: $250K-$500K capital, $50K-$150K annual profit

---

## Common Failure Modes (How to Avoid)

| Mistake | Why Traders Fail | Our Mitigation |
|---------|-----------------|----------------|
| **Overfitting** | Optimize on test data | Walk-forward, out-of-sample testing |
| **Rushing Live** | Skip paper trading | Mandatory 3+ months paper trading |
| **No Stops** | Hope losing trades recover | Always use stop losses, 1% risk max |
| **Over-Leverage** | Chase big returns | Max 100% exposure, 50% typical |
| **Emotion** | Override discipline in drawdown | Automated execution, strict rules |
| **Single Strategy** | All eggs in one basket | 3 uncorrelated strategies |
| **Ignoring Costs** | Forget fees and slippage | Include all costs in backtest |
| **No Monitoring** | Set and forget | Daily review required |

---

## Regulatory Quick Reference (US)

**Tax**:
- Crypto taxed as property (capital gains)
- Report all trades (Form 8949)
- Set aside 30-40% for taxes
- Use crypto tax software (CoinTracker, Koinly)

**Trading Rules**:
- Pattern Day Trader: Doesn't apply to crypto (applies to stocks)
- Wash Sale: Currently doesn't apply to crypto (may change)

**Prohibited**:
- Spoofing (fake orders)
- Layering (manipulative order placement)
- Pump and dump
- Insider trading

**Best Practice**:
- Keep detailed records (7 years)
- Consult tax professional
- Consider LLC for liability protection

---

## Team Requirements

### Minimum (Solo)
- **Skills**: Python, basic stats, market knowledge, DevOps basics
- **Time**: 3-6 months development, then 10-20 hours/week ongoing
- **Limitations**: Single point of failure, slower development

### Recommended (2-3 People)
- **Quant Developer**: Strategy research and implementation
- **Infrastructure Engineer**: Systems and performance
- **Operator**: Daily monitoring and risk management
- **Time**: 2-4 months development, then 5-10 hours/week each

### Professional (5-8 People)
- Multiple quants, engineers, DevOps, risk manager, operators
- Required for $1M+ scale

---

## Decision Framework

### GO (Proceed) If:
- [x] Have $30K-$65K to commit (2+ year horizon)
- [x] Can allocate 3-6 months to development
- [x] Have technical skills or can hire developer
- [x] Comfortable with 20% drawdowns
- [x] Can monitor daily
- [x] Willing to start small and scale
- [x] Realistic expectations (20-30% target)

### NO-GO If:
- [ ] Need capital in next 12 months
- [ ] Expecting guaranteed profits
- [ ] Can't handle 20%+ drawdowns
- [ ] Don't have time for monitoring
- [ ] Want "set and forget" income
- [ ] Expecting 100%+ returns

---

## Key Performance Indicators (KPIs)

### Track Daily
- Total P&L ($ and %)
- P&L by strategy
- Open positions and exposure
- System uptime
- Error rate

### Review Weekly
- Win rate per strategy
- Average win vs. loss
- Current drawdown
- Sharpe ratio
- Slippage (actual vs. expected)

### Review Monthly
- Performance vs. backtest
- Strategy correlation
- Infrastructure costs
- Time spent on maintenance

### Review Quarterly
- Keep/modify/kill strategy decisions
- Market regime analysis
- Team and capability needs
- Capital allocation adjustments

---

## Emergency Procedures

### When to Halt All Trading
- Hit 20% max drawdown
- System instability (crashes, errors)
- Exchange major issues
- Major market disruption
- Systematic bugs detected

### How to Emergency Exit
1. Cancel all open orders immediately
2. Close all positions (market orders if necessary)
3. Alert team
4. Log all actions taken
5. Conduct post-mortem review
6. Require manual approval to restart

### Contact Escalation
- Level 1: Operator handles (normal operations)
- Level 2: Developer/quant (bugs, strategy issues)
- Level 3: CEO (major capital loss, legal issues)

---

## Resources

### Books
1. "Algorithmic Trading" by Ernest Chan (start here)
2. "Quantitative Trading" by Ernest Chan (strategies)
3. "Python for Finance" by Yves Hilpisch (coding)

### Tools
- **Exchanges**: Binance, Coinbase Pro, Kraken
- **API Library**: CCXT (Python)
- **Backtesting**: Backtrader, vectorbt
- **Database**: PostgreSQL + TimescaleDB
- **Monitoring**: Prometheus + Grafana

### Communities
- r/algotrading (Reddit)
- QuantConnect forums
- Elite Trader forums

---

## Next Steps (Week by Week)

### Week 1: Decision
- [ ] Review executive summary (30 min)
- [ ] Read relevant sections of full report (1-2 hours)
- [ ] Team discussion (1 hour)
- [ ] Make GO/NO-GO decision
- [ ] If GO: Allocate budget and assign team

### Week 2: Setup
- [ ] Register exchange accounts (Binance, Coinbase, Kraken)
- [ ] Get API keys (read-only first)
- [ ] Set up git repository
- [ ] Set up cloud servers (AWS/GCP)
- [ ] Begin collecting historical data

### Week 3-4: Foundation
- [ ] Build exchange integration (CCXT)
- [ ] Set up database (PostgreSQL + TimescaleDB)
- [ ] Create backtesting framework skeleton
- [ ] Implement basic monitoring

### Weeks 5-8: Strategy Development
- [ ] Implement arbitrage strategy
- [ ] Backtest on 3-5 years data
- [ ] Implement risk management
- [ ] Build operator dashboard

### Weeks 9-12: Additional Strategies
- [ ] Implement momentum strategy
- [ ] Implement mean reversion strategy
- [ ] Cross-strategy correlation analysis
- [ ] Prepare for paper trading

### Months 4-6: Paper Trading
- [ ] Deploy to staging environment
- [ ] Run strategies in real-time (no real money)
- [ ] Monitor and compare to backtest
- [ ] Fix bugs and tune parameters

### Month 7: Go-Live Decision
- [ ] Review paper trading results
- [ ] Make GO/NO-GO decision for live trading
- [ ] If GO: Transfer initial capital ($5K)
- [ ] Enable live trading for lowest-risk strategy

### Months 8-12: Scale Up
- [ ] Month 8: $5K (10%) single strategy
- [ ] Month 9: Add second strategy if positive
- [ ] Month 10: $12.5K (25%) if still positive
- [ ] Month 11: Add third strategy
- [ ] Month 12: $25K (50%) if consistently positive

---

## Success Metrics by Phase

### Development Phase (Months 1-4)
**Success**:
- All systems built and tested
- Strategies backtested with Sharpe >1.0
- Ready for paper trading on schedule

### Paper Trading Phase (Months 5-7)
**Success**:
- Positive returns over 3 months
- Performance within 30% of backtest
- System stable (>99% uptime)
- Team comfortable with operations

### Small Live Phase (Month 8)
**Success**:
- Positive or break-even results
- Actual slippage/fees match expectations
- No major surprises or bugs
- Confidence to scale up

### Scale Up Phase (Months 9-12)
**Success**:
- Consistent profitability (4+ profitable months out of 6)
- Drawdowns within planned limits (<20%)
- System reliability maintained
- Team capable of ongoing operations

### Year 1 Overall
**Success Criteria**:
- Net positive returns (any amount)
- Proven system works in live markets
- Team trained and confident
- Ready to scale in Year 2

---

## Contact and Support

**For Questions on**:
- **Strategy Details**: See full report sections 1, 2, 6
- **Technical Implementation**: See full report sections 2.2, 2.3, Appendix B
- **Risk Management**: See full report section 4
- **Timeline and Milestones**: See full report sections 5.2, 10
- **Financial Projections**: See full report sections 11, Executive Summary section 10
- **Regulatory**: See full report section 8
- **Decision Framework**: See executive summary section 12

**Swarm Resources**:
- Full Report: `COMPREHENSIVE_TRADING_BOT_RESEARCH.md` (47 pages)
- Executive Summary: `EXECUTIVE_SUMMARY.md` (6 pages)
- This Guide: `QUICK_REFERENCE_GUIDE.md` (current document)

**Orchestrator**: Available for detailed discussion and clarification

---

**Last Updated**: 2026-01-02
**Version**: 1.0
**Status**: Research Complete - Decision Pending
