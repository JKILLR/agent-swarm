"""
Swing Trading System
====================

Just run it:

    python trading_system.py

Scans 100 stocks + crypto, analyzes fundamentals + technicals + news sentiment,
tells you the single best trade AND suggests a specific options contract.

WHAT IT DOES:
  1. Fetches news from Yahoo Finance for each ticker
  2. Sends to Claude for semantic analysis:
     - Source attribution (who said it)
     - Credibility weighting (analyst > news > unknown)
     - Certainty detection (definite vs speculative)
     - Sentiment extraction
  3. Fetches fundamentals from Yahoo:
     - Insider transactions
     - Analyst upgrades/downgrades
     - Earnings surprises
  4. Calculates technicals:
     - Trend (price vs 20/50 day MA)
     - RSI, relative strength vs SPY
  5. Combines all → BUY/SELL/HOLD + options contract

SETUP:
    pip install yfinance anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

COST: ~$0.50-1.00 per scan (Claude API for news analysis)

OPTIONS STRATEGY:
    STRONG BUY → Call, 2-8% OTM, 2-4 weeks
    BUY        → Call, 2-5% OTM, 2-4 weeks  
    SELL       → Put, 2-5% OTM, 2-4 weeks
"""

import os
import sys
import json
import re
import time
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

# =============================================================================
# CONFIG
# =============================================================================

# S&P 500 / NASDAQ - Major stocks
STOCKS = [
    # Mega Cap Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
    # Tech
    'AVGO', 'AMD', 'QCOM', 'INTC', 'CSCO', 'ORCL', 'ADBE', 'CRM', 'NOW', 'IBM', 'MU', 'AMAT', 'LRCX', 'ADI', 'TXN',
    # Financials
    'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'C', 'BRK-B',
    # Healthcare
    'UNH', 'JNJ', 'LLY', 'PFE', 'MRK', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'GILD', 'CVS', 'ISRG', 'MDT',
    # Consumer
    'COST', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TGT', 'TJX', 'BKNG', 'CMG', 'LULU', 'YUM', 'DG',
    # Industrials
    'CAT', 'BA', 'HON', 'UPS', 'GE', 'RTX', 'LMT', 'DE', 'UNP', 'MMM',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY',
    # Communications
    'DIS', 'NFLX', 'CMCSA', 'T', 'VZ', 'TMUS',
    # Other Notable
    'COIN', 'PLTR', 'SNOW', 'CRWD', 'PANW', 'ZS', 'NET', 'DDOG', 'SHOP', 'SQ', 'PYPL', 'MSTR', 'SOFI',
]

# Main cryptos only
CRYPTO = [
    'BTC-USD',   # Bitcoin
    'ETH-USD',   # Ethereum
    'XRP-USD',   # Ripple
    'DOGE-USD',  # Dogecoin
    'SOL-USD',   # Solana
    'ADA-USD',   # Cardano
    'LTC-USD',   # Litecoin
]

# Weight for final score
LAYER_WEIGHTS = {'sentiment': 0.30, 'events': 0.40, 'technicals': 0.30}

# Credibility weights
SOURCE_CREDIBILITY = {
    'INSIDER': 1.0, 'COMPANY': 0.9, 'INSTITUTION': 0.85,
    'ANALYST': 0.75, 'NEWS': 0.5, 'RETAIL': 0.2, 'UNKNOWN': 0.3,
}

CERTAINTY_WEIGHTS = {
    'DEFINITE': 1.0, 'PROBABLE': 0.75, 'POSSIBLE': 0.5,
    'CONDITIONAL': 0.4, 'SPECULATIVE': 0.2,
}

# Claude model
CLAUDE_MODEL = "claude-sonnet-4-20250514"


# =============================================================================
# NEWS SENTIMENT ANALYSIS (Yahoo Finance + Claude)
# =============================================================================

NEWS_ANALYSIS_PROMPT = '''Analyze this financial news for sentiment. Return JSON only:

{{
  "sentiment": <-1.0 to 1.0>,
  "intensity": <0.0 to 1.0>,
  "source_type": "INSIDER|COMPANY|INSTITUTION|ANALYST|NEWS|UNKNOWN",
  "certainty": "DEFINITE|PROBABLE|POSSIBLE|SPECULATIVE",
  "action": "UPGRADE|DOWNGRADE|BUY|SELL|BEAT|MISS|GUIDANCE_UP|GUIDANCE_DOWN|NEUTRAL",
  "key_point": "one sentence summary"
}}

Source types:
- INSIDER: CEO/CFO/board member of the company
- COMPANY: Official company announcement
- INSTITUTION: Hedge fund, bank as investor (Berkshire, ARK)
- ANALYST: Research analyst, price target, ratings
- NEWS: Journalist reporting facts
- UNKNOWN: Can't determine

Certainty:
- DEFINITE: Already happened ("announced", "reported", "beat")
- PROBABLE: Very likely ("will", "expects to")
- POSSIBLE: Hedged ("may", "could", "considering")
- SPECULATIVE: Opinion/rumor ("sources say", "reportedly")

News headline: "{headline}"
Summary: "{summary}"
Ticker: {ticker}'''


class SentimentAnalyzer:
    def __init__(self):
        self.client = None
        self.total_cost = 0
        self.api_key = os.environ.get('ANTHROPIC_API_KEY')
    
    def _init_client(self):
        if self.client is None:
            if not self.api_key:
                return False
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                return False
            except Exception as e:
                return False
        return True
    
    def analyze_ticker(self, ticker: str, debug: bool = False) -> Tuple[Optional[float], List[str], List[str]]:
        """
        Fetch news for ticker and analyze with Claude.
        Returns: (sentiment_score, bullish_factors, bearish_factors)
        """
        if not self._init_client():
            if debug: print(f"  [DEBUG] Client init failed")
            return None, [], []
        
        # Get news from Yahoo Finance
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                if debug: print(f"  [DEBUG] No news for {ticker}")
                return None, [], []
            if debug: print(f"  [DEBUG] Found {len(news)} news items")
        except Exception as e:
            if debug: print(f"  [DEBUG] News fetch error: {e}")
            return None, [], []
        
        # Analyze up to 5 most recent news items
        sentiments = []
        bullish = []
        bearish = []
        
        for item in news[:5]:
            # Yahoo Finance nests content - title and summary are inside 'content'
            content_obj = item.get('content', {})
            headline = (
                content_obj.get('title') or 
                item.get('title') or 
                ''
            )
            summary = (
                content_obj.get('summary') or 
                item.get('summary') or 
                ''
            )[:500]
            
            if not headline:
                if debug: print(f"  [DEBUG] Empty headline, skipping")
                continue
            
            if debug: print(f"  [DEBUG] Processing: {headline[:50]}...")
            
            try:
                prompt = NEWS_ANALYSIS_PROMPT.format(
                    headline=headline,
                    summary=summary,
                    ticker=ticker
                )
                
                if debug: print(f"  [DEBUG] Calling Claude API...")
                response = self.client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )
                if debug: print(f"  [DEBUG] Got response")
                
                # Track cost
                tokens = response.usage.input_tokens + response.usage.output_tokens
                self.total_cost += tokens * 3.0 / 1_000_000 + tokens * 0.3 * 15.0 / 1_000_000
                
                # Parse response
                raw_content = response.content[0].text.strip()
                if debug: print(f"  [DEBUG] Raw: {raw_content[:80]}...")
                
                # Find JSON by locating braces
                start = raw_content.find('{')
                end = raw_content.rfind('}')
                
                if start == -1 or end == -1 or end <= start:
                    if debug: print(f"  [DEBUG] No JSON braces found")
                    continue
                
                json_str = raw_content[start:end+1]
                
                try:
                    data = json.loads(json_str)
                    if debug: print(f"  [DEBUG] Parsed: {data}")
                except json.JSONDecodeError as e:
                    if debug: print(f"  [DEBUG] JSON error: {e}")
                    # Try removing any trailing content after the JSON
                    # Sometimes Claude adds explanation after the JSON
                    continue
                
                sentiment = float(data.get('sentiment', 0))
                intensity = float(data.get('intensity', 0.5))
                source_type = data.get('source_type', 'UNKNOWN')
                certainty = data.get('certainty', 'PROBABLE')
                action = data.get('action', 'NEUTRAL')
                key_point = data.get('key_point', '')
                
                # Weight by credibility and certainty
                cred = SOURCE_CREDIBILITY.get(source_type, 0.3)
                cert = CERTAINTY_WEIGHTS.get(certainty, 0.5)
                weight = cred * cert * intensity
                
                sentiments.append((sentiment, weight))
                
                # Track factors
                if sentiment > 0.3 and key_point:
                    bullish.append(f"{key_point} ({source_type.lower()})")
                elif sentiment < -0.3 and key_point:
                    bearish.append(f"{key_point} ({source_type.lower()})")
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                if debug: print(f"  [DEBUG] Exception: {type(e).__name__}: {e}")
                # Silently skip failed analysis
                continue
        
        if not sentiments:
            return None, [], []
        
        # Weighted average
        total_weight = sum(w for _, w in sentiments)
        if total_weight > 0:
            score = sum(s * w for s, w in sentiments) / total_weight
        else:
            score = 0
        
        return round(score, 3), bullish[:3], bearish[:3]


# Global analyzer instance
_sentiment_analyzer = None

def get_sentiment_analyzer():
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer


# =============================================================================
# OPTIONS ANALYSIS (Yahoo Finance)
# =============================================================================

def get_options_suggestion(ticker: str, signal: str, current_price: float, target_price: float = None) -> dict:
    """
    Suggest an options contract based on the signal.
    Returns contract details or None if no good options found.
    """
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
    except ImportError:
        return None
    
    if not current_price or current_price <= 0:
        return None
    
    # Skip crypto - no options
    if '-USD' in ticker:
        return None
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get available expiration dates
        expirations = stock.options
        if not expirations:
            return None
        
        # Find expiration 2-4 weeks out (ideal for swing trades)
        today = datetime.now()
        target_min = today + timedelta(days=14)
        target_max = today + timedelta(days=35)
        
        best_exp = None
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d')
            if target_min <= exp_date <= target_max:
                best_exp = exp
                break
        
        # If nothing in 2-4 weeks, take the nearest one that's at least 7 days out
        if not best_exp:
            for exp in expirations:
                exp_date = datetime.strptime(exp, '%Y-%m-%d')
                if exp_date >= today + timedelta(days=7):
                    best_exp = exp
                    break
        
        if not best_exp:
            return None
        
        # Get options chain
        chain = stock.option_chain(best_exp)
        
        # Determine if we want calls or puts
        if signal in ['STRONG BUY', 'BUY', 'LEAN BUY']:
            options_df = chain.calls
            contract_type = 'Call'
            # For calls: slightly OTM (2-8% above current price)
            target_strike_low = current_price * 1.02
            target_strike_high = current_price * 1.08
        elif signal in ['STRONG SELL', 'SELL', 'LEAN SELL']:
            options_df = chain.puts
            contract_type = 'Put'
            # For puts: slightly OTM (2-8% below current price)
            target_strike_low = current_price * 0.92
            target_strike_high = current_price * 0.98
        else:
            return None  # HOLD = no options play
        
        if options_df.empty:
            return None
        
        # Filter for good liquidity (volume > 10, open interest > 50)
        liquid = options_df[
            (options_df['volume'].fillna(0) >= 10) | 
            (options_df['openInterest'].fillna(0) >= 50)
        ]
        
        if liquid.empty:
            liquid = options_df  # Fall back to all options
        
        # Find strike in our target range
        if contract_type == 'Call':
            candidates = liquid[
                (liquid['strike'] >= target_strike_low) & 
                (liquid['strike'] <= target_strike_high)
            ]
            if candidates.empty:
                # Get nearest OTM call
                otm = liquid[liquid['strike'] >= current_price]
                if not otm.empty:
                    candidates = otm.head(3)
        else:  # Put
            candidates = liquid[
                (liquid['strike'] >= target_strike_low) & 
                (liquid['strike'] <= target_strike_high)
            ]
            if candidates.empty:
                # Get nearest OTM put
                otm = liquid[liquid['strike'] <= current_price]
                if not otm.empty:
                    candidates = otm.tail(3)
        
        if candidates.empty:
            return None
        
        # Pick the one with best liquidity (highest open interest)
        best = candidates.loc[candidates['openInterest'].fillna(0).idxmax()]
        
        strike = best['strike']
        bid = best.get('bid', 0) or 0
        ask = best.get('ask', 0) or 0
        last_price = best.get('lastPrice', 0) or 0
        volume = int(best.get('volume', 0) or 0)
        open_interest = int(best.get('openInterest', 0) or 0)
        implied_vol = best.get('impliedVolatility', 0) or 0
        
        # Use mid price if bid/ask available, otherwise last price
        if bid > 0 and ask > 0:
            mid_price = (bid + ask) / 2
        else:
            mid_price = last_price
        
        if mid_price <= 0:
            return None
        
        # Calculate break-even
        if contract_type == 'Call':
            breakeven = strike + mid_price
            breakeven_pct = (breakeven / current_price - 1) * 100
        else:
            breakeven = strike - mid_price
            breakeven_pct = (1 - breakeven / current_price) * 100
        
        # Calculate potential profit if target hit
        profit_at_target = None
        profit_pct = None
        if target_price and target_price > 0:
            if contract_type == 'Call' and target_price > strike:
                intrinsic_at_target = target_price - strike
                profit_at_target = intrinsic_at_target - mid_price
                profit_pct = (profit_at_target / mid_price) * 100
            elif contract_type == 'Put' and target_price < strike:
                intrinsic_at_target = strike - target_price
                profit_at_target = intrinsic_at_target - mid_price
                profit_pct = (profit_at_target / mid_price) * 100
        
        # Calculate approximate delta (simplified)
        if contract_type == 'Call':
            moneyness = current_price / strike
            if moneyness > 1.05:
                delta = 0.7
            elif moneyness > 1.0:
                delta = 0.55
            elif moneyness > 0.95:
                delta = 0.45
            else:
                delta = 0.30
        else:  # Put
            moneyness = strike / current_price
            if moneyness > 1.05:
                delta = -0.7
            elif moneyness > 1.0:
                delta = -0.55
            elif moneyness > 0.95:
                delta = -0.45
            else:
                delta = -0.30
        
        # Days to expiration
        exp_date = datetime.strptime(best_exp, '%Y-%m-%d')
        dte = (exp_date - today).days
        
        # Contract symbol
        contract_symbol = f"{ticker} {best_exp} ${strike:.0f} {contract_type}"
        
        return {
            'symbol': contract_symbol,
            'type': contract_type,
            'strike': strike,
            'expiration': best_exp,
            'dte': dte,
            'bid': bid,
            'ask': ask,
            'mid_price': mid_price,
            'volume': volume,
            'open_interest': open_interest,
            'implied_vol': implied_vol,
            'delta': delta,
            'breakeven': breakeven,
            'breakeven_pct': breakeven_pct,
            'profit_at_target': profit_at_target,
            'profit_pct': profit_pct,
            'cost_per_contract': mid_price * 100,  # Options are 100 shares
        }
        
    except Exception as e:
        return None


# =============================================================================
# EVENTS (Yahoo Finance)
# =============================================================================

def analyze_events(ticker: str) -> Tuple[Optional[float], List[str], List[str], Optional[float]]:
    """
    Returns: (score, bullish_factors, bearish_factors, price_target)
    """
    try:
        import yfinance as yf
    except ImportError:
        print("\n❌ yfinance not installed. Run: pip install yfinance")
        sys.exit(1)
    
    try:
        stock = yf.Ticker(ticker)
        score = 0
        bullish, bearish = [], []
        
        # --- Insider Transactions ---
        try:
            ins = stock.insider_transactions
            if ins is not None and not ins.empty:
                buys, sells = 0, 0
                
                for _, row in ins.head(20).iterrows():
                    txn = str(row.get('Text', '')).lower()
                    
                    if 'buy' in txn or 'purchase' in txn:
                        buys += 1
                    elif 'sell' in txn or 'sale' in txn:
                        sells += 1
                
                if buys >= 3 and sells == 0:
                    score += 0.5
                    bullish.append(f"🔥 Insider cluster buying ({buys} buys, 0 sells)")
                elif buys >= 2 and buys > sells:
                    score += 0.3
                    bullish.append(f"Insider buying ({buys} buys, {sells} sells)")
                elif sells >= 3 and sells > buys * 2:
                    score -= 0.4
                    bearish.append(f"Heavy insider selling ({sells} sells)")
                elif sells > buys:
                    score -= 0.2
                    bearish.append(f"Net insider selling ({sells} sells, {buys} buys)")
        except:
            pass
        
        # --- Analyst Ratings ---
        try:
            recs = stock.recommendations
            if recs is not None and not recs.empty:
                ups, downs = 0, 0
                for _, row in recs.tail(10).iterrows():
                    action = str(row.get('Action', '')).lower()
                    grade = str(row.get('To Grade', '')).lower()
                    if 'upgrade' in action or 'buy' in grade or 'outperform' in grade:
                        ups += 1
                    elif 'downgrade' in action or 'sell' in grade or 'underperform' in grade:
                        downs += 1
                
                if ups >= 3 and ups > downs:
                    score += 0.35
                    bullish.append(f"Strong analyst upgrades ({ups} recent)")
                elif ups >= 2 and ups > downs:
                    score += 0.2
                    bullish.append(f"Analyst upgrades ({ups} recent)")
                elif downs >= 3 and downs > ups:
                    score -= 0.35
                    bearish.append(f"Analyst downgrades ({downs} recent)")
                elif downs >= 2 and downs > ups:
                    score -= 0.2
                    bearish.append(f"Analyst downgrades ({downs} recent)")
        except:
            pass
        
        # --- Earnings Surprise ---
        try:
            earnings = stock.earnings_history
            if earnings is not None and not earnings.empty:
                latest = earnings.iloc[-1]
                actual = latest.get('epsActual', 0) or 0
                est = latest.get('epsEstimate', 0) or 0
                if est != 0:
                    surprise = (actual - est) / abs(est) * 100
                    if surprise > 15:
                        score += 0.35
                        bullish.append(f"🔥 Earnings crushed estimates (+{surprise:.0f}%)")
                    elif surprise > 5:
                        score += 0.2
                        bullish.append(f"Earnings beat (+{surprise:.0f}%)")
                    elif surprise < -15:
                        score -= 0.35
                        bearish.append(f"Earnings badly missed ({surprise:.0f}%)")
                    elif surprise < -5:
                        score -= 0.2
                        bearish.append(f"Earnings missed ({surprise:.0f}%)")
        except:
            pass
        
        # --- Price Target vs Current ---
        pt = None
        try:
            info = stock.info or {}
            pt = info.get('targetMeanPrice')
            current = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if pt and current and current > 0:
                upside = (pt / current - 1) * 100
                if upside > 30:
                    score += 0.2
                    bullish.append(f"High upside to target (+{upside:.0f}%)")
                elif upside < -10:
                    score -= 0.15
                    bearish.append(f"Trading above analyst targets ({upside:.0f}%)")
        except:
            pass
        
        return score, bullish, bearish, pt
        
    except Exception as e:
        return None, [], [str(e)], None


# =============================================================================
# TECHNICALS (Yahoo Finance)
# =============================================================================

def analyze_technicals(ticker: str) -> Tuple[Optional[float], List[str], List[str], Optional[float]]:
    """
    Returns: (score, bullish_factors, bearish_factors, current_price)
    """
    try:
        import yfinance as yf
    except ImportError:
        return None, [], ["yfinance not installed"], None
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        
        if hist.empty or len(hist) < 20:
            return None, [], ["Insufficient data"], None
        
        price = hist['Close'].iloc[-1]
        score = 0
        bullish, bearish = [], []
        
        # --- Trend (Moving Averages) ---
        ma20 = hist['Close'].rolling(20).mean().iloc[-1]
        ma50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else ma20
        
        above_20 = price > ma20
        above_50 = price > ma50
        
        if above_20 and above_50:
            score += 0.3
            bullish.append("Uptrend (above 20 & 50 MA)")
        elif above_20 and not above_50:
            score += 0.1
            bullish.append("Above 20-day MA")
        elif not above_20 and not above_50:
            score -= 0.3
            bearish.append("Downtrend (below 20 & 50 MA)")
        else:
            score -= 0.1
            bearish.append("Below 20-day MA")
        
        # --- RSI ---
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        rsi = 100 - (100 / (1 + gain / loss)) if loss != 0 else 50
        
        if rsi < 30:
            score += 0.25
            bullish.append(f"RSI oversold ({rsi:.0f}) - bounce potential")
        elif rsi < 40:
            score += 0.1
            bullish.append(f"RSI low ({rsi:.0f})")
        elif rsi > 70:
            score -= 0.2
            bearish.append(f"RSI overbought ({rsi:.0f})")
        
        # --- Relative Strength vs SPY ---
        try:
            spy = yf.Ticker("SPY").history(period="3mo")
            if len(hist) >= 22 and len(spy) >= 22:
                ticker_ret = (hist['Close'].iloc[-1] / hist['Close'].iloc[-22] - 1) * 100
                spy_ret = (spy['Close'].iloc[-1] / spy['Close'].iloc[-22] - 1) * 100
                rs = ticker_ret - spy_ret
                
                if rs > 15:
                    score += 0.25
                    bullish.append(f"🔥 Crushing the S&P (+{rs:.0f}% vs SPY)")
                elif rs > 5:
                    score += 0.1
                    bullish.append(f"Outperforming S&P (+{rs:.0f}%)")
                elif rs < -15:
                    score -= 0.2
                    bearish.append(f"Lagging S&P badly ({rs:.0f}%)")
                elif rs < -5:
                    score -= 0.1
                    bearish.append(f"Underperforming S&P ({rs:.0f}%)")
        except:
            pass
        
        # --- Recent Momentum ---
        if len(hist) >= 5:
            week_ret = (hist['Close'].iloc[-1] / hist['Close'].iloc[-5] - 1) * 100
            if week_ret > 10:
                score += 0.15
                bullish.append(f"Hot momentum (+{week_ret:.0f}% this week)")
            elif week_ret < -10:
                score -= 0.1
                bearish.append(f"Weak momentum ({week_ret:.0f}% this week)")
        
        return score, bullish, bearish, price
        
    except Exception as e:
        return None, [], [str(e)], None


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_ticker(ticker: str) -> dict:
    """Full analysis of one ticker."""
    
    all_bullish, all_bearish = [], []
    scores, weights = [], []
    
    # Sentiment (Claude + Yahoo News)
    analyzer = get_sentiment_analyzer()
    sent_score, sent_bull, sent_bear = analyzer.analyze_ticker(ticker)
    if sent_score is not None:
        scores.append(sent_score)
        weights.append(LAYER_WEIGHTS['sentiment'])
        all_bullish.extend(sent_bull)
        all_bearish.extend(sent_bear)
    
    # Events (insider, earnings, analysts)
    ev_score, ev_bull, ev_bear, price_target = analyze_events(ticker)
    if ev_score is not None:
        scores.append(ev_score)
        weights.append(LAYER_WEIGHTS['events'])
        all_bullish.extend(ev_bull)
        all_bearish.extend(ev_bear)
    
    # Technicals
    tech_score, tech_bull, tech_bear, price = analyze_technicals(ticker)
    if tech_score is not None:
        scores.append(tech_score)
        weights.append(LAYER_WEIGHTS['technicals'])
        all_bullish.extend(tech_bull)
        all_bearish.extend(tech_bear)
    
    # Combine
    final = sum(s * w for s, w in zip(scores, weights)) / sum(weights) if weights else 0
    
    return {
        'ticker': ticker,
        'score': final,
        'price': price,
        'target': price_target,
        'sentiment': sent_score,
        'events': ev_score,
        'technicals': tech_score,
        'bullish': all_bullish,
        'bearish': all_bearish,
    }


def main():
    print("\n🔍 Scanning market for best trade...\n")
    
    # Check for yfinance
    try:
        import yfinance
    except ImportError:
        print("❌ Missing required package. Run:")
        print("   pip install yfinance")
        return
    
    # Check for anthropic (optional but recommended)
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set - running without sentiment analysis")
        print("   For full analysis: export ANTHROPIC_API_KEY=sk-ant-...")
        print()
    else:
        try:
            import anthropic
        except ImportError:
            print("⚠️  anthropic package not installed - running without sentiment")
            print("   Run: pip install anthropic")
            print()
    
    tickers = STOCKS + CRYPTO
    results = []
    
    for i, ticker in enumerate(tickers):
        sys.stdout.write(f"\r   [{i+1}/{len(tickers)}] Analyzing {ticker:<12}")
        sys.stdout.flush()
        
        try:
            result = analyze_ticker(ticker)
            results.append(result)
        except Exception as e:
            pass
        
        time.sleep(0.1)  # Be nice to Yahoo
    
    print("\r" + " " * 50 + "\r", end="")
    
    if not results:
        print("❌ No results. Check your internet connection.")
        return
    
    # Find the best
    best = max(results, key=lambda x: x['score'])
    
    # Determine signal
    if best['score'] > 0.4:
        signal, emoji = "STRONG BUY", "🟢"
    elif best['score'] > 0.2:
        signal, emoji = "BUY", "🟢"
    elif best['score'] > 0.05:
        signal, emoji = "LEAN BUY", "🟡"
    elif best['score'] < -0.4:
        signal, emoji = "STRONG SELL", "🔴"
    elif best['score'] < -0.2:
        signal, emoji = "SELL", "🔴"
    elif best['score'] < -0.05:
        signal, emoji = "LEAN SELL", "🟠"
    else:
        signal, emoji = "HOLD", "⚪"
    
    # === OUTPUT ===
    print(f"\n{'═'*55}")
    print(f"  {emoji}  BEST TRADE: {signal} {best['ticker']}")
    print(f"{'═'*55}")
    
    # Price + target
    if best['price']:
        line = f"\n  💰 Price: ${best['price']:.2f}"
        if best['target']:
            upside = (best['target'] / best['price'] - 1) * 100
            arrow = "↑" if upside > 0 else "↓"
            line += f"  →  Target: ${best['target']:.2f} ({arrow}{abs(upside):.0f}%)"
        print(line)
    
    # Scores
    print(f"\n  📊 Score: {best['score']:+.2f}")
    print(f"     ├─ Sentiment:  {best['sentiment']:+.2f}" if best['sentiment'] is not None else "     ├─ Sentiment:  N/A")
    print(f"     ├─ Events:     {best['events']:+.2f}" if best['events'] is not None else "     ├─ Events:     N/A")
    print(f"     └─ Technicals: {best['technicals']:+.2f}" if best['technicals'] is not None else "     └─ Technicals: N/A")
    
    # Options suggestion
    if best['price'] and signal != 'HOLD':
        options = get_options_suggestion(
            best['ticker'], 
            signal, 
            best['price'], 
            best['target']
        )
        if options:
            print(f"\n  🎯 Suggested Options Play:")
            print(f"     Contract:   {options['symbol']}")
            print(f"     Price:      ${options['bid']:.2f} (bid) - ${options['ask']:.2f} (ask)")
            print(f"     Delta:      {options['delta']:.2f}")
            print(f"     Expires:    {options['expiration']} ({options['dte']} days)")
            print(f"     Break-even: ${options['breakeven']:.2f} ({options['breakeven_pct']:+.1f}%)")
            print(f"     Cost:       ${options['cost_per_contract']:.0f} per contract")
            if options['profit_pct'] and options['profit_pct'] > 0:
                print(f"     If target hit: +{options['profit_pct']:.0f}% return")
            print(f"     Volume/OI:  {options['volume']:,} / {options['open_interest']:,}")
    
    # Bullish factors
    if best['bullish']:
        print(f"\n  ✅ Bullish Factors:")
        for f in best['bullish'][:6]:
            print(f"     • {f}")
    
    # Bearish factors
    if best['bearish']:
        print(f"\n  ❌ Bearish Factors:")
        for f in best['bearish'][:6]:
            print(f"     • {f}")
    
    # Top 5
    top5 = sorted(results, key=lambda x: -x['score'])[:5]
    if len(top5) > 1:
        print(f"\n  📋 Other Opportunities:")
        for r in top5[1:]:
            p = f"${r['price']:.2f}" if r['price'] else ""
            print(f"     {r['ticker']:<10} {r['score']:+.2f}  {p}")
    
    # API cost
    analyzer = get_sentiment_analyzer()
    if analyzer.total_cost > 0:
        print(f"\n  💰 Claude API Cost: ${analyzer.total_cost:.2f}")
    
    print()


if __name__ == "__main__":
    main()
