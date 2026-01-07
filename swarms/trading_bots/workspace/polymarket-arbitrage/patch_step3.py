
# Read the file
with open("/Users/jellingson/agent-swarm/swarms/trading_bots/workspace/polymarket-arbitrage/ultimate_arb_bot.py", "r") as f:
    content = f.read()

# Find the marker and insert after it
marker = """            }

# =============================================================================
# MARKET DISCOVERY
# =============================================================================
"""

database_momentum_classes = """            }


# =============================================================================
# DATABASE
# =============================================================================

class Database:
    \"\"\"SQLite database for trade history and statistics\"\"\"

    def __init__(self, db_path: str = None):
        self.db_path = db_path or CONFIG.get("DB_PATH", "ultimate_arb_data.db")
        self._create_tables()

    def _create_tables(self):
        \"\"\"Create database tables if they do not exist\"\"\"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(\"\"\"
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                asset TEXT NOT NULL,
                spread_pct REAL NOT NULL,
                up_ask REAL,
                down_ask REAL,
                total_cost REAL,
                kelly_size REAL,
                expected_profit REAL,
                actual_profit REAL,
                success INTEGER NOT NULL,
                error TEXT
            )
        \"\"\")

        cursor.execute(\"\"\"
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                opportunities_found INTEGER DEFAULT 0,
                opportunities_executed INTEGER DEFAULT 0,
                total_profit REAL DEFAULT 0.0,
                total_loss REAL DEFAULT 0.0,
                win_count INTEGER DEFAULT 0,
                loss_count INTEGER DEFAULT 0
            )
        \"\"\")

        conn.commit()
        conn.close()

    def record_trade(self, opp, result):
        \"\"\"Record a trade to the database\"\"\"
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(\"\"\"
                INSERT INTO trades (timestamp, asset, spread_pct, up_ask, down_ask,
                                   total_cost, kelly_size, expected_profit, actual_profit, success, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            \"\"\", (
                datetime.now().isoformat(),
                opp.market.asset,
                opp.spread_pct,
                opp.up_ask,
                opp.down_ask,
                opp.total_cost,
                opp.kelly_size,
                opp.expected_profit,
                result.get("profit", 0.0),
                1 if result.get("success") else 0,
                result.get("error")
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database error recording trade: {e}")

    def get_stats(self, days: int = 30):
        \"\"\"Get statistics for the last N days\"\"\"
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            cursor.execute(\"\"\"
                SELECT COUNT(*), SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END),
                       SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END),
                       SUM(CASE WHEN success = 1 THEN actual_profit ELSE 0 END),
                       SUM(CASE WHEN success = 0 THEN actual_profit ELSE 0 END),
                       AVG(spread_pct), AVG(kelly_size)
                FROM trades WHERE date(timestamp) >= ?
            \"\"\", (cutoff,))
            row = cursor.fetchone()
            conn.close()
            if row:
                total_trades, wins, losses, total_profit, total_loss, avg_spread, avg_size = row
                return {
                    "period_days": days, "total_trades": total_trades or 0,
                    "wins": wins or 0, "losses": losses or 0,
                    "win_rate": (wins / total_trades * 100) if total_trades else 0,
                    "total_profit": total_profit or 0.0, "total_loss": total_loss or 0.0,
                    "net_profit": (total_profit or 0) - abs(total_loss or 0),
                    "avg_spread_pct": avg_spread or 0.0, "avg_position_size": avg_size or 0.0
                }
            return {}
        except Exception as e:
            logger.error(f"Database error getting stats: {e}")
            return {}

    def print_stats(self, days: int = 30):
        \"\"\"Print formatted statistics\"\"\"
        stats = self.get_stats(days)
        if not stats:
            print("No trading data available.")
            return
        print("\\n" + "=" * 70)
        print(f"HISTORICAL STATISTICS (Last {days} Days)")
        print("=" * 70)
        print(f"Total Trades:          {stats["total_trades"]}")
        print(f"Wins / Losses:         {stats["wins"]} / {stats["losses"]}")
        print(f"Win Rate:              {stats["win_rate"]:.1f}%")
        print(f"Total Profit:          \${stats["total_profit"]:.2f}")
        print(f"Total Loss:            \${stats["total_loss"]:.2f}")
        print(f"Net P&L:               \${stats["net_profit"]:.2f}")
        print(f"Avg Spread:            {stats["avg_spread_pct"]:.3f}%")
        print(f"Avg Position Size:     \${stats["avg_position_size"]:.2f}")
        print("=" * 70 + "\\n")


# =============================================================================
# MOMENTUM TRACKER
# =============================================================================

class MomentumTracker:
    \"\"\"Track price momentum from Binance for crypto assets\"\"\"

    BINANCE_API = "https://api.binance.com/api/v3"
    SYMBOL_MAP = {"btc": "BTCUSDT", "eth": "ETHUSDT", "sol": "SOLUSDT", "xrp": "XRPUSDT"}

    def __init__(self, assets: List[str] = None, window_seconds: int = None):
        self.assets = assets or CONFIG.get("ASSETS", ["btc", "eth", "sol", "xrp"])
        self.window_seconds = window_seconds or CONFIG.get("MOMENTUM_WINDOW_SECONDS", 60)
        self.price_history: Dict[str, deque] = {asset: deque(maxlen=120) for asset in self.assets}
        self.current_prices: Dict[str, float] = {}
        self.running = False
        self._thread = None
        self._lock = threading.Lock()
        self.session = requests.Session()

    def start(self):
        \"\"\"Start the momentum tracking thread\"\"\"
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._track_loop, daemon=True)
        self._thread.start()
        logger.info("Momentum tracker started")

    def stop(self):
        \"\"\"Stop the momentum tracking thread\"\"\"
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Momentum tracker stopped")

    def _track_loop(self):
        \"\"\"Background loop to fetch prices\"\"\"
        while self.running:
            for asset in self.assets:
                try:
                    price = self._fetch_price(asset)
                    if price:
                        with self._lock:
                            self.current_prices[asset] = price
                            self.price_history[asset].append((time.time(), price))
                except Exception as e:
                    logger.debug(f"Momentum fetch error for {asset}: {e}")
            time.sleep(1)

    def _fetch_price(self, asset: str) -> Optional[float]:
        \"\"\"Fetch current price from Binance\"\"\"
        symbol = self.SYMBOL_MAP.get(asset.lower())
        if not symbol:
            return None
        try:
            url = f"{self.BINANCE_API}/ticker/price"
            resp = self.session.get(url, params={"symbol": symbol}, timeout=2)
            if resp.ok:
                return float(resp.json().get("price", 0))
        except Exception:
            pass
        return None

    def get_momentum(self, asset: str) -> Tuple[str, float, float]:
        \"\"\"Get momentum for an asset. Returns: (direction, strength, price_change_pct)\"\"\"
        with self._lock:
            history = list(self.price_history.get(asset.lower(), []))
        if len(history) < 2:
            return "neutral", 0.0, 0.0
        now = time.time()
        cutoff = now - self.window_seconds
        window_prices = [(t, p) for t, p in history if t >= cutoff]
        if len(window_prices) < 2:
            return "neutral", 0.0, 0.0
        start_price = window_prices[0][1]
        end_price = window_prices[-1][1]
        if start_price == 0:
            return "neutral", 0.0, 0.0
        price_change_pct = ((end_price - start_price) / start_price) * 100
        threshold = CONFIG.get("MOMENTUM_THRESHOLD", 0.001) * 100
        if abs(price_change_pct) < threshold:
            return "neutral", 0.0, price_change_pct
        direction = "up" if price_change_pct > 0 else "down"
        strength = min(abs(price_change_pct) / 2.0, 1.0)
        return direction, strength, price_change_pct

    def get_current_price(self, asset: str) -> Optional[float]:
        \"\"\"Get the current price for an asset\"\"\"
        with self._lock:
            return self.current_prices.get(asset.lower())


# =============================================================================
# MARKET DISCOVERY
# =============================================================================
"""

content = content.replace(marker, database_momentum_classes, 1)

with open("/Users/jellingson/agent-swarm/swarms/trading_bots/workspace/polymarket-arbitrage/ultimate_arb_bot.py", "w") as f:
    f.write(content)

print("Step 3: Database and MomentumTracker classes added!")
