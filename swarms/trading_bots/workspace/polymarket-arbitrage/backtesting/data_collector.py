#!/usr/bin/env python3
"""Data Collector for Polymarket Arbitrage Backtesting"""

import sqlite3
import json
import time
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path
import threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class OrderBookSnapshot:
    token_id: str
    asset: str
    side: str
    best_ask: float
    best_bid: float
    ask_liquidity: float
    bid_liquidity: float
    spread: float
    timestamp: float = field(default_factory=time.time)
    epoch: int = 0
    hour: int = 0
    day_of_week: int = 0
    
    def __post_init__(self):
        dt = datetime.fromtimestamp(self.timestamp, tz=timezone.utc)
        self.hour = dt.hour
        self.day_of_week = dt.weekday()
        if self.epoch == 0:
            self.epoch = (int(self.timestamp) // 900) * 900

@dataclass
class MarketPair:
    market_id: str
    asset: str
    timeframe: str
    up_token_id: str
    down_token_id: str
    question: str
    end_time: Optional[float] = None
    resolution: Optional[str] = None

@dataclass
class OpportunityRecord:
    opportunity_id: str
    market_id: str
    asset: str
    timestamp: float
    up_ask: float
    down_ask: float
    combined_cost: float
    spread_pct: float
    expected_profit: float
    up_liquidity: float
    down_liquidity: float
    executed: bool = False
    execution_result: Optional[Dict] = None
    epoch: int = 0
    hour: int = 0
    day_of_week: int = 0
    
    def __post_init__(self):
        dt = datetime.fromtimestamp(self.timestamp, tz=timezone.utc)
        self.hour = dt.hour
        self.day_of_week = dt.weekday()
        if self.epoch == 0:
            self.epoch = (int(self.timestamp) // 900) * 900

@dataclass
class ResolutionRecord:
    market_id: str
    asset: str
    resolution: str
    timestamp: float
    epoch: int
    final_up_price: float = 0.0
    final_down_price: float = 0.0

class DataStore:
    def __init__(self, db_path: str = "backtest_data.db"):
        self.db_path = Path(db_path)
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("""CREATE TABLE IF NOT EXISTS order_book_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT, token_id TEXT, asset TEXT, side TEXT,
                best_ask REAL, best_bid REAL, ask_liquidity REAL, bid_liquidity REAL,
                spread REAL, timestamp REAL, epoch INTEGER, hour INTEGER, day_of_week INTEGER)""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_snap_asset ON order_book_snapshots(asset, timestamp)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_snap_hour ON order_book_snapshots(hour)")
            c.execute("""CREATE TABLE IF NOT EXISTS markets (
                market_id TEXT PRIMARY KEY, asset TEXT, timeframe TEXT, up_token_id TEXT,
                down_token_id TEXT, question TEXT, end_time REAL, resolution TEXT)""")
            c.execute("""CREATE TABLE IF NOT EXISTS opportunities (
                opportunity_id TEXT PRIMARY KEY, market_id TEXT, asset TEXT, timestamp REAL,
                up_ask REAL, down_ask REAL, combined_cost REAL, spread_pct REAL,
                expected_profit REAL, up_liquidity REAL, down_liquidity REAL,
                executed INTEGER DEFAULT 0, execution_result TEXT, epoch INTEGER, hour INTEGER, day_of_week INTEGER)""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_opp_asset ON opportunities(asset)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_opp_hour ON opportunities(hour)")
            c.execute("""CREATE TABLE IF NOT EXISTS resolutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, market_id TEXT, asset TEXT,
                resolution TEXT, timestamp REAL, epoch INTEGER, final_up_price REAL, final_down_price REAL)""")
            conn.commit()
            conn.close()
    
    def record_snapshot(self, s: OrderBookSnapshot):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("INSERT INTO order_book_snapshots (token_id,asset,side,best_ask,best_bid,ask_liquidity,bid_liquidity,spread,timestamp,epoch,hour,day_of_week) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (s.token_id, s.asset, s.side, s.best_ask, s.best_bid, s.ask_liquidity, s.bid_liquidity, s.spread, s.timestamp, s.epoch, s.hour, s.day_of_week))
            conn.commit()
            conn.close()
    
    def record_snapshots_batch(self, snapshots: List[OrderBookSnapshot]):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            data = [(s.token_id, s.asset, s.side, s.best_ask, s.best_bid, s.ask_liquidity, s.bid_liquidity, s.spread, s.timestamp, s.epoch, s.hour, s.day_of_week) for s in snapshots]
            conn.executemany("INSERT INTO order_book_snapshots (token_id,asset,side,best_ask,best_bid,ask_liquidity,bid_liquidity,spread,timestamp,epoch,hour,day_of_week) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", data)
            conn.commit()
            conn.close()
    
    def record_market(self, m: MarketPair):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("INSERT OR REPLACE INTO markets (market_id,asset,timeframe,up_token_id,down_token_id,question,end_time,resolution) VALUES (?,?,?,?,?,?,?,?)",
                (m.market_id, m.asset, m.timeframe, m.up_token_id, m.down_token_id, m.question, m.end_time, m.resolution))
            conn.commit()
            conn.close()
    
    def record_opportunity(self, o: OpportunityRecord):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            exec_json = json.dumps(o.execution_result) if o.execution_result else None
            conn.execute("INSERT OR REPLACE INTO opportunities (opportunity_id,market_id,asset,timestamp,up_ask,down_ask,combined_cost,spread_pct,expected_profit,up_liquidity,down_liquidity,executed,execution_result,epoch,hour,day_of_week) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (o.opportunity_id, o.market_id, o.asset, o.timestamp, o.up_ask, o.down_ask, o.combined_cost, o.spread_pct, o.expected_profit, o.up_liquidity, o.down_liquidity, 1 if o.executed else 0, exec_json, o.epoch, o.hour, o.day_of_week))
            conn.commit()
            conn.close()
    
    def record_resolution(self, r: ResolutionRecord):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("INSERT INTO resolutions (market_id,asset,resolution,timestamp,epoch,final_up_price,final_down_price) VALUES (?,?,?,?,?,?,?)",
                (r.market_id, r.asset, r.resolution, r.timestamp, r.epoch, r.final_up_price, r.final_down_price))
            conn.execute("UPDATE markets SET resolution=? WHERE market_id=?", (r.resolution, r.market_id))
            conn.commit()
            conn.close()
    
    def get_snapshots(self, asset=None, start_time=None, end_time=None, hour=None):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            q, p = "SELECT * FROM order_book_snapshots WHERE 1=1", []
            if asset: q += " AND asset=?"; p.append(asset)
            if start_time: q += " AND timestamp>=?"; p.append(start_time)
            if end_time: q += " AND timestamp<=?"; p.append(end_time)
            if hour is not None: q += " AND hour=?"; p.append(hour)
            q += " ORDER BY timestamp"
            rows = [dict(r) for r in conn.execute(q, p).fetchall()]
            conn.close()
            return rows
    
    def get_opportunities(self, asset=None, executed_only=False, start_time=None, end_time=None):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            q, p = "SELECT * FROM opportunities WHERE 1=1", []
            if asset: q += " AND asset=?"; p.append(asset)
            if executed_only: q += " AND executed=1"
            if start_time: q += " AND timestamp>=?"; p.append(start_time)
            if end_time: q += " AND timestamp<=?"; p.append(end_time)
            q += " ORDER BY timestamp"
            rows = [dict(r) for r in conn.execute(q, p).fetchall()]
            conn.close()
            return rows
    
    def get_resolutions(self, asset=None, start_time=None, end_time=None):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            q, p = "SELECT * FROM resolutions WHERE 1=1", []
            if asset: q += " AND asset=?"; p.append(asset)
            if start_time: q += " AND timestamp>=?"; p.append(start_time)
            if end_time: q += " AND timestamp<=?"; p.append(end_time)
            q += " ORDER BY timestamp"
            rows = [dict(r) for r in conn.execute(q, p).fetchall()]
            conn.close()
            return rows
    
    def get_statistics(self):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            stats = {}
            stats["total_snapshots"] = c.execute("SELECT COUNT(*) FROM order_book_snapshots").fetchone()[0]
            stats["total_opportunities"] = c.execute("SELECT COUNT(*) FROM opportunities").fetchone()[0]
            stats["executed_opportunities"] = c.execute("SELECT COUNT(*) FROM opportunities WHERE executed=1").fetchone()[0]
            stats["total_resolutions"] = c.execute("SELECT COUNT(*) FROM resolutions").fetchone()[0]
            row = c.execute("SELECT MIN(timestamp), MAX(timestamp) FROM order_book_snapshots").fetchone()
            stats["time_range"] = {"start": row[0], "end": row[1]}
            conn.close()
            return stats

class SampleDataGenerator:
    def __init__(self, db_path="backtest_sample.db"):
        self.store = DataStore(db_path)
    
    def generate(self, num_days=7, assets=None, opportunity_rate=0.05):
        import random, uuid
        if not assets: assets = ["BTC", "ETH", "SOL", "XRP"]
        logger.info(f"Generating {num_days} days of data for {assets}")
        
        end_time = time.time()
        start_time = end_time - (num_days * 86400)
        interval = 5
        snapshots, opportunities, resolutions = [], [], []
        current_time, epoch_outcomes = start_time, {}
        
        while current_time < end_time:
            epoch = (int(current_time) // 900) * 900
            for asset in assets:
                base_up = max(0.1, min(0.9, 0.5 + random.gauss(0, 0.08)))
                base_down = max(0.1, min(0.9, 1.0 - base_up + random.gauss(0, 0.03)))
                up_ask = base_up + random.uniform(0.0025, 0.01)
                up_bid = base_up - random.uniform(0.0025, 0.01)
                down_ask = base_down + random.uniform(0.0025, 0.01)
                down_bid = base_down - random.uniform(0.0025, 0.01)
                
                if random.random() < opportunity_rate:
                    c = random.uniform(0.02, 0.08)
                    up_ask -= c/2; down_ask -= c/2
                
                liq_m = {"BTC": 1.5, "ETH": 1.2, "SOL": 0.8, "XRP": 0.6}.get(asset, 1.0)
                ask_liq = random.uniform(500, 2000) * liq_m
                
                snapshots.append(OrderBookSnapshot(f"{asset}-UP-{epoch}", asset, "UP", up_ask, up_bid, ask_liq, ask_liq*0.9, up_ask-up_bid, current_time))
                snapshots.append(OrderBookSnapshot(f"{asset}-DOWN-{epoch}", asset, "DOWN", down_ask, down_bid, ask_liq*0.9, ask_liq*0.8, down_ask-down_bid, current_time))
                
                cost = up_ask + down_ask
                if cost < 1.0:
                    opportunities.append(OpportunityRecord(str(uuid.uuid4()), f"{asset}-15m-{epoch}", asset, current_time,
                        up_ask, down_ask, cost, ((1.0-cost)/cost)*100, 1.0-cost, ask_liq, ask_liq*0.9, random.random()<0.7))
                
                ek = f"{asset}-{epoch}"
                if ek not in epoch_outcomes:
                    res = "UP" if random.random() < up_ask/(up_ask+down_ask) else "DOWN"
                    epoch_outcomes[ek] = res
                    resolutions.append(ResolutionRecord(f"{asset}-15m-{epoch}", asset, res, epoch+900, epoch,
                        1.0 if res=="UP" else 0.0, 0.0 if res=="UP" else 1.0))
            
            current_time += interval
            if len(snapshots) >= 1000:
                self.store.record_snapshots_batch(snapshots); snapshots = []
        
        if snapshots: self.store.record_snapshots_batch(snapshots)
        for o in opportunities: self.store.record_opportunity(o)
        for r in resolutions: self.store.record_resolution(r)
        return self.store.get_statistics()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--generate", action="store_true")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--db", default="backtest_data.db")
    p.add_argument("--stats", action="store_true")
    args = p.parse_args()
    if args.generate: print(SampleDataGenerator(args.db).generate(args.days))
    elif args.stats: print(DataStore(args.db).get_statistics())
    else: print("Use --generate or --stats")
