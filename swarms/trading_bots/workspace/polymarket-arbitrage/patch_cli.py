
# Read the file
with open("/Users/jellingson/agent-swarm/swarms/trading_bots/workspace/polymarket-arbitrage/ultimate_arb_bot.py", "r") as f:
    content = f.read()

# First fix: Add db.record_trade to handle_opportunity
old_handle = """        # Execute if live mode
        if self.live_mode and self.executor:
            result = self.executor.execute_trade(opp)
            self.tracker.record_opportunity(opp, executed=True, result=result)
            self.last_trade_time = now

            if result["success"]:
                logger.info(f"         ✅ SUCCESS | Profit: \${result["profit"]:.3f}")
            else:
                logger.error(f"         ❌ FAILED | {result["error"]}")
        else:
            # Simulation mode - just track
            self.tracker.record_opportunity(opp, executed=False)"""

new_handle = """        # Execute if live mode
        if self.live_mode and self.executor:
            result = self.executor.execute_trade(opp)
            self.tracker.record_opportunity(opp, executed=True, result=result)
            self.db.record_trade(opp, result)  # Record to database
            self.last_trade_time = now

            if result["success"]:
                logger.info(f"         ✅ SUCCESS | Profit: \${result["profit"]:.3f}")
            else:
                logger.error(f"         ❌ FAILED | {result["error"]}")
        else:
            # Simulation mode - just track
            self.tracker.record_opportunity(opp, executed=False)"""

content = content.replace(old_handle, new_handle, 1)

# Add new CLI arguments after --no-kelly
old_args = """    parser.add_argument("--no-kelly", action="store_true",
                       help="Disable Kelly Criterion position sizing (use fixed ORDER_SIZE instead)")

    args = parser.parse_args()"""

new_args = """    parser.add_argument("--no-kelly", action="store_true",
                       help="Disable Kelly Criterion position sizing (use fixed ORDER_SIZE instead)")
    parser.add_argument("--stats", action="store_true",
                       help="Show historical statistics from database and exit")
    parser.add_argument("--test", action="store_true",
                       help="Run quick sanity test and exit")
    parser.add_argument("--momentum", action="store_true",
                       help="Enable momentum overlay for trading decisions")
    parser.add_argument("--assets", type=str, default=None,
                       help="Comma-separated list of assets (e.g., btc,eth)")

    args = parser.parse_args()

    # Handle --stats: show historical statistics and exit
    if args.stats:
        db = Database()
        db.print_stats(days=30)
        return

    # Handle --test: run quick sanity test and exit
    if args.test:
        print("\\nRunning sanity test...")
        print("=" * 50)
        try:
            db = Database()
            print("[PASS] Database initialization")
        except Exception as e:
            print(f"[FAIL] Database initialization: {e}")
            return
        try:
            resp = requests.get(f"{CONFIG[CLOB_API]}/book", params={"token_id": "test"}, timeout=5)
            print(f"[PASS] CLOB API reachable (status: {resp.status_code})")
        except Exception as e:
            print(f"[WARN] CLOB API unreachable: {e}")
        try:
            discovery = MarketDiscovery()
            print("[PASS] MarketDiscovery initialized")
        except Exception as e:
            print(f"[FAIL] MarketDiscovery: {e}")
        if args.momentum:
            try:
                mt = MomentumTracker()
                mt.start()
                time.sleep(2)
                direction, strength, change = mt.get_momentum("btc")
                mt.stop()
                print(f"[PASS] MomentumTracker: BTC direction={direction}, strength={strength:.3f}")
            except Exception as e:
                print(f"[FAIL] MomentumTracker: {e}")
        print("=" * 50)
        print("Sanity test complete.\\n")
        return

    # Parse assets if provided
    assets_list = None
    if args.assets:
        assets_list = [a.strip().lower() for a in args.assets.split(",")]
        print(f"Using custom assets: {assets_list}")"""

content = content.replace(old_args, new_args, 1)

# Update bot instantiation to use new parameters
old_create = """    # Create and run bot
    bot = AdvancedArbitrageBot(live_mode=args.live)
    bot.run()"""

new_create = """    # Create and run bot
    bot = AdvancedArbitrageBot(
        live_mode=args.live,
        use_momentum=args.momentum,
        assets=assets_list
    )
    bot.run()"""

content = content.replace(old_create, new_create, 1)

with open("/Users/jellingson/agent-swarm/swarms/trading_bots/workspace/polymarket-arbitrage/ultimate_arb_bot.py", "w") as f:
    f.write(content)

print("CLI patches applied\!")
