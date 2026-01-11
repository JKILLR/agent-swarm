#!/usr/bin/env python3
"""
Polymarket Trade Execution Helper
=================================

Executes both legs of an arbitrage trade (buy UP + buy DOWN).

SETUP:
    1. Export your private key from https://reveal.magic.link/polymarket
    2. Set environment variables:
       export POLY_PRIVATE_KEY="your-key"
       export POLY_FUNDER_ADDRESS="your-address"

USAGE:
    # Dry run (no actual trades)
    python execute_trade.py --up-token <id> --down-token <id> --size 100 --dry-run
    
    # Live execution
    python execute_trade.py --up-token <id> --down-token <id> --size 100

WARNING: This executes real trades with real money. Always dry-run first!
"""

import os
import sys
import json
import argparse
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
import requests

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY
    CLOB_AVAILABLE = True
except ImportError:
    CLOB_AVAILABLE = False
    print("❌ py-clob-client not installed!")
    print("   Run: pip install py-clob-client")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'CLOB_HOST': 'https://clob.polymarket.com',
    'CHAIN_ID': 137,  # Polygon
    'MAX_SLIPPAGE': 0.02,  # 2% max slippage per leg
}


# =============================================================================
# EXECUTION ENGINE
# =============================================================================

class TradeExecutor:
    """Executes arbitrage trades on Polymarket"""
    
    def __init__(self, private_key: str, funder_address: str, signature_type: int = 1):
        """
        Initialize executor with wallet credentials.
        
        Args:
            private_key: Your wallet's private key
            funder_address: Address that holds your USDC
            signature_type: 1 for Magic/email, 2 for browser wallet, 0 for EOA
        """
        self.client = ClobClient(
            host=CONFIG['CLOB_HOST'],
            key=private_key,
            chain_id=CONFIG['CHAIN_ID'],
            signature_type=signature_type,
            funder=funder_address
        )
        
        # Create/derive API credentials
        self.client.set_api_creds(self.client.create_or_derive_api_creds())
        print("✅ Initialized CLOB client")
    
    def get_price(self, token_id: str) -> dict:
        """Get current price and spread for a token"""
        try:
            mid = self.client.get_midpoint(token_id)
            price = self.client.get_price(token_id, side="BUY")
            return {
                'midpoint': float(mid) if mid else 0,
                'buy_price': float(price) if price else 0
            }
        except Exception as e:
            print(f"⚠️ Error getting price: {e}")
            return {'midpoint': 0, 'buy_price': 0}
    
    def get_balance(self) -> dict:
        """Get current wallet balances"""
        # Note: This endpoint may not be officially documented
        try:
            # Try to get open orders as a proxy for account status
            orders = self.client.get_orders()
            return {'status': 'connected', 'open_orders': len(orders) if orders else 0}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def calculate_trade(
        self,
        up_token_id: str,
        down_token_id: str,
        size_usd: float
    ) -> dict:
        """
        Calculate trade details before execution.
        
        Args:
            up_token_id: Token ID for UP outcome
            down_token_id: Token ID for DOWN outcome
            size_usd: Total position size in USD
        
        Returns:
            Trade calculation details
        """
        # Get current prices
        up_price = self.get_price(up_token_id)
        down_price = self.get_price(down_token_id)
        
        up_cost = up_price['buy_price'] or up_price['midpoint']
        down_cost = down_price['buy_price'] or down_price['midpoint']
        
        if up_cost == 0 or down_cost == 0:
            return {'error': 'Could not fetch prices'}
        
        combined_cost = up_cost + down_cost
        spread_profit = 1.0 - combined_cost
        spread_percent = (spread_profit / combined_cost) * 100
        
        # Calculate shares
        shares = size_usd / combined_cost
        
        return {
            'up_token_id': up_token_id,
            'down_token_id': down_token_id,
            'up_price': up_cost,
            'down_price': down_cost,
            'combined_cost': combined_cost,
            'spread_profit_per_share': spread_profit,
            'spread_percent': spread_percent,
            'shares': shares,
            'up_cost_total': shares * up_cost,
            'down_cost_total': shares * down_cost,
            'total_cost': size_usd,
            'expected_profit': shares * spread_profit,
            'is_profitable': spread_profit > 0
        }
    
    def execute_leg(
        self,
        token_id: str,
        size: float,
        price: float,
        order_type: str = "GTC",
        dry_run: bool = True
    ) -> dict:
        """
        Execute one leg of the trade.
        
        Args:
            token_id: Token to buy
            size: Number of shares
            price: Price per share
            order_type: GTC, FOK, or FAK
            dry_run: If True, don't actually submit
        
        Returns:
            Order result
        """
        result = {
            'token_id': token_id,
            'side': 'BUY',
            'size': size,
            'price': price,
            'order_type': order_type,
            'dry_run': dry_run
        }
        
        if dry_run:
            result['status'] = 'DRY_RUN'
            result['message'] = 'Order not submitted (dry run mode)'
            return result
        
        try:
            # Create order
            order_args = OrderArgs(
                price=price,
                size=size,
                side=BUY,
                token_id=token_id
            )
            
            signed_order = self.client.create_order(order_args)
            
            # Submit order
            if order_type == "GTC":
                response = self.client.post_order(signed_order, OrderType.GTC)
            elif order_type == "FOK":
                response = self.client.post_order(signed_order, OrderType.FOK)
            else:
                response = self.client.post_order(signed_order)
            
            result['status'] = 'SUBMITTED'
            result['response'] = response
            
        except Exception as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)
        
        return result
    
    def execute_arbitrage(
        self,
        up_token_id: str,
        down_token_id: str,
        size_usd: float,
        dry_run: bool = True
    ) -> dict:
        """
        Execute complete arbitrage trade (both legs).
        
        Args:
            up_token_id: Token ID for UP outcome
            down_token_id: Token ID for DOWN outcome
            size_usd: Total position size in USD
            dry_run: If True, don't execute
        
        Returns:
            Complete trade result
        """
        print(f"\n{'='*60}")
        print(f"{'DRY RUN - ' if dry_run else ''}EXECUTING ARBITRAGE TRADE")
        print(f"{'='*60}")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Size: ${size_usd:.2f}")
        print(f"{'='*60}\n")
        
        # Calculate trade
        calc = self.calculate_trade(up_token_id, down_token_id, size_usd)
        
        if 'error' in calc:
            print(f"❌ Error: {calc['error']}")
            return calc
        
        # Print calculation
        print("📊 TRADE CALCULATION:")
        print(f"   UP price:    ${calc['up_price']:.4f}")
        print(f"   DOWN price:  ${calc['down_price']:.4f}")
        print(f"   Combined:    ${calc['combined_cost']:.4f}")
        print(f"   Spread:      {calc['spread_percent']:.2f}%")
        print(f"   Shares:      {calc['shares']:.2f}")
        print(f"   Expected P&L: ${calc['expected_profit']:.2f}")
        print()
        
        if not calc['is_profitable']:
            print("❌ Trade is not profitable (spread <= 0)")
            return {'error': 'Not profitable', 'calculation': calc}
        
        # Execute legs
        print("📤 EXECUTING ORDERS...")
        
        # Leg 1: Buy UP
        print(f"\n   [1/2] Buying UP...")
        up_result = self.execute_leg(
            token_id=up_token_id,
            size=calc['shares'],
            price=calc['up_price'] * (1 + CONFIG['MAX_SLIPPAGE']),  # Add slippage buffer
            dry_run=dry_run
        )
        print(f"         Status: {up_result['status']}")
        
        # Leg 2: Buy DOWN
        print(f"\n   [2/2] Buying DOWN...")
        down_result = self.execute_leg(
            token_id=down_token_id,
            size=calc['shares'],
            price=calc['down_price'] * (1 + CONFIG['MAX_SLIPPAGE']),
            dry_run=dry_run
        )
        print(f"         Status: {down_result['status']}")
        
        # Summary
        result = {
            'calculation': calc,
            'up_order': up_result,
            'down_order': down_result,
            'dry_run': dry_run
        }
        
        print(f"\n{'='*60}")
        if dry_run:
            print("✅ DRY RUN COMPLETE - No orders submitted")
        else:
            up_ok = up_result.get('status') == 'SUBMITTED'
            down_ok = down_result.get('status') == 'SUBMITTED'
            
            if up_ok and down_ok:
                print("✅ BOTH ORDERS SUBMITTED SUCCESSFULLY")
            elif up_ok or down_ok:
                print("⚠️ PARTIAL EXECUTION - One leg failed!")
            else:
                print("❌ BOTH ORDERS FAILED")
        print(f"{'='*60}\n")
        
        return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Execute Polymarket arbitrage trades")
    
    # Token IDs
    parser.add_argument('--up-token', '-u', type=str, required=True,
                        help='Token ID for UP outcome')
    parser.add_argument('--down-token', '-d', type=str, required=True,
                        help='Token ID for DOWN outcome')
    
    # Trade parameters
    parser.add_argument('--size', '-s', type=float, required=True,
                        help='Position size in USD')
    parser.add_argument('--dry-run', action='store_true',
                        help='Simulate without executing')
    
    # Wallet (can also use env vars)
    parser.add_argument('--private-key', type=str,
                        help='Private key (or set POLY_PRIVATE_KEY env var)')
    parser.add_argument('--funder', type=str,
                        help='Funder address (or set POLY_FUNDER_ADDRESS env var)')
    
    args = parser.parse_args()
    
    # Get credentials
    private_key = args.private_key or os.environ.get('POLY_PRIVATE_KEY')
    funder = args.funder or os.environ.get('POLY_FUNDER_ADDRESS')
    
    if not private_key:
        print("❌ No private key provided!")
        print("   Set POLY_PRIVATE_KEY env var or use --private-key")
        sys.exit(1)
    
    if not funder:
        print("❌ No funder address provided!")
        print("   Set POLY_FUNDER_ADDRESS env var or use --funder")
        sys.exit(1)
    
    # Confirm if not dry run
    if not args.dry_run:
        print("\n⚠️  WARNING: This will execute REAL trades with REAL money!")
        print(f"   UP Token:   {args.up_token[:30]}...")
        print(f"   DOWN Token: {args.down_token[:30]}...")
        print(f"   Size:       ${args.size:.2f}")
        print()
        confirm = input("Type 'EXECUTE' to confirm: ")
        
        if confirm != 'EXECUTE':
            print("❌ Cancelled")
            sys.exit(0)
    
    # Initialize executor
    try:
        executor = TradeExecutor(private_key, funder)
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        sys.exit(1)
    
    # Execute
    result = executor.execute_arbitrage(
        up_token_id=args.up_token,
        down_token_id=args.down_token,
        size_usd=args.size,
        dry_run=args.dry_run
    )
    
    # Output result as JSON
    print("\n📋 RESULT (JSON):")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
