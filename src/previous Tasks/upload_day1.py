from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import numpy as np
from statistics import mean, stdev
import math

class Trader:
    
    def __init__(self):
        self.position_limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50
        }
        
        # Product-specific parameters
        self.product_params = {
            "RAINFOREST_RESIN": {
                "mean_window": 100,
                "spread_multiplier": 1.0,
                "max_spread": 2.0,
                "min_spread": 0.5
            },
            "KELP": {
                "mean_window": 50,
                "spread_multiplier": 1.5,
                "max_spread": 3.0,
                "min_spread": 1.0,
                "cycle_length": 20  # Estimated cycle length for Kelp
            },
            "SQUID_INK": {
                "mean_window": 30,
                "spread_multiplier": 2.0,
                "max_spread": 5.0,
                "min_spread": 1.5,
                "pattern_window": 10  # For pattern recognition
            }
        }
        
        # Data storage
        self.price_history = {product: [] for product in self.position_limits}
        self.trade_history = {product: [] for product in self.position_limits}
        
    def update_history(self, product: str, state: TradingState):
        """Update price and trade history for a product"""
        # Update price history from order book
        if product in state.order_depths:
            order_depth = state.order_depths[product]
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            
            if best_bid and best_ask:
                mid_price = (best_bid + best_ask) / 2
                self.price_history[product].append(mid_price)
            elif best_bid:
                self.price_history[product].append(best_bid)
            elif best_ask:
                self.price_history[product].append(best_ask)
            
            # Keep only the most recent prices
            window_size = self.product_params[product]["mean_window"]
            self.price_history[product] = self.price_history[product][-window_size:]
        
        # Update trade history
        if product in state.market_trades:
            for trade in state.market_trades[product]:
                self.trade_history[product].append((trade.price, trade.quantity, trade.timestamp))
    
    def calculate_fair_value(self, product: str) -> float:
        """Calculate product-specific fair value"""
        history = self.price_history.get(product, [])
        if not history:
            return 0
        
        params = self.product_params[product]
        
        if product == "RAINFOREST_RESIN":
            # Simple mean for stable product
            return mean(history)
        
        elif product == "KELP":
            # Incorporate cyclical behavior
            cycle_length = params["cycle_length"]
            recent_prices = history[-cycle_length:] if len(history) >= cycle_length else history
            return mean(recent_prices)
        
        elif product == "SQUID_INK":
            # Try to detect patterns - simple version looks for recent trends
            if len(history) < 2:
                return history[-1] if history else 0
            
            # Weighted average favoring recent prices
            weights = np.linspace(1, 3, len(history))
            return np.average(history, weights=weights)
        
        return mean(history)
    
    def calculate_spread(self, product: str, position: int) -> float:
        """Calculate dynamic spread based on product characteristics"""
        params = self.product_params[product]
        position_limit = self.position_limits[product]
        
        # Base spread component
        volatility = stdev(self.price_history[product]) if len(self.price_history[product]) > 1 else 0
        spread = params["min_spread"] + volatility * params["spread_multiplier"]
        spread = min(params["max_spread"], max(params["min_spread"], spread))
        
        # Inventory adjustment
        position_ratio = abs(position) / position_limit
        spread *= (1 + position_ratio * 0.5)  # Increase spread by up to 50% when near limit
        
        return spread
    
    def detect_patterns(self, product: str) -> float:
        """Detect patterns in Squid Ink prices"""
        if product != "SQUID_INK" or len(self.price_history[product]) < 10:
            return 0
        
        # Simple pattern detection - look for recent trends
        recent_prices = self.price_history[product][-10:]
        price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
        
        # If mostly increasing, predict continuation
        if sum(1 for change in price_changes if change > 0) > 6:
            return 1.0  # Positive adjustment
        # If mostly decreasing, predict continuation
        elif sum(1 for change in price_changes if change < 0) > 6:
            return -1.0  # Negative adjustment
        
        return 0
    
    def run(self, state: TradingState):
        result = {}
        conversions = 0
        trader_data = ""
        
        # Update history for all products
        for product in self.position_limits:
            self.update_history(product, state)
        
        for product in state.order_depths:
            if product not in self.position_limits:
                continue
                
            order_depth = state.order_depths[product]
            orders = []
            current_position = state.position.get(product, 0)
            position_limit = self.position_limits[product]
            
            # Calculate fair value and spread
            fair_value = self.calculate_fair_value(product)
            spread = self.calculate_spread(product, current_position)
            
            # Pattern adjustment for Squid Ink
            if product == "SQUID_INK":
                pattern_adjustment = self.detect_patterns(product)
                fair_value *= (1 + 0.05 * pattern_adjustment)  # 5% adjustment
            
            # Calculate bid and ask prices
            bid_price = fair_value - spread
            ask_price = fair_value + spread
            
            # Calculate maximum order sizes considering position limits
            max_buy_size = position_limit - current_position
            max_sell_size = position_limit + current_position
            
            # Place buy orders (bids)
            if max_buy_size > 0:
                # Look for existing sell orders below our bid price
                for price, quantity in sorted(order_depth.sell_orders.items()):
                    if price <= bid_price and max_buy_size > 0:
                        buy_size = min(-quantity, max_buy_size)
                        orders.append(Order(product, price, buy_size))
                        max_buy_size -= buy_size
                
                # If we still have capacity, place our own bid
                if max_buy_size > 0:
                    # More aggressive bidding when price is below historical mean
                    if fair_value < mean(self.price_history[product]):
                        adjusted_bid = bid_price + spread * 0.3  # Bid more aggressively
                    else:
                        adjusted_bid = bid_price
                    orders.append(Order(product, int(adjusted_bid), max_buy_size))
            
            # Place sell orders (asks)
            if max_sell_size > 0:
                # Look for existing buy orders above our ask price
                for price, quantity in sorted(order_depth.buy_orders.items(), reverse=True):
                    if price >= ask_price and max_sell_size > 0:
                        sell_size = min(quantity, max_sell_size)
                        orders.append(Order(product, price, -sell_size))
                        max_sell_size -= sell_size
                
                # If we still have capacity, place our own ask
                if max_sell_size > 0:
                    # More aggressive asking when price is above historical mean
                    if fair_value > mean(self.price_history[product]):
                        adjusted_ask = ask_price - spread * 0.3  # Ask more aggressively
                    else:
                        adjusted_ask = ask_price
                    orders.append(Order(product, int(adjusted_ask), -max_sell_size))
            
            result[product] = orders
        
        return result, conversions, trader_data