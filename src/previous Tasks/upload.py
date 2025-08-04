from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import numpy as np
from statistics import mean, stdev
import math
import json

class Trader:   
    def __init__(self):
        # Initialize parameters first
        self.params = {
            "VOLCANIC_ROCK": {
                "window_size": 30,
                "spread_factor": 1.0,
                "min_spread": 0.5,
                "max_spread": 2.0,
                "reversion_factor": 0.3,
                "momentum_factor": 0.1
            },
            "VOLCANIC_ROCK_VOUCHER_9500": {
                "window_size": 20,
                "spread_factor": 1.2,
                "min_spread": 0.8,
                "max_spread": 2.5
            },
            "VOLCANIC_ROCK_VOUCHER_9750": {
                "window_size": 20,
                "spread_factor": 1.2,
                "min_spread": 0.8,
                "max_spread": 2.5
            },
            "VOLCANIC_ROCK_VOUCHER_10000": {
                "window_size": 20,
                "spread_factor": 1.2,
                "min_spread": 0.8,
                "max_spread": 2.5
            },
            "VOLCANIC_ROCK_VOUCHER_10250": {
                "window_size": 20,
                "spread_factor": 1.2,
                "min_spread": 0.8,
                "max_spread": 2.5
            },
            "VOLCANIC_ROCK_VOUCHER_10500": {
                "window_size": 20,
                "spread_factor": 1.2,
                "min_spread": 0.8,
                "max_spread": 2.5
            },
            # Other products from your original implementation
            "RAINFOREST_RESIN": {
                "window_size": 80,
                "spread_factor": 0.8,
                "min_spread": 0.3,
                "max_spread": 1.5,
                "reversion_factor": 0.4,
                "momentum_factor": 0.1,
                "cycle_length": 15,
                "cycle_strength": 0.1
            },
            # ... (include all other product parameters from your original code)
        }
        
        # Then initialize position limits
        self.position_limits = {
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 250,
            "JAM": 350,
            "DJEMBE": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100
        }
        
        # Strike prices for vouchers
        self.strike_prices = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500,
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500
        }
        
        # Days remaining until expiration (starts at 7)
        self.days_to_expiry = 7
        
        # Initialize data structures
        self.price_history = {product: [] for product in self.position_limits}
        self.volatility = {product: 0 for product in self.position_limits}
        self.fair_values = {product: 0 for product in self.position_limits}
        
        # Basket compositions (if still needed)
        self.basket1_contents = {
            "CROISSANTS": 6,
            "JAM": 3,
            "DJEMBE": 1
        }
        self.basket2_contents = {
            "CROISSANTS": 4,
            "JAM": 2
        }

    def update_days_to_expiry(self, state: TradingState):
        """Update days remaining until voucher expiration"""
        self.days_to_expiry = max(2, 7 - (state.timestamp // 1000000))

    def update_market_data(self, product: str, order_depth: OrderDepth):
        """Update market data with weighted mid-price"""
        bid_prices = list(order_depth.buy_orders.keys())
        ask_prices = list(order_depth.sell_orders.keys())
        
        if bid_prices and ask_prices:
            best_bid = max(bid_prices)
            best_ask = min(ask_prices)
            bid_vol = sum(abs(qty) for price, qty in order_depth.buy_orders.items() if price >= best_bid)
            ask_vol = sum(abs(qty) for price, qty in order_depth.sell_orders.items() if price <= best_ask)
            total_vol = bid_vol + ask_vol
            
            if total_vol > 0:
                mid_price = (best_bid * ask_vol + best_ask * bid_vol) / total_vol
            else:
                mid_price = (best_bid + best_ask) / 2
                
            self.price_history[product].append(mid_price)
            self.price_history[product] = self.price_history[product][-self.params[product]["window_size"]:]
            
            if len(self.price_history[product]) > 2:
                returns = np.diff(self.price_history[product]) / self.price_history[product][:-1]
                self.volatility[product] = stdev(returns) * math.sqrt(252) if len(returns) > 0 else 0

    def calculate_voucher_fair_value(self, product: str, state: TradingState) -> float:
        """Calculate fair value of a voucher"""
        if "VOLCANIC_ROCK" not in state.order_depths or product not in self.strike_prices:
            return 0
        
        # Get underlying price
        rock_bids = state.order_depths["VOLCANIC_ROCK"].buy_orders
        rock_asks = state.order_depths["VOLCANIC_ROCK"].sell_orders
        if not rock_bids or not rock_asks:
            return 0
            
        best_bid = max(rock_bids.keys())
        best_ask = min(rock_asks.keys())
        underlying_price = (best_bid + best_ask) / 2
        
        strike = self.strike_prices[product]
        time_value = max(0.1, self.days_to_expiry / 7)  # Normalized time decay
        
        # Intrinsic value
        intrinsic = max(0, underlying_price - strike)
        
        # Time value component
        volatility = self.volatility.get("VOLCANIC_ROCK", 0.2)
        time_component = underlying_price * volatility * math.sqrt(time_value)
        
        return intrinsic + time_component

    def calculate_fair_value(self, product: str) -> float:
        """Calculate fair value for any product"""
        if not self.price_history.get(product):
            return 0
            
        if product in self.strike_prices:
            # This shouldn't happen as we call calculate_voucher_fair_value directly
            return 0
            
        # Original fair value calculation for non-voucher products
        params = self.params[product]
        prices = self.price_history[product]
        
        if len(prices) == 0:
            return 0
            
        recent_price = prices[-1]
        window = min(len(prices), params["window_size"])
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        ewma = np.sum(np.array(prices[-window:]) * weights)
        
        historical_mean = mean(prices[-window:]) if window > 0 else recent_price
        reversion = ewma * (1 - params["reversion_factor"]) + historical_mean * params["reversion_factor"]
        
        momentum = 0
        if len(prices) > 5:
            recent_returns = np.diff(prices[-5:]) / prices[-5:-1]
            momentum = recent_price + recent_price * params["momentum_factor"] * np.mean(recent_returns)
        
        return max(0.1, reversion * 0.5 + momentum * 0.3)

    def calculate_spread(self, product: str, position: int) -> float:
        """Dynamic spread adjustment"""
        params = self.params.get(product, {})
        position_limit = self.position_limits.get(product, 1)
        
        # Get parameters with defaults
        min_spread = params.get("min_spread", 0.5)
        max_spread = params.get("max_spread", 2.0)
        spread_factor = params.get("spread_factor", 1.0)
        
        # Volatility-based spread
        base_spread = min_spread + self.volatility[product] * spread_factor
        spread = min(max_spread, max(min_spread, base_spread))
        
        # Position-based adjustment
        position_ratio = abs(position) / position_limit
        if position_ratio > 0.8:
            spread *= (1 + position_ratio * 2.5)
        elif position_ratio > 0.4:
            spread *= (1 + position_ratio * 1.2)
        
        return spread

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        trader_data = ""
        
        # Update days to expiry
        self.update_days_to_expiry(state)
        
        # Update market data for all products
        for product in state.order_depths:
            if product in self.position_limits:
                self.update_market_data(product, state.order_depths[product])
                if product in self.strike_prices:
                    self.fair_values[product] = self.calculate_voucher_fair_value(product, state)
                else:
                    self.fair_values[product] = self.calculate_fair_value(product)
        
        # Market making for all products
        for product in state.order_depths:
            if product not in self.position_limits:
                continue
                
            order_depth = state.order_depths[product]
            orders = []
            current_position = state.position.get(product, 0)
            position_limit = self.position_limits[product]
            
            fair_value = self.fair_values[product]
            if fair_value == 0:
                continue
                
            spread = self.calculate_spread(product, current_position)
            
            # Adjust spread for vouchers near expiration
            if product in self.strike_prices:
                spread *= min(1.5, 1 + (7 - self.days_to_expiry)/7)
            
            bid_price = fair_value - spread
            ask_price = fair_value + spread
            
            # Position-based adjustments
            position_ratio = current_position / position_limit
            if position_ratio > 0.5:
                bid_price -= spread * position_ratio
                ask_price -= spread * position_ratio * 0.5
            elif position_ratio < -0.5:
                bid_price += spread * abs(position_ratio) * 0.5
                ask_price += spread * abs(position_ratio)
            
            bid_price = math.floor(bid_price)
            ask_price = math.ceil(ask_price)
            
            # Calculate quantities
            bid_qty = min(
                position_limit - current_position,
                sum(abs(qty) for price, qty in order_depth.sell_orders.items() if price <= bid_price + 1)
            )
            ask_qty = min(
                position_limit + current_position,
                sum(abs(qty) for price, qty in order_depth.buy_orders.items() if price >= ask_price - 1)
            )
            
            max_trade_size = position_limit // (3 if product in self.strike_prices else 5)
            bid_qty = min(bid_qty, max_trade_size)
            ask_qty = min(ask_qty, max_trade_size)
            
            if bid_qty > 0 and bid_price > 0:
                orders.append(Order(product, bid_price, bid_qty))
            if ask_qty > 0 and ask_price > 0:
                orders.append(Order(product, ask_price, -ask_qty))
            
            if orders:
                result[product] = orders
        
        return result, conversions, trader_data