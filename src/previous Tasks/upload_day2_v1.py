from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import numpy as np
from statistics import mean, stdev
import math
import json

class Trader:
    
    def __init__(self):
        # Position limits for all products
        self.position_limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 250,
            "JAM": 350,
            "DJEMBE": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100
        }
        
        # Basket compositions
        self.basket1_contents = {
            "CROISSANTS": 6,
            "JAM": 3,
            "DJEMBE": 1
        }
        self.basket2_contents = {
            "CROISSANTS": 4,
            "JAM": 2
        }
        
        # Initialize data structures
        self.price_history = {product: [] for product in self.position_limits}
        self.trade_history = {product: [] for product in self.position_limits}
        self.volatility = {product: 0 for product in self.position_limits}
        self.fair_values = {product: 0 for product in self.position_limits}
        
        # Product-specific parameters with all required fields
        self.params = {
            "RAINFOREST_RESIN": {
                "window_size": 100,
                "spread_factor": 1.0,
                "min_spread": 0.5,
                "max_spread": 2.0,
                "reversion_factor": 0.3,
                "momentum_factor": 0.0,  # Added default
                "cycle_length": 20,      # Added default
                "cycle_strength": 0.0    # Added default
            },
            "KELP": {
                "window_size": 50,
                "spread_factor": 1.5,
                "min_spread": 1.0,
                "max_spread": 3.0,
                "cycle_length": 20,
                "cycle_strength": 0.0,
                "reversion_factor": 0.0,  # Added default
                "momentum_factor": 0.0    # Added default
            },
            "SQUID_INK": {
                "window_size": 30,
                "spread_factor": 2.0,
                "min_spread": 1.5,
                "max_spread": 5.0,
                "pattern_window": 10,
                "momentum_factor": 0.0,
                "reversion_factor": 0.0,  # Added default
                "cycle_length": 20,       # Added default
                "cycle_strength": 0.0      # Added default
            },
            "CROISSANTS": {
                "window_size": 50,
                "spread_factor": 1.2,
                "min_spread": 0.8,
                "max_spread": 2.5,
                "reversion_factor": 0.2,
                "momentum_factor": 0.0,    # Added default
                "cycle_length": 20,        # Added default
                "cycle_strength": 0.0      # Added default
            },
            "JAM": {
                "window_size": 60,
                "spread_factor": 1.0,
                "min_spread": 0.6,
                "max_spread": 2.0,
                "reversion_factor": 0.25,
                "momentum_factor": 0.0,    # Added default
                "cycle_length": 20,       # Added default
                "cycle_strength": 0.0      # Added default
            },
            "DJEMBE": {
                "window_size": 40,
                "spread_factor": 1.5,
                "min_spread": 1.0,
                "max_spread": 3.0,
                "momentum_factor": 0.0,
                "reversion_factor": 0.0,   # Added default
                "cycle_length": 20,       # Added default
                "cycle_strength": 0.0      # Added default
            },
            "PICNIC_BASKET1": {
                "window_size": 30,
                "spread_factor": 1.8,
                "min_spread": 1.2,
                "max_spread": 4.0,
                "arb_window": 10,
                "momentum_factor": 0.0,   # Added default
                "reversion_factor": 0.0,  # Added default
                "cycle_length": 20,       # Added default
                "cycle_strength": 0.0      # Added default
            },
            "PICNIC_BASKET2": {
                "window_size": 30,
                "spread_factor": 1.5,
                "min_spread": 1.0,
                "max_spread": 3.5,
                "arb_window": 10,
                "momentum_factor": 0.0,   # Added default
                "reversion_factor": 0.0,  # Added default
                "cycle_length": 20,       # Added default
                "cycle_strength": 0.0      # Added default
            }
        }
        
        self.data_loaded = False
    
    def update_market_data(self, product: str, order_depth: OrderDepth):
        """Update our market data with the latest order book information"""
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        
        if best_bid and best_ask:
            mid_price = (best_bid + best_ask) / 2
            self.price_history[product].append(mid_price)
            self.price_history[product] = self.price_history[product][-self.params[product]["window_size"]:]
            
            if len(self.price_history[product]) > 1:
                self.volatility[product] = stdev(self.price_history[product])
    
    def calculate_basket_fair_value(self, basket: str) -> float:
        """Calculate fair value of picnic baskets based on components"""
        if basket == "PICNIC_BASKET1":
            components = self.basket1_contents
        else:
            components = self.basket2_contents
        
        basket_value = 0
        for product, quantity in components.items():
            if product in self.fair_values:
                basket_value += self.fair_values[product] * quantity
            else:
                # If we don't have a fair value for a component, use last price
                if self.price_history.get(product):
                    basket_value += self.price_history[product][-1] * quantity
        
        # Apply basket-specific adjustment
        if basket in self.price_history and len(self.price_history[basket]) > 5:
            # Compare to historical basket premium/discount
            historical_avg = mean(self.price_history[basket][-5:])
            component_value = basket_value
            premium = (historical_avg - component_value) / component_value if component_value != 0 else 0
            basket_value *= (1 + premium * 0.5)  # Adjust partially toward historical premium
        
        return basket_value
    
    def simple_sine_fit(self, prices: List[float], cycle_length: int) -> float:
        """Simplified sine wave fitting using numpy"""
        if len(prices) < cycle_length or cycle_length <= 0:
            return mean(prices)
        
        try:
            t = np.arange(len(prices))
            target = prices - np.mean(prices)
            
            # Simple sine wave estimation
            sin_wave = np.sin(2 * np.pi * t / cycle_length)
            cos_wave = np.cos(2 * np.pi * t / cycle_length)
            
            # Solve for amplitude and phase
            A = np.sum(target * sin_wave) / np.sum(sin_wave**2)
            B = np.sum(target * cos_wave) / np.sum(cos_wave**2)
            
            amplitude = np.sqrt(A**2 + B**2)
            phase = np.arctan2(B, A)
            
            current_phase = 2 * np.pi * (len(prices) % cycle_length) / cycle_length + phase
            return np.mean(prices) + amplitude * np.sin(current_phase)
        except:
            return mean(prices)
    
    def calculate_fair_value(self, product: str) -> float:
        """Calculate the fair value for a product using product-specific logic"""
        if not self.price_history.get(product):
            return 0
        
        params = self.params[product]
        prices = self.price_history[product]
        
        # Special handling for picnic baskets
        if product in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            return self.calculate_basket_fair_value(product)
        
        # Existing products logic
        if product in ["RAINFOREST_RESIN", "CROISSANTS", "JAM"]:
            historical_mean = mean(prices)
            current_price = prices[-1]
            return current_price * (1 - params["reversion_factor"]) + historical_mean * params["reversion_factor"]
        
        elif product in ["KELP", "DJEMBE"]:
            return self.simple_sine_fit(prices, params["cycle_length"])
        
        elif product in ["SQUID_INK"]:
            pattern_window = params["pattern_window"]
            if len(prices) >= pattern_window:
                recent_prices = prices[-pattern_window:]
                x = np.arange(len(recent_prices))
                slope, intercept = np.polyfit(x, recent_prices, 1)
                momentum = slope * params.get("momentum_factor", 0) * pattern_window
                return recent_prices[-1] + momentum
            return prices[-1]
        
        return prices[-1]
    
    def calculate_spread(self, product: str, position: int) -> float:
        """Calculate dynamic spread based on volatility and inventory"""
        params = self.params[product]
        position_limit = self.position_limits[product]
        
        spread = params["min_spread"] + self.volatility[product] * params["spread_factor"]
        spread = min(params["max_spread"], max(params["min_spread"], spread))
        
        position_ratio = abs(position) / position_limit
        spread *= (1 + position_ratio * 0.5)
        
        return spread
    
    def find_arbitrage_opportunities(self, state: TradingState) -> Dict[str, List[Order]]:
        """Identify arbitrage opportunities between baskets and components"""
        arb_orders = {}
        
        # Calculate component values if we have price data
        component_prices = {}
        for product in ["CROISSANTS", "JAM", "DJEMBE"]:
            if product in state.order_depths:
                best_bid = max(state.order_depths[product].buy_orders.keys()) if state.order_depths[product].buy_orders else None
                best_ask = min(state.order_depths[product].sell_orders.keys()) if state.order_depths[product].sell_orders else None
                if best_bid and best_ask:
                    component_prices[product] = (best_bid + best_ask) / 2
        
        # Check PICNIC_BASKET1 arbitrage
        if "PICNIC_BASKET1" in state.order_depths and len(component_prices) >= 3:
            basket1_value = (6 * component_prices["CROISSANTS"] + 
                           3 * component_prices["JAM"] + 
                           1 * component_prices["DJEMBE"])
            
            best_basket1_bid = max(state.order_depths["PICNIC_BASKET1"].buy_orders.keys()) if state.order_depths["PICNIC_BASKET1"].buy_orders else None
            best_basket1_ask = min(state.order_depths["PICNIC_BASKET1"].sell_orders.keys()) if state.order_depths["PICNIC_BASKET1"].sell_orders else None
            
            # Buy components, sell basket if profitable
            if best_basket1_bid and best_basket1_bid > basket1_value * 1.02:  # 2% premium
                max_qty = min(
                    state.order_depths["PICNIC_BASKET1"].buy_orders[best_basket1_bid],
                    self.position_limits["PICNIC_BASKET1"] - state.position.get("PICNIC_BASKET1", 0)
                )
                if max_qty > 0:
                    arb_orders["PICNIC_BASKET1"] = [Order("PICNIC_BASKET1", best_basket1_bid, -max_qty)]
            
            # Sell components, buy basket if profitable
            elif best_basket1_ask and best_basket1_ask < basket1_value * 0.98:  # 2% discount
                max_qty = min(
                    -state.order_depths["PICNIC_BASKET1"].sell_orders[best_basket1_ask],
                    self.position_limits["PICNIC_BASKET1"] + state.position.get("PICNIC_BASKET1", 0)
                )
                if max_qty > 0:
                    arb_orders["PICNIC_BASKET1"] = [Order("PICNIC_BASKET1", best_basket1_ask, max_qty)]
        
        # Similar logic for PICNIC_BASKET2
        if "PICNIC_BASKET2" in state.order_depths and "CROISSANTS" in component_prices and "JAM" in component_prices:
            basket2_value = (4 * component_prices["CROISSANTS"] + 
                           2 * component_prices["JAM"])
            
            best_basket2_bid = max(state.order_depths["PICNIC_BASKET2"].buy_orders.keys()) if state.order_depths["PICNIC_BASKET2"].buy_orders else None
            best_basket2_ask = min(state.order_depths["PICNIC_BASKET2"].sell_orders.keys()) if state.order_depths["PICNIC_BASKET2"].sell_orders else None
            
            if best_basket2_bid and best_basket2_bid > basket2_value * 1.02:
                max_qty = min(
                    state.order_depths["PICNIC_BASKET2"].buy_orders[best_basket2_bid],
                    self.position_limits["PICNIC_BASKET2"] - state.position.get("PICNIC_BASKET2", 0)
                )
                if max_qty > 0:
                    if "PICNIC_BASKET2" not in arb_orders:
                        arb_orders["PICNIC_BASKET2"] = []
                    arb_orders["PICNIC_BASKET2"].append(Order("PICNIC_BASKET2", best_basket2_bid, -max_qty))
            
            elif best_basket2_ask and best_basket2_ask < basket2_value * 0.98:
                max_qty = min(
                    -state.order_depths["PICNIC_BASKET2"].sell_orders[best_basket2_ask],
                    self.position_limits["PICNIC_BASKET2"] + state.position.get("PICNIC_BASKET2", 0)
                )
                if max_qty > 0:
                    if "PICNIC_BASKET2" not in arb_orders:
                        arb_orders["PICNIC_BASKET2"] = []
                    arb_orders["PICNIC_BASKET2"].append(Order("PICNIC_BASKET2", best_basket2_ask, max_qty))
        
        return arb_orders
    
    def run(self, state: TradingState):
        result = {}
        conversions = 0
        trader_data = ""
        
        # First update all market data and calculate fair values
        for product in state.order_depths:
            if product in self.position_limits:
                self.update_market_data(product, state.order_depths[product])
                self.fair_values[product] = self.calculate_fair_value(product)
        
        # Check for arbitrage opportunities between baskets and components
        arb_orders = self.find_arbitrage_opportunities(state)
        
        # Generate regular trading orders
        for product in state.order_depths:
            if product not in self.position_limits:
                continue
                
            order_depth = state.order_depths[product]
            orders = []
            current_position = state.position.get(product, 0)
            position_limit = self.position_limits[product]
            
            # Skip if we already have arbitrage orders for this product
            if product in arb_orders:
                result[product] = arb_orders[product]
                continue
            
            fair_value = self.fair_values[product]
            spread = self.calculate_spread(product, current_position)
            
            bid_price = fair_value - spread
            ask_price = fair_value + spread
            
            max_buy_size = position_limit - current_position
            max_sell_size = position_limit + current_position
            
            # Place buy orders (bids)
            if max_buy_size > 0:
                for price, quantity in sorted(order_depth.sell_orders.items()):
                    if price <= bid_price and max_buy_size > 0:
                        buy_size = min(-quantity, max_buy_size)
                        orders.append(Order(product, price, buy_size))
                        max_buy_size -= buy_size
                
                if max_buy_size > 0:
                    # Adjust bid based on product type
                    params = self.params[product]
                    if product in ["RAINFOREST_RESIN", "CROISSANTS", "JAM"] and fair_value < mean(self.price_history[product]):
                        adjusted_bid = bid_price + spread * 0.3
                    elif product in ["KELP", "DJEMBE"]:
                        cycle_pos = len(self.price_history[product]) % params.get("cycle_length", 20)
                        cycle_phase = 2 * np.pi * cycle_pos / params.get("cycle_length", 20)
                        adjusted_bid = bid_price + 0.2 * spread * np.sin(cycle_phase)
                    elif product in ["SQUID_INK", "PICNIC_BASKET1", "PICNIC_BASKET2"]:
                        adjusted_bid = bid_price + spread * 0.1 * params.get("momentum_factor", 0)
                    else:
                        adjusted_bid = bid_price
                    
                    orders.append(Order(product, int(adjusted_bid), max_buy_size))
            
            # Place sell orders (asks)
            if max_sell_size > 0:
                for price, quantity in sorted(order_depth.buy_orders.items(), reverse=True):
                    if price >= ask_price and max_sell_size > 0:
                        sell_size = min(quantity, max_sell_size)
                        orders.append(Order(product, price, -sell_size))
                        max_sell_size -= sell_size
                
                if max_sell_size > 0:
                    params = self.params[product]
                    if product in ["RAINFOREST_RESIN", "CROISSANTS", "JAM"] and fair_value > mean(self.price_history[product]):
                        adjusted_ask = ask_price - spread * 0.3
                    elif product in ["KELP", "DJEMBE"]:
                        cycle_pos = len(self.price_history[product]) % params.get("cycle_length", 20)
                        cycle_phase = 2 * np.pi * cycle_pos / params.get("cycle_length", 20)
                        adjusted_ask = ask_price - 0.2 * spread * np.sin(cycle_phase)
                    elif product in ["SQUID_INK", "PICNIC_BASKET1", "PICNIC_BASKET2"]:
                        adjusted_ask = ask_price - spread * 0.1 * params.get("momentum_factor", 0)
                    else:
                        adjusted_ask = ask_price
                    
                    orders.append(Order(product, int(adjusted_ask), -max_sell_size))
            
            result[product] = orders
        
        return result, conversions, trader_data