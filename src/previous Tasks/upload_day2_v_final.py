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
        
        # Optimized product-specific parameters
        self.params = {
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
            "KELP": {
                "window_size": 40,
                "spread_factor": 1.2,
                "min_spread": 0.8,
                "max_spread": 2.5,
                "cycle_length": 15,
                "cycle_strength": 0.15,
                "reversion_factor": 0.1,
                "momentum_factor": 0.05
            },
            "SQUID_INK": {
                "window_size": 25,
                "spread_factor": 1.5,
                "min_spread": 1.0,
                "max_spread": 4.0,
                "pattern_window": 8,
                "momentum_factor": 0.2,
                "reversion_factor": 0.05,
                "cycle_length": 15,
                "cycle_strength": 0.1
            },
            "CROISSANTS": {
                "window_size": 40,
                "spread_factor": 1.0,
                "min_spread": 0.5,
                "max_spread": 2.0,
                "reversion_factor": 0.3,
                "momentum_factor": 0.1,
                "cycle_length": 15,
                "cycle_strength": 0.1
            },
            "JAM": {
                "window_size": 50,
                "spread_factor": 0.9,
                "min_spread": 0.4,
                "max_spread": 1.8,
                "reversion_factor": 0.35,
                "momentum_factor": 0.1,
                "cycle_length": 15,
                "cycle_strength": 0.1
            },
            "DJEMBE": {
                "window_size": 30,
                "spread_factor": 1.2,
                "min_spread": 0.8,
                "max_spread": 2.5,
                "momentum_factor": 0.15,
                "reversion_factor": 0.1,
                "cycle_length": 15,
                "cycle_strength": 0.15
            },
            "PICNIC_BASKET1": {
                "window_size": 25,
                "spread_factor": 1.4,
                "min_spread": 0.8,
                "max_spread": 3.0,
                "arb_window": 8,
                "momentum_factor": 0.05,
                "reversion_factor": 0.1,
                "cycle_length": 15,
                "cycle_strength": 0.05
            },
            "PICNIC_BASKET2": {
                "window_size": 25,
                "spread_factor": 1.3,
                "min_spread": 0.8,
                "max_spread": 3.0,
                "arb_window": 8,
                "momentum_factor": 0.05,
                "reversion_factor": 0.1,
                "cycle_length": 15,
                "cycle_strength": 0.05
            }
        }
        
        self.data_loaded = False
    
    def update_market_data(self, product: str, order_depth: OrderDepth):
        """Update market data with weighted mid-price for better accuracy"""
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
    
    def calculate_basket_fair_value(self, basket: str) -> float:
        """Calculate fair value of picnic baskets with dynamic premium adjustment"""
        if basket == "PICNIC_BASKET1":
            components = self.basket1_contents
        else:
            components = self.basket2_contents
        
        basket_value = 0
        for product, quantity in components.items():
            if self.fair_values[product] != 0:
                basket_value += self.fair_values[product] * quantity
            elif self.price_history.get(product):
                basket_value += self.price_history[product][-1] * quantity
        
        # Dynamic premium adjustment based on recent trades
        if basket in self.price_history and len(self.price_history[basket]) > 3:
            recent_prices = self.price_history[basket][-3:]
            historical_avg = mean(recent_prices)
            component_value = basket_value
            premium = (historical_avg - component_value) / component_value if component_value != 0 else 0
            adjustment_factor = 0.7 if premium > 0 else 0.3
            basket_value *= (1 + premium * adjustment_factor)
        
        return basket_value
    
    def simple_sine_fit(self, prices: List[float], cycle_length: int) -> float:
        """Optimized sine wave fitting with amplitude dampening"""
        if len(prices) < cycle_length or cycle_length <= 0:
            return mean(prices) if prices else 0
        
        try:
            t = np.arange(len(prices))
            target = np.array(prices) - np.mean(prices)
            
            sin_wave = np.sin(2 * np.pi * t / cycle_length)
            cos_wave = np.cos(2 * np.pi * t / cycle_length)
            
            A = np.sum(target * sin_wave) / np.sum(sin_wave**2)
            B = np.sum(target * cos_wave) / np.sum(cos_wave**2)
            
            amplitude = np.sqrt(A**2 + B**2) * 0.8
            phase = np.arctan2(B, A)
            
            current_phase = 2 * np.pi * (len(prices) % cycle_length) / cycle_length + phase
            return np.mean(prices) + amplitude * np.sin(current_phase)
        except:
            return mean(prices) if prices else 0
    
    def calculate_fair_value(self, product: str) -> float:
        """Enhanced fair value with momentum and cycle integration"""
        if not self.price_history.get(product):
            return 0
        
        params = self.params[product]
        prices = self.price_history[product]
        recent_price = prices[-1]
        
        # Basket products
        if product in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            return self.calculate_basket_fair_value(product)
        
        # Base fair value: EWMA for smoother trends
        window = min(len(prices), params["window_size"])
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        ewma = np.sum(np.array(prices[-window:]) * weights)
        
        # Mean reversion component
        historical_mean = mean(prices[-window:]) if window > 0 else recent_price
        reversion = ewma * (1 - params["reversion_factor"]) + historical_mean * params["reversion_factor"]
        
        # Momentum component
        momentum = 0
        if len(prices) > 5:
            recent_returns = np.diff(prices[-5:]) / prices[-5:-1]
            momentum = recent_price + recent_price * params["momentum_factor"] * np.mean(recent_returns)
        
        # Cycle component
        cycle = self.simple_sine_fit(prices, params["cycle_length"])
        
        # Combine components
        fair_value = (
            reversion * 0.5 +
            momentum * 0.3 +
            cycle * params["cycle_strength"]
        )
        
        # Ensure non-negative and reasonable bounds
        return max(0.1, fair_value)
    
    def calculate_spread(self, product: str, position: int) -> float:
        """Dynamic spread adjustment with position and volatility"""
        params = self.params[product]
        position_limit = self.position_limits[product]
        
        # Volatility-based spread
        base_spread = params["min_spread"] + self.volatility[product] * params["spread_factor"]
        spread = min(params["max_spread"], max(params["min_spread"], base_spread))
        
        # Position-based adjustment
        position_ratio = abs(position) / position_limit
        if position_ratio > 0.8:
            spread *= (1 + position_ratio * 2.5)
        elif position_ratio > 0.4:
            spread *= (1 + position_ratio * 1.2)
        
        # Tighten spread in low volatility
        if self.volatility[product] < 0.5:
            spread *= 0.7
        
        return spread
    
    def find_arbitrage_opportunities(self, state: TradingState) -> Dict[str, List[Order]]:
        """Aggressive arbitrage with tighter thresholds and hedging"""
        arb_orders = {}
        
        # Component prices with volume-weighted mid
        component_prices = {}
        for product in ["CROISSANTS", "JAM", "DJEMBE"]:
            if product in state.order_depths:
                bids = state.order_depths[product].buy_orders
                asks = state.order_depths[product].sell_orders
                if bids and asks:
                    best_bid = max(bids.keys())
                    best_ask = min(asks.keys())
                    bid_vol = sum(abs(q) for p, q in bids.items() if p >= best_bid)
                    ask_vol = sum(abs(q) for p, q in asks.items() if p <= best_ask)
                    total_vol = bid_vol + ask_vol
                    component_prices[product] = (best_bid * ask_vol + best_ask * bid_vol) / total_vol if total_vol > 0 else (best_bid + best_ask) / 2
        
        # PICNIC_BASKET1 arbitrage
        if "PICNIC_BASKET1" in state.order_depths and len(component_prices) >= 3:
            basket1_value = (
                6 * component_prices["CROISSANTS"] +
                3 * component_prices["JAM"] +
                1 * component_prices["DJEMBE"]
            )
            
            bids = state.order_depths["PICNIC_BASKET1"].buy_orders
            asks = state.order_depths["PICNIC_BASKET1"].sell_orders
            best_basket1_bid = max(bids.keys()) if bids else 0
            best_basket1_ask = min(asks.keys()) if asks else float('inf')
            
            if best_basket1_bid > basket1_value * 1.005:
                max_qty = min(
                    bids.get(best_basket1_bid, 0),
                    self.position_limits["PICNIC_BASKET1"] - state.position.get("PICNIC_BASKET1", 0),
                    (self.position_limits["CROISSANTS"] - state.position.get("CROISSANTS", 0)) // 6,
                    (self.position_limits["JAM"] - state.position.get("JAM", 0)) // 3,
                    (self.position_limits["DJEMBE"] - state.position.get("DJEMBE", 0)) // 1
                )
                if max_qty > 0:
                    arb_orders["PICNIC_BASKET1"] = [Order("PICNIC_BASKET1", best_basket1_bid, -max_qty)]
                    for product, qty_per_basket in self.basket1_contents.items():
                        if product in state.order_depths:
                            asks = state.order_depths[product].sell_orders
                            if asks:
                                best_ask = min(asks.keys())
                                hedge_qty = min(
                                    -asks.get(best_ask, 0),
                                    max_qty * qty_per_basket,
                                    self.position_limits[product] - state.position.get(product, 0)
                                )
                                if hedge_qty > 0:
                                    arb_orders[product] = arb_orders.get(product, []) + [Order(product, best_ask, hedge_qty)]
            
            elif best_basket1_ask < basket1_value * 0.995:
                max_qty = min(
                    -asks.get(best_basket1_ask, 0),
                    self.position_limits["PICNIC_BASKET1"] + state.position.get("PICNIC_BASKET1", 0),
                    (self.position_limits["CROISSANTS"] + state.position.get("CROISSANTS", 0)) // 6,
                    (self.position_limits["JAM"] + state.position.get("JAM", 0)) // 3,
                    (self.position_limits["DJEMBE"] + state.position.get("DJEMBE", 0)) // 1
                )
                if max_qty > 0:
                    arb_orders["PICNIC_BASKET1"] = [Order("PICNIC_BASKET1", best_basket1_ask, max_qty)]
                    for product, qty_per_basket in self.basket1_contents.items():
                        if product in state.order_depths:
                            bids = state.order_depths[product].buy_orders
                            if bids:
                                best_bid = max(bids.keys())
                                hedge_qty = min(
                                    bids.get(best_bid, 0),
                                    max_qty * qty_per_basket,
                                    self.position_limits[product] + state.position.get(product, 0)
                                )
                                if hedge_qty > 0:
                                    arb_orders[product] = arb_orders.get(product, []) + [Order(product, best_bid, -hedge_qty)]
        
        # PICNIC_BASKET2 arbitrage
        if "PICNIC_BASKET2" in state.order_depths and "CROISSANTS" in component_prices and "JAM" in component_prices:
            basket2_value = (
                4 * component_prices["CROISSANTS"] +
                2 * component_prices["JAM"]
            )
            
            bids = state.order_depths["PICNIC_BASKET2"].buy_orders
            asks = state.order_depths["PICNIC_BASKET2"].sell_orders
            best_basket2_bid = max(bids.keys()) if bids else 0
            best_basket2_ask = min(asks.keys()) if asks else float('inf')
            
            if best_basket2_bid > basket2_value * 1.005:
                max_qty = min(
                    bids.get(best_basket2_bid, 0),
                    self.position_limits["PICNIC_BASKET2"] - state.position.get("PICNIC_BASKET2", 0),
                    (self.position_limits["CROISSANTS"] - state.position.get("CROISSANTS", 0)) // 4,
                    (self.position_limits["JAM"] - state.position.get("JAM", 0)) // 2
                )
                if max_qty > 0:
                    arb_orders["PICNIC_BASKET2"] = [Order("PICNIC_BASKET2", best_basket2_bid, -max_qty)]
                    for product, qty_per_basket in self.basket2_contents.items():
                        if product in state.order_depths:
                            asks = state.order_depths[product].sell_orders
                            if asks:
                                best_ask = min(asks.keys())
                                hedge_qty = min(
                                    -asks.get(best_ask, 0),
                                    max_qty * qty_per_basket,
                                    self.position_limits[product] - state.position.get(product, 0)
                                )
                                if hedge_qty > 0:
                                    arb_orders[product] = arb_orders.get(product, []) + [Order(product, best_ask, hedge_qty)]
            
            elif best_basket2_ask < basket2_value * 0.995:
                max_qty = min(
                    -asks.get(best_basket2_ask, 0),
                    self.position_limits["PICNIC_BASKET2"] + state.position.get("PICNIC_BASKET2", 0),
                    (self.position_limits["CROISSANTS"] + state.position.get("CROISSANTS", 0)) // 4,
                    (self.position_limits["JAM"] + state.position.get("JAM", 0)) // 2
                )
                if max_qty > 0:
                    arb_orders["PICNIC_BASKET2"] = [Order("PICNIC_BASKET2", best_basket2_ask, max_qty)]
                    for product, qty_per_basket in self.basket2_contents.items():
                        if product in state.order_depths:
                            bids = state.order_depths[product].buy_orders
                            if bids:
                                best_bid = max(bids.keys())
                                hedge_qty = min(
                                    bids.get(best_bid, 0),
                                    max_qty * qty_per_basket,
                                    self.position_limits[product] + state.position.get(product, 0)
                                )
                                if hedge_qty > 0:
                                    arb_orders[product] = arb_orders.get(product, []) + [Order(product, best_bid, -hedge_qty)]
        
        return arb_orders
    
    def find_statistical_arbitrage(self, state: TradingState) -> Dict[str, List[Order]]:
        """Enhanced stat arb with dynamic ratios and hedging"""
        stat_arb_orders = {}
        
        # CROISSANTS vs JAM
        if "CROISSANTS" in self.price_history and "JAM" in self.price_history:
            window = min(len(self.price_history["CROISSANTS"]), len(self.price_history["JAM"]), 15)
            if window > 5:
                croissant_prices = self.price_history["CROISSANTS"][-window:]
                jam_prices = self.price_history["JAM"][-window:]
                ratios = [c/j for c, j in zip(croissant_prices, jam_prices) if j != 0]
                if ratios:
                    typical_ratio = np.mean(ratios)
                    ratio_std = np.std(ratios) if len(ratios) > 1 else 0.1
                    current_croissant = self.price_history["CROISSANTS"][-1]
                    current_jam = self.price_history["JAM"][-1]
                    current_ratio = current_croissant / current_jam if current_jam != 0 else typical_ratio
                    
                    if current_ratio > typical_ratio + ratio_std * 1.5:
                        if "CROISSANTS" in state.order_depths and "JAM" in state.order_depths:
                            best_croissant_bid = max(state.order_depths["CROISSANTS"].buy_orders.keys())
                            best_jam_ask = min(state.order_depths["JAM"].sell_orders.keys())
                            
                            max_croissant_sell = min(
                                state.order_depths["CROISSANTS"].buy_orders.get(best_croissant_bid, 0),
                                self.position_limits["CROISSANTS"] + state.position.get("CROISSANTS", 0)
                            )
                            max_jam_buy = min(
                                -state.order_depths["JAM"].sell_orders.get(best_jam_ask, 0),
                                self.position_limits["JAM"] - state.position.get("JAM", 0)
                            )
                            
                            qty_ratio = current_jam / best_croissant_bid if best_croissant_bid != 0 else 1
                            qty = min(max_croissant_sell, int(max_jam_buy * qty_ratio), 50)
                            if qty > 0:
                                stat_arb_orders["CROISSANTS"] = [Order("CROISSANTS", best_croissant_bid, -qty)]
                                jam_qty = int(qty / qty_ratio)
                                if jam_qty > 0:
                                    stat_arb_orders["JAM"] = [Order("JAM", best_jam_ask, jam_qty)]
                    
                    elif current_ratio < typical_ratio - ratio_std * 1.5:
                        if "CROISSANTS" in state.order_depths and "JAM" in state.order_depths:
                            best_croissant_ask = min(state.order_depths["CROISSANTS"].sell_orders.keys())
                            best_jam_bid = max(state.order_depths["JAM"].buy_orders.keys())
                            
                            max_croissant_buy = min(
                                -state.order_depths["CROISSANTS"].sell_orders.get(best_croissant_ask, 0),
                                self.position_limits["CROISSANTS"] - state.position.get("CROISSANTS", 0)
                            )
                            max_jam_sell = min(
                                state.order_depths["JAM"].buy_orders.get(best_jam_bid, 0),
                                self.position_limits["JAM"] + state.position.get("JAM", 0)
                            )
                            
                            qty_ratio = best_jam_bid / current_croissant if current_croissant != 0 else 1
                            qty = min(max_croissant_buy, int(max_jam_sell * qty_ratio), 50)
                            if qty > 0:
                                stat_arb_orders["CROISSANTS"] = [Order("CROISSANTS", best_croissant_ask, qty)]
                                jam_qty = int(qty / qty_ratio)
                                if jam_qty > 0:
                                    stat_arb_orders["JAM"] = [Order("JAM", best_jam_bid, -jam_qty)]
        
        return stat_arb_orders
    
    def run(self, state: TradingState):
        result = {}
        conversions = 0
        trader_data = ""
        
        # Update market data
        for product in state.order_depths:
            if product in self.position_limits:
                self.update_market_data(product, state.order_depths[product])
                self.fair_values[product] = self.calculate_fair_value(product)
        
        # Execute arbitrage strategies
        arb_orders = self.find_arbitrage_opportunities(state)
        stat_arb_orders = self.find_statistical_arbitrage(state)
        
        # Market making
        for product in state.order_depths:
            if product not in self.position_limits:
                continue
                
            if product in arb_orders or product in stat_arb_orders:
                continue
                
            order_depth = state.order_depths[product]
            orders = []
            current_position = state.position.get(product, 0)
            position_limit = self.position_limits[product]
            
            fair_value = self.fair_values[product]
            if fair_value == 0:
                continue
                
            spread = self.calculate_spread(product, current_position)
            
            if self.volatility[product] < 0.3 and len(self.price_history[product]) > 5:
                spread *= 0.6
            elif self.volatility[product] > 1.0:
                spread *= 1.2
            
            bid_price = fair_value - spread
            ask_price = fair_value + spread
            
            position_ratio = current_position / position_limit
            if position_ratio > 0.5:
                bid_price -= spread * position_ratio
                ask_price -= spread * position_ratio * 0.5
            elif position_ratio < -0.5:
                bid_price += spread * abs(position_ratio) * 0.5
                ask_price += spread * abs(position_ratio)
            
            bid_price = math.floor(bid_price)
            ask_price = math.ceil(ask_price)
            
            bid_qty = min(
                position_limit - current_position,
                sum(abs(qty) for price, qty in order_depth.sell_orders.items() if price <= bid_price + 1)
            )
            ask_qty = min(
                position_limit + current_position,
                sum(abs(qty) for price, qty in order_depth.buy_orders.items() if price >= ask_price - 1)
            )
            
            max_trade_size = position_limit // 5
            bid_qty = min(bid_qty, max_trade_size)
            ask_qty = min(ask_qty, max_trade_size)
            
            if bid_qty > 0 and bid_price > 0:
                orders.append(Order(product, bid_price, bid_qty))
            if ask_qty > 0 and ask_price > 0:
                orders.append(Order(product, ask_price, -ask_qty))
            
            result[product] = orders
        
        # Merge orders
        for product in arb_orders:
            result[product] = arb_orders[product]
        for product in stat_arb_orders:
            if product in result:
                result[product].extend(stat_arb_orders[product])
            else:
                result[product] = stat_arb_orders[product]
        
        return result, conversions, trader_data