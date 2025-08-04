from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import numpy as np
from statistics import mean, stdev
import math

class Trader:
    def __init__(self):
        self.position_limits = {
            "RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50,
            "CROISSANTS": 250, "JAM": 350, "DJEMBE": 60,
            "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200, "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200, "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200
        }

        self.params = {
            "VOLCANIC_ROCK": {
                "window_size": 50, "spread_factor": 1.0, "min_spread": 5.0,
                "max_spread": 20.0, "reversion_factor": 0.4, "momentum_factor": 0.1
            },
            **{voucher: {
                "window_size": 30, "spread_factor": 1.5, "min_spread": 10.0,
                "max_spread": 50.0, "reversion_factor": 0.2, "momentum_factor": 0.05
            } for voucher in [
                "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750",
                "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250",
                "VOLCANIC_ROCK_VOUCHER_10500"
            ]}
        }

        self.price_history = {p: [] for p in self.position_limits}
        self.volatility = {p: 0 for p in self.position_limits}
        self.fair_values = {p: 0 for p in self.position_limits}
        self.implied_vols = {p: [] for p in self.position_limits}
        self.strike_map = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500, "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000, "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500
        }

    def update_market_data(self, product: str, order_depth: OrderDepth) -> None:
        best_bid = max(order_depth.buy_orders.keys(), default=0)
        best_ask = min(order_depth.sell_orders.keys(), default=float('inf'))
        if best_bid > 0 and best_ask < float('inf'):
            mid_price = (best_bid + best_ask) / 2
            self.price_history[product].append(mid_price)
            window_size = self.params[product]["window_size"]
            self.price_history[product] = self.price_history[product][-window_size:]
            if len(self.price_history[product]) > 1:
                returns = np.diff(self.price_history[product]) / self.price_history[product][:-1]
                self.volatility[product] = np.std(returns) if len(returns) > 0 else 0

    def estimate_implied_volatility(self, S: float, K: float, T: float, market_price: float) -> float:
        if T <= 0:
            return 0
        intrinsic = max(0, S - K)
        if market_price <= intrinsic:
            return self.volatility.get("VOLCANIC_ROCK", 0.2)
        
        time_value = market_price - intrinsic
        if time_value <= 0 or S <= 0:
            return 0.2
        iv = (time_value / (S * math.sqrt(T))) * 10
        return max(0.1, min(iv, 1.0))

    def calculate_fair_value(self, product: str, timestamp: int) -> float:
        if not self.price_history[product]:
            return 0
        params = self.params[product]
        recent_prices = self.price_history[product][-params["window_size"]:]
        fair_value = np.mean(recent_prices) if recent_prices else 0
        if len(recent_prices) > 1:
            last_price = recent_prices[-1]
            fair_value += params["reversion_factor"] * (fair_value - last_price)
            if len(recent_prices) > 2:
                momentum = recent_prices[-1] - recent_prices[-2]
                fair_value += params["momentum_factor"] * momentum
        return fair_value

    def calculate_voucher_fair_value(self, voucher: str, rock_price: float, timestamp: int) -> float:
        strike = self.strike_map[voucher]
        days_to_expiry = max(7 - (timestamp // 100), 0)
        T = days_to_expiry / 365
        intrinsic = max(0, rock_price - strike)
        
        if days_to_expiry == 0:
            return intrinsic
        
        recent_prices = self.price_history[voucher][-10:]
        if recent_prices:
            market_price = np.mean(recent_prices)
            iv = self.estimate_implied_volatility(rock_price, strike, T, market_price)
            self.implied_vols[voucher].append(iv)
            self.implied_vols[voucher] = self.implied_vols[voucher][-50:]
            
            if len(self.implied_vols[voucher]) > 10:
                iv_mean = np.mean(self.implied_vols[voucher])
                time_value = iv_mean * rock_price * math.sqrt(T) * 0.1
                fair_value = intrinsic + time_value
                fair_value = 0.7 * fair_value + 0.3 * market_price
            else:
                fair_value = intrinsic + iv * rock_price * math.sqrt(T) * 0.1
        else:
            fair_value = intrinsic
            
        return max(0.1, fair_value)

    def calculate_spread(self, product: str, current_position: int) -> float:
        params = self.params[product]
        spread = self.volatility[product] * params["spread_factor"] * self.fair_values[product]
        spread = max(params["min_spread"], min(params["max_spread"], spread))
        position_ratio = current_position / self.position_limits[product]
        spread *= (1 + abs(position_ratio) * 0.5)
        return spread

    def trade_volcanic_products(self, state: TradingState) -> Dict[str, List[Order]]:
        orders = {}
        rock_product = "VOLCANIC_ROCK"
        
        if rock_product in state.order_depths:
            self.update_market_data(rock_product, state.order_depths[rock_product])
            self.fair_values[rock_product] = self.calculate_fair_value(rock_product, state.timestamp)

        rock_price = self.fair_values[rock_product] if self.fair_values[rock_product] else 10000

        # Trade VOLCANIC_ROCK
        if rock_product in state.order_depths:
            order_depth = state.order_depths[rock_product]
            current_position = state.position.get(rock_product, 0)
            fair_value = self.fair_values[rock_product]
            
            if fair_value > 0:
                spread = self.calculate_spread(rock_product, current_position)
                spread *= 0.6 if self.volatility[rock_product] < 0.3 else 1.2 if self.volatility[rock_product] > 1.0 else 1.0
                
                bid_price = math.floor(fair_value - spread)
                ask_price = math.ceil(fair_value + spread)
                
                position_ratio = current_position / self.position_limits[rock_product]
                if position_ratio > 0.5:
                    bid_price -= math.floor(spread * position_ratio)
                    ask_price -= math.floor(spread * position_ratio * 0.5)
                elif position_ratio < -0.5:
                    bid_price += math.floor(spread * abs(position_ratio) * 0.5)
                    ask_price += math.floor(spread * abs(position_ratio))
                
                bid_qty = min(
                    self.position_limits[rock_product] - current_position,
                    sum(abs(qty) for price, qty in order_depth.sell_orders.items() if price <= bid_price + 5),
                    self.position_limits[rock_product] // 5
                )
                ask_qty = min(
                    self.position_limits[rock_product] + current_position,
                    sum(abs(qty) for price, qty in order_depth.buy_orders.items() if price >= ask_price - 5),
                    self.position_limits[rock_product] // 5
                )
                
                orders[rock_product] = []
                if bid_qty > 0 and bid_price > 0:
                    orders[rock_product].append(Order(rock_product, bid_price, bid_qty))
                if ask_qty > 0 and ask_price > 0:
                    orders[rock_product].append(Order(rock_product, ask_price, -ask_qty))

        # Trade Vouchers
        vouchers = list(self.strike_map.keys())
        for voucher in vouchers:
            if voucher not in state.order_depths:
                continue
                
            self.update_market_data(voucher, state.order_depths[voucher])
            self.fair_values[voucher] = self.calculate_voucher_fair_value(voucher, rock_price, state.timestamp)
            
            order_depth = state.order_depths[voucher]
            current_position = state.position.get(voucher, 0)
            fair_value = self.fair_values[voucher]
            
            if fair_value <= 0:
                continue
                
            spread = self.calculate_spread(voucher, current_position)
            bid_price = math.floor(fair_value - spread)
            ask_price = math.ceil(fair_value + spread)
            
            position_ratio = current_position / self.position_limits[voucher]
            if position_ratio > 0.5:
                bid_price -= math.floor(spread * position_ratio)
            elif position_ratio < -0.5:
                ask_price += math.floor(spread * abs(position_ratio))
            
            bid_qty = min(
                self.position_limits[voucher] - current_position,
                sum(abs(qty) for price, qty in order_depth.sell_orders.items() if price <= bid_price + 10),
                self.position_limits[voucher] // 10
            )
            ask_qty = min(
                self.position_limits[voucher] + current_position,
                sum(abs(qty) for price, qty in order_depth.buy_orders.items() if price >= ask_price - 10),
                self.position_limits[voucher] // 10
            )
            
            orders[voucher] = []
            if bid_qty > 0 and bid_price > 0:
                orders[voucher].append(Order(voucher, bid_price, bid_qty))
            if ask_qty > 0 and ask_price > 0:
                orders[voucher].append(Order(voucher, ask_price, -ask_qty))
            
            # Opportunistic trading
            best_ask = min(order_depth.sell_orders.keys(), default=float('inf'))
            if best_ask < fair_value * 0.95:
                qty = min(
                    -order_depth.sell_orders.get(best_ask, 0),
                    self.position_limits[voucher] - current_position,
                    self.position_limits[voucher] // 10
                )
                if qty > 0:
                    orders[voucher].append(Order(voucher, math.floor(best_ask), qty))

        # Delta hedging
        for voucher in vouchers:
            if voucher in orders and orders[voucher] and rock_product in state.order_depths:
                delta = 0.5
                rock_qty = int(sum(o.quantity for o in orders[voucher]) * delta)
                if rock_qty != 0:
                    rock_order_depth = state.order_depths[rock_product]
                    rock_position = state.position.get(rock_product, 0)
                    if rock_qty > 0 and rock_position - rock_qty >= -self.position_limits[rock_product]:
                        best_bid = max(rock_order_depth.buy_orders.keys(), default=0)
                        hedge_qty = min(rock_qty, rock_order_depth.buy_orders.get(best_bid, 0))
                        if hedge_qty > 0:
                            orders[rock_product] = orders.get(rock_product, []) + [Order(rock_product, math.floor(best_bid), -hedge_qty)]
                    elif rock_qty < 0 and rock_position - rock_qty <= self.position_limits[rock_product]:
                        best_ask = min(rock_order_depth.sell_orders.keys(), default=float('inf'))
                        hedge_qty = min(-rock_qty, -rock_order_depth.sell_orders.get(best_ask, 0))
                        if hedge_qty > 0:
                            orders[rock_product] = orders.get(rock_product, []) + [Order(rock_product, math.floor(best_ask), hedge_qty)]

        return orders

    def find_arbitrage_opportunities(self, state: TradingState) -> Dict[str, List[Order]]:
        orders = {}
        rock_price = self.fair_values.get("VOLCANIC_ROCK", 10000)
        vouchers = list(self.strike_map.keys())
        
        for i, v1 in enumerate(vouchers):
            for v2 in vouchers[i+1:]:
                if v1 not in state.order_depths or v2 not in state.order_depths:
                    continue
                k1, k2 = self.strike_map[v1], self.strike_map[v2]
                spread = abs(k2 - k1)
                v1_ask = min(state.order_depths[v1].sell_orders.keys(), default=float('inf'))
                v2_bid = max(state.order_depths[v2].buy_orders.keys(), default=0)
                
                if v1_ask < v2_bid - spread:
                    qty = min(
                        -state.order_depths[v1].sell_orders.get(v1_ask, 0),
                        state.order_depths[v2].buy_orders.get(v2_bid, 0),
                        self.position_limits[v1] // 10
                    )
                    if qty > 0:
                        orders[v1] = orders.get(v1, []) + [Order(v1, math.floor(v1_ask), qty)]
                        orders[v2] = orders.get(v2, []) + [Order(v2, math.floor(v2_bid), -qty)]
        
        return orders

    def consolidate_orders(self, orders: Dict[str, List[Order]]) -> Dict[str, List[Order]]:
        """Consolidate multiple orders for the same product at the same price."""
        consolidated = {}
        for product, order_list in orders.items():
            # Group orders by price
            price_to_qty = {}
            for order in order_list:
                price = order.price
                qty = order.quantity
                price_to_qty[price] = price_to_qty.get(price, 0) + qty
            
            # Create consolidated orders
            consolidated_orders = []
            for price, qty in price_to_qty.items():
                if qty != 0:  # Only include non-zero quantities
                    consolidated_orders.append(Order(product, price, qty))
            
            consolidated[product] = consolidated_orders
        
        return consolidated

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result = self.trade_volcanic_products(state)
        arb_orders = self.find_arbitrage_opportunities(state)
        
        # Merge orders
        for product, orders in arb_orders.items():
            if product in result:
                result[product].extend(orders)
            else:
                result[product] = orders
        
        # Consolidate orders to avoid multiple orders at the same price
        result = self.consolidate_orders(result)
        
        return result, 0, ""