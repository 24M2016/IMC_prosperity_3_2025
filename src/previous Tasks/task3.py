from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import numpy as np
from statistics import mean
import math
import pandas as pd
import jsonpickle

class MLModel:
    def __init__(self, input_dim: int, hidden_dim: int = 12):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Initialize weights with smaller values
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.001
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * 0.001
        self.b2 = np.zeros((1, 1))
        # Smaller learning rate
        self.lr = 0.001
        # For numerical stability
        self.epsilon = 1e-8

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, X):
        self.X = X
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2

    def backward(self, y_pred, y_true):
        m = y_true.shape[0]
        
        # Gradient of output layer with clipping
        dz2 = np.clip(y_pred - y_true, -1e6, 1e6)
        
        # Gradient clipping for weights and biases
        dW2 = np.clip(np.dot(self.a1.T, dz2) / m, -1e4, 1e4)
        db2 = np.clip(np.sum(dz2, axis=0, keepdims=True) / m, -1e4, 1e4)
        
        # Gradient of hidden layer with numerical stability
        da1 = np.clip(np.dot(dz2, self.W2.T), -1e6, 1e6)
        relu_grad = np.where(self.z1 > 0, 1.0, 0.0)
        dz1 = np.clip(da1, -1e6, 1e6) * relu_grad
        
        # More gradient clipping
        dW1 = np.clip(np.dot(self.X.T, dz1) / m, -1e4, 1e4)
        db1 = np.clip(np.sum(dz1, axis=0, keepdims=True) / m, -1e4, 1e4)

        # Update parameters
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y):
        y_pred = self.forward(X)
        
        # Calculate loss with numerical stability
        error = np.clip(y_pred - y, -1e6, 1e6)
        squared_error = np.clip(error ** 2, -1e12, 1e12)
        loss = np.mean(squared_error)
        
        self.backward(y_pred, y)
        return loss
    

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
                "window_size": 15, "spread_factor": 0.6, "min_spread": 2.0,
                "max_spread": 8.0, "reversion_factor": 0.15, "momentum_factor": 0.02
            },
            **{voucher: {
                "window_size": 10, "spread_factor": 0.9, "min_spread": 3.0,
                "max_spread": 20.0, "reversion_factor": 0.08, "momentum_factor": 0.01
            } for voucher in [
                "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750",
                "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250",
                "VOLCANIC_ROCK_VOUCHER_10500"
            ]}
        }

        self.price_history = {p: [] for p in self.position_limits}
        self.volatility = {p: 0.2 for p in self.position_limits}
        self.fair_values = {p: 0 for p in self.position_limits}
        self.implied_vols = {p: [] for p in self.position_limits}
        self.strike_map = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500, "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000, "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500
        }
        self.base_iv = []
        self.iv_params = {}
        self.training_data = {p: [] for p in self.position_limits}

        input_dim = 8  # mid_price, volatility, T, m_t, v_t, position_ratio, base_iv, prev_fair_value
        self.ml_fair_value = {p: MLModel(input_dim) for p in self.position_limits}
        self.ml_implied_vol = {v: MLModel(input_dim) for v in self.strike_map.keys()}

    def update_market_data(self, product: str, order_depth: OrderDepth) -> None:
        best_bid = max(order_depth.buy_orders.keys(), default=0)
        best_ask = min(order_depth.sell_orders.keys(), default=float('inf'))
        if best_bid > 0 and best_ask < float('inf'):
            mid_price = (best_bid + best_ask) / 2
            self.price_history[product].append(mid_price)
            self.price_history[product] = self.price_history[product][-50:]  # Cap at 50
            if len(self.price_history[product]) > 1:
                returns = np.diff(self.price_history[product]) / np.array(self.price_history[product][:-1])
                self.volatility[product] = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.2

    def estimate_implied_volatility(self, S: float, K: float, T: float, market_price: float) -> float:
        if T <= 0 or market_price <= 0 or S <= 0:
            return 0.2
        intrinsic = max(0, S - K)
        if market_price <= intrinsic:
            return 0.2
        time_value = market_price - intrinsic
        iv = np.sqrt(2 * np.pi / T) * time_value / S
        return max(0.1, min(iv, 1.0))

    def fit_implied_volatility_curve(self, rock_price: float, timestamp: int) -> None:
        if rock_price <= 0:
            return
        vouchers = list(self.strike_map.keys())
        m_t = []
        v_t = []
        days_to_expiry = max(7 - (timestamp // 100), 0)
        T = days_to_expiry / 365

        for voucher in vouchers:
            if not self.price_history[voucher]:
                continue
            strike = self.strike_map[voucher]
            market_price = mean(self.price_history[voucher][-5:]) if len(self.price_history[voucher]) >= 5 else mean(self.price_history[voucher])
            iv = self.estimate_implied_volatility(rock_price, strike, T, market_price)
            self.implied_vols[voucher].append(iv)
            self.implied_vols[voucher] = self.implied_vols[voucher][-50:]
            if T > 0 and rock_price > 0:
                m = np.log(strike / rock_price) / np.sqrt(T)
                if not np.isnan(m) and not np.isinf(m):
                    m_t.append(m)
                    v_t.append(iv)

        if len(m_t) > 2:
            try:
                # Scale the data to improve numerical stability
                m_t_scaled = np.array(m_t)
                v_t_scaled = np.array(v_t)
                
                # Remove any remaining NaN or inf values
                mask = ~np.isnan(m_t_scaled) & ~np.isinf(m_t_scaled) & ~np.isnan(v_t_scaled) & ~np.isinf(v_t_scaled)
                m_t_scaled = m_t_scaled[mask]
                v_t_scaled = v_t_scaled[mask]
                
                if len(m_t_scaled) > 2:
                    # Add some regularization by using a weighted least squares approach
                    coeffs = np.polyfit(m_t_scaled, v_t_scaled, 2, w=np.ones_like(v_t_scaled)/np.maximum(0.1, v_t_scaled))
                    self.iv_params = {'a': coeffs[0], 'b': coeffs[1], 'c': coeffs[2]}
                    self.base_iv.append(coeffs[2])
                    self.base_iv = self.base_iv[-100:]
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Failed to fit IV curve: {e}")
                # Fall back to previous parameters or default values
                if not self.base_iv:
                    self.base_iv.append(0.2)

    def prepare_features(self, product: str, rock_price: float, timestamp: int, current_position: int) -> np.ndarray:
        days_to_expiry = max(7 - (timestamp // 100), 0)
        T = days_to_expiry / 365
        mid_price = mean(self.price_history[product][-5:]) if self.price_history[product] and len(self.price_history[product]) >= 5 else 0
        volatility = self.volatility[product]
        position_ratio = current_position / self.position_limits[product]
        base_iv = self.base_iv[-1] if self.base_iv else 0.2
        prev_fair_value = self.fair_values[product]

        if product in self.strike_map:
            strike = self.strike_map[product]
            m_t = np.log(strike / rock_price) / np.sqrt(T) if T > 0 and rock_price > 0 else 0
            v_t = self.implied_vols[product][-1] if self.implied_vols[product] else 0.2
        else:
            m_t = 0
            v_t = volatility

        features = np.array([[mid_price, volatility, T, m_t, v_t, position_ratio, base_iv, prev_fair_value]])
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    def train_ml_models(self, product: str, rock_price: float, timestamp: int, current_position: int):
        if not self.price_history[product]:
            return
        features = self.prepare_features(product, rock_price, timestamp, current_position)
        market_price = mean(self.price_history[product][-5:]) if len(self.price_history[product]) >= 5 else mean(self.price_history[product])

        fair_value_target = np.array([[market_price]])
        self.ml_fair_value[product].train(features, fair_value_target)

        if product in self.strike_map:
            strike = self.strike_map[product]
            days_to_expiry = max(7 - (timestamp // 100), 0)
            T = days_to_expiry / 365
            iv = self.estimate_implied_volatility(rock_price, strike, T, market_price)
            iv_target = np.array([[iv]])
            self.ml_implied_vol[product].train(features, iv_target)

    def calculate_fair_value(self, product: str, rock_price: float, timestamp: int, current_position: int) -> float:
        if not self.price_history[product]:
            return 0
        features = self.prepare_features(product, rock_price, timestamp, current_position)
        fair_value = self.ml_fair_value[product].forward(features)[0, 0]
        fair_value = max(0.1, fair_value)

        params = self.params[product]
        recent_prices = self.price_history[product][-params["window_size"]:]
        heuristic_value = mean(recent_prices) if recent_prices else fair_value
        if len(recent_prices) > 1:
            last_price = recent_prices[-1]
            heuristic_value += params["reversion_factor"] * (heuristic_value - last_price)
            if len(recent_prices) > 2:
                momentum = recent_prices[-1] - recent_prices[-2]
                heuristic_value += params["momentum_factor"] * momentum
        return 0.8 * fair_value + 0.2 * heuristic_value

    def calculate_voucher_fair_value(self, voucher: str, rock_price: float, timestamp: int, current_position: int) -> float:
        strike = self.strike_map[voucher]
        days_to_expiry = max(7 - (timestamp // 100), 0)
        T = days_to_expiry / 365
        intrinsic = max(0, rock_price - strike)

        if days_to_expiry == 0:
            return intrinsic

        features = self.prepare_features(voucher, rock_price, timestamp, current_position)
        iv = self.ml_implied_vol[voucher].forward(features)[0, 0]
        iv = max(0.1, min(iv, 1.0))

        time_value = iv * rock_price * np.sqrt(T) * 0.35
        fair_value = intrinsic + time_value

        recent_prices = self.price_history[voucher][-5:]
        if recent_prices:
            market_price = mean(recent_prices)
            fair_value = 0.7 * fair_value + 0.3 * market_price

        return max(0.1, fair_value)

    def calculate_spread(self, product: str, current_position: int) -> float:
        params = self.params[product]
        spread = self.volatility[product] * params["spread_factor"] * self.fair_values[product]
        spread = max(params["min_spread"], min(params["max_spread"], spread))
        position_ratio = current_position / self.position_limits[product]
        spread *= (1 + abs(position_ratio) * 0.25)
        return spread

    def trade_volcanic_products(self, state: TradingState) -> Dict[str, List[Order]]:
        orders = {}
        rock_product = "VOLCANIC_ROCK"

        if rock_product in state.order_depths:
            self.update_market_data(rock_product, state.order_depths[rock_product])
            rock_position = state.position.get(rock_product, 0)
            self.train_ml_models(rock_product, self.fair_values.get(rock_product, 10000), state.timestamp, rock_position)
            self.fair_values[rock_product] = self.calculate_fair_value(rock_product, self.fair_values.get(rock_product, 10000), state.timestamp, rock_position)
            self.fit_implied_volatility_curve(self.fair_values[rock_product], state.timestamp)

        rock_price = self.fair_values[rock_product] if self.fair_values[rock_product] else 10000

        if rock_product in state.order_depths:
            order_depth = state.order_depths[rock_product]
            current_position = state.position.get(rock_product, 0)
            fair_value = self.fair_values[rock_product]

            if fair_value > 0:
                spread = self.calculate_spread(rock_product, current_position)
                volatility_adj = 0.4 if self.volatility[rock_product] < 0.15 else 1.6 if self.volatility[rock_product] > 0.7 else 1.0
                spread *= volatility_adj

                bid_price = math.floor(fair_value - spread)
                ask_price = math.ceil(fair_value + spread)

                position_ratio = current_position / self.position_limits[rock_product]
                if position_ratio > 0.8:
                    bid_price -= math.floor(spread * position_ratio * 0.4)
                    ask_price -= math.floor(spread * position_ratio * 0.2)
                elif position_ratio < -0.8:
                    bid_price += math.floor(spread * abs(position_ratio) * 0.2)
                    ask_price += math.floor(spread * abs(position_ratio) * 0.4)

                bid_qty = min(
                    self.position_limits[rock_product] - current_position,
                    sum(abs(qty) for price, qty in order_depth.sell_orders.items() if price <= bid_price + 2),
                    self.position_limits[rock_product] // 12
                )
                ask_qty = min(
                    self.position_limits[rock_product] + current_position,
                    sum(abs(qty) for price, qty in order_depth.buy_orders.items() if price >= ask_price - 2),
                    self.position_limits[rock_product] // 12
                )

                orders[rock_product] = []
                if bid_qty > 0 and bid_price > 0:
                    orders[rock_product].append(Order(rock_product, bid_price, bid_qty))
                if ask_qty > 0 and ask_price > 0:
                    orders[rock_product].append(Order(rock_product, ask_price, -ask_qty))

        vouchers = list(self.strike_map.keys())
        for voucher in vouchers:
            if voucher not in state.order_depths:
                continue

            self.update_market_data(voucher, state.order_depths[voucher])
            current_position = state.position.get(voucher, 0)
            self.train_ml_models(voucher, rock_price, state.timestamp, current_position)
            self.fair_values[voucher] = self.calculate_voucher_fair_value(voucher, rock_price, state.timestamp, current_position)

            order_depth = state.order_depths[voucher]
            fair_value = self.fair_values[voucher]

            if fair_value <= 0:
                continue

            spread = self.calculate_spread(voucher, current_position)
            bid_price = math.floor(fair_value - spread)
            ask_price = math.ceil(fair_value + spread)

            position_ratio = current_position / self.position_limits[voucher]
            if position_ratio > 0.8:
                bid_price -= math.floor(spread * position_ratio * 0.3)
            elif position_ratio < -0.8:
                ask_price += math.floor(spread * abs(position_ratio) * 0.3)

            bid_qty = min(
                self.position_limits[voucher] - current_position,
                sum(abs(qty) for price, qty in order_depth.sell_orders.items() if price <= bid_price + 4),
                self.position_limits[voucher] // 20
            )
            ask_qty = min(
                self.position_limits[voucher] + current_position,
                sum(abs(qty) for price, qty in order_depth.buy_orders.items() if price >= ask_price - 4),
                self.position_limits[voucher] // 20
            )

            orders[voucher] = []
            if bid_qty > 0 and bid_price > 0:
                orders[voucher].append(Order(voucher, bid_price, bid_qty))
            if ask_qty > 0 and ask_price > 0:
                orders[voucher].append(Order(voucher, ask_price, -ask_qty))

            best_ask = min(order_depth.sell_orders.keys(), default=float('inf'))
            if best_ask < fair_value * 0.96:
                qty = min(
                    -order_depth.sell_orders.get(best_ask, 0),
                    self.position_limits[voucher] - current_position,
                    self.position_limits[voucher] // 20
                )
                if qty > 0:
                    orders[voucher].append(Order(voucher, math.floor(best_ask), qty))

        for voucher in vouchers:
            if voucher in orders and orders[voucher] and rock_product in state.order_depths:
                delta = 0.6 if rock_price > self.strike_map[voucher] else 0.4
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

                if v1_ask < v2_bid - spread * 1.03:
                    qty = min(
                        -state.order_depths[v1].sell_orders.get(v1_ask, 0),
                        state.order_depths[v2].buy_orders.get(v2_bid, 0),
                        self.position_limits[v1] // 20
                    )
                    if qty > 0:
                        orders[v1] = orders.get(v1, []) + [Order(v1, math.floor(v1_ask), qty)]
                        orders[v2] = orders.get(v2, []) + [Order(v2, math.floor(v2_bid), -qty)]

        return orders

    def consolidate_orders(self, orders: Dict[str, List[Order]]) -> Dict[str, List[Order]]:
        consolidated = {}
        for product, order_list in orders.items():
            price_to_qty = {}
            for order in order_list:
                price = order.price
                qty = order.quantity
                price_to_qty[price] = price_to_qty.get(price, 0) + qty

            consolidated_orders = []
            for price, qty in price_to_qty.items():
                if qty != 0:
                    consolidated_orders.append(Order(product, price, qty))

            consolidated[product] = consolidated_orders

        return consolidated

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result = self.trade_volcanic_products(state)
        arb_orders = self.find_arbitrage_opportunities(state)

        for product, orders in arb_orders.items():
            if product in result:
                result[product].extend(orders)
            else:
                result[product] = orders

        result = self.consolidate_orders(result)
        return result, 0, ""