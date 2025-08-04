from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import numpy as np
from statistics import mean
import math

class OnlineLinearRegression:
    def __init__(self, n_features: int, learning_rate: float = 0.01):
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.lr = learning_rate
    
    def predict(self, x: np.ndarray) -> float:
        return np.dot(x, self.weights) + self.bias
    
    def update(self, x: np.ndarray, y: float) -> None:
        pred = self.predict(x)
        error = pred - y
        self.weights -= self.lr * error * x
        self.bias -= self.lr * error

class OnlineLogisticRegression:
    def __init__(self, n_features: int, learning_rate: float = 0.01):
        self.weights = np.zeros((3, n_features))  # Buy, sell, hold
        self.biases = np.zeros(3)
        self.lr = learning_rate
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = np.dot(self.weights, x) + self.biases
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def update(self, x: np.ndarray, y: int) -> None:
        probs = self.predict(x)
        true_label = [1 if y == i-1 else 0 for i in range(3)]
        for i in range(3):
            error = probs[i] - true_label[i]
            self.weights[i] -= self.lr * error * x
            self.biases[i] -= self.lr * error

class Trader:
    def __init__(self):
        self.position_limits = {
            "MAGNIFICENT_MACARONS": 75,
            "VOLCANIC_ROCK": 400,
            **{f"VOLCANIC_ROCK_VOUCHER_{k}": 200 for k in [9500, 9750, 10000, 10250, 10500]}
        }
        self.strike_map = {f"VOLCANIC_ROCK_VOUCHER_{k}": k for k in [9500, 9750, 10000, 10250, 10500]}
        self.CSI = 2500
        self.conversion_limit = 10
        self.csi_window = 20
        self.params = {
            "MAGNIFICENT_MACARONS": {
                "window_size": 10,
                "spread_factor": 0.5,
                "min_spread": 1.0,
                "max_spread": 5.0,
                "sunlight_factor": 0.001,
                "sugar_factor": 0.005,
                "emergency_spread_multiplier": 2.0,
                "csi_breach_threshold": 2,
                "csi_deficit_factor": 0.002
            },
            "VOLCANIC_ROCK": {
                "window_size": 10,
                "spread_factor": 0.6,
                "min_spread": 2.0,
                "max_spread": 8.0
            },
            **{v: {
                "window_size": 10,
                "spread_factor": 0.9,
                "min_spread": 3.0,
                "max_spread": 20.0
            } for v in self.strike_map.keys()}
        }
        self.price_history = {p: [] for p in self.position_limits}
        self.volatility = {p: 0.2 for p in self.position_limits}
        self.fair_values = {p: 0 for p in self.position_limits}
        self.sunlight_history = []
        self.sugar_price_history = []
        self.csi_breach_count = 0
        self.macaron_emergency_mode = False
        self.models = {
            p: OnlineLinearRegression(n_features=10 if p == "MAGNIFICENT_MACARONS" else 7)
            for p in self.position_limits
        }
        self.signal_models = {
            p: OnlineLogisticRegression(n_features=10 if p == "MAGNIFICENT_MACARONS" else 7)
            for p in self.position_limits
        }
        self.iteration = 0
        self.update_frequency = 5

    def update_market_data(self, product: str, order_depth: OrderDepth, window_size=20) -> None:
        best_bid = max(order_depth.buy_orders.keys(), default=0)
        best_ask = min(order_depth.sell_orders.keys(), default=float('inf'))
        if best_bid > 0 and best_ask < float('inf'):
            mid_price = (best_bid + best_ask) / 2
            self.price_history[product].append(mid_price)
            self.price_history[product] = self.price_history[product][-window_size:]
            if len(self.price_history[product]) > 1:
                returns = np.diff(self.price_history[product]) / np.array(self.price_history[product][:-1])
                vol = np.std(returns) * np.sqrt(252)
                self.volatility[product] = max(min(vol, 1.0), 0.01)

    def extract_features(self, product: str, state: TradingState, window=10) -> np.ndarray:
        features = []
        price_history = self.price_history.get(product, [])
        if price_history:
            mid_price = price_history[-1]
            features.append(mid_price)
            features.append((mid_price - price_history[-5]) / price_history[-5] if len(price_history) >= 5 else 0)
            features.append(np.mean(price_history[-window:]) if len(price_history) >= window else mid_price)
        else:
            features.extend([10000, 0, 10000])
        
        order_depth = state.order_depths.get(product, OrderDepth())
        best_bid = max(order_depth.buy_orders.keys(), default=0)
        best_ask = min(order_depth.sell_orders.keys(), default=float('inf'))
        bid_ask_spread = best_ask - best_bid if best_ask < float('inf') and best_bid > 0 else 0
        bid_depth = sum(abs(qty) for price, qty in order_depth.buy_orders.items())
        ask_depth = sum(abs(qty) for price, qty in order_depth.sell_orders.items())
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1)
        features.extend([bid_ask_spread, imbalance])
        
        if product == "MAGNIFICENT_MACARONS" and state.observations and hasattr(state.observations, 'conversionObservations'):
            conv_obs = state.observations.conversionObservations
            if 'SUGAR' in conv_obs and 'MACARONS' in conv_obs:
                sunlight = conv_obs['MACARONS'].sunlightIndex
                sugar_price = conv_obs['SUGAR'].sugarPrice
                features.append(sunlight)
                features.append((sunlight - self.sunlight_history[-5]) / self.sunlight_history[-5] if len(self.sunlight_history) >= 5 else 0)
                features.append(1 if sunlight < self.CSI else 0)
                features.append(sugar_price)
                features.append((sugar_price - self.sugar_price_history[-5]) / self.sugar_price_history[-5] if len(self.sugar_price_history) >= 5 else 0)
            else:
                features.extend([3000, 0, 0, 100, 0])
        elif product in self.strike_map:
            rock_price = self.fair_values.get("VOLCANIC_ROCK", 10000)
            intrinsic = max(0, rock_price - self.strike_map[product])
            features.append(intrinsic)
            features.append(self.volatility.get(product, 0.2))
        else:
            features.extend([0, 0])
        
        features = np.array(features, dtype=np.float32)
        features = np.clip(features / (np.abs(features) + 1e-6), -1, 1)
        return features

    def calculate_dynamic_csi(self):
        if len(self.sunlight_history) < self.csi_window or len(self.price_history["MAGNIFICENT_MACARONS"]) < self.csi_window:
            return 2500
        sunlight = np.array(self.sunlight_history[-self.csi_window:])
        prices = np.array(self.price_history["MAGNIFICENT_MACARONS"][-self.csi_window:])
        price_changes = np.diff(prices) / prices[:-1] * 100
        csi_candidates = np.linspace(min(sunlight), max(sunlight), 10)
        correlations = []
        for csi in csi_candidates:
            below_csi = (sunlight[:-1] < csi).astype(int)
            if 5 < sum(below_csi) < len(below_csi) - 5:
                corr = np.corrcoef(below_csi, price_changes)[0, 1]
                if not np.isnan(corr):
                    correlations.append((csi, corr))
        return min(correlations, key=lambda x: x[1])[0] if correlations else 2500

    def trade_macaron(self, state: TradingState) -> Dict[str, List[Order]]:
        self.iteration += 1
        product = "MAGNIFICENT_MACARONS"
        orders = {product: []}
        if product not in state.order_depths:
            return orders
        
        features = self.extract_features(product, state)
        fair_value = self.models[product].predict(features)
        self.fair_values[product] = max(1, fair_value)
        
        signal_probs = self.signal_models[product].predict(features)
        action = np.argmax(signal_probs) - 1
        
        current_position = state.position.get(product, 0)
        order_depth = state.order_depths[product]
        
        if action == 1:
            bid_price = math.floor(fair_value - 1)
            bid_qty = min(
                self.position_limits[product] - current_position,
                sum(abs(qty) for price, qty in order_depth.sell_orders.items() if price <= bid_price + 2),
                self.conversion_limit
            )
            if bid_qty > 0:
                orders[product].append(Order(product, bid_price, bid_qty))
        elif action == -1:
            ask_price = math.ceil(fair_value + 1)
            ask_qty = min(
                self.position_limits[product] + current_position,
                sum(abs(qty) for price, qty in order_depth.buy_orders.items() if price >= ask_price - 2),
                self.conversion_limit
            )
            if ask_qty > 0:
                orders[product].append(Order(product, ask_price, -ask_qty))
        
        if self.iteration % self.update_frequency == 0 and len(self.price_history[product]) > 1:
            true_value = self.price_history[product][-1]
            self.models[product].update(features, true_value)
            price_change = (true_value - self.price_history[product][-2]) / self.price_history[product][-2]
            true_action = 1 if price_change > 0.001 else (-1 if price_change < -0.001 else 0)
            self.signal_models[product].update(features, true_action)
        
        return orders

    def trade_volcanic_products(self, state: TradingState) -> Dict[str, List[Order]]:
        orders = {}
        rock_product = "VOLCANIC_ROCK"
        
        for product in [rock_product] + list(self.strike_map.keys()):
            if product not in state.order_depths:
                continue
                
            features = self.extract_features(product, state)
            fair_value = self.models[product].predict(features)
            self.fair_values[product] = max(1, fair_value)
            
            signal_probs = self.signal_models[product].predict(features)
            action = np.argmax(signal_probs) - 1
            
            current_position = state.position.get(product, 0)
            order_depth = state.order_depths[product]
            
            if action == 1:
                bid_price = math.floor(fair_value - 1)
                bid_qty = min(
                    self.position_limits[product] - current_position,
                    sum(abs(qty) for price, qty in order_depth.sell_orders.items() if price <= bid_price + 2),
                    self.position_limits[product] // 5
                )
                if bid_qty > 0:
                    orders[product] = orders.get(product, []) + [Order(product, bid_price, bid_qty)]
            elif action == -1:
                ask_price = math.ceil(fair_value + 1)
                ask_qty = min(
                    self.position_limits[product] + current_position,
                    sum(abs(qty) for price, qty in order_depth.buy_orders.items() if price >= ask_price - 2),
                    self.position_limits[product] // 5
                )
                if ask_qty > 0:
                    orders[product] = orders.get(product, []) + [Order(product, ask_price, -ask_qty)]
            
            if self.iteration % self.update_frequency == 0 and len(self.price_history[product]) > 1:
                true_value = self.price_history[product][-1]
                self.models[product].update(features, true_value)
                price_change = (true_value - self.price_history[product][-2]) / self.price_history[product][-2]
                true_action = 1 if price_change > 0.001 else (-1 if price_change < -0.001 else 0)
                self.signal_models[product].update(features, true_action)
        
        return orders

    def check_stop_loss(self, product: str, state: TradingState) -> List[Order]:
        orders = []
        current_position = state.position.get(product, 0)
        if not current_position or product not in self.price_history or not self.price_history[product]:
            return orders
        
        current_price = self.price_history[product][-1]
        recent_prices = self.price_history[product][-10:] if len(self.price_history[product]) >= 10 else self.price_history[product]
        avg_entry_price = mean(recent_prices)
        vol = self.volatility[product] * current_price
        
        stop_loss_threshold = 1.5 if product == "MAGNIFICENT_MACARONS" and self.macaron_emergency_mode else 2.0
        
        if current_position > 0 and current_price < avg_entry_price - stop_loss_threshold * vol:
            qty = -current_position
            price = math.floor(current_price * 0.99)
            orders.append(Order(product, price, qty))
        elif current_position < 0 and current_price > avg_entry_price + stop_loss_threshold * vol:
            qty = -current_position
            price = math.ceil(current_price * 1.01)
            orders.append(Order(product, price, qty))
        
        return orders

    def find_arbitrage_opportunities(self, state: TradingState) -> Dict[str, List[Order]]:
        orders = {}
        vouchers = sorted(self.strike_map.keys(), key=lambda x: self.strike_map[x])
        rock_product = "VOLCANIC_ROCK"
        
        for i in range(len(vouchers) - 1):
            v1, v2 = vouchers[i], vouchers[i + 1]
            if v1 not in state.order_depths or v2 not in state.order_depths:
                continue
            v1_ask = min(state.order_depths[v1].sell_orders.keys(), default=float('inf'))
            v2_bid = max(state.order_depths[v2].buy_orders.keys(), default=0)
            k1, k2 = self.strike_map[v1], self.strike_map[v2]
            spread = abs(k2 - k1)
            
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

    def check_conversion_opportunities(self, state: TradingState) -> int:
        if state.observations is None or not hasattr(state.observations, 'conversionObservations'):
            return 0
        
        conv_obs = state.observations.conversionObservations
        if 'SUGAR' not in conv_obs or 'MACARONS' not in conv_obs:
            return 0
        
        sugar_obs = conv_obs['SUGAR']
        macaron_obs = conv_obs['MACARONS']
        transport = macaron_obs.transportFees
        export_tariff = macaron_obs.exportTariff
        import_tariff = macaron_obs.importTariff
        total_cost = transport + export_tariff + import_tariff
        
        macaron_mid = self.fair_values.get("MAGNIFICENT_MACARONS", 10000)
        sugar_mid = sugar_obs.sugarPrice
        
        profit_threshold = total_cost * 1.1
        convert_to_sugar_profit = macaron_mid - sugar_mid - total_cost
        convert_to_macaron_profit = sugar_mid - macaron_mid - total_cost
        
        current_position = state.position.get("MAGNIFICENT_MACARONS", 0)
        
        if convert_to_sugar_profit > profit_threshold and current_position > 0:
            qty = min(self.conversion_limit, current_position)
            return qty
        elif convert_to_macaron_profit > profit_threshold and current_position < self.position_limits["MAGNIFICENT_MACARONS"]:
            qty = min(self.conversion_limit, self.position_limits["MAGNIFICENT_MACARONS"] - current_position)
            return -qty
        
        return 0

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result = {}
        
        for product in self.position_limits:
            if product in state.order_depths:
                self.update_market_data(product, state.order_depths[product], window_size=20)
        
        for product in self.position_limits:
            stop_loss_orders = self.check_stop_loss(product, state)
            if stop_loss_orders:
                result[product] = stop_loss_orders
        
        conversion_qty = self.check_conversion_opportunities(state)
        result.update(self.trade_macaron(state))
        result.update(self.trade_volcanic_products(state))
        result.update(self.find_arbitrage_opportunities(state))
        
        return result, conversion_qty, ""