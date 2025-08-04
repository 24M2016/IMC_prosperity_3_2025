from datamodel import OrderDepth, UserId, TradingState, Order, Trade
from typing import Dict, List, Tuple
import jsonpickle
import numpy as np
from statistics import mean
from math import ceil, floor
import pandas  # Unused but included for compliance
import json
import time

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class TraderState:
    def __init__(self, **kwargs):
        self.fair_values: Dict[str, float] = kwargs.get('fair_values', {})
        self.counterparty_trades: Dict[str, Dict[str, List[Dict]]] = kwargs.get('counterparty_trades', {})
        self.counterparty_avg_price: Dict[str, Dict[str, float]] = kwargs.get('counterparty_avg_price', {})
        self.prev_sunlight: float = kwargs.get('prev_sunlight', 1000)
        self.prev_sugar: float = kwargs.get('prev_sugar', 100)
        self.position_limits: Dict[str, int] = kwargs.get('position_limits', {})
        self.historical_prices: Dict[str, List[Dict]] = kwargs.get('historical_prices', {})
        self.last_regression_timestamp: Dict[str, float] = kwargs.get('last_regression_timestamp', {})
        self.last_regression_result: Dict[str, float] = kwargs.get('last_regression_result', {})

class Trader:
    def __init__(self):
        self.position_limits = {
            "PEARLS": 20,
            "VOLCANIC_ROCK_VOUCHER_10500": 20,
            "CROISSANTS": 100,
            "DJEMBES": 50,
            "JAMS": 80,
            "KELP": 150,
            "MAGNIFICENT_MACARONS": 200,
            "PICNIC_BASKET1": 30,
            "PICNIC_BASKET2": 40,
            "RAINFOREST_RESIN": 50,
            "SQUID_INK": 50,
            "VOLCANIC_ROCK": 70,
            "VOLCANIC_ROCK_VOUCHER_10000": 20,
            "VOLCANIC_ROCK_VOUCHER_10250": 20,
            "VOLCANIC_ROCK_VOUCHER_9500": 20,
            "VOLCANIC_ROCK_VOUCHER_9750": 20
        }

    def simple_linear_regression(self, x: List[float], y: List[float], product: str, state: TraderState) -> float:
        n = len(x)
        if n < 3:
            return y[-1] if y else 1000
        x = x[-5:]  # Last 5 trades for recency
        y = y[-5:]
        n = len(x)
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)
        if denominator == 0:
            return y_mean
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        result = intercept + slope * max(x)
        state.last_regression_timestamp[product] = max(x)
        state.last_regression_result[product] = float(result)  # Ensure float
        return result

    def calculate_fair_value(self, product: str, order_depth: OrderDepth, state: TradingState, trader_state: TraderState) -> float:
        start_time = time.time()
        if product not in trader_state.historical_prices:
            trader_state.historical_prices[product] = []

        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders
        trades = state.own_trades.get(product, []) + state.market_trades.get(product, [])
        trade_prices = [float(t.price) for t in trades if t.timestamp >= state.timestamp - 1000]
        trade_timestamps = [float(t.timestamp) for t in trades if t.timestamp >= state.timestamp - 1000]

        # Order book analysis
        best_bid = max(buy_orders.keys(), default=None)
        best_ask = min(sell_orders.keys(), default=None)
        mid_price = None
        imbalance = 0
        if best_bid and best_ask:
            mid_price = (best_bid + best_ask) / 2
            bid_volume = sum(buy_orders.values())
            ask_volume = sum(abs(q) for q in sell_orders.values())
            total_volume = bid_volume + ask_volume
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
        elif best_bid:
            mid_price = best_bid
            imbalance = 0.5
        elif best_ask:
            mid_price = best_ask
            imbalance = -0.5
        else:
            mid_price = trader_state.fair_values.get(product, 1000)

        # VWAP for significant trades
        vwap = None
        total_qty = sum(abs(t.quantity) for t in trades if t.timestamp >= state.timestamp - 1000)
        if total_qty > 5:
            vwap = sum(t.price * abs(t.quantity) for t in trades if t.timestamp >= state.timestamp - 1000) / total_qty

        # Linear regression with caching
        fair_value = mid_price
        if len(trade_prices) >= 3 and trader_state.last_regression_timestamp.get(product, 0) != trade_timestamps[-1]:
            predicted_price = self.simple_linear_regression(trade_timestamps, trade_prices, product, trader_state)
            fair_value = 0.5 * mid_price + 0.3 * predicted_price + 0.2 * (vwap or mid_price)

        fair_value += fair_value * imbalance * 0.003

        # Observation adjustments
        conv_obs = state.observations.conversionObservations.get(product)
        if conv_obs:
            sunlight = conv_obs.sunlightIndex
            sugar = conv_obs.sugarPrice
            sunlight_adjustment = 0
            if sunlight > trader_state.prev_sunlight + 50 and sunlight > 1050:
                sunlight_adjustment = fair_value * 0.0005
            elif sunlight < trader_state.prev_sunlight - 50 and sunlight < 950:
                sunlight_adjustment = -fair_value * 0.0005
            sugar_adjustment = 0
            if sugar > 200:
                sugar_adjustment = -fair_value * 0.0002
            elif sugar < 50:
                sugar_adjustment = fair_value * 0.0002
            fair_value += sunlight_adjustment + sugar_adjustment
            trader_state.prev_sunlight = sunlight
            trader_state.prev_sugar = sugar

        trader_state.historical_prices[product].append({"timestamp": state.timestamp, "price": float(fair_value)})
        trader_state.historical_prices[product] = trader_state.historical_prices[product][-10:]
        trader_state.fair_values[product] = float(fair_value)
        print(f"calculate_fair_value for {product} took {time.time() - start_time:.3f}s")
        return fair_value

    def analyze_counterparty(self, product: str, own_trades: List[Trade], state_data: TraderState, current_timestamp: int) -> Tuple[Dict[str, float], float]:
        start_time = time.time()
        if product not in state_data.counterparty_trades:
            state_data.counterparty_trades[product] = {}
        if product not in state_data.counterparty_avg_price:
            state_data.counterparty_avg_price[product] = {}

        aggression_score = 0.0
        trade_count = 0
        for trade in own_trades[-5:]:
            counter_party = trade.seller if trade.buyer == "SUBMISSION" else trade.buyer if trade.seller == "SUBMISSION" else None
            if not counter_party or counter_party == "SUBMISSION" or counter_party == "":
                continue

            if counter_party not in state_data.counterparty_trades[product]:
                state_data.counterparty_trades[product][counter_party] = []
            state_data.counterparty_trades[product][counter_party].append({
                "price": float(trade.price),
                "quantity": int(trade.quantity),
                "timestamp": float(trade.timestamp)
            })
            state_data.counterparty_trades[product][counter_party] = state_data.counterparty_trades[product][counter_party][-5:]

            recent_trades = state_data.counterparty_trades[product][counter_party]
            prices = [t["price"] for t in recent_trades]
            quantities = [abs(t["quantity"]) for t in recent_trades]
            if prices:
                state_data.counterparty_avg_price[product][counter_party] = mean(prices)
                aggression_score += sum(quantities) / (len(quantities) * 10)
                trade_count += 1

        aggression_score = min(aggression_score / max(trade_count, 1), 0.15)
        print(f"analyze_counterparty for {product} took {time.time() - start_time:.3f}s")
        return state_data.counterparty_avg_price[product], aggression_score

    def handle_conversions(self, product: str, state: TradingState, position: int) -> int:
        conv_obs = state.observations.conversionObservations.get(product)
        if not conv_obs:
            return 0
        total_cost = conv_obs.transportFees + conv_obs.importTariff + conv_obs.exportTariff
        conversion_limit = 20
        if position < 0 and conv_obs.bidPrice > total_cost + 2:
            return min(abs(position), conversion_limit)
        elif position > 0 and conv_obs.askPrice < -total_cost - 2:
            return min(position, conversion_limit)
        return 0

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        start_time = time.time()
        result = {}
        conversions = 0

        # Deserialize traderData once
        try:
            trader_state_dict = json.loads(state.traderData) if state.traderData else {}
            trader_state = TraderState(**trader_state_dict)
        except Exception as e:
            print(f"Error decoding traderData: {e}")
            trader_state = TraderState()
        trader_state.position_limits = self.position_limits

        # Prioritize active products
        active_products = [p for p in state.order_depths if state.own_trades.get(p) or state.market_trades.get(p)]
        active_products = active_products[:5] or state.order_depths.keys()

        for product in active_products:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            current_position = state.position.get(product, 0)
            position_limit = trader_state.position_limits.get(product, 20)

            if abs(current_position) >= position_limit:
                print(f"{product} at position limit {current_position}/{position_limit}, skipping orders")
                result[product] = orders
                continue

            fair_value = self.calculate_fair_value(product, order_depth, state, trader_state)
            print(f"{product} Fair Value: {fair_value:.2f}, Position: {current_position}")

            max_buy_qty = position_limit - current_position
            max_sell_qty = -(position_limit + current_position)

            # Counterparty analysis
            counter_party_prices, aggression_score = self.analyze_counterparty(
                product, state.own_trades.get(product, []), trader_state, state.timestamp
            )

            # Volatility and spread
            trade_prices = [float(t.price) for t in state.own_trades.get(product, [])[-5:]]
            volatility = np.std(trade_prices) if trade_prices else 1.0
            min_spread = 0.3
            spread = max(min_spread, volatility * 0.1) * (1.0 - aggression_score * 0.2)

            conv_obs = state.observations.conversionObservations.get(product)
            if conv_obs:
                conversion_cost = conv_obs.transportFees + max(conv_obs.exportTariff, conv_obs.importTariff)
                spread += conversion_cost * 0.003

            position_ratio = abs(current_position) / position_limit
            order_qty = min(5, max(1, int(position_limit / (8 + volatility))))
            if position_ratio > 0.8:
                spread *= 1.2
                order_qty = max(1, int(2 * (1 - position_ratio)))

            buy_price = floor(fair_value - spread)
            sell_price = ceil(fair_value + spread)

            if counter_party_prices:
                avg_counter_price = mean(counter_party_prices.values())
                spread_limit = spread * 1.05
                if avg_counter_price > fair_value + spread_limit:
                    sell_price = min(sell_price + 1, floor(avg_counter_price * 1.001))
                    print(f"Adjusting sell_price for {product} to {sell_price} due to counterparty")
                elif avg_counter_price < fair_value - spread_limit:
                    buy_price = max(buy_price - 1, ceil(avg_counter_price * 0.999))
                    print(f"Adjusting buy_price for {product} to {buy_price} due to counterparty")

            # Order placement
            if max_buy_qty > 0:
                for price in sorted(order_depth.sell_orders.keys())[:2]:
                    qty = order_depth.sell_orders[price]
                    if price <= buy_price and qty < 0:
                        qty_to_buy = min(-qty, max_buy_qty, order_qty)
                        orders.append(Order(product, price, qty_to_buy))
                        max_buy_qty -= qty_to_buy
                        print(f"BUY {product} {qty_to_buy}x {price}")
                    if max_buy_qty <= 0:
                        break
                if max_buy_qty > 0:
                    orders.append(Order(product, buy_price, min(order_qty, max_buy_qty)))
                    print(f"PROVIDE BID {product} {min(order_qty, max_buy_qty)}x {buy_price}")

            if max_sell_qty < 0:
                for price in sorted(order_depth.buy_orders.keys(), reverse=True)[:2]:
                    qty = order_depth.buy_orders[price]
                    if price >= sell_price and qty > 0:
                        qty_to_sell = min(qty, -max_sell_qty, order_qty)
                        orders.append(Order(product, price, -qty_to_sell))
                        max_sell_qty += qty_to_sell
                        print(f"SELL {product} {qty_to_sell}x {price}")
                    if max_sell_qty >= 0:
                        break
                if max_sell_qty < 0:
                    orders.append(Order(product, sell_price, max(-order_qty, max_sell_qty)))
                    print(f"PROVIDE ASK {product} {-max(-order_qty, max_sell_qty)}x {sell_price}")

            # Unwind if at limit
            if current_position >= position_limit * 0.9 and max_sell_qty < 0:
                orders.append(Order(product, sell_price, max(-order_qty, max_sell_qty)))
                print(f"UNWIND SELL {product} {-max(-order_qty, max_sell_qty)}x {sell_price}")
            elif current_position <= -position_limit * 0.9 and max_buy_qty > 0:
                orders.append(Order(product, buy_price, min(order_qty, max_buy_qty)))
                print(f"UNWIND BUY {product} {min(order_qty, max_buy_qty)}x {buy_price}")

            result[product] = orders
            conversions += self.handle_conversions(product, state, current_position)

        traderData = json.dumps(trader_state.__dict__, cls=NumpyJSONEncoder)
        print(f"run took {time.time() - start_time:.3f}s")
        return result, conversions, traderData