from datamodel import OrderDepth, UserId, TradingState, Order, Trade
from typing import Dict, List, Tuple
import jsonpickle
import numpy as np
from statistics import mean
from math import ceil, floor

class TraderState:
    def __init__(self):
        self.fair_values: Dict[str, float] = {}
        self.counterparty_trades: Dict[str, Dict[str, List[Dict]]] = {}
        self.counterparty_avg_price: Dict[str, Dict[str, float]] = {}
        self.prev_sunlight: float = 1000
        self.prev_sugar: float = 100
        self.position_limits: Dict[str, int] = {}

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
            "RAINFOREST_RESIN": 60,
            "SQUID_INK": 120,
            "VOLCANIC_ROCK": 70,
            "VOLCANIC_ROCK_VOUCHER_10000": 20,
            "VOLCANIC_ROCK_VOUCHER_10250": 20,
            "VOLCANIC_ROCK_VOUCHER_9500": 20,
            "VOLCANIC_ROCK_VOUCHER_9750": 20
        }

    def calculate_fair_value(self, product: str, order_depth: OrderDepth, state: TradingState) -> float:
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders
        trade_prices = [t.price for t in state.own_trades.get(product, []) + state.market_trades.get(product, [])
                        if t.timestamp >= state.timestamp - 2000]

        best_bid = max(buy_orders.keys(), default=None)
        best_ask = min(sell_orders.keys(), default=None)
        if best_bid and best_ask:
            bid_volume = sum(buy_orders.values())
            ask_volume = sum(abs(q) for q in sell_orders.values())
            total_volume = bid_volume + ask_volume
            mid_price = (best_bid + best_ask) / 2
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
        elif best_bid:
            mid_price = best_bid
            imbalance = 0.5
        elif best_ask:
            mid_price = best_ask
            imbalance = -0.5
        else:
            mid_price = trade_prices[-1] if trade_prices else 1000
            imbalance = 0

        vwap = None
        if trade_prices:
            trades = state.own_trades.get(product, []) + state.market_trades.get(product, [])
            total_qty = sum(t.quantity for t in trades if t.timestamp >= state.timestamp - 2000)
            if total_qty > 0:
                vwap = sum(t.price * t.quantity for t in trades if t.timestamp >= state.timestamp - 2000) / total_qty

        fair_value = mid_price
        if vwap:
            fair_value = 0.7 * mid_price + 0.3 * vwap
        fair_value += fair_value * imbalance * 0.015

        conv_obs = state.observations.conversionObservations.get(product)
        if conv_obs:
            sunlight = conv_obs.sunlightIndex
            sugar = conv_obs.sugarPrice
            trader_state = jsonpickle.decode(state.traderData) if state.traderData else TraderState()

            sunlight_adjustment = 0
            if sunlight > trader_state.prev_sunlight + 50 and sunlight > 1050:
                sunlight_adjustment = fair_value * 0.003
            elif sunlight < trader_state.prev_sunlight - 50 and sunlight < 950:
                sunlight_adjustment = -fair_value * 0.003

            sugar_adjustment = 0
            if sugar > 200:
                sugar_adjustment = -fair_value * 0.001
            elif sugar < 50:
                sugar_adjustment = fair_value * 0.001

            fair_value += sunlight_adjustment + sugar_adjustment
            trader_state.prev_sunlight = sunlight
            trader_state.prev_sugar = sugar
            state.traderData = jsonpickle.encode(trader_state)

        return fair_value

    def analyze_counterparty(self, product: str, own_trades: List[Trade], state_data: TraderState, current_timestamp: int) -> Tuple[Dict[str, float], float]:
        if product not in state_data.counterparty_trades:
            state_data.counterparty_trades[product] = {}
        if product not in state_data.counterparty_avg_price:
            state_data.counterparty_avg_price[product] = {}

        # Simplified logic: return empty prices and default aggression score
        # since Trade object lacks counter_party attribute
        aggression_score = 0.0
        return state_data.counterparty_avg_price[product], aggression_score

    def handle_conversions(self, product: str, state: TradingState, position: int) -> int:
        conv_obs = state.observations.conversionObservations.get(product)
        if not conv_obs:
            return 0

        total_cost = conv_obs.transportFees + conv_obs.importTariff + conv_obs.exportTariff
        conversion_limit = 100

        if position < 0 and conv_obs.bidPrice > total_cost + 1:
            return min(abs(position), conversion_limit)
        elif position > 0 and conv_obs.askPrice < -total_cost - 1:
            return min(position, conversion_limit)
        return 0

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result = {}
        conversions = 0
        trader_state = jsonpickle.decode(state.traderData) if state.traderData else TraderState()
        trader_state.position_limits = self.position_limits

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            current_position = state.position.get(product, 0)
            position_limit = trader_state.position_limits.get(product, 20)

            fair_value = self.calculate_fair_value(product, order_depth, state)
            trader_state.fair_values[product] = fair_value
            print(f"{product} Fair Value: {fair_value:.2f}, Position: {current_position}")

            counter_prices, aggression_score = self.analyze_counterparty(
                product, state.own_trades.get(product, []), trader_state, state.timestamp
            )

            trade_prices = [t.price for t in state.own_trades.get(product, []) + state.market_trades.get(product, [])]
            volatility = np.std(trade_prices) if trade_prices else 1.0
            spread = max(1.0, volatility * 0.25) * (1.0 - aggression_score * 0.2)

            conv_obs = state.observations.conversionObservations.get(product)
            if conv_obs:
                conversion_cost = conv_obs.transportFees + max(conv_obs.exportTariff, conv_obs.importTariff)
                spread += conversion_cost * 0.015

            buy_price = floor(fair_value - spread)
            sell_price = ceil(fair_value + spread)

            if counter_prices:
                avg_cp = mean(counter_prices.values())
                if avg_cp > fair_value + spread:
                    sell_price = min(sell_price + 1, floor(avg_cp * 1.01))
                elif avg_cp < fair_value - spread:
                    buy_price = max(buy_price - 1, ceil(avg_cp * 0.99))

            order_qty = min(10, max(2, int(position_limit / (5 + volatility))))
            max_buy_qty = position_limit - current_position
            max_sell_qty = -(position_limit + current_position)

            # Arbitrage
            if max_buy_qty > 0:
                for price in sorted(order_depth.sell_orders.keys()):
                    qty = order_depth.sell_orders[price]
                    if price <= buy_price and qty < 0:
                        qty_to_buy = min(-qty, max_buy_qty, order_qty)
                        orders.append(Order(product, price, qty_to_buy))
                        max_buy_qty -= qty_to_buy
                        current_position += qty_to_buy

            if max_sell_qty < 0:
                for price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    qty = order_depth.buy_orders[price]
                    if price >= sell_price and qty > 0:
                        qty_to_sell = min(qty, -max_sell_qty, order_qty)
                        orders.append(Order(product, price, -qty_to_sell))
                        max_sell_qty += qty_to_sell
                        current_position -= qty_to_sell

            # Market making
            if max_buy_qty > 0 and (not order_depth.sell_orders or min(order_depth.sell_orders.keys()) > buy_price):
                orders.append(Order(product, buy_price, min(order_qty, max_buy_qty)))

            if max_sell_qty < 0 and (not order_depth.buy_orders or max(order_depth.buy_orders.keys()) < sell_price):
                orders.append(Order(product, sell_price, max(-order_qty, max_sell_qty)))

            result[product] = orders
            conversions += self.handle_conversions(product, state, current_position)

        traderData = jsonpickle.encode(trader_state)
        return result, conversions, traderData