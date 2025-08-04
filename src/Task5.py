from datamodel import OrderDepth, UserId, TradingState, Order, Trade
from typing import Dict, List, Tuple
import json
import numpy as np
from statistics import mean
from math import ceil, floor

class TraderState:
    def __init__(self):
        self.fair_values: Dict[str, float] = {}
        self.counterparty_trades: Dict[str, Dict[str, List[Dict]]] = {}
        self.position_limits: Dict[str, int] = {}
        self.prev_sunlight: float = 1000
        self.prev_sugar: float = 100

    def to_dict(self):
        return {
            "fair_values": self.fair_values,
            "counterparty_trades": self.counterparty_trades,
            "position_limits": self.position_limits,
            "prev_sunlight": self.prev_sunlight,
            "prev_sugar": self.prev_sugar
        }

    @staticmethod
    def from_dict(data):
        state = TraderState()
        state.fair_values = data.get("fair_values", {})
        state.counterparty_trades = data.get("counterparty_trades", {})
        state.position_limits = data.get("position_limits", {})
        state.prev_sunlight = data.get("prev_sunlight", 1000)
        state.prev_sugar = data.get("prev_sugar", 100)
        return state

class Trader:
    def __init__(self):
        self.position_limits = {
            "PEARLS": 20,
            "VOLCANIC_ROCK_VOUCHER_10500": 20
        }

    def calculate_fair_value(self, product: str, order_depth: OrderDepth, state: TradingState) -> float:
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders
        trade_prices = [trade.price for trade in state.market_trades.get(product, []) if trade.timestamp >= state.timestamp - 1000]

        # Calculate mid-price from order book
        best_bid = max(buy_orders.keys(), default=None)
        best_ask = min(sell_orders.keys(), default=None)
        if best_bid and best_ask:
            bid_volume = buy_orders[best_bid]
            ask_volume = abs(sell_orders[best_ask])
            total_volume = bid_volume + ask_volume
            mid_price = (best_bid * ask_volume + best_ask * bid_volume) / total_volume if total_volume > 0 else (best_bid + best_ask) / 2
        elif best_bid:
            mid_price = best_bid
        elif best_ask:
            mid_price = best_ask
        else:
            mid_price = trade_prices[-1] if trade_prices else 1000

        # Blend with recent trade prices
        fair_value = 0.7 * mid_price + 0.3 * mean(trade_prices) if trade_prices else mid_price

        # Adjust based on observation data
        conv_obs = state.observations.conversionObservations.get(product)
        if conv_obs:
            sunlight = conv_obs.sunlightIndex
            sugar = conv_obs.sugarPrice
            trader_state = TraderState.from_dict(json.loads(state.traderData)) if state.traderData else TraderState()

            sunlight_adjustment = 0
            if sunlight > trader_state.prev_sunlight and sunlight > 1000:
                sunlight_adjustment = fair_value * 0.005
            elif sunlight < trader_state.prev_sunlight and sunlight < 1000:
                sunlight_adjustment = -fair_value * 0.005

            sugar_adjustment = 0
            if sugar > 150:
                sugar_adjustment = -fair_value * 0.0025
            elif sugar < 50:
                sugar_adjustment = fair_value * 0.0025

            fair_value += sunlight_adjustment + sugar_adjustment
            trader_state.prev_sunlight = sunlight
            trader_state.prev_sugar = sugar
            state.traderData = json.dumps(trader_state.to_dict())

        return fair_value

    def analyze_counterparty(self, product: str, own_trades: List[Trade], state_data: TraderState, current_timestamp: int) -> Dict[str, float]:
        if product not in state_data.counterparty_trades:
            state_data.counterparty_trades[product] = {}

        for trade in own_trades[-5:]:  # Process only the last 5 trades to reduce computation
            if trade.buyer == "SUBMISSION":
                counter_party = trade.seller
                side = "sell"
            elif trade.seller == "SUBMISSION":
                counter_party = trade.buyer
                side = "buy"
            else:
                continue

            if counter_party:
                if counter_party not in state_data.counterparty_trades[product]:
                    state_data.counterparty_trades[product][counter_party] = []
                state_data.counterparty_trades[product][counter_party].append({
                    "price": trade.price,
                    "quantity": trade.quantity,
                    "side": side,
                    "timestamp": trade.timestamp
                })
                # Limit trade history to last 10 trades per counterparty
                state_data.counterparty_trades[product][counter_party] = state_data.counterparty_trades[product][counter_party][-10:]

        counter_party_prices = {}
        for cp, trades in state_data.counterparty_trades[product].items():
            recent_trades = [t for t in trades if t["timestamp"] >= current_timestamp - 1000]
            prices = [t["price"] for t in recent_trades[-5:]]
            if prices:
                counter_party_prices[cp] = mean(prices)

        return counter_party_prices

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result = {}
        conversions = 0

        # Deserialize traderData
        trader_state = TraderState.from_dict(json.loads(state.traderData)) if state.traderData else TraderState()
        trader_state.position_limits = self.position_limits

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            current_position = state.position.get(product, 0)
            position_limit = trader_state.position_limits.get(product, 20)

            # Calculate fair value
            fair_value = self.calculate_fair_value(product, order_depth, state)
            trader_state.fair_values[product] = fair_value

            # Analyze counterparty behavior
            counter_party_prices = self.analyze_counterparty(product, state.own_trades.get(product, []), trader_state, state.timestamp)

            # Calculate spread
            trade_prices = [t.price for t in state.market_trades.get(product, [])]
            spread = np.std(trade_prices) * 0.5 if trade_prices else 2.0
            spread = max(spread, 1.0)

            # Adjust spread based on conversion costs
            conv_obs = state.observations.conversionObservations.get(product)
            if conv_obs:
                conversion_cost = conv_obs.transportFees + max(conv_obs.exportTariff, conv_obs.importTariff)
                spread += conversion_cost * 0.05

            # Calculate buy and sell prices
            buy_price = floor(fair_value - spread)
            sell_price = ceil(fair_value + spread)

            # Adjust prices based on counterparty behavior
            if counter_party_prices:
                avg_counter_price = mean(counter_party_prices.values())
                spread_limit = spread * 2
                if avg_counter_price > fair_value + spread_limit:
                    sell_price = min(sell_price + 1, floor(avg_counter_price))
                elif avg_counter_price < fair_value - spread_limit:
                    buy_price = max(buy_price - 1, ceil(avg_counter_price))

            # Calculate max quantities
            max_buy_qty = position_limit - current_position
            max_sell_qty = -(position_limit + current_position)
            order_qty = min(5, max(1, position_limit // 4))  # Simplified division

            # Match sell orders (buying)
            if max_buy_qty > 0:
                for price in sorted(order_depth.sell_orders.keys()):  # Lowest price first
                    qty = order_depth.sell_orders[price]
                    if price <= buy_price and qty < 0:
                        qty_to_buy = min(-qty, max_buy_qty, order_qty)
                        if qty_to_buy > 0:
                            orders.append(Order(product, price, qty_to_buy))
                            max_buy_qty -= qty_to_buy
                            current_position += qty_to_buy
                if max_buy_qty > 0:
                    orders.append(Order(product, buy_price, min(order_qty, max_buy_qty)))

            # Match buy orders (selling)
            if max_sell_qty < 0:
                for price in sorted(order_depth.buy_orders.keys(), reverse=True):  # Highest price first
                    qty = order_depth.buy_orders[price]
                    if price >= sell_price and qty > 0:
                        qty_to_sell = min(qty, -max_sell_qty, order_qty)
                        if qty_to_sell > 0:
                            orders.append(Order(product, price, -qty_to_sell))
                            max_sell_qty += qty_to_sell
                            current_position -= qty_to_sell
                if max_sell_qty < 0:
                    orders.append(Order(product, sell_price, max(-order_qty, max_sell_qty)))

            result[product] = orders

        # Serialize state
        traderData = json.dumps(trader_state.to_dict())

        return result, conversions, traderData