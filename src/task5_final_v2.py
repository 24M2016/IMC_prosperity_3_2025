from datamodel import OrderDepth, UserId, TradingState, Order, Trade
from typing import Dict, List, Tuple
import jsonpickle
import numpy as np
from statistics import mean, stdev
from math import ceil, floor

class TraderState:
    def __init__(self):
        self.fair_values: Dict[str, float] = {}
        self.counterparty_trades: Dict[str, Dict[str, List[Dict]]] = {}
        self.counterparty_avg_price: Dict[str, Dict[str, float]] = {}
        self.trade_history: Dict[str, List[Dict]] = {}
        self.prev_sunlight: float = 1000
        self.prev_sugar: float = 100
        self.position_limits: Dict[str, int] = {}
        self.volatility_history: Dict[str, List[float]] = {}

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
                       if t.timestamp >= state.timestamp - 1500]

        # Calculate order book imbalance and mid-price
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

        # VWAP with volume-weighted trades
        vwap = None
        if trade_prices:
            trades = state.own_trades.get(product, []) + state.market_trades.get(product, [])
            recent_trades = [t for t in trades if t.timestamp >= state.timestamp - 1500]
            total_qty = sum(abs(t.quantity) for t in recent_trades)
            if total_qty > 0:
                vwap = sum(t.price * abs(t.quantity) for t in recent_trades) / total_qty

        # Blend mid-price, VWAP, and imbalance
        fair_value = mid_price
        if vwap:
            fair_value = 0.6 * mid_price + 0.4 * vwap  # Adjusted weights for VWAP
        fair_value += fair_value * imbalance * 0.01  # Further reduced imbalance impact

        # Adjust based on observation data
        conv_obs = state.observations.conversionObservations.get(product)
        if conv_obs:
            sunlight = conv_obs.sunlightIndex
            sugar = conv_obs.sugarPrice
            trader_state = jsonpickle.decode(state.traderData) if state.traderData else TraderState()

            sunlight_adjustment = 0
            if sunlight > trader_state.prev_sunlight + 75 and sunlight > 1075:
                sunlight_adjustment = fair_value * 0.002
            elif sunlight < trader_state.prev_sunlight - 75 and sunlight < 925:
                sunlight_adjustment = -fair_value * 0.002

            sugar_adjustment = 0
            if sugar > 225:
                sugar_adjustment = -fair_value * 0.0008
            elif sugar < 25:
                sugar_adjustment = fair_value * 0.0008

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

        aggression_score = 0.0
        for trade in own_trades[-15:]:  # Look at more trades for better analysis
            counter_party = trade.buyer if trade.seller == "SUBMISSION" else trade.seller if trade.buyer == "SUBMISSION" else None

            if not counter_party or counter_party == "SUBMISSION" or counter_party == "":
                continue

            if counter_party not in state_data.counterparty_trades[product]:
                state_data.counterparty_trades[product][counter_party] = []
            state_data.counterparty_trades[product][counter_party].append({
                "price": trade.price,
                "quantity": trade.quantity,
                "timestamp": trade.timestamp
            })
            state_data.counterparty_trades[product][counter_party] = state_data.counterparty_trades[product][counter_party][-15:]

            recent_trades = [t for t in state_data.counterparty_trades[product][counter_party]
                            if t["timestamp"] >= current_timestamp - 1500]
            prices = [t["price"] for t in recent_trades]
            quantities = [abs(t["quantity"]) for t in recent_trades]
            if prices:
                state_data.counterparty_avg_price[product][counter_party] = mean(prices)
                trade_freq = len(recent_trades)
                avg_qty = mean(quantities) if quantities else 1
                aggression_score += (trade_freq * avg_qty) / (15 * 10)  # Normalize aggression

        return state_data.counterparty_avg_price[product], min(aggression_score, 0.5)

    def handle_conversions(self, product: str, state: TradingState, position: int) -> int:
        conv_obs = state.observations.conversionObservations.get(product)
        if not conv_obs:
            return 0

        total_cost = conv_obs.transportFees + conv_obs.importTariff + conv_obs.exportTariff
        conversion_limit = 80  # Reduced limit to avoid over-conversion

        if position < 0 and conv_obs.bidPrice > total_cost + 2:  # Stricter threshold
            return min(abs(position), conversion_limit)
        elif position > 0 and conv_obs.askPrice < -total_cost - 2:
            return min(position, conversion_limit)
        return 0

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result = {}
        conversions = 0
        trader_state = jsonpickle.decode(state.traderData) if state.traderData else TraderState()
        trader_state.position_limits = self.position_limits
        if not trader_state.volatility_history:
            trader_state.volatility_history = {p: [] for p in state.order_depths}

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            current_position = state.position.get(product, 0)
            position_limit = trader_state.position_limits.get(product, 20)

            fair_value = self.calculate_fair_value(product, order_depth, state)
            trader_state.fair_values[product] = fair_value
            print(f"{product} Fair Value: {fair_value:.2f}, Position: {current_position}")

            counter_party_prices, aggression_score = self.analyze_counterparty(
                product, state.own_trades.get(product, []), trader_state, state.timestamp
            )

            trade_prices = [t.price for t in state.own_trades.get(product, []) + state.market_trades.get(product, [])
                           if t.timestamp >= state.timestamp - 1500]
            volatility = stdev(trade_prices) if len(trade_prices) > 1 else 1.0
            trader_state.volatility_history[product].append(volatility)
            trader_state.volatility_history[product] = trader_state.volatility_history[product][-10:]
            avg_volatility = mean(trader_state.volatility_history[product]) if trader_state.volatility_history[product] else 1.0
            spread = max(1.0, avg_volatility * 0.3) * (1.0 - aggression_score * 0.25)

            if conv_obs := state.observations.conversionObservations.get(product):
                conversion_cost = conv_obs.transportFees + max(conv_obs.exportTariff, conv_obs.importTariff)
                spread += conversion_cost * 0.01

            buy_price = floor(fair_value - spread)
            sell_price = ceil(fair_value + spread)
            if counter_party_prices:
                avg_counter_price = mean(counter_party_prices.values())
                spread_limit = spread * 1.2
                if avg_counter_price > fair_value + spread_limit:
                    sell_price = min(sell_price + 2, floor(avg_counter_price * 1.01))
                    print(f"Adjusting sell_price for {product} to {sell_price} due to counterparty")
                elif avg_counter_price < fair_value - spread_limit:
                    buy_price = max(buy_price - 2, ceil(avg_counter_price * 0.99))
                    print(f"Adjusting buy_price for {product} to {buy_price} due to counterparty")

            order_qty = min(8, max(1, int(position_limit / (6 + avg_volatility))))  # Dynamic order size
            max_buy_qty = position_limit - current_position
            max_sell_qty = -(position_limit + current_position)

            # Arbitrage
            if max_buy_qty > 0:
                for price in sorted(order_depth.sell_orders.keys()):
                    qty = order_depth.sell_orders[price]
                    if price <= buy_price and qty < 0:
                        qty_to_buy = min(-qty, max_buy_qty, order_qty)
                        if qty_to_buy > 0:
                            orders.append(Order(product, price, qty_to_buy))
                            max_buy_qty -= qty_to_buy
                            current_position += qty_to_buy
                            print(f"BUY {product} {qty_to_buy}x {price}")

            if max_sell_qty < 0:
                for price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    qty = order_depth.buy_orders[price]
                    if price >= sell_price and qty > 0:
                        qty_to_sell = min(qty, -max_sell_qty, order_qty)
                        if qty_to_sell > 0:
                            orders.append(Order(product, price, -qty_to_sell))
                            max_sell_qty += qty_to_sell
                            current_position -= qty_to_sell
                            print(f"SELL {product} {qty_to_sell}x {price}")

            # Market making
            if max_buy_qty > 0 and (not order_depth.sell_orders or min(order_depth.sell_orders.keys()) > buy_price):
                orders.append(Order(product, buy_price, min(order_qty, max_buy_qty)))
                print(f"PROVIDE BID {product} {min(order_qty, max_buy_qty)}x {buy_price}")

            if max_sell_qty < 0 and (not order_depth.buy_orders or max(order_depth.buy_orders.keys()) < sell_price):
                orders.append(Order(product, sell_price, max(-order_qty, max_sell_qty)))
                print(f"PROVIDE ASK {product} {-max(-order_qty, max_sell_qty)}x {sell_price}")

            result[product] = orders
            conversions += self.handle_conversions(product, state, current_position)

        traderData = jsonpickle.encode(trader_state)
        return result, conversions, traderData