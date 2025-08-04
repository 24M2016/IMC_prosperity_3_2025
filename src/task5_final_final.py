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
        self.counterparty_volume: Dict[str, Dict[str, float]] = {}
        self.trade_history: Dict[str, List[Dict]] = {}
        self.prev_sunlight: float = 1000
        self.prev_sugar: float = 100
        self.position_limits: Dict[str, int] = {}
        self.loss_threshold: float = -10000

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
                        if t.timestamp >= state.timestamp - 500]  # Shorter lookback for recent trends

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
            total_qty = sum(t.quantity for t in trades if t.timestamp >= state.timestamp - 500)
            if total_qty > 0:
                vwap = sum(t.price * t.quantity for t in trades if t.timestamp >= state.timestamp - 500) / total_qty

        fair_value = mid_price
        if vwap:
            fair_value = 0.7 * mid_price + 0.3 * vwap  # Increased weight on mid_price for responsiveness
        fair_value += fair_value * imbalance * 0.015  # Reduced imbalance impact for tighter pricing

        trader_state = jsonpickle.decode(state.traderData) if state.traderData else TraderState()
        if product in trader_state.counterparty_avg_price and trader_state.counterparty_avg_price[product]:
            total_volume = sum(trader_state.counterparty_volume[product].values())
            if total_volume > 0:
                weighted_counter_price = sum(
                    trader_state.counterparty_avg_price[product][cp] * trader_state.counterparty_volume[product][cp]
                    for cp in trader_state.counterparty_avg_price[product]
                ) / total_volume
                fair_value = 0.85 * fair_value + 0.15 * weighted_counter_price  # Slightly less counterparty influence

        conv_obs = state.observations.conversionObservations.get(product)
        if conv_obs:
            sunlight = conv_obs.sunlightIndex
            sugar = conv_obs.sugarPrice
            sunlight_adjustment = 0
            if sunlight > trader_state.prev_sunlight + 50 and sunlight > 1050:  # Tighter thresholds
                sunlight_adjustment = fair_value * 0.004
            elif sunlight < trader_state.prev_sunlight - 50 and sunlight < 950:
                sunlight_adjustment = -fair_value * 0.004
            sugar_adjustment = 0
            if sugar > 200:
                sugar_adjustment = -fair_value * 0.0015
            elif sugar < 50:
                sugar_adjustment = fair_value * 0.0015
            fair_value += sunlight_adjustment + sugar_adjustment
            trader_state.prev_sunlight = sunlight
            trader_state.prev_sugar = sugar
            state.traderData = jsonpickle.encode(trader_state)

        return max(fair_value, 1.0)  # Ensure positive fair value

    def analyze_counterparty(self, product: str, own_trades: List[Trade], state_data: TraderState, current_timestamp: int) -> Tuple[Dict[str, float], float]:
        if product not in state_data.counterparty_trades:
            state_data.counterparty_trades[product] = {}
            state_data.counterparty_avg_price[product] = {}
            state_data.counterparty_volume[product] = {}

        aggression_score = 0.0
        for trade in own_trades[-20:]:  # Increased trade lookback for better counterparty analysis
            counter_party = trade.buyer if trade.seller == "SUBMISSION" else trade.seller
            if not counter_party or counter_party == "SUBMISSION" or counter_party == "":
                continue

            if counter_party not in state_data.counterparty_trades[product]:
                state_data.counterparty_trades[product][counter_party] = []
                state_data.counterparty_volume[product][counter_party] = 0
                state_data.counterparty_avg_price[product][counter_party] = 0  # Initialize to avoid KeyError
            trade_data = {
                "price": trade.price,
                "quantity": trade.quantity,
                "timestamp": trade.timestamp
            }
            state_data.counterparty_trades[product][counter_party].append(trade_data)
            state_data.counterparty_trades[product][counter_party] = state_data.counterparty_trades[product][counter_party][-20:]
            state_data.counterparty_volume[product][counter_party] += abs(trade.quantity)

            recent_trades = [t for t in state_data.counterparty_trades[product][counter_party]
                             if t["timestamp"] >= current_timestamp - 500]  # Shorter timestamp window
            prices = [t["price"] for t in recent_trades]
            quantities = [abs(t["quantity"]) for t in recent_trades]
            if prices:
                state_data.counterparty_avg_price[product][counter_party] = mean(prices)
                aggression_score += sum(quantities) / (len(quantities) * 10) if quantities else 0  # Adjusted normalization

        for cp in list(state_data.counterparty_volume[product].keys()):
            if state_data.counterparty_volume[product][cp] < 15:  # Increased volume threshold
                if cp in state_data.counterparty_avg_price[product]:  # Fix for KeyError
                    del state_data.counterparty_avg_price[product][cp]
                del state_data.counterparty_trades[product][cp]
                del state_data.counterparty_volume[product][cp]

        return state_data.counterparty_avg_price[product], min(aggression_score * 1.2, 0.6)  # Adjusted aggression cap

    def handle_conversions(self, product: str, state: TradingState, position: int, fair_value: float) -> int:
        conv_obs = state.observations.conversionObservations.get(product)
        if not conv_obs:
            return 0

        total_cost = conv_obs.transportFees + conv_obs.importTariff + conv_obs.exportTariff
        conversion_limit = 150  # Increased limit for more conversions
        conversions = 0

        # More aggressive conversion thresholds
        if position < 0 and conv_obs.bidPrice > fair_value + total_cost + 1.5:
            conversions = min(abs(position), conversion_limit)
        elif position > 0 and conv_obs.askPrice < fair_value - total_cost - 1.5:
            conversions = min(position, conversion_limit)

        trader_state = jsonpickle.decode(state.traderData) if state.traderData else TraderState()
        if product in trader_state.counterparty_avg_price and trader_state.counterparty_avg_price[product]:
            avg_counter_price = mean([p for p in trader_state.counterparty_avg_price[product].values() if p > 0])
            if position < 0 and avg_counter_price > fair_value + total_cost + 2:
                conversions = max(conversions, min(abs(position), conversion_limit // 2))
            elif position > 0 and avg_counter_price < fair_value - total_cost - 2:
                conversions = max(conversions, min(position, conversion_limit // 2))

        return conversions

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result = {}
        conversions = 0
        trader_state = jsonpickle.decode(state.traderData) if state.traderData else TraderState()
        trader_state.position_limits = self.position_limits

        total_pnl = sum(state.position.get(p, 0) * trader_state.fair_values.get(p, 1000) for p in state.position)
        if total_pnl < trader_state.loss_threshold:
            print("STOP-LOSS TRIGGERED: Reducing activity")
            return result, conversions, jsonpickle.encode(trader_state)

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

            trade_prices = [t.price for t in state.own_trades.get(product, []) + state.market_trades.get(product, [])]
            volatility = np.std(trade_prices) if trade_prices else 0.8  # Lower default volatility
            min_spread = 0.8 if fair_value > 100 else max(0.4, fair_value * 0.04)  # Tighter min spread
            spread = max(min_spread, volatility * 0.25) * (1.0 - aggression_score * 0.4)  # Tighter spread
            if state.timestamp < 500:  # Even tighter early on
                spread *= 0.7

            if conv_obs := state.observations.conversionObservations.get(product):
                conversion_cost = conv_obs.transportFees + max(conv_obs.exportTariff, conv_obs.importTariff)
                spread += conversion_cost * 0.015  # Reduced conversion cost impact

            buy_price = max(1, floor(fair_value - spread))
            sell_price = max(1, ceil(fair_value + spread))
            if counter_party_prices:
                valid_prices = [p for p in counter_party_prices.values() if p > 0]
                if valid_prices:
                    avg_counter_price = mean(valid_prices)
                    spread_limit = spread * 1.1  # Tighter spread limit
                    if avg_counter_price > fair_value + spread_limit:
                        sell_price = max(1, min(sell_price + 1, floor(avg_counter_price * 1.005)))  # More aggressive
                        print(f"Adjusting sell_price for {product} to {sell_price} due to counterparty")
                    elif avg_counter_price < fair_value - spread_limit:
                        buy_price = max(1, ceil(avg_counter_price * 0.995))
                        print(f"Adjusting buy_price for {product} to {buy_price} due to counterparty")

            order_qty = min(20, max(3, int(position_limit / (2 + volatility * 1.5))))  # More aggressive sizing
            if abs(current_position) > position_limit * 0.7:  # Adjusted threshold
                order_qty //= 2
            if aggression_score > 0.2:  # Lower threshold for aggression
                order_qty = int(order_qty * 1.3)
            if state.timestamp < 500:
                order_qty = min(order_qty, position_limit // 3)  # Slightly less conservative

            max_buy_qty = position_limit - current_position
            max_sell_qty = -(position_limit + current_position)

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

            if max_buy_qty > 0:
                orders.append(Order(product, buy_price, min(order_qty // 2, max_buy_qty)))
                if buy_price - 1 > min(order_depth.sell_orders.keys(), default=buy_price) and buy_price > 1:
                    orders.append(Order(product, buy_price - 1, min(order_qty // 3, max_buy_qty)))  # Larger secondary order
                print(f"PROVIDE BID {product} {min(order_qty // 2, max_buy_qty)}x {buy_price}")

            if max_sell_qty < 0:
                orders.append(Order(product, sell_price, max(-order_qty // 2, max_sell_qty)))
                if sell_price + 1 < max(order_depth.buy_orders.keys(), default=sell_price):
                    orders.append(Order(product, sell_price + 1, max(-order_qty // 3, max_sell_qty)))  # Larger secondary order
                print(f"PROVIDE ASK {product} {-max(-order_qty // 2, max_sell_qty)}x {sell_price}")

            result[product] = orders
            conversions += self.handle_conversions(product, state, current_position, fair_value)

        traderData = jsonpickle.encode(trader_state)
        return result, conversions, traderData