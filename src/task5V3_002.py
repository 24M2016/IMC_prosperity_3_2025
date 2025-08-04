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
        self.position_limits: Dict[str, int] = {}
        self.prev_sunlight: float = 1000
        self.prev_sugar: float = 100
        self.counterparty_avg_price: Dict[str, Dict[str, float]] = {}
        self.momentum: Dict[str, float] = {}
        self.volatility_history: Dict[str, List[float]] = {}  # Track volatility over time
        self.price_history: Dict[str, List[float]] = {}  # Track price trends
        self.last_counterparty_analysis: int = 0 # Timestamp of last counterparty analysis

class Trader:
    def __init__(self):
        self.position_limits = {
            "PEARLS": 20,
            "VOLCANIC_ROCK_VOUCHER_10500": 20
        }
        self.max_spread_factor = 1.5  # Reduced max spread
        self.min_spread_factor = 0.7  # Increased min spread

    def calculate_fair_value(self, product: str, order_depth: OrderDepth, state: TradingState) -> float:
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders
        trade_prices = [t.price for t in state.market_trades.get(product, []) if t.timestamp >= state.timestamp - 1000] # Shorter window
        trade_volumes = [t.quantity for t in state.market_trades.get(product, []) if t.timestamp >= state.timestamp - 1000] # Shorter window

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

        trader_state = jsonpickle.decode(state.traderData) if state.traderData else TraderState()
        if product not in trader_state.price_history:
            trader_state.price_history[product] = []
        trader_state.price_history[product].append(mid_price)
        trader_state.price_history[product] = trader_state.price_history[product][-5:]  # Keep last 5 prices

        momentum = trader_state.momentum.get(product, 0.0)
        if len(trade_prices) >= 3 and len(trade_volumes) >= 3: # Shorter window
            recent_prices = trade_prices[-3:]
            recent_volumes = trade_volumes[-3:]
            total_volume = sum(recent_volumes)
            if total_volume > 0:
                volume_weighted_prices = [p * v / total_volume for p, v in zip(recent_prices, recent_volumes)]
                price_diffs = np.diff(volume_weighted_prices)
                momentum = np.mean(price_diffs) / mid_price if mid_price != 0 else 0.0
                momentum = np.clip(momentum, -0.01, 0.01)  # Slightly tighter momentum range
            trader_state.momentum[product] = momentum

        fair_value = 0.6 * mid_price + 0.4 * mean(trade_prices) if trade_prices else mid_price
        fair_value += fair_value * momentum * 4.0  # Reduced momentum effect

        if product not in trader_state.volatility_history:
            trader_state.volatility_history[product] = []
        volatility = np.std(trade_prices) if trade_prices else 1.0
        trader_state.volatility_history[product].append(volatility)
        trader_state.volatility_history[product] = trader_state.volatility_history[product][-5:] # Keep last 5 volatilities
        avg_volatility = mean(trader_state.volatility_history[product]) if trader_state.volatility_history[product] else 1.0

        conv_obs = state.observations.conversionObservations.get(product)
        if conv_obs:
            sunlight = conv_obs.sunlightIndex
            sugar = conv_obs.sugarPrice

            sunlight_adjustment = 0
            sunlight_trend = sunlight - trader_state.prev_sunlight
            if sunlight_trend > 15 and sunlight > 1000: # Less sensitive
                sunlight_adjustment = fair_value * 0.007
            elif sunlight_trend < -15 and sunlight < 1000: # Less sensitive
                sunlight_adjustment = -fair_value * 0.007

            sugar_adjustment = 0
            if sugar > 160: # Less sensitive
                sugar_adjustment = -fair_value * 0.003
            elif sugar < 40: # Less sensitive
                sugar_adjustment = fair_value * 0.003

            fair_value += sunlight_adjustment + sugar_adjustment
            trader_state.prev_sunlight = sunlight
            trader_state.prev_sugar = sugar

        trader_state.fair_values[product] = fair_value
        state.traderData = jsonpickle.encode(trader_state)
        return fair_value

    def analyze_counterparty(self, product: str, own_trades: List[Trade], state_data: TraderState, current_timestamp: int) -> Tuple[Dict[str, float], float]:
        if product not in state_data.counterparty_trades:
            state_data.counterparty_trades[product] = {}
        if product not in state_data.counterparty_avg_price:
            state_data.counterparty_avg_price[product] = {}

        aggression_score = 0.0
        for trade in own_trades[-3:]:  # Look at last 3 trades - reduced scope
            counter_party = trade.buyer if trade.seller == "SUBMISSION" else trade.seller
            if counter_party and counter_party != "SUBMISSION":
                if counter_party not in state_data.counterparty_trades[product]:
                    state_data.counterparty_trades[product][counter_party] = []
                state_data.counterparty_trades[product][counter_party].append({
                    "price": trade.price,
                    "quantity": trade.quantity,
                    "timestamp": trade.timestamp
                })
                state_data.counterparty_trades[product][counter_party] = state_data.counterparty_trades[product][counter_party][-5:] # Keep last 5

                recent_trades = [t for t in state_data.counterparty_trades[product][counter_party] if t["timestamp"] >= current_timestamp - 1000] # Shorter window
                prices = [t["price"] for t in recent_trades]
                quantities = [t["quantity"] for t in recent_trades]
                if prices:
                    state_data.counterparty_avg_price[product][counter_party] = mean(prices)
                    avg_quantity = sum(quantities) / len(quantities) if quantities else 1
                    aggression_score += avg_quantity * 0.1 # Reduced weight

        return state_data.counterparty_avg_price[product], aggression_score

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result = {}
        conversions = 0
        trader_state = jsonpickle.decode(state.traderData) if state.traderData else TraderState()
        trader_state.position_limits = self.position_limits

        market_trade_prices = {
            product: [t.price for t in trades if t.timestamp >= state.timestamp - 500] # Even shorter window for direct price checks
            for product, trades in state.market_trades.items()
        }

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            current_position = state.position.get(product, 0)
            position_limit = trader_state.position_limits.get(product, 20)

            fair_value = self.calculate_fair_value(product, order_depth, state)

            # Analyze counterparty less frequently
            if state.timestamp - trader_state.last_counterparty_analysis > 500: # Analyze every 0.5 seconds
                counter_party_prices, aggression_score = self.analyze_counterparty(product, state.own_trades.get(product, []), trader_state, state.timestamp)
                trader_state.last_counterparty_analysis = state.timestamp
            else:
                counter_party_prices = trader_state.counterparty_avg_price.get(product, {})
                aggression_score = 0.0 # Don't use the score if not recently updated

            trade_prices = market_trade_prices.get(product, [])
            base_spread = np.std(trade_prices) * 0.5 if len(trade_prices) > 1 else 1.5 # Simplified base spread
            volatility_factor = mean(trader_state.volatility_history.get(product, [1.0]))
            position_factor = abs(current_position) / position_limit if position_limit != 0 else 0
            spread = max(base_spread, 1.0) * (self.min_spread_factor + (self.max_spread_factor - self.min_spread_factor) * volatility_factor / 1.5) # Reduced impact
            spread *= (1.0 - aggression_score * 0.3) * (1.0 + position_factor * 0.4) # Reduced impact

            conv_obs = state.observations.conversionObservations.get(product)
            if conv_obs:
                conversion_cost = conv_obs.transportFees + max(conv_obs.exportTariff, conv_obs.importTariff)
                spread += conversion_cost * 0.03 # Reduced impact

            buy_price = floor(fair_value - spread)
            sell_price = ceil(fair_value + spread)

            # Less aggressive adjustment based on counterparty
            if counter_party_prices:
                avg_counter_price = mean(counter_party_prices.values()) if counter_party_prices.values() else fair_value
                spread_limit = spread * 1.5
                if avg_counter_price > fair_value + spread_limit:
                    sell_price = min(sell_price + 2, floor(avg_counter_price * 1.02))
                elif avg_counter_price < fair_value - spread_limit:
                    buy_price = max(buy_price - 2, ceil(avg_counter_price * 0.98))

            volatility = np.std(trade_prices) if trade_prices else 1.0
            order_qty = min(5, max(1, int(position_limit / (4 + volatility * 1.5)))) # Less aggressive sizing
            max_buy_qty = position_limit - current_position
            max_sell_qty = -(position_limit + current_position)

            price_history = trader_state.price_history.get(product, [])
            if price_history and len(price_history) >= 3: # Less strict stop-loss
                price_range = max(price_history) - min(price_history)
                if price_range > fair_value * 0.08: # Less sensitive stop-loss
                    order_qty = max(1, order_qty // 2)
                    spread *= 1.3
                    buy_price = floor(fair_value - spread)
                    sell_price = ceil(fair_value + spread)

            if max_buy_qty > 0:
                for price in sorted(order_depth.sell_orders.keys()):
                    qty = order_depth.sell_orders[price]
                    if price <= buy_price and qty < 0:
                        qty_to_buy = min(-qty, max_buy_qty, order_qty)
                        if qty_to_buy > 0:
                            orders.append(Order(product, price, qty_to_buy))
                            max_buy_qty -= qty_to_buy
                            current_position += qty_to_buy
                if max_buy_qty > 0:
                    orders.append(Order(product, buy_price, min(order_qty, max_buy_qty)))

            if max_sell_qty < 0:
                for price in sorted(order_depth.buy_orders.keys(), reverse=True):
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

        traderData = jsonpickle.encode(trader_state)
        return result, conversions, traderData