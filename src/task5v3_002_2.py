from datamodel import OrderDepth, UserId, TradingState, Order, Trade
from typing import Dict, List, Tuple
import jsonpickle
import numpy as np
from statistics import mean
from math import ceil, floor
from time import time
import traceback

class TraderState:
    def __init__(self):
        self.fair_values: Dict[str, float] = {}
        self.counterparty_trades: Dict[str, Dict[str, List[Dict]]] = {}
        self.position_limits: Dict[str, int] = {}
        self.prev_sunlight: float = 1000
        self.prev_sugar: float = 100
        self.counterparty_avg_price: Dict[str, Dict[str, float]] = {}
        self.momentum: Dict[str, float] = {}
        self.volatility_history: Dict[str, List[float]] = {}
        self.price_history: Dict[str, List[float]] = {}
        self.q_table: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.prev_profit: Dict[str, float] = {}
        self.prev_state: Dict[str, str] = {}
        self.prev_action: Dict[str, str] = {}

class Trader:
    def __init__(self):
        self.default_position_limit = 20
        self.position_limits = {
            "PEARLS": self.default_position_limit,
            "VOLCANIC_ROCK_VOUCHER_10500": self.default_position_limit
        }
        self.max_spread_factor = 2.0
        self.min_spread_factor = 0.5
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.9
        # Reduced state/action spaces
        self.volatility_bins = [0, 2, float('inf')]  # 2 states
        self.position_bins = [-0.5, 0.5]  # 1 state
        self.momentum_bins = [-0.01, 0.01]  # 1 state
        self.aggression_bins = [0.0, 1.0]  # 1 state
        self.spread_multipliers = [0.5, 1.0]  # 2 actions
        self.order_sizes = [1, 3]  # 2 actions
        self.aggression_adjustments = [0.0]  # 1 action
        # Pre-initialize Q-table for known products
        self.q_table = {
            "PEARLS": {},
            "VOLCANIC_ROCK_VOUCHER_10500": {}
        }
        action_space = self._get_action_space()
        for product in self.q_table:
            self.q_table[product] = {
                f"{v}_{p}_{m}_{a}": {act: 0.0 for act in action_space}
                for v in range(len(self.volatility_bins)-1)
                for p in range(len(self.position_bins)-1)
                for m in range(len(self.momentum_bins)-1)
                for a in range(len(self.aggression_bins)-1)
            }
        self.max_run_time = 0.8  # Stop processing if exceeding 80% of 1s timeout

    def _initialize_product(self, product: str, trader_state: TraderState):
        if product not in trader_state.q_table:
            action_space = self._get_action_space()
            trader_state.q_table[product] = {
                f"{v}_{p}_{m}_{a}": {act: 0.0 for act in action_space}
                for v in range(len(self.volatility_bins)-1)
                for p in range(len(self.position_bins)-1)
                for m in range(len(self.momentum_bins)-1)
                for a in range(len(self.aggression_bins)-1)
            }
        if product not in trader_state.position_limits:
            trader_state.position_limits[product] = self.default_position_limit

    def _discretize_state(self, volatility: float, position_ratio: float, momentum: float, aggression: float) -> str:
        vol_bin = next(i for i, v in enumerate(self.volatility_bins) if volatility <= v)
        pos_bin = next(i for i, v in enumerate(self.position_bins) if position_ratio <= v)
        mom_bin = next(i for i, v in enumerate(self.momentum_bins) if momentum <= v)
        agg_bin = next(i for i, v in enumerate(self.aggression_bins) if aggression <= v)
        return f"{vol_bin}_{pos_bin}_{mom_bin}_{agg_bin}"

    def _get_action_space(self) -> List[str]:
        actions = []
        for sm in self.spread_multipliers:
            for os in self.order_sizes:
                for aa in self.aggression_adjustments:
                    actions.append(f"{sm}_{os}_{aa}")
        return actions

    def _choose_action(self, state: str, product: str, trader_state: TraderState) -> str:
        if np.random.random() < self.epsilon:
            return np.random.choice(self._get_action_space())
        return max(trader_state.q_table[product][state], key=trader_state.q_table[product][state].get)

    def _update_q_table(self, product: str, state: str, action: str, reward: float, next_state: str, trader_state: TraderState):
        current_q = trader_state.q_table[product][state][action]
        max_future_q = max(trader_state.q_table[product][next_state].values())
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        trader_state.q_table[product][state][action] = new_q

    def calculate_fair_value(self, product: str, order_depth: OrderDepth, state: TradingState, trader_state: TraderState) -> float:
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        trades = state.market_trades.get(product, [])
        trade_prices = [t.price for t in trades if t.timestamp >= state.timestamp - 2000][:10]
        trade_volumes = [t.quantity for t in trades if t.timestamp >= state.timestamp - 2000][:10]

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

        if product not in trader_state.price_history:
            trader_state.price_history[product] = []
        trader_state.price_history[product].append(mid_price)
        trader_state.price_history[product] = trader_state.price_history[product][-5:]

        momentum = trader_state.momentum.get(product, 0.0)
        if len(trade_prices) >= 3:
            recent_prices = trade_prices[-3:]
            momentum = (recent_prices[-1] - recent_prices[0]) / mid_price if mid_price != 0 else 0.0
            momentum = np.clip(momentum, -0.015, 0.015)
        trader_state.momentum[product] = momentum

        fair_value = mid_price if not trade_prices else 0.6 * mid_price + 0.4 * mean(trade_prices)
        fair_value += fair_value * momentum * 4.0

        volatility = 1.0
        if trade_prices:
            volatility = np.std(trade_prices[-5:])
        if product not in trader_state.volatility_history:
            trader_state.volatility_history[product] = []
        trader_state.volatility_history[product].append(volatility)
        trader_state.volatility_history[product] = trader_state.volatility_history[product][-5:]

        conv_obs = state.observations.conversionObservations.get(product)
        if conv_obs:
            sunlight = conv_obs.sunlightIndex
            sugar = conv_obs.sugarPrice
            sunlight_adjustment = fair_value * 0.01 if sunlight > trader_state.prev_sunlight + 10 else 0
            sugar_adjustment = fair_value * (-0.005 if sugar > 150 else 0.005 if sugar < 50 else 0)
            fair_value += sunlight_adjustment + sugar_adjustment
            trader_state.prev_sunlight = sunlight
            trader_state.prev_sugar = sugar

        trader_state.fair_values[product] = fair_value
        return fair_value

    def analyze_counterparty(self, product: str, own_trades: List[Trade], state_data: TraderState, current_timestamp: int) -> Tuple[Dict[str, float], float]:
        if product not in state_data.counterparty_trades:
            state_data.counterparty_trades[product] = {}
        if product not in state_data.counterparty_avg_price:
            state_data.counterparty_avg_price[product] = {}

        aggression_score = 0.0
        for trade in own_trades[-5:]:
            counter_party = trade.buyer if trade.seller == "SUBMISSION" else trade.seller
            if counter_party and counter_party != "SUBMISSION":
                if counter_party not in state_data.counterparty_trades[product]:
                    state_data.counterparty_trades[product][counter_party] = []
                state_data.counterparty_trades[product][counter_party].append({
                    "price": trade.price,
                    "quantity": trade.quantity,
                    "timestamp": trade.timestamp
                })
                state_data.counterparty_trades[product][counter_party] = state_data.counterparty_trades[product][counter_party][-5:]
                prices = [t["price"] for t in state_data.counterparty_trades[product][counter_party]]
                quantities = [t["quantity"] for t in state_data.counterparty_trades[product][counter_party]]
                if prices:
                    state_data.counterparty_avg_price[product][counter_party] = mean(prices)
                    aggression_score += sum(quantities) / len(quantities) * 0.1

        return state_data.counterparty_avg_price[product], aggression_score

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        start_time = time()
        result = {}
        conversions = 0
        trader_state = jsonpickle.decode(state.traderData) if state.traderData else TraderState()
        trader_state.position_limits = self.position_limits.copy()
        trader_state.q_table = self.q_table
        print(f"Deserialization time: {time() - start_time:.3f}s")

        # Initialize Q-table for all products in order_depths
        for product in state.order_depths:
            self._initialize_product(product, trader_state)

        # Prioritize known products
        product_priority = ["PEARLS", "VOLCANIC_ROCK_VOUCHER_10500"] + [p for p in state.order_depths if p not in ["PEARLS", "VOLCANIC_ROCK_VOUCHER_10500"]]

        for product in product_priority:
            if product not in state.order_depths:
                continue
            if time() - start_time > self.max_run_time:
                print(f"Skipping product {product} to avoid timeout; current time: {time() - start_time:.3f}s")
                continue

            section_time = time()
            try:
                order_depth = state.order_depths[product]
                orders: List[Order] = []
                current_position = state.position.get(product, 0)
                position_limit = trader_state.position_limits.get(product, self.default_position_limit)
                position_ratio = current_position / position_limit if position_limit != 0 else 0.0

                if product in trader_state.price_history:
                    trader_state.price_history[product] = trader_state.price_history[product][-5:]
                if product in trader_state.volatility_history:
                    trader_state.volatility_history[product] = trader_state.volatility_history[product][-5:]
                if product in trader_state.counterparty_trades:
                    for counter_party in trader_state.counterparty_trades[product]:
                        trader_state.counterparty_trades[product][counter_party] = trader_state.counterparty_trades[product][counter_party][-5:]

                fair_value = self.calculate_fair_value(product, order_depth, state, trader_state)
                print(f"Fair value time ({product}): {time() - section_time:.3f}s")
                section_time = time()

                counter_party_prices, aggression_score = self.analyze_counterparty(product, state.own_trades.get(product, []), trader_state, state.timestamp)
                print(f"Counterparty time ({product}): {time() - section_time:.3f}s")
                section_time = time()

                trade_prices = [t.price for t in state.market_trades.get(product, [])][:10]
                volatility = np.std(trade_prices) if trade_prices else 1.0
                momentum = trader_state.momentum.get(product, 0.0)

                state_str = self._discretize_state(volatility, position_ratio, momentum, aggression_score)
                action = self._choose_action(state_str, product, trader_state)
                spread_multiplier, order_qty, aggression_adj = map(float, action.split('_'))
                order_qty = int(order_qty)

                base_spread = np.std(trade_prices) * 0.7 if trade_prices else 2.0
                spread = max(base_spread, 1.0) * spread_multiplier * (1.0 - aggression_score * (0.4 + aggression_adj))
                conv_obs = state.observations.conversionObservations.get(product)
                if conv_obs:
                    conversion_cost = conv_obs.transportFees + max(conv_obs.exportTariff, conv_obs.importTariff)
                    spread += conversion_cost * 0.05

                buy_price = floor(fair_value - spread)
                sell_price = ceil(fair_value + spread)
                if counter_party_prices:
                    avg_counter_price = mean(counter_party_prices.values())
                    spread_limit = spread * 2.0
                    if avg_counter_price > fair_value + spread_limit:
                        sell_price = min(sell_price + 3, floor(avg_counter_price * 1.03))
                    elif avg_counter_price < fair_value - spread_limit:
                        buy_price = max(buy_price - 3, ceil(avg_counter_price * 0.97))

                price_history = trader_state.price_history.get(product, [])
                if price_history and len(price_history) >= 5:
                    price_range = max(price_history) - min(price_history)
                    if price_range > fair_value * 0.1:
                        order_qty = max(1, order_qty // 2)
                        spread *= 1.5
                        buy_price = floor(fair_value - spread)
                        sell_price = ceil(fair_value + spread)

                max_buy_qty = position_limit - current_position
                max_sell_qty = -(position_limit + current_position)

                if max_buy_qty > 0:
                    sell_prices = sorted(order_depth.sell_orders.keys())[:3]
                    for price in sell_prices:
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
                    buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)[:3]
                    for price in buy_prices:
                        qty = order_depth.buy_orders[price]
                        if price >= sell_price and qty > 0:
                            qty_to_sell = min(qty, -max_sell_qty, order_qty)
                            if qty_to_sell > 0:
                                orders.append(Order(product, price, -qty_to_sell))
                                max_sell_qty += qty_to_sell
                                current_position -= qty_to_sell
                    if max_sell_qty < 0:
                        orders.append(Order(product, sell_price, max(-order_qty, max_sell_qty)))

                profit = 0.0
                for trade in state.own_trades.get(product, [])[:5]:
                    if trade.timestamp >= state.timestamp - 100:
                        if trade.buyer == "SUBMISSION":
                            profit += (fair_value - trade.price) * trade.quantity
                        elif trade.seller == "SUBMISSION":
                            profit += (trade.price - fair_value) * trade.quantity
                inventory_risk = -0.01 * abs(current_position) * fair_value
                reward = profit + inventory_risk

                if product in trader_state.prev_state and product in trader_state.prev_action:
                    self._update_q_table(
                        product,
                        trader_state.prev_state[product],
                        trader_state.prev_action[product],
                        reward,
                        state_str,
                        trader_state
                    )
                trader_state.prev_state[product] = state_str
                trader_state.prev_action[product] = action

                result[product] = orders
                print(f"RL and order time ({product}): {time() - section_time:.3f}s")
            except Exception as e:
                print(f"Error processing product {product}: {str(e)}")
                print(traceback.format_exc())

        traderData = jsonpickle.encode(trader_state)
        total_time = time() - start_time
        print(f"Total run time: {total_time:.3f}s")
        return result, conversions, traderData