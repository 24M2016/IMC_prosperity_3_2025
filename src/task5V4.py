from datamodel import OrderDepth, UserId, TradingState, Order, Trade
from typing import Dict, List, Tuple
import jsonpickle
import numpy as np
from statistics import mean, stdev
from math import ceil, floor, isnan, isinf
import pandas as pd

class TraderState:
    def __init__(self):
        self.fair_values: Dict[str, float] = {}
        self.price_history: Dict[str, List[float]] = {}
        self.volatility: Dict[str, float] = {}
        self.momentum: Dict[str, float] = {}
        self.regime: Dict[str, str] = {}  # 'trending' or 'ranging'
        self.profit_tracker: Dict[str, float] = {}
        self.drawdown: float = 0.0
        self.daily_loss: float = 0.0
        self.position_limits: Dict[str, int] = {}

class Trader:
    def __init__(self):
        self.position_limits = {
            "PEARLS": 20,
            "VOLCANIC_ROCK_VOUCHER_10500": 20
        }
        self.MAX_PRICE = 1000000
        self.MIN_PRICE = 1
        self.MAX_DRAWDOWN = -5000  # Max allowable drawdown
        self.DAILY_LOSS_LIMIT = -2000  # Max daily loss
        self.MAX_POSITION_RISK = 0.1  # Max 10% of position limit per trade
        self.ATR_PERIOD = 5  # For volatility calculation
        self.MIN_RR_RATIO = 1.5  # Minimum reward-to-risk ratio

    def calculate_atr(self, product: str, prices: List[float], period: int = 5) -> float:
        if len(prices) < period:
            return 1.0
        highs = prices[1:]
        lows = prices[:-1]
        tr = [max(h - l, abs(h - prices[i-1]), abs(l - prices[i-1])) for i, (h, l) in enumerate(zip(highs, lows))]
        return mean(tr[-period:]) if tr else 1.0

    def detect_regime(self, product: str, prices: List[float], trader_state: TraderState) -> str:
        if len(prices) < 5:
            return 'ranging'
        returns = np.diff(prices) / prices[:-1]
        std_returns = stdev(returns) if len(returns) > 1 else 0.01
        momentum = trader_state.momentum.get(product, 0.0)
        if abs(momentum) > 0.01 and std_returns > 0.015:
            return 'trending'
        return 'ranging'

    def predict_price(self, product: str, trader_state: TraderState, prices: List[float]) -> float:
        if len(prices) < 5:
            return trader_state.fair_values.get(product, 1000)
        # Simple exponential moving average with momentum adjustment
        weights = np.exp(np.linspace(-1., 0., 5))
        weights /= weights.sum()
        ema = np.average(prices[-5:], weights=weights)
        momentum = trader_state.momentum.get(product, 0.0)
        return ema * (1 + momentum * 2.0)

    def calculate_fair_value(self, product: str, order_depth: OrderDepth, state: TradingState, trader_state: TraderState) -> float:
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders
        trade_prices = [t.price for t in state.market_trades.get(product, []) if t.timestamp >= state.timestamp - 1000]

        default_fair_value = trader_state.fair_values.get(product, 1000)
        if product not in trader_state.price_history:
            trader_state.price_history[product] = []

        # Calculate mid-price
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
            mid_price = trade_prices[-1] if trade_prices else default_fair_value

        trader_state.price_history[product].append(mid_price)
        trader_state.price_history[product] = trader_state.price_history[product][-20:]

        # Update volatility and momentum
        if len(trade_prices) >= 3:
            trader_state.volatility[product] = stdev(trade_prices[-3:]) / mid_price if mid_price != 0 else 1.0
            trader_state.momentum[product] = np.mean(np.diff(trade_prices[-3:])) / mid_price if mid_price != 0 else 0.0
        else:
            trader_state.volatility[product] = 1.0
            trader_state.momentum[product] = 0.0

        # Get statistical price prediction
        predicted_price = self.predict_price(product, trader_state, trader_state.price_history[product])

        # Combine mid-price and prediction
        fair_value = 0.7 * mid_price + 0.3 * predicted_price

        # Apply observation adjustments
        conv_obs = state.observations.conversionObservations.get(product)
        if conv_obs:
            sunlight = conv_obs.sunlightIndex
            sugar = conv_obs.sugarPrice
            fair_value *= (1 + 0.01 * (sunlight / 1000 - 1))
            fair_value *= (1 - 0.005 * (sugar / 100 - 1))

        # Ensure valid fair value
        fair_value = np.clip(fair_value, self.MIN_PRICE, self.MAX_PRICE)
        trader_state.fair_values[product] = fair_value
        trader_state.regime[product] = self.detect_regime(product, trader_state.price_history[product], trader_state)
        return fair_value

    def calculate_position_size(self, product: str, volatility: float, position_limit: int, fair_value: float) -> int:
        # Volatility-adjusted sizing
        risk_per_trade = min(0.05, 0.02 / (volatility + 0.01))
        base_size = position_limit * risk_per_trade
        atr = self.calculate_atr(product, [fair_value] * 5)  # Placeholder if no history
        size = base_size / (atr / fair_value + 0.01) if fair_value != 0 else base_size
        return min(max(1, int(size)), int(position_limit * self.MAX_POSITION_RISK))

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result = {}
        conversions = 0
        trader_state = jsonpickle.decode(state.traderData) if state.traderData else TraderState()
        trader_state.position_limits = self.position_limits

        # Track drawdown and daily loss
        total_pnl = sum(trader_state.profit_tracker.values())
        trader_state.drawdown = min(trader_state.drawdown, total_pnl)
        if trader_state.drawdown < self.MAX_DRAWDOWN or trader_state.daily_loss < self.DAILY_LOSS_LIMIT:
            return result, conversions, jsonpickle.encode(trader_state)

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            current_position = state.position.get(product, 0)
            position_limit = trader_state.position_limits.get(product, 20)
            volatility = trader_state.volatility.get(product, 1.0)
            fair_value = self.calculate_fair_value(product, order_depth, state, trader_state)

            # Update profit tracker
            trades = state.own_trades.get(product, [])
            avg_position_price = mean([t.price for t in trades if t.quantity != 0]) if trades else fair_value
            trader_state.profit_tracker[product] = current_position * (fair_value - avg_position_price)

            # Calculate dynamic spread and stops
            atr = self.calculate_atr(product, trader_state.price_history[product])
            spread = max(atr * 0.5, 1.0)
            stop_loss = atr * 1.5
            take_profit = atr * self.MIN_RR_RATIO

            # Regime-based strategy
            regime = trader_state.regime.get(product, 'ranging')
            if regime == 'trending':
                spread *= 0.8  # Tighter spread in trends
                take_profit *= 1.2  # Larger targets in trends
            else:
                spread *= 1.2  # Wider spread in ranging markets
                take_profit *= 0.8  # Smaller targets in ranging markets

            buy_price = floor(fair_value - spread)
            sell_price = ceil(fair_value + spread)
            buy_price = max(buy_price, self.MIN_PRICE)
            sell_price = min(sell_price, self.MAX_PRICE)

            # Volatility-adjusted position sizing
            order_qty = self.calculate_position_size(product, volatility, position_limit, fair_value)
            max_buy_qty = position_limit - current_position
            max_sell_qty = -(position_limit + current_position)

            # Signal confirmation (momentum + volatility breakout)
            momentum = trader_state.momentum.get(product, 0.0)
            is_valid_trade = abs(momentum) > 0.005 and volatility > 0.01

            if is_valid_trade:
                # Buy logic
                if max_buy_qty > 0:
                    for price in sorted(order_depth.sell_orders.keys())[:2]:
                        qty = order_depth.sell_orders[price]
                        if price <= buy_price + stop_loss and qty < 0:
                            qty_to_buy = min(-qty, max_buy_qty, order_qty)
                            if qty_to_buy > 0:
                                orders.append(Order(product, price, qty_to_buy))
                                max_buy_qty -= qty_to_buy
                    if max_buy_qty > 0:
                        orders.append(Order(product, buy_price, min(order_qty, max_buy_qty)))

                # Sell logic
                if max_sell_qty < 0:
                    for price in sorted(order_depth.buy_orders.keys(), reverse=True)[:2]:
                        qty = order_depth.buy_orders[price]
                        if price >= sell_price - stop_loss and qty > 0:
                            qty_to_sell = min(qty, -max_sell_qty, order_qty)
                            if qty_to_sell > 0:
                                orders.append(Order(product, price, -qty_to_sell))
                                max_sell_qty += qty_to_sell
                    if max_sell_qty < 0:
                        orders.append(Order(product, sell_price, max(-order_qty, max_sell_qty)))

            result[product] = orders

        trader_state.daily_loss = min(trader_state.daily_loss, total_pnl)
        traderData = jsonpickle.encode(trader_state)
        return result, conversions, traderData