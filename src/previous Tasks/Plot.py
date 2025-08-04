import pandas as pd
import numpy as np
from statistics import mean, stdev
import matplotlib.pyplot as plt
#from trader import Trader
from task4 import Trader
from datamodel import OrderDepth, Order, TradingState

# Load data
prices_df = pd.read_csv("round-4-island-data-bottle/prices.csv", delimiter=';')
prices_df = prices_df.sort_values(by='timestamp')
timestamps = sorted(prices_df['timestamp'].unique())

PRODUCTS = [
    "VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750",
    "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250", "VOLCANIC_ROCK_VOUCHER_10500"
]

# Create dict of product -> timestamp -> mid_price
mid_prices = {}
for product in PRODUCTS:
    df = prices_df[prices_df['product'] == product].copy()
    df['mid_price'] = df[['bid_price_1', 'ask_price_1']].mean(axis=1)
    mid_prices[product] = dict(zip(df['timestamp'], df['mid_price']))

# Initialize trader and tracking
trader = Trader()
positions = {p: 0 for p in PRODUCTS}
cash = 0
portfolio_values = []
pnl_history = []
iv_history = {p: [] for p in PRODUCTS if "VOUCHER" in p}

# Run simulation
for ts in timestamps:
    order_depths = {}
    for product in PRODUCTS:
        snap = prices_df[(prices_df['timestamp'] == ts) & (prices_df['product'] == product)]
        if snap.empty:
            continue
        row = snap.iloc[0]
        buy_orders = {row[f'bid_price_{i}']: row[f'bid_volume_{i}'] for i in range(1, 4) if row[f'bid_price_{i}'] > 0}
        sell_orders = {row[f'ask_price_{i}']: -row[f'ask_volume_{i}'] for i in range(1, 4) if row[f'ask_price_{i}'] > 0}
        od = OrderDepth()
        od.buy_orders = buy_orders
        od.sell_orders = sell_orders
        order_depths[product] = od

    state = TradingState(
        timestamp=ts, listings={}, order_depths=order_depths,
        own_trades={}, market_trades={}, position=positions.copy(),
        observations=None, traderData=""
    )

    orders_dict, _, _ = trader.run(state)

    # Simulate fills
    for product, orders in orders_dict.items():
        for order in orders:
            od = order_depths[product]
            if order.quantity > 0:
                best_ask = min(od.sell_orders.keys(), default=float('inf'))
                if order.price >= best_ask:
                    qty = min(order.quantity, -od.sell_orders.get(best_ask, 0))
                    positions[product] += qty
                    cash -= qty * best_ask
            elif order.quantity < 0:
                best_bid = max(od.buy_orders.keys(), default=0)
                if order.price <= best_bid:
                    qty = min(-order.quantity, od.buy_orders.get(best_bid, 0))
                    positions[product] -= qty
                    cash += qty * best_bid

    # Calculate portfolio value
    value = cash
    for product in PRODUCTS:
        mid = mid_prices[product].get(ts, None)
        if mid is not None:
            value += positions[product] * mid
    portfolio_values.append(value)
    if len(portfolio_values) > 1:
        pnl_history.append(portfolio_values[-1] - portfolio_values[-2])

    # Track implied volatilities
    for voucher in [p for p in PRODUCTS if "VOUCHER" in p]:
        if trader.implied_vols[voucher]:
            iv_history[voucher].append(trader.implied_vols[voucher][-1])

# Calculate metrics
total_pnl = portfolio_values[-1] - portfolio_values[0]
returns = [pnl / abs(portfolio_values[i-1]) for i, pnl in enumerate(pnl_history) if portfolio_values[i-1] != 0]
sharpe_ratio = (mean(returns) / stdev(returns)) * np.sqrt(252) if returns and stdev(returns) > 0 else 0

print(f"Total PnL: {total_pnl:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Plot portfolio value
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(timestamps, portfolio_values)
plt.xlabel("Timestamp")
plt.ylabel("Portfolio Value")
plt.title("Strategy Performance")
plt.grid(True)

# Plot implied volatilities
plt.subplot(2, 1, 2)
for voucher in iv_history:
    if iv_history[voucher]:
        plt.plot(timestamps[:len(iv_history[voucher])], iv_history[voucher], label=voucher)
plt.xlabel("Timestamp")
plt.ylabel("Implied Volatility")
plt.title("Implied Volatility Across Vouchers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()