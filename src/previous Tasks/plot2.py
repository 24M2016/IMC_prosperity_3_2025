import pandas as pd
import numpy as np
from statistics import mean, stdev
import matplotlib.pyplot as plt
from Task5 import Trader
from datamodel import OrderDepth, Order, TradingState, Observation, ConversionObservation

# Load data
prices_df = pd.read_csv("round-4-island-data-bottle/prices.csv", delimiter=';')
prices_df = prices_df.sort_values(by='timestamp')
try:
    trades_df = pd.read_csv("round-4-island-data-bottle/trades.csv", delimiter=';')
except FileNotFoundError:
    trades_df = pd.DataFrame(columns=['timestamp', 'symbol', 'price', 'quantity'])

timestamps = sorted(prices_df['timestamp'].unique())

PRODUCTS = [
    "VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750",
    "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250", "VOLCANIC_ROCK_VOUCHER_10500",
    "MAGNIFICENT_MACARONS"
]

# Create dict of product -> timestamp -> mid_price
mid_prices = {}
for product in PRODUCTS:
    df = prices_df[prices_df['product'] == product].copy()
    df['mid_price'] = df[['bid_price_1', 'ask_price_1']].mean(axis=1)
    mid_prices[product] = dict(zip(df['timestamp'], df['mid_price']))

# Simulate conversion observations (fallback if not in data)
# Assuming trades.csv might have some conversion-related data or we simulate defaults
conversion_data = {
    'timestamp': [],
    'sunlightIndex': [],
    'sugarPrice': [],
    'transportFees': [],
    'exportTariff': [],
    'importTariff': [],
    'bidPrice': [],
    'askPrice': []
}

# If trades.csv contains conversion data, extract it; otherwise, simulate
for ts in timestamps:
    conversion_data['timestamp'].append(ts)
    conversion_data['sunlightIndex'].append(1000)  # Default
    conversion_data['sugarPrice'].append(100)      # Default
    conversion_data['transportFees'].append(0)     # Default
    conversion_data['exportTariff'].append(0)      # Default
    conversion_data['importTariff'].append(0)      # Default
    conversion_data['bidPrice'].append(100)       # Default
    conversion_data['askPrice'].append(102)       # Default

conversion_df = pd.DataFrame(conversion_data)

# Initialize trader and tracking
trader = Trader()
positions = {p: 0 for p in PRODUCTS}
cash = 0
portfolio_values = []
pnl_history = []
iv_history = {p: [] for p in PRODUCTS if "VOUCHER" in p}
position_history = {p: [] for p in PRODUCTS}
conversion_history = []
cumulative_conversions = 0

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

    # Create conversion observations
    conv_row = conversion_df[conversion_df['timestamp'] == ts]
    if not conv_row.empty:
        conv_obs = ConversionObservation(
            bidPrice=conv_row['bidPrice'].iloc[0],
            askPrice=conv_row['askPrice'].iloc[0],
            transportFees=conv_row['transportFees'].iloc[0],
            exportTariff=conv_row['exportTariff'].iloc[0],
            importTariff=conv_row['importTariff'].iloc[0],
            sugarPrice=conv_row['sugarPrice'].iloc[0],
            sunlightIndex=conv_row['sunlightIndex'].iloc[0]
        )
        conversion_observations = {"MAGNIFICENT_MACARONS": conv_obs}
    else:
        conversion_observations = {}

    observations = Observation(
        plainValueObservations={},
        conversionObservations=conversion_observations
    )

    state = TradingState(
        timestamp=ts,
        listings={},
        order_depths=order_depths,
        own_trades={},
        market_trades={},
        position=positions.copy(),
        observations=observations,
        traderData=""
    )

    orders_dict, conversions, _ = trader.run(state)

    # Simulate fills
    for product, orders in orders_dict.items():
        for order in orders:
            od = order_depths[product]
            if order.quantity > 0:
                best_ask = min(od.sell_orders.keys(), default=float('inf'))
                if order.price >= best_ask and best_ask < float('inf'):
                    qty = min(order.quantity, -od.sell_orders.get(best_ask, 0))
                    positions[product] += qty
                    cash -= qty * best_ask
            elif order.quantity < 0:
                best_bid = max(od.buy_orders.keys(), default=0)
                if order.price <= best_bid and best_bid > 0:
                    qty = min(-order.quantity, od.buy_orders.get(best_bid, 0))
                    positions[product] -= qty
                    cash += qty * best_bid

    # Handle conversions
    if conversions != 0:
        conv_obs = conversion_observations.get("MAGNIFICENT_MACARONS")
        if conv_obs:
            if conversions > 0:
                # Convert macarons to cash
                cash += conversions * (conv_obs.bidPrice - conv_obs.transportFees - conv_obs.exportTariff)
                positions["MAGNIFICENT_MACARONS"] -= conversions
            elif conversions < 0:
                # Convert cash to macarons
                cash -= abs(conversions) * (conv_obs.askPrice + conv_obs.transportFees + conv_obs.importTariff)
                positions["MAGNIFICENT_MACARONS"] -= conversions  # conversions is negative
        cumulative_conversions += conversions
        conversion_history.append(cumulative_conversions)
    else:
        conversion_history.append(cumulative_conversions)

    # Calculate portfolio value
    value = cash
    for product in PRODUCTS:
        mid = mid_prices[product].get(ts, None)
        if mid is not None:
            value += positions[product] * mid
        position_history[product].append(positions[product])
    portfolio_values.append(value)
    if len(portfolio_values) > 1:
        pnl_history.append(portfolio_values[-1] - portfolio_values[-2])

    # Track implied volatilities
    for voucher in [p for p in PRODUCTS if "VOUCHER" in p]:
        if hasattr(trader, 'implied_vols') and voucher in trader.implied_vols and trader.implied_vols[voucher]:
            iv_history[voucher].append(trader.implied_vols[voucher][-1])
        else:
            iv_history[voucher].append(0)

# Calculate metrics
total_pnl = portfolio_values[-1] - portfolio_values[0] if portfolio_values else 0
returns = [pnl / abs(portfolio_values[i-1]) for i, pnl in enumerate(pnl_history) if i > 0 and portfolio_values[i-1] != 0]
sharpe_ratio = (mean(returns) / stdev(returns)) * np.sqrt(252) if returns and stdev(returns) > 0 else 0
max_drawdown = 0
peak = portfolio_values[0]
for value in portfolio_values:
    if value > peak:
        peak = value
    drawdown = (peak - value) / peak
    max_drawdown = max(max_drawdown, drawdown)

print(f"Total PnL: {total_pnl:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# Plotting
plt.figure(figsize=(12, 12))

# Portfolio Value
plt.subplot(3, 1, 1)
plt.plot(timestamps, portfolio_values)
plt.xlabel("Timestamp")
plt.ylabel("Portfolio Value")
plt.title("Portfolio Value Over Time")
plt.grid(True)

# Positions
plt.subplot(3, 1, 2)
for product in PRODUCTS:
    plt.plot(timestamps[:len(position_history[product])], position_history[product], label=product)
plt.xlabel("Timestamp")
plt.ylabel("Position")
plt.title("Positions Over Time")
plt.legend()
plt.grid(True)

# Implied Volatilities and Conversions
plt.subplot(3, 1, 3)
for voucher in iv_history:
    if any(iv != 0 for iv in iv_history[voucher]):
        plt.plot(timestamps[:len(iv_history[voucher])], iv_history[voucher], label=f"{voucher} IV")
plt.plot(timestamps[:len(conversion_history)], conversion_history, label="Cumulative Conversions", linestyle='--')
plt.xlabel("Timestamp")
plt.ylabel("Value")
plt.title("Implied Volatility and Cumulative Conversions")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()