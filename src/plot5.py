import pandas as pd
import numpy as np
from statistics import mean, stdev
import matplotlib.pyplot as plt
import logging
from Tsk5_final_final_v2 import Trader
from datamodel import OrderDepth, Order, Trade, TradingState, Observation, ConversionObservation, Listing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TraderTester:
    def __init__(self, prices_file: str, trades_file: str, observations_file: str):
        self.prices_file = prices_file
        self.trades_file = trades_file
        self.observations_file = observations_file
        self.trader = Trader()
        self.products = []
        self.positions = {}
        self.cash = 0
        self.portfolio_values = []
        self.pnl_history = []
        self.position_history = {}
        self.conversion_history = []
        self.cumulative_conversions = 0
        self.mid_prices = {}
        self.timestamps = []
        self.prices_df = None
        self.observations_df = None
        self.trader_data = ""  # Initialize trader data

    def load_data(self):
        logging.info("Loading data...")
        try:
            # Load prices data
            self.prices_df = pd.read_csv(self.prices_file, delimiter=';')
            self.prices_df = self.prices_df.sort_values(by='timestamp')
            self.timestamps = sorted(self.prices_df['timestamp'].unique())
            self.products = self.prices_df['product'].unique().tolist()

            # Initialize positions and position_history
            self.positions = {p: 0 for p in self.products}
            self.position_history = {p: [] for p in self.products}

            # Calculate mid prices
            for product in self.products:
                df = self.prices_df[self.prices_df['product'] == product].copy()
                df['mid_price'] = df[['bid_price_1', 'ask_price_1']].mean(axis=1)
                self.mid_prices[product] = dict(zip(df['timestamp'], df['mid_price']))

            # Load trades data
            try:
                trades_df = pd.read_csv(self.trades_file, delimiter=';')
            except FileNotFoundError:
                logging.warning("Trades file not found. Initializing empty trades DataFrame.")
                trades_df = pd.DataFrame(columns=['timestamp', 'symbol', 'price', 'quantity', 'buyer', 'seller'])

            # Group trades by timestamp and symbol
            self.market_trades = {}
            for (ts, symbol), group in trades_df.groupby(['timestamp', 'symbol']):
                trades = [
                    Trade(
                        symbol=row['symbol'],
                        price=row['price'],
                        quantity=row['quantity'],
                        buyer=row.get('buyer', ''),
                        seller=row.get('seller', ''),
                        timestamp=row['timestamp']
                    )
                    for _, row in group.iterrows()
                ]
                self.market_trades[(ts, symbol)] = trades

            # Load observations data
            try:
                self.observations_df = pd.read_csv(self.observations_file)
                self.observations_df = self.observations_df.sort_values(by='timestamp')
                # Validate required columns
                required_cols = ['timestamp', 'bidPrice', 'askPrice', 'transportFees', 'exportTariff', 'importTariff', 'sugarPrice', 'sunlightIndex']
                if not all(col in self.observations_df.columns for col in required_cols):
                    missing = [col for col in required_cols if col not in self.observations_df.columns]
                    logging.error(f"Missing columns in observations.csv: {missing}")
                    raise ValueError(f"Missing columns in observations.csv: {missing}")
                # Ensure numeric types
                for col in required_cols[1:]:  # Skip timestamp
                    self.observations_df[col] = pd.to_numeric(self.observations_df[col], errors='coerce')
                if self.observations_df.isnull().any().any():
                    logging.warning("NaN values detected in observations.csv. Filling with default values.")
                    self.observations_df.fillna({
                        'bidPrice': 100,
                        'askPrice': 102,
                        'transportFees': 0,
                        'exportTariff': 0,
                        'importTariff': 0,
                        'sugarPrice': 100,
                        'sunlightIndex': 1000
                    }, inplace=True)
                logging.info("Observations data loaded successfully.")
            except FileNotFoundError:
                logging.warning("Observations file not found. Using default ConversionObservation values.")
                self.observations_df = pd.DataFrame(columns=['timestamp', 'bidPrice', 'askPrice', 'transportFees', 'exportTariff', 'importTariff', 'sugarPrice', 'sunlightIndex'])

            # Update trader's position limits
            self.trader.position_limits = {p: 20 for p in self.products}
            logging.info(f"Loaded {len(self.products)} products: {self.products}")

        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def get_conversion_observation(self, timestamp: int, product: str) -> ConversionObservation:
        """Retrieve or create a ConversionObservation for a given timestamp and product."""
        if self.observations_df is not None and not self.observations_df.empty:
            obs = self.observations_df[self.observations_df['timestamp'] == timestamp]
            if not obs.empty:
                row = obs.iloc[0]
                return ConversionObservation(
                    bidPrice=float(row['bidPrice']),
                    askPrice=float(row['askPrice']),
                    transportFees=float(row['transportFees']),
                    exportTariff=float(row['exportTariff']),
                    importTariff=float(row['importTariff']),
                    sugarPrice=float(row['sugarPrice']),
                    sunlightIndex=float(row['sunlightIndex'])
                )
        logging.warning(f"No observation data for timestamp {timestamp}, product {product}. Using default values.")
        return ConversionObservation(
            bidPrice=100,
            askPrice=102,
            transportFees=0,
            exportTariff=0,
            importTariff=0,
            sugarPrice=100,
            sunlightIndex=1000
        )

    def simulate_trading(self):
        """Run the trading simulation and compute metrics."""
        logging.info("Starting trading simulation...")
        trade_count = 0

        for ts in self.timestamps:
            logging.info(f"Processing timestamp {ts}")
            # Construct order depths
            order_depths = {}
            for product in self.products:
                snap = self.prices_df[(self.prices_df['timestamp'] == ts) & (self.prices_df['product'] == product)]
                if snap.empty:
                    logging.warning(f"No data for {product} at timestamp {ts}")
                    continue
                row = snap.iloc[0]
                buy_orders = {row[f'bid_price_{i}']: row[f'bid_volume_{i}'] for i in range(1, 4) if pd.notna(row[f'bid_price_{i}']) and row[f'bid_price_{i}'] > 0}
                sell_orders = {row[f'ask_price_{i}']: -row[f'ask_volume_{i}'] for i in range(1, 4) if pd.notna(row[f'ask_price_{i}']) and row[f'ask_price_{i}'] > 0}
                od = OrderDepth()
                od.buy_orders = buy_orders
                od.sell_orders = sell_orders
                order_depths[product] = od

            # Create listings
            listings = {p: Listing(symbol=p, product=p, denomination="SEASHELLS") for p in self.products}

            # Create conversion observations
            conversion_observations = {p: self.get_conversion_observation(ts, p) for p in self.products}

            observations = Observation(
                plainValueObservations={},
                conversionObservations=conversion_observations
            )

            # Construct TradingState
            state = TradingState(
                timestamp=ts,
                listings=listings,
                order_depths=order_depths,
                own_trades={},  # Will be updated after trader run
                market_trades={p: self.market_trades.get((ts, p), []) for p in self.products},
                position=self.positions.copy(),
                observations=observations,
                traderData=self.trader_data
            )

            # Run trader
            try:
                orders_dict, conversions, trader_data = self.trader.run(state)
                self.trader_data = trader_data  # Update trader data for next iteration
            except Exception as e:
                logging.error(f"Error running trader at timestamp {ts}: {e}")
                continue

            # Simulate fills
            own_trades = {}
            for product, orders in orders_dict.items():
                own_trades[product] = []
                od = order_depths.get(product)
                if not od:
                    continue
                for order in orders:
                    if order.quantity > 0:  # Buy order
                        best_ask = min(od.sell_orders.keys(), default=float('inf'))
                        if order.price >= best_ask and best_ask < float('inf'):
                            qty = min(order.quantity, -od.sell_orders.get(best_ask, 0))
                            if qty > 0:
                                self.positions[product] += qty
                                self.cash -= qty * best_ask
                                own_trades[product].append(
                                    Trade(
                                        symbol=product,
                                        price=best_ask,
                                        quantity=qty,
                                        buyer="SUBMISSION",
                                        seller="",  # Counterparty unknown in simulation
                                        timestamp=ts
                                    )
                                )
                                trade_count += 1
                    elif order.quantity < 0:  # Sell order
                        best_bid = max(od.buy_orders.keys(), default=0)
                        if order.price <= best_bid and best_bid > 0:
                            qty = min(-order.quantity, od.buy_orders.get(best_bid, 0))
                            if qty > 0:
                                self.positions[product] -= qty
                                self.cash += qty * best_bid
                                own_trades[product].append(
                                    Trade(
                                        symbol=product,
                                        price=best_bid,
                                        quantity=qty,
                                        buyer="",  # Counterparty unknown in simulation
                                        seller="SUBMISSION",
                                        timestamp=ts
                                    )
                                )
                                trade_count += 1

            # Handle conversions
            if conversions != 0:
                for product in self.products:
                    conv_obs = conversion_observations.get(product)
                    if conv_obs:
                        if conversions > 0 and self.positions[product] >= conversions:
                            self.cash += conversions * (conv_obs.bidPrice - conv_obs.transportFees - conv_obs.exportTariff)
                            self.positions[product] -= conversions
                        elif conversions < 0 and self.cash >= abs(conversions) * (
                                conv_obs.askPrice + conv_obs.transportFees + conv_obs.importTariff):
                            self.cash -= abs(conversions) * (
                                    conv_obs.askPrice + conv_obs.transportFees + conv_obs.importTariff)
                            self.positions[product] -= conversions
                self.cumulative_conversions += conversions
            self.conversion_history.append(self.cumulative_conversions)

            # Calculate portfolio value
            value = self.cash
            for product in self.products:
                mid = self.mid_prices[product].get(ts, None)
                if mid is not None:
                    value += self.positions[product] * mid
                self.position_history[product].append(self.positions[product])
            self.portfolio_values.append(value)
            if len(self.portfolio_values) > 1:
                self.pnl_history.append(self.portfolio_values[-1] - self.portfolio_values[-2])

            # Update state
            state.own_trades = own_trades

        # Calculate metrics
        self.total_pnl = self.portfolio_values[-1] - self.portfolio_values[0] if self.portfolio_values else 0
        returns = [pnl / abs(self.portfolio_values[i - 1]) for i, pnl in enumerate(self.pnl_history) if
                   i > 0 and self.portfolio_values[i - 1] != 0]
        self.sharpe_ratio = (mean(returns) / stdev(returns)) * np.sqrt(252) if returns and stdev(returns) > 0 else 0
        self.max_drawdown = 0
        peak = self.portfolio_values[0] if self.portfolio_values else 0
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak != 0 else 0
            self.max_drawdown = max(self.max_drawdown, drawdown)
        self.trade_frequency = trade_count / len(self.timestamps) if self.timestamps else 0
        logging.info(f"Simulation completed. Total trades: {trade_count}")

    def print_results(self):
        """Print performance metrics."""
        print(f"Total PnL: {self.total_pnl:.2f}")
        print(f"Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {self.max_drawdown:.2%}")
        print(f"Average Trade Frequency (trades per timestamp): {self.trade_frequency:.4f}")

    def plot_results(self):
        """Generate plots for portfolio value, positions, conversions, and observations."""
        logging.info("Generating plots...")
        plt.style.use('ggplot')  # Use a clean, built-in style
        fig = plt.figure(figsize=(14, 12))

        # Validate data lengths
        n_timestamps = len(self.timestamps)
        n_portfolio = len(self.portfolio_values)
        n_conversions = len(self.conversion_history)
        logging.info(f"Data lengths - Timestamps: {n_timestamps}, Portfolio: {n_portfolio}, Conversions: {n_conversions}")

        if n_portfolio == 0:
            logging.warning("No portfolio data to plot. Generating empty plot.")
            plt.text(0.5, 0.5, "No portfolio data available", ha='center', va='center')
            plt.savefig('trader_performance.png', dpi=300)
            plt.close()
            return

        # Portfolio Value
        ax1 = fig.add_subplot(3, 1, 1)
        plot_timestamps = self.timestamps[:min(n_timestamps, n_portfolio)]
        ax1.plot(plot_timestamps, self.portfolio_values[:len(plot_timestamps)], color='blue', label='Portfolio Value')
        ax1.set_ylabel("Portfolio Value (SEASHELLS)", fontsize=12)
        ax1.set_title("Portfolio Value Over Time", fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

        # Positions and Conversions
        ax2 = fig.add_subplot(3, 1, 2)
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.products)))
        for i, product in enumerate(self.products):
            pos_data = self.position_history[product]
            if pos_data:
                ax2.plot(self.timestamps[:min(n_timestamps, len(pos_data))], pos_data[:min(n_timestamps, len(pos_data))],
                         label=f"{product} Position", color=colors[i])
        if self.conversion_history:
            ax2.plot(self.timestamps[:min(n_timestamps, n_conversions)], self.conversion_history[:min(n_timestamps, n_conversions)],
                     label="Cumulative Conversions", linestyle='--', color='black')
        ax2.set_ylabel("Quantity", fontsize=12)
        ax2.set_title("Positions and Cumulative Conversions", fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0)