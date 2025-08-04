import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from datamodel import Order  # Import Order if needed for type hints

class Backtester:
    def __init__(self):
        self.trades = pd.DataFrame()
        self.prices = pd.DataFrame()
        self.portfolio_values = pd.DataFrame()
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.initial_capital = 1000000  # $1M starting capital
        
    def load_data(self, trades_path: str, prices_path: str):
        """Load both trades and prices data"""
        # Load trades data
        self.trades = pd.read_csv(trades_path, sep=';')
        self.trades['timestamp'] = pd.to_datetime(self.trades['timestamp'])
        
        # Standardize trade sides (assuming 'SUBMISSION' is our bot)
        self.trades['side'] = np.where(self.trades['buyer'] == 'SUBMISSION', 'BUY', 
                                      np.where(self.trades['seller'] == 'SUBMISSION', 'SELL', None))
        self.trades = self.trades.dropna(subset=['side'])  # Remove trades not involving our bot
        
        # Load prices data
        self.prices = pd.read_csv(prices_path, sep=';')
        self.prices['timestamp'] = pd.to_datetime(self.prices['timestamp'])
        
        # Calculate mid price if not present
        if 'mid_price' not in self.prices.columns:
            self.prices['mid_price'] = (self.prices['bid_price_1'] + self.prices['ask_price_1']) / 2
        
        print(f"Loaded {len(self.trades)} trades and {len(self.prices)} price records")
        self.preprocess_data()
        
    def preprocess_data(self):
        """Prepare data for analysis and calculate portfolio values"""
        if self.trades.empty or self.prices.empty:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Calculate trade values and costs (0.05% transaction cost)
        self.trades['trade_value'] = self.trades['price'] * abs(self.trades['quantity'])
        self.trades['trade_cost'] = self.trades['trade_value'] * 0.0005
        
        # Initialize portfolio tracking
        portfolio_value = self.initial_capital
        daily_values = []
        
        # Group trades by day
        trades_by_day = self.trades.groupby(pd.Grouper(key='timestamp', freq='D'))
        
        # Get unique trading days from prices
        trading_days = sorted(self.prices['timestamp'].dt.normalize().unique())
        
        for day in trading_days:
            # Get trades for this day
            day_trades = self.trades[self.trades['timestamp'].dt.normalize() == day]
            
            # Calculate daily P&L from trades
            day_pnl = 0
            for _, trade in day_trades.iterrows():
                if trade['side'] == 'BUY':
                    day_pnl -= (trade['trade_value'] + trade['trade_cost'])
                else:
                    day_pnl += (trade['trade_value'] - trade['trade_cost'])
            
            # Update portfolio value
            portfolio_value += day_pnl
            daily_values.append({
                'date': day,
                'portfolio_value': portfolio_value,
                'daily_pnl': day_pnl
            })
        
        self.portfolio_values = pd.DataFrame(daily_values)
        
        # Calculate daily returns
        self.portfolio_values['daily_return'] = self.portfolio_values['portfolio_value'].pct_change()
        
    def calculate_sharpe_ratio(self, annualize=True):
        """Calculate Sharpe ratio with optional annualization"""
        if len(self.portfolio_values) == 0:
            raise ValueError("No portfolio values available. Run preprocess_data() first.")
            
        excess_returns = self.portfolio_values['daily_return'] - (self.risk_free_rate / 252)
        sharpe = excess_returns.mean() / excess_returns.std()
        
        if annualize:
            sharpe *= np.sqrt(252)
            
        return sharpe
    
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown and duration"""
        if len(self.portfolio_values) == 0:
            raise ValueError("No portfolio values available. Run preprocess_data() first.")
            
        cumulative_max = self.portfolio_values['portfolio_value'].cummax()
        drawdown = (self.portfolio_values['portfolio_value'] - cumulative_max) / cumulative_max
        max_dd = drawdown.min()
        end_date = self.portfolio_values.loc[drawdown.idxmin(), 'date']
        start_date = self.portfolio_values.loc[
            self.portfolio_values['portfolio_value'][:drawdown.idxmin()].idxmax(), 'date']
            
        return max_dd, start_date, end_date
    
    def plot_performance(self, save_path=None):
        """Plot portfolio performance metrics"""
        if len(self.portfolio_values) == 0:
            raise ValueError("No portfolio values available. Run preprocess_data() first.")
            
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Portfolio value
        ax1.plot(self.portfolio_values['date'], self.portfolio_values['portfolio_value'])
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Value ($)')
        ax1.grid(True)
        
        # Daily P&L
        ax2.bar(self.portfolio_values['date'], self.portfolio_values['daily_pnl'])
        ax2.set_title('Daily P&L')
        ax2.set_ylabel('P&L ($)')
        ax2.grid(True)
        
        # Returns distribution
        ax3.hist(self.portfolio_values['daily_return'].dropna(), bins=50)
        ax3.set_title('Daily Returns Distribution')
        ax3.set_xlabel('Daily Return')
        ax3.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def analyze_product(self, product_symbol):
        """Analyze performance for a specific product"""
        product_trades = self.trades[self.trades['symbol'] == product_symbol]
        if len(product_trades) == 0:
            print(f"No trades found for {product_symbol}")
            return
            
        # Calculate approximate P&L per trade
        product_trades = product_trades.copy()
        product_trades['exit_price'] = np.nan
        product_trades['pnl'] = 0
        
        for idx, trade in product_trades.iterrows():
            # Find next available price after trade
            future_prices = self.prices[
                (self.prices['product'] == product_symbol) & 
                (self.prices['timestamp'] > trade['timestamp'])]
                
            if len(future_prices) > 0:
                exit_price = future_prices.iloc[0]['mid_price']
                product_trades.at[idx, 'exit_price'] = exit_price
                
                if trade['side'] == 'BUY':
                    pnl = (exit_price - trade['price']) * trade['quantity']
                else:
                    pnl = (trade['price'] - exit_price) * abs(trade['quantity'])
                
                product_trades.at[idx, 'pnl'] = pnl
        
        # Calculate metrics
        total_pnl = product_trades['pnl'].sum()
        win_rate = len(product_trades[product_trades['pnl'] > 0]) / len(product_trades)
        avg_win = product_trades[product_trades['pnl'] > 0]['pnl'].mean()
        avg_loss = product_trades[product_trades['pnl'] <= 0]['pnl'].mean()
        
        print(f"\n{product_symbol} Performance Analysis:")
        print(f"Total Trades: {len(product_trades)}")
        print(f"Total P&L: ${total_pnl:,.2f}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Avg Win: ${avg_win:,.2f}")
        print(f"Avg Loss: ${avg_loss:,.2f}")
        if avg_loss != 0:
            print(f"Profit Factor: {abs(avg_win * win_rate / (avg_loss * (1-win_rate))):.2f}")
        
        # Plot cumulative P&L
        product_trades['cumulative_pnl'] = product_trades['pnl'].cumsum()
        plt.figure(figsize=(12, 6))
        plt.plot(product_trades['timestamp'], product_trades['cumulative_pnl'])
        plt.title(f'{product_symbol} Cumulative P&L')
        plt.ylabel('P&L ($)')
        plt.xlabel('Date')
        plt.grid(True)
        plt.show()
    
    def generate_report(self, save_path=None):
        """Generate comprehensive performance report"""
        if len(self.portfolio_values) == 0:
            raise ValueError("No portfolio values available. Run preprocess_data() first.")
            
        sharpe = self.calculate_sharpe_ratio()
        max_dd, dd_start, dd_end = self.calculate_max_drawdown()
        total_return = (self.portfolio_values['portfolio_value'].iloc[-1] / 
                       self.portfolio_values['portfolio_value'].iloc[0] - 1)
        annualized_return = (1 + total_return) ** (252/len(self.portfolio_values)) - 1
        volatility = self.portfolio_values['daily_return'].std() * np.sqrt(252)
        
        report = f"""
        ===== ALGORITHM PERFORMANCE REPORT =====
        
        Time Period: {self.portfolio_values['date'].iloc[0].date()} to {self.portfolio_values['date'].iloc[-1].date()}
        Trading Days: {len(self.portfolio_values)}
        
        --- Returns ---
        Total Return: {total_return:.2%}
        Annualized Return: {annualized_return:.2%}
        
        --- Risk ---
        Annualized Volatility: {volatility:.2%}
        Max Drawdown: {max_dd:.2%}
        Drawdown Period: {dd_start.date()} to {dd_end.date()}
        
        --- Risk-Adjusted Performance ---
        Sharpe Ratio: {sharpe:.2f}
        
        --- Portfolio ---
        Final Value: ${self.portfolio_values['portfolio_value'].iloc[-1]:,.2f}
        """
        
        print(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
                
    def plot_rolling_metrics(self, window=30, save_path=None):
        """Plot rolling Sharpe ratio and volatility"""
        if len(self.portfolio_values) == 0:
            raise ValueError("No portfolio values available. Run preprocess_data() first.")
            
        returns = self.portfolio_values['daily_return']
        
        # Calculate rolling metrics
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_sharpe = (rolling_mean - (self.risk_free_rate/252)) / rolling_std * np.sqrt(252)
        rolling_vol = rolling_std * np.sqrt(252)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Rolling Sharpe
        ax1.plot(self.portfolio_values['date'], rolling_sharpe)
        ax1.set_title(f'{window}-Day Rolling Sharpe Ratio')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.grid(True)
        
        # Rolling Volatility
        ax2.plot(self.portfolio_values['date'], rolling_vol)
        ax2.set_title(f'{window}-Day Rolling Volatility')
        ax2.set_ylabel('Annualized Volatility')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    # Initialize backtester
    backtester = Backtester()
    
    # Load your CSV files (using raw strings for Windows paths)
    backtester.load_data(
        r'round-3-island-data-bottle\trades.csv', 
        r'round-3-island-data-bottle\prices.csv'
    )
    
    # Generate performance metrics
    print(f"Sharpe Ratio: {backtester.calculate_sharpe_ratio():.2f}")
    
    # Show performance plots
    backtester.plot_performance()
    backtester.plot_rolling_metrics()
    
    # Analyze specific products
    backtester.analyze_product('VOLCANIC_ROCK_VOUCHER_10000')
    backtester.analyze_product('VOLCANIC_ROCK')
    
    # Generate full report
    backtester.generate_report('performance_report.txt')