import pandas as pd
import numpy as np
from pathlib import Path

class HistoricalData:
    def __init__(self):
        self.price_data = {}  # Dictionary to store price data by product
        self.trade_data = {}  # Dictionary to store trade data by product
    
    def load_csv_files(self, prices_path: str, trades_path: str):
        """Load both price and trade CSV files"""
        self._load_price_data(prices_path)
        self._load_trade_data(trades_path)
    
    def _load_price_data(self, file_path: str):
        """Load and process price CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            # Assuming CSV has columns: ['timestamp', 'product', 'bid', 'ask', ...]
            for product in df['product'].unique():
                product_df = df[df['product'] == product]
                self.price_data[product] = product_df.sort_values('timestamp')
                
        except Exception as e:
            print(f"Error loading price data: {e}")
    
    def _load_trade_data(self, file_path: str):
        """Load and process trade CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            # Assuming CSV has columns: ['timestamp', 'product', 'price', 'quantity', ...]
            for product in df['product'].unique():
                product_df = df[df['product'] == product]
                self.trade_data[product] = product_df.sort_values('timestamp')
                
        except Exception as e:
            print(f"Error loading trade data: {e}")