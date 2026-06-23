import numpy as np
import yfinance as yf

class StockDataset():
    def __init__(self, symbol):
        self.symbol = symbol if symbol.endswith('.VN') else f"{symbol}.VN"
        self.ticker = yf.Ticker(self.symbol)

    def load_dataset(self, start, end, features, target_feature='Close'):
        data = self.ticker.history(start=start, end=end, interval='1d')
        if data.empty: return None, None, None

        data = data.dropna()

        if 'Date' in features or 'date' in features:
            data = data.reset_index()

        x_values = data[features].values
        y_values = data[target_feature].values

        return x_values, y_values

    def split_dataset(self, data, train_ratio, val_ratio=None):
        if val_ratio is None:
            return data[:int(len(data) * train_ratio)], data[int(len(data) * train_ratio):]

        train_size = int(len(data) * train_ratio)
        val_size = int(len(data) * val_ratio)
        return (data[:train_size],
                data[train_size:train_size + val_size],
                data[train_size + val_size:])

    def create_dataset(self, data_x, data_y, lookback):
        x, y = [], []

        for i in range(len(data_y) - lookback):
            x.append(data_x[i : i + lookback])
            y.append(data_y[i + lookback])

        return np.array(x), np.array(y)
