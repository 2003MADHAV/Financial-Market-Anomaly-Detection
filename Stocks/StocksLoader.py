import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class Stocks:
    def __init__(self, wkns:[]=None, names:[]=None, start=None, stop=None):
        self.wkns = wkns
        self.names = names
        self.start = start
        self.stop = stop if stop is not None else datetime.now().strftime('%Y-%m-%d')
        self.df_stocks = self.read_and_merge()
        self.stocks = self.df_stocks


    def read_and_merge(self) -> pd.DataFrame:
        """
        This function uses yfinance to get the wkn's from Yahoo Finance API
        :return: DataFrame with the wanted wkn's
        """
        data = yf.download(" ".join(self.wkns), start=self.start, end=self.stop, group_by='ticker')
        df_list = []
        for name, wkn in zip(self.names, self.wkns):
            if len(self.wkns) > 1:
                df = data[wkn]
            else:
                df = data
            df.columns = pd.MultiIndex.from_product([[name], df.columns])
            df_list.append(df)

        stocks = pd.concat(df_list, axis=1)
        stocks.columns.names = ['Stock Ticker', 'Stock Info']
        return stocks

# def read_and_merge(self) -> pd.DataFrame:
#         """
#         This function uses yfinance to get the wkn's from Yahoo Finance API
#         :return: DataFrame with the wanted wkn's
#         """
#         # yf.download(stock_wkn, start="1980-01-01", end=datetime.now().strftime('%Y-%m-%d'))
#         df_list = []
#         for wkn in self.wkns:
#             stock = yf.Ticker(wkn)
#             df = stock.history(start=self.start, end=self.stop)
#             df_list.append(df)
#         stocks = pd.concat(df_list, axis=1, keys=self.names)
#         stocks.columns.names = ['Stock Ticker', 'Stock Info']
#         return stocks

    def plot_stocks_plt(self):
        """
        This function plots the stocks that self contains
        """
        for name in self.stocks.names:
            self.stocks.df_stocks[name]['Close'].plot(figsize=(16, 10), label=name)
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_stocks_df(self):
        """
        This function plots the closing prices
        """
        self.stocks.df_stocks.xs(key='Close', axis=1, level='Stock Info').iplot()

    def clustermap(self):
        """
        This function plots a clustermap with the given stocks of self
        """
        sns.clustermap(self.stocks.df_stocks.xs(key='Close', axis=1, level='Stock Info').corr(), annot=True)

    def heatmap(self):
        """
        This function plots a heatmap of self and the given stocks
        """
        sns.heatmap(self.stocks.df_stocks.xs(key='Close', axis=1, level='Stock Info').corr(), annot=True)
