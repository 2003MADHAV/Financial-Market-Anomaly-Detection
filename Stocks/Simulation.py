import sys
sys.path.append("C:\\Users\\ASUS\\project\\anomaly-detection")
import pandas as pd
import numpy as np
import time
from Stocks.StocksLoader import *
from Stocks.BuilderStock import acc_score, build_stock
from StocksLoader import *


def stock_simulation(labels:[str]=None, predict:int=90)->pd.DataFrame:
    """
    This function runs a simulation of stocks for different contaminations of the isolation forest.
    :param labels: wkn's of the wanted Stocks
    :param predict: amount of days to look for after an recognized anomaly
    :return: DataFrame with contamination and the comparison between isolation forest predicting buying
             and random buying of stocks
    """
    stocks = Stocks(labels, labels, start="2015-01-01", stop="2021-10-01")
    stocks = stocks.df_stocks.dropna()

    conta_list = [0.005, 0.008, 0.015, 0.09, 0.25]

    a_score, c_score, mean_a, mean_c = [], [], [], []
    start = time.time()
    for conta in conta_list:
        print('Conta', conta)
        for label in labels:
            df = pd.DataFrame(stocks[label]['Close'])
            df = build_stock(df, N=1, contamin=conta, tage_pred=predict)
            anom_score, comp_score = acc_score(df)
            a_score.append(anom_score)
            c_score.append(comp_score)
        mean_a.append(round(np.mean(a_score), 3))
        mean_c.append(round(np.mean(c_score), 3))

    end = time.time()
    sek = end - start
    print('running time: {} min'.format(round(sek / 60, 2)))
    table = pd.DataFrame(data=[conta_list, mean_a, mean_c])
    table = table.transpose()
    table.columns = ['Kontamination', 'Anomalie Ergebnis', 'Zufall Ergebnis']

    return table
