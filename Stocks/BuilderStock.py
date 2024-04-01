import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
#from pandas.core.common import SettingWithCopyWarning
from pandas.errors import SettingWithCopyWarning
from sklearn.metrics import accuracy_score

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)


def build_stock(stockDf, N=5, contamin=0.02, tage_pred:int=90)->pd.DataFrame:
    """
    This method calculates the features returns, realized variance, realized bipower variation,
    difference, positive and negative returns and signed jumps of a given DataFrame. Then for each features the
    isolation forest anomaly recognition will be applied.
    :param stockDf: source data to build the features from
    :param N: size of the rolling window calculation
    :param contamin: contamination parameter for the isolation forest method
    :param tage_pred: amount of days to look for after an recognized anomaly
    :return: DataFrame of the input data, features and anomalies of these features, analysed by the isolation forest
    """
    stockDf['Return'] = stockDf['Close'].pct_change()
    stockDf['Return log'] = np.log(stockDf['Close']) - np.log(stockDf['Close'].shift(1))
    stockDf = stockDf.fillna(0)

    # Realized variance
    stockDf['RV'] = stockDf['Return log'] ** 2
    stockDf['RV'] = stockDf['RV'].rolling(window=N).sum()
    stockDf = stockDf.fillna(0)

    # Bipower variance
    stockDf['BPV'] = (np.log(stockDf['Close']).shift(-1) - np.log(stockDf['Close'])).abs() * (
            np.log(stockDf['Close']) - np.log(stockDf['Close'].shift(1))).abs()
    stockDf['BPV'] = stockDf['BPV'].rolling(window=N).sum() * (np.pi / 2)
    stockDf = stockDf.fillna(0)

    # Difference RV - BPV
    stockDf['Diff'] = stockDf['RV'] - stockDf['BPV']

    # RV+ and RV-
    RV_pos = stockDf[['Return log', 'RV']]
    RV_pos.loc[RV_pos['Return log'] < 0.0, 'RV'] = 0.0
    RV_pos = RV_pos['RV']
    RV_neg = stockDf[['Return log', 'RV']]
    RV_neg.loc[RV_neg['Return log'] > 0.0, 'RV'] = 0.0
    RV_neg = RV_neg['RV']

    # Signed Jumps SJ
    stockDf['SJ'] = RV_pos - RV_neg

    # Realized semi-variation RSV
    stockDf['RSV'] = stockDf['SJ']

    # Prediction
    stockDf['Prediction'] = np.where(stockDf["Close"].shift(-tage_pred) > stockDf["Close"], 1, 0)

    # IF and features
    stockDf['Anomaly Close'] = isolationForest(stockDf[['Close']], contamin=contamin)
    stockDf['Anomaly pct Return'] = isolationForest(stockDf[['Return']], contamin=contamin)
    stockDf['Anomaly Returns IF'] = isolationForest(stockDf[['Return log']], contamin=contamin)
    stockDf['Anomaly RSV IF'] = isolationForest(stockDf[['RSV']], contamin=contamin)
    stockDf['Anomaly Diff IF'] = isolationForest(stockDf[['Diff']], contamin=contamin)
    stockDf['Amomaly RSV Diff'] = isolationForest(stockDf[['RSV', 'Diff']], contamin=contamin, max_features=2)
    stockDf['Amomaly Returns RSV Diff'] = isolationForest(stockDf[['Return log', 'RSV', 'Diff']],contamin=contamin, max_features=3)

    return stockDf


def isolationForest(data: [str], contamin: float, max_features: int = 1) -> [int]:
    """
    Creates an isolation forest based on the transferred data
    :param data: dataset
    :param contamin: the jump-rate of the dataset
    :param max_features: parameter for sklearn.ensemble.IsolationForest
    :return: dataset of anomaly valus where 0 = inlier and 1 = outlier
    """

    model = IsolationForest(n_estimators=100,
                            max_samples=0.25,
                            contamination=contamin,
                            max_features=max_features)

    data_predicted = model.fit_predict(data)
    ret = [1 if (i == -1) else 0 for i in data_predicted]

    return ret


def acc_score(data=None, label=None) -> (float, float):
    """
    This method calculates the accuracy of the found anomalies with
    :param data: data to draw the recognized anomalies from
    :param label: name of the compared stock
    :return: accuracy score for each label
    """

    pred = pd.DataFrame()
    pred['Prediction'] = \
        data[(data['Anomaly Returns IF'] == 1) & ((data['Anomaly RSV IF'] == 1) | (data['Anomaly Diff IF'] == 1))][
            'Prediction']
    list = pred.value_counts().to_list()
    len_pred = sum(list)
    hit = list[0]
    fail = 0
    if len(list) == 2: fail = list[1]

    compare = data.sample(n=len_pred)
    compare['Compare'] = 1
    acc_score_comp = accuracy_score(compare['Compare'], compare['Prediction'], normalize=True) * 100

    pred = data[(data['Anomaly Returns IF'] == 1) & ((data['Anomaly RSV IF'] == 1) | (data['Anomaly Diff IF'] == 1))]
    acc_score = accuracy_score(pred['Anomaly Returns IF'], pred['Prediction'], normalize=True) * 100

    if label is not None:
        print('Treffer: ', hit, 'Fehler:', fail)
        print(label, 'Anomalie:', round(acc_score, 2), '%')
        print(label, 'Zufällig:', round(acc_score_comp, 2), '%')
        print('------------------------')
    return round(acc_score, 2), round(acc_score_comp, 2)

