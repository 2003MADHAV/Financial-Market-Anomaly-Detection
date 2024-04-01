import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_style('darkgrid')


def plot_signed_anomaly(df:pd.DataFrame=None, label:str=None):
    """
    This method plots the signed anomalies (if at least two methods work) from a given DataFrame.
    :param df: data to plot
    :param label: stock name [String]
    :return: The stock chart wiht signed jumps as a plot and png
    """
    data = pd.DataFrame()
    data['Anomalie'] = \
        df[(df['Anomaly Returns IF'] == 1) & ((df['Anomaly RSV IF'] == 1) | (df['Anomaly Diff IF'] == 1))]['Close']

    plt.figure(figsize=(10, 8))
    sns.lineplot(data=df['Close'], legend='auto', label=label)
    sns.scatterplot(data=data['Anomalie'], label='Signed anomaly', color='red', alpha=.6, s=110)
    plt.savefig('../../Pictures/Stockdata/AdjustetAnomalies_Stockdata.png')


def plotter(df: pd.DataFrame = None, label: str = None, features=False):
    """
    Graphic example output of a merton-jump-diffusion process
    with signed anomalies and detected anomalies, as well as the feature output and cutoff method.
    :param features: enable plotting features
    :param label: title and name of the plotted data
    :param df: DataFrame with features and signed jumps
    """

    plt.figure(figsize=(10, 8))
    plt.title(label)
    sns.lineplot(data=df['Close'], legend='auto', label=label)

    # IF Return anomalies points
    ret = df.loc[(df['Anomaly Returns IF'] == 1)]
    ret_return = ret['Return log']
    ret = ret['Close']
    sns.scatterplot(data=ret, label='IF Return', color='red', alpha=.6, s=110)

    # IF Diff anomalies points
    diff = df.loc[(df['Anomaly Diff IF'] == 1)]
    diff_diff = diff['Diff']
    diff = diff['Close']
    sns.scatterplot(data=diff, label='IF Diff', color='green', alpha=.6, marker="v", s=70)

    # RSV IF anomalie points
    rsv = df.loc[(df['Anomaly RSV IF'] == 1)]
    rsv_rsv = rsv['SJ']
    rsv = rsv['Close']
    sns.scatterplot(data=rsv, label='IF RSV', color='orange', alpha=1, marker="v", s=70)
    plt.savefig('../../Pictures/Stockdata/ChartAnomalies_Stockdata.png')

    if features:
        # Return log
        plt.figure(figsize=(10, 6))

        sns.lineplot(data=df['Return log'], legend='auto', label='Returns (log)')
        sns.scatterplot(data=ret_return, legend='auto', label='IF Return', color='red', s=110)
        plt.savefig('../../Pictures/Stockdata/Return_Stockdata.png')
        rsv_diff = df.loc[(df['Amomaly RSV Diff'] == 1)]
        rsv_diff = rsv_diff['SJ']

        # plot features
        fig, axes = plt.subplots(4, 1, figsize=(9, 12))

        fig.suptitle('Merkmale')
        fig.subplots_adjust(hspace=0.6, wspace=0.6)

        sns.lineplot(ax=axes[0], data=df['Return log'], legend='auto', label='Returns (log)')
        sns.scatterplot(ax=axes[0], data=ret_return, color='red', legend='auto', label='Anomaly')

        sns.lineplot(ax=axes[1], data=df['Diff'], legend='auto', label='Difference')
        sns.scatterplot(ax=axes[1], data=diff_diff, color='red', legend='auto', label='Anomaly')

        sns.lineplot(ax=axes[2], data=df['SJ'], legend='auto', label='RSV')
        sns.scatterplot(ax=axes[2], data=rsv_rsv, color='red', legend='auto', label='Anomaly')

        sns.lineplot(ax=axes[3], data=df['SJ'], legend='auto', label='RSV and Diff')
        sns.scatterplot(ax=axes[3], data=rsv_diff, color='red', legend='auto', label='Anomaly')
        plt.savefig('../../Pictures/Stockdata/Features_Stockdata.png')

    plt.show()
