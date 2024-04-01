import sys
sys.path.append("C:\\Users\\ASUS\\project\\anomaly-detection")
import Testdata.Builder as Builder

import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def sim_table(n:int=10):
    """ Creates a simulation for six given jump rates.
    :param n: number of runs per jump rate, per default 10 [int]
    :return: table with f1 scores per method and feature for the respective jump rate [DataFrame]
    """

    jump_steps = [0.0002, 0.001, 0.002, 0.005,0.01, 0.02]

    header = [np.array(['Features', 'Returns', 'Returns', 'Diff', 'Diff','RSV','RSV','RSV and Diff','Returns,RSV and Diff']),
              np.array(['Jumps/steps','CutOff','IF','CutOff','IF','CutOff','IF','IF','IF'])]
    f1_score = pd.DataFrame(columns=header)
    f1_score['Features'] = jump_steps

    cutOff_returns,if_returns,cutOff_diff,if_difference,if_rsv_list,cut_rsv_list,rsv_diff_list,r_rsv_diff_list = [],[],[],[],[],[],[],[]
    returns_cutoff, returns_if, diff_cutoff, diff_if,rsv_cutoff,rsv_if,rsv_diff,r_rsv_diff = [],[],[],[],[],[],[],[]

    start = time.time()
    for rate in jump_steps:
        for _ in range(0,n,1):
            data, if_ret, if_diff, cut_ret, cut_diff, if_rsv, cut_rsv, rsv_diff_val,r_rsv_diff_val = Builder.simulation(jump_rate=rate)
            cutOff_returns.append(cut_ret)
            if_returns.append(if_ret)
            cutOff_diff.append(cut_diff)
            if_difference.append(if_diff)
            cut_rsv_list.append(cut_rsv)
            if_rsv_list.append(if_rsv)
            rsv_diff_list.append(rsv_diff_val)
            r_rsv_diff_list.append(r_rsv_diff_val)

        returns_cutoff.append(np.round(np.mean(cutOff_returns),2))
        returns_if.append(np.round(np.mean(if_returns),2))
        diff_cutoff.append(np.round(np.mean(cutOff_diff),2))
        diff_if.append(np.round(np.mean(if_difference),2))
        rsv_cutoff.append(np.round(np.mean(cut_rsv_list),2))
        rsv_if.append(np.round(np.mean(if_rsv_list),2))
        rsv_diff.append(np.round(np.mean(rsv_diff_list),2))
        r_rsv_diff.append(np.round(np.mean(r_rsv_diff_list),2))

    end = time.time()
    sek = end - start
    print('running time: {} min, with n={}'.format(round(sek/60,2),n))

    df = pd.DataFrame([jump_steps,returns_cutoff,returns_if,diff_cutoff,diff_if,rsv_cutoff,rsv_if,rsv_diff,r_rsv_diff])
    df = df.transpose()
    df.columns = header

    return df


def table_heatmap(df=None):
    fig = plt.figure(facecolor='w', edgecolor='k')
    sns.heatmap(df, annot=True, cmap='viridis', cbar=False)
    plt.savefig('../../Pictures/Testdata/F1_scores_Project.png')
    plt.show()