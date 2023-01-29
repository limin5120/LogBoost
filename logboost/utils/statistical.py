import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import pdist
import sys

sys.path.append('../../')
from logboost.dataGenerator.sample import read_file


def unique_data(data, max_len=22):
    tmp = dict()
    for i in data:
        if len(i) <= 22:
            continue
        tmp[str(i)] = i
    new_data = []
    for _, v in tmp.items():
        new_data.append(v)
    return np.array(new_data, dtype=object)


def jaccard(u, v):
    u = set(u)
    v = set(v)
    or_ = len(u | v)
    and_ = len(u & v)
    return and_ / or_

def contrast(nor, abnor):
    res = []
    for i in nor:
        for j in abnor:
            res.append(jaccard(i, j))
    return res


if __name__ == "__main__":
    # normal = '../../data/hdfs/swiss/swiss_hdfs_test_normal'
    # abnormal = '../../data/hdfs/swiss/swiss_hdfs_test_abnormal'

    # normal_ = read_file(normal, 'normal')
    # abnormal_ = read_file(abnormal, 'abnormal')
    # print(len(normal_), len(abnormal_))
    # new_nor = unique_data(normal_)
    # new_abnor = unique_data(abnormal_)
    # print(len(new_nor), len(new_abnor))
    # res = contrast(new_nor, new_abnor)
    # res_df = pd.DataFrame(res, columns=['jaccard'])
    # res_df.to_csv('jaccard.csv', index=False)

    res = pd.read_csv('jaccard.csv')
    a = res['jaccard']
    print(a.min(), a.max(), a.mean())
    fs_xk = np.sort(a)
    val, cnt = np.unique(a, return_counts=True)
    #print(val,cnt)
    pmf = cnt / len(a)

    fs_rv_dist2 = stats.rv_discrete(name='fs_rv_dist2', values=(val, pmf))

    plt.rc('font', family='Times New Roman')
    plt.rcParams['xtick.labelsize'] = 28
    plt.rcParams['ytick.labelsize'] = 28
    plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.titlepad"] = 16.18
    fig = plt.figure(1, figsize=(15, 4))
    axes = fig.subplots(nrows=1, ncols=1)
    for idx, ax in enumerate(fig.axes):
        ax.set_title("CDF of Jaccard Similarity of Normal and Abnormal Sequences on HDFS", fontsize=32)
        ax.plot(val, fs_rv_dist2.cdf(val), lw=5, label='Frequency Counts of Jaccard Similarity')
        ax.set_xlabel('Jaccard Similarity of Normal and Abnormal Sequences', fontsize = 28)
        ax.set_ylabel('Cumulative Frequency', fontsize = 26)
        ax.plot([0.5], [0.188],  marker = "o", c='r', markersize=12)
        ax.text(0.4, 0.25, '[0.5, 0.188]', fontsize = 28)
    plt.legend(fontsize=26)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.22)
    # plt.show()
    plt.savefig('./figures/cdf.svg')