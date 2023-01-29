import matplotlib.pyplot as plt
from sklearn import manifold
import numpy as np
import sys

sys.path.append('../../')
from logboost.dataGenerator.sample import read_file
from logboost.utils.utils import *

HDFS_PATH = '../../data/hdfs/'
SPARK_PATH = '../../data/spark/'


def unique_data(data):
    tmp = dict()
    for i in data:
        tmp[str(i)] = i
    new_data = []
    for _, v in tmp.items():
        new_data.append(v)
    return np.array(new_data, dtype=object)


def TSNE(data, max_len):
    data = padding_data(np.array(data, dtype=object), max_len)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x = tsne.fit_transform(data)
    x_min, x_max = x.min(0), x.max(0)
    return (x - x_min) / (x_max - x_min)


def read_npz(name):
    data = np.load(name + '.npz')
    return data['normal'], data['abnormal']


def visualize(normal, abnormal, name):
    normal = unique_data(read_file(normal, ''))
    abnormal = unique_data(read_file(abnormal, ''))
    max_len = get_max_seq_len(normal, abnormal)
    normal_ = TSNE(normal, max_len)
    abnormal_ = TSNE(abnormal, max_len)
    np.savez(name + '.npz', normal=normal_, abnormal=abnormal_)


def boost_hdfs_deep():
    normal = HDFS_PATH + '/deep/hdfs_test_normal'
    abnormal = HDFS_PATH + '/deep/hdfs_test_abnormal'
    visualize(normal, abnormal, 'f1')
    normal = HDFS_PATH + '/boost_deep/boost_hdfs_test_normal'
    abnormal = HDFS_PATH + '/boost_deep/boost_hdfs_test_abnormal'
    visualize(normal, abnormal, 'of1')


def boost_hdfs_swiss():
    normal = HDFS_PATH + '/swiss/swiss_hdfs_test_normal'
    abnormal = HDFS_PATH + '/swiss/swiss_hdfs_test_abnormal'
    visualize(normal, abnormal, 'f2')
    normal = HDFS_PATH + '/boost_swiss/boost_hdfs_test_normal'
    abnormal = HDFS_PATH + '/boost_swiss/boost_hdfs_test_abnormal'
    visualize(normal, abnormal, 'of2')


def boost_spark_swiss():
    normal = SPARK_PATH + '/swiss/swiss_spark_test_normal'
    abnormal = SPARK_PATH + '/swiss/swiss_spark_test_abnormal'
    visualize(normal, abnormal, 'f3')
    normal = SPARK_PATH + '/boost_swiss/boost_spark_test_normal'
    abnormal = SPARK_PATH + '/boost_swiss/boost_spark_test_abnormal'
    visualize(normal, abnormal, 'of3')


if __name__ == "__main__":
    title_list = [
        'HDFS-A', 'Optimized HDFS-A', 'HDFS-B',
        'Optimized HDFS-B', 'SPARK',
        'Otimized SPARK'
    ]
    data_list = ['f1', 'of1', 'f2', 'of2', 'f3', 'of3']
    boost_hdfs_deep()
    boost_hdfs_swiss()
    boost_spark_swiss()
    plt.rc('font', family='Times New Roman')
    plt.rcParams['xtick.labelsize'] = 32
    plt.rcParams['ytick.labelsize'] = 32
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.titlepad"] = 16.18
    # print(plt.rcParams.keys())

    fig = plt.figure(1, figsize=(42, 6))
    axes = fig.subplots(nrows=1, ncols=6)
    for idx, ax in enumerate(fig.axes):
        ax.set_title(title_list[idx], fontsize=36)
        normal, abnormal = read_npz(data_list[idx])
        ax.scatter(normal[:, 0], normal[:, 1])
        ax.scatter(abnormal[:, 0], abnormal[:, 1])
    plt.subplots_adjust(left=0.02, right=0.98, top=0.8, bottom=0.1, wspace=0.35)
    plt.savefig('./figures/features.svg')
