#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

sys.path.append('../')
from logboost.utils.utils import *
from logboost.boost.boost import boost

# -- boost HDFS-A for deeplog , loganomaly , robustlog
# options = {
#     'datapath': '../data/hdfs/deep/',
#     'data_prefix': 'hdfs_',
#     'savepath': '../data/hdfs/boost_deep/',
#     'alpha': 0.2,
#     'topk': 2,
#     'templates_num': 29,  # deeplog 29 swiss 49 159
#     'mode': 'deep',  # deep, swiss
#     'size': 500  # if swiss
# }

# -- boost HDFS-A for randomforest, xgb
# options = {
#     'datapath': '../data/hdfs/deep/',
#     'data_prefix': 'hdfs_',
#     'savepath': '../data/hdfs/boost_deep_/',
#     'alpha': 0.2,
#     'topk': 4,
#     'templates_num': 29,  # deeplog 29 swiss 49 159
#     'mode': 'deep',  # deep, swiss
#     'size': 500  # if swiss
# }

# -- boost HDFS-B for deeplog , loganomaly , robustlog
# options = {
#     'datapath': '../data/hdfs/swiss/',
#     'data_prefix': 'swiss_hdfs_',
#     'savepath': '../data/hdfs/boost_swiss_/',
#     'alpha': 0.2,
#     'topk': 1,
#     'templates_num': 49,  # deeplog 29 swiss 49 159
#     'mode': 'deep',  # deep, swiss
#     'size': 500  # if swiss
# }

# -- boost HDFS-B for randomforest, xgb
# options = {
#     'datapath': '../data/hdfs/swiss/',
#     'data_prefix': 'swiss_hdfs_',
#     'savepath': '../data/hdfs/boost_swiss/',
#     'alpha': 0.2,
#     'topk': 4,
#     'templates_num': 49,  # deeplog 29 swiss 49 159
#     'mode': 'swiss',  # deep, swiss
#     'size': 500  # if swiss
# }

options = {
    'datapath': '../data/hdfs/swiss/',
    'data_prefix': 'swiss_hdfs_',
    'savepath': '../data/hdfs/boost_swiss/',
    'alpha': 0.2,
    'topk': 4,
    'templates_num': 49,  # deeplog 29 swiss 49 159
    'mode': 'swiss',  # deep, swiss
    'size': 500  # if swiss
}

seed_everything(1)


def boosting():
    logboost = boost(options)
    logboost.boost_analysis()
    logboost.boost_dataset()


if __name__ == "__main__":
    boosting()