#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import argparse

sys.path.append('../')
from logboost.utils.utils import *
from logboost.models.ml import rflog

options = {
    'train_size': 500,
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'random_state': None,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', choices=['hdfs', 'spark'])
    parser.add_argument('type', choices=['seq', 'frq'])
    parser.add_argument('target', choices=['deep', 'swiss', 'spark'])
    parser.add_argument('boost', choices=['origin', 'boost'])
    args = parser.parse_args()
    # data
    if args.data == 'hdfs':
        datapath = '../data/hdfs/'
    else:
        datapath = '../data/spark/'
    options['type'] = 'seq' if args.type == 'seq' else 'frq'
    # target
    if args.target == 'deep':
        seed = 389 if args.type == 'seq' else 68
        # boost
        if args.boost == 'origin':
            options['datapath'] = datapath + 'deep/'
            options[
                'data_prefix'] = 'hdfs_' if args.type == 'seq' else 'frq_hdfs_'
        else:
            options['datapath'] = datapath + 'boost_deep_/'
            options[
                'data_prefix'] = 'boost_hdfs_' if args.type == 'seq' else 'frq_boost_hdfs_'
    elif args.target == 'swiss':
        seed = 457 if args.type == 'seq' else 463
        # boost
        if args.boost == 'origin':
            options['datapath'] = datapath + 'swiss/'
            options[
                'data_prefix'] = 'swiss_hdfs_' if args.type == 'seq' else 'frq_swiss_hdfs_'
        else:
            options['datapath'] = datapath + 'boost_swiss/'
            options[
                'data_prefix'] = 'boost_hdfs_' if args.type == 'seq' else 'frq_boost_hdfs_'
    else:
        seed = 308 if args.type == 'seq' else 633
        # boost
        if args.boost == 'origin':
            options['datapath'] = datapath + 'swiss/'
            options[
                'data_prefix'] = 'swiss_spark_' if args.type == 'seq' else 'frq_swiss_spark_'
        else:
            options['datapath'] = datapath + 'boost_swiss/'
            options[
                'data_prefix'] = 'boost_spark_' if args.type == 'seq' else 'frq_boost_spark_'
    rf = rflog(options)
    rf.load_dataset()
    seed_everything(seed)
    rf.session_random()
    rf.train()
    rf.predict()

# ------ sequence vetcor ------
# -- HDFS-A origin
# > python randomforest.py hdfs seq deep origin

# -- HDFS-A boost
# > python randomforest.py hdfs seq deep boost

# -- HDFS-B origin
# > python randomforest.py hdfs seq swiss origin

# -- HDFS-B boost
# > python randomforest.py hdfs seq swiss boost

# -- SPARK origin
# > python randomforest.py spark seq spark origin

# -- SPARK boost
# > python randomforest.py spark seq spark boost

# ------ frequency vetcor ------
# -- HDFS-A origin
# > python randomforest.py hdfs frq deep origin

# -- HDFS-A boost
# > python randomforest.py hdfs frq deep boost

# -- HDFS-B origin
# > python randomforest.py hdfs frq swiss origin

# -- HDFS-B boost
# > python randomforest.py hdfs frq swiss boost

# -- SPARK origin
# > python randomforest.py spark frq spark origin

# -- SPARK boost
# > python randomforest.py spark frq spark boost