#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import sys

sys.path.append('../')
from logboost.models.lstm import deeplog
from logboost.utils.train import Trainer
from logboost.utils.predict import Predicter
from logboost.utils.utils import *

options = {
    # Smaple
    'is_fix': False,
    'sample': "sliding_window",
    'window_size': 10,
    'session_size': 0,
    # Features
    'sequentials': True,
    'quantitatives': False,
    'semantics': False,
    # Model
    'input_size': 1,
    'hidden_size': 64,
    'num_layers': 2,
    # Train
    'batch_size': 2048,
    'accumulation_step': 1,
    'optimizer': 'adam',  # sgd, adam
    'lr': 0.001,
    'max_epoch': 500,
    'lr_step': (300, 350),
    'lr_decay_ratio': 0.1,
    'model_name': "deeplog",
    'save_dir': "../result/deeplog/",
    'save_interval': 100,
    # Predict
    'model_path': "../result/deeplog/deeplog_last.pth",
    'resume_path': None,
}

seed_everything(seed=1234)


def train():
    Model = deeplog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    trainer = Trainer(Model, options)
    trainer.start_train()


def predict():
    Model = deeplog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    predicter = Predicter(Model, options)
    predicter.predict_unsupervised()


def evaluation():
    Model = deeplog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    predicter = Predicter(Model, options)
    predicter.evaluation(start=options['eval'][0],
                         end=options['eval'][1],
                         test_abnormal_length=options['test_abnormal_length'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict', 'evaluation'])
    parser.add_argument('data', choices=['hdfs', 'spark'])
    parser.add_argument('target', choices=['deep', 'swiss', 'spark'])
    parser.add_argument('boost', choices=['origin', 'boost'])
    parser.add_argument('device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    options['device'] = args.device
    # data
    if args.data == 'hdfs':
        datapath = '../data/hdfs/'
        options['test_abnormal_length'] = 16838
        options['eval'] = [1, 30]
    else:
        datapath = '../data/spark/'
        options['test_abnormal_length'] = 1267
        options['num_classes'] = 158
        options['eval'] = [100, 130]
    # target
    if args.target == 'deep':
        options['num_classes'] = 28
        # boost
        if args.boost == 'origin':
            options['data_dir'] = datapath + 'deep/'
            options['data_prefix'] = 'hdfs_'
        else:
            options['data_dir'] = datapath + 'boost_deep/'
            options['data_prefix'] = 'boost_hdfs_'
    elif args.target == 'swiss':
        options['num_classes'] = 49
        options['is_fix'] = True
        # boost
        if args.boost == 'origin':
            options['data_dir'] = datapath + 'swiss/'
            options['data_prefix'] = 'swiss_hdfs_'
        else:
            options['data_dir'] = datapath + 'boost_swiss_/'
            options['data_prefix'] = 'boost_hdfs_'
    else:
        # boost
        if args.boost == 'origin':
            options['data_dir'] = datapath + 'swiss/'
            options['data_prefix'] = 'swiss_spark_'
        else:
            options['data_dir'] = datapath + 'boost_swiss/'
            options['data_prefix'] = 'boost_spark_'
    # mode
    if args.mode == 'train':
        train()
    elif args.mode == 'predict':
        predict()
    else:
        evaluation()

# -- HDFS-A origin
# > python deeplog.py train hdfs deep origin cpu
# > python deeplog.py predict hdfs deep origin cpu
# > python deeplog.py evaluation hdfs deep origin cpu

# -- HDFS-A boost
# > python deeplog.py train hdfs deep boost cpu
# > python deeplog.py predict hdfs deep boost cpu
# > python deeplog.py evaluation hdfs deep boost cpu

# -- HDFS-B origin
# > python deeplog.py train hdfs swiss origin cpu
# > python deeplog.py predict hdfs swiss origin cpu
# > python deeplog.py evaluation hdfs swiss origin cpu

# -- HDFS-B boost
# > python deeplog.py train hdfs swiss boost cpu
# > python deeplog.py predict hdfs swiss boost cpu
# > python deeplog.py evaluation hdfs swiss boost cpu

# -- SPARK origin
# > python deeplog.py train spark spark origin cpu
# > python deeplog.py predict spark spark origin cpu
# > python deeplog.py evaluation spark spark origin cpu

# -- SPARK boost
# > python deeplog.py train spark spark boost cpu
# > python deeplog.py predict spark spark boost cpu
# > python deeplog.py evaluation spark spark boost cpu