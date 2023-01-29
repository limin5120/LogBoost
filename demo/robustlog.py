#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import sys

sys.path.append('../')

from logboost.models.lstm import robustlog
from logboost.utils.train import Trainer
from logboost.utils.predict import Predicter
from logboost.utils.utils import *

options = {
    # Smaple
    'is_fix': False,
    'sample': "session_window",
    'window_size': 10,
    # Features
    'sequentials': False,
    'quantitatives': False,
    'semantics': True,
    # Model
    'input_size': 300,
    'hidden_size': 128,
    'num_layers': 2,
    'num_classes': 2,
    # Train
    'batch_size': 256,
    'accumulation_step': 1,
    'optimizer': 'adam',  # sgd, adam
    'lr': 0.001,
    'max_epoch': 400,
    'lr_step': (300, 350),
    'lr_decay_ratio': 0.1,
    'model_name': "robustlog",
    'save_dir': "../result/robustlog/",
    'save_interval': 50,
    # Predict
    'model_path': "../result/robustlog/robustlog_last.pth",
    'resume_path': None,
    'num_candidates': -1,
}

seed_everything(seed=1234)


def train():
    Model = robustlog(input_size=options['input_size'],
                      hidden_size=options['hidden_size'],
                      num_layers=options['num_layers'],
                      num_keys=options['num_classes'])
    trainer = Trainer(Model, options)
    trainer.start_train()


def predict():
    Model = robustlog(input_size=options['input_size'],
                      hidden_size=options['hidden_size'],
                      num_layers=options['num_layers'],
                      num_keys=options['num_classes'])
    predicter = Predicter(Model, options)
    predicter.predict_supervised()


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
        options['session_size'] = 50 if args.boost == 'origin' else 30
    else:
        datapath = '../data/spark/'
        options['session_size'] = 100
    # target
    if args.target in ['deep', 'spark']:
        # boost
        if args.boost == 'origin':
            options['data_dir'] = datapath + 'robust/'
            options['data_prefix'] = ''
        else:
            options['data_dir'] = datapath + 'boost_robust/'
            options['data_prefix'] = 'boost_'
    elif args.target == 'swiss':
        # boost
        if args.boost == 'origin':
            options['data_dir'] = datapath + 'robust_swiss/'
            options['data_prefix'] = ''
        else:
            options['data_dir'] = datapath + 'boost_robust_swiss/'
            options['data_prefix'] = 'boost_'
    # mode
    if args.mode == 'train':
        train()
    else:
        predict()

# -- HDFS-A origin
# > python robustlog.py train hdfs deep origin cpu
# > python robustlog.py predict hdfs deep origin cpu

# -- HDFS-A boost
# > python robustlog.py train hdfs deep boost cpu
# > python robustlog.py predict hdfs deep boost cpu

# -- HDFS-B origin
# > python robustlog.py train hdfs swiss origin cpu
# > python robustlog.py predict hdfs swiss origin cpu

# -- HDFS-B boost
# > python robustlog.py train hdfs swiss boost cpu
# > python robustlog.py predict hdfs swiss boost cpu

# -- SPARK origin
# > python robustlog.py train spark spark origin cpu
# > python robustlog.py predict spark spark origin cpu

# -- SPARK boost
# > python robustlog.py train spark spark boost cpu
# > python robustlog.py predict spark spark boost cpu