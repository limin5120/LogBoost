#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def save_parameters(options, filename):
    with open(filename, "w+") as f:
        for key in options.keys():
            f.write("{}: {}\n".format(key, options[key]))


def seed_everything(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def padding_data(data, max_len):
    if len(data[0]) < max_len:
        diff = max_len - len(data[0])
        data[0] = np.append(data[0], [0 for i in range(diff)])
    return np.array(
        pad_sequence([torch.from_numpy(np.array(x)) for x in data],
                     batch_first=True).tolist())


def get_max_seq_len(normal, abnormal):
    max_len = 0
    for seq in normal:
        if len(seq) > max_len:
            max_len = len(seq)
    for seq in abnormal:
        if len(seq) > max_len:
            max_len = len(seq)
    print('The max seq len of this dataset is', max_len)
    return max_len


def evaluation_model(name, y_pre, Y_test, n1, n2):
    diff = Y_test - y_pre
    # FP , TN , FN , TP
    FP = len(diff[diff == -1])
    TN = n1 - FP
    FN = len(diff[diff == 1])
    TP = n2 - FN
    R = TP / (TP + FN)
    if (TP + FP) == 0:
        P = 1
    else:
        P = TP / (TP + FP)
    if (P + R) == 0:
        F = 0
    else:
        F = (2 * P * R) / (P + R)
    A = (TP + TN) / (TP + TN + FP + FN)
    print('model:', name)
    print('TP:', TP)
    print('FN: failure 1 ', FN)
    print('TN:', TN)
    print('FP: failure 2', FP)
    print('F(a=1) =', F)
    print('Recall =', R)
    print('Precision =', P)
    print('Accuracy =', A)
    return TP, FN, TN, FP, F, R, P, A