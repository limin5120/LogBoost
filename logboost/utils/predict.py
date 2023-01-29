#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gc
import sys
import time
from collections import Counter

sys.path.append('../../')

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from logboost.dataGenerator.tensor import log_to_tensor
from logboost.dataGenerator.sample import session_window


def generate(path, name, is_fix):
    window_size = 10
    hdfs = {}
    length = 0
    with open(path + name, 'r') as f:
        for ln in f.readlines():
            if is_fix:
                ln = list(map(int, ln.strip().split()))
            else:
                ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            hdfs[tuple(ln)] = hdfs.get(tuple(ln), 0) + 1
            length += 1
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs, length


class Predicter():
    def __init__(self, model, options):
        self.data_dir = options['data_dir']
        self.data_prefix = options['data_prefix']
        self.save_dir = options['save_dir']
        self.device = options['device']
        self.model = model
        self.model_path = options['model_path']
        self.window_size = options['window_size']
        self.session_size = options['session_size']
        self.num_classes = options['num_classes']
        self.input_size = options['input_size']
        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.batch_size = options['batch_size']
        self.is_fix = options['is_fix']

    def predict_unsupervised(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        test_normal_loader, test_normal_length = generate(
            self.data_dir, self.data_prefix + 'test_normal', self.is_fix)
        test_abnormal_loader, test_abnormal_length = generate(
            self.data_dir, self.data_prefix + 'test_abnormal', self.is_fix)
        print('test_abnormal_length:', test_abnormal_length)
        test_res_normal = []
        # Test the model
        start_time = time.time()
        with torch.no_grad():
            for line in tqdm(test_normal_loader.keys()):
                line_res = []
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    seq1 = [0] * self.num_classes
                    log_conuter = Counter(seq0)
                    for key in log_conuter:
                        seq1[key] = log_conuter[key]

                    seq0 = torch.tensor(seq0, dtype=torch.float).view(
                        -1, self.window_size, self.input_size).to(self.device)
                    seq1 = torch.tensor(seq1, dtype=torch.float).view(
                        -1, self.num_classes, self.input_size).to(self.device)
                    label = torch.tensor(label).view(-1).to(self.device)
                    output = model(features=[seq0, seq1], device=self.device)
                    line_res.append(torch.argsort(output, 1)[0].tolist())
                test_res_normal.append(
                    [line, line_res, test_normal_loader[line]])
        test_res_abnormal = []
        with torch.no_grad():
            for line in tqdm(test_abnormal_loader.keys()):
                line_res = []
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    seq1 = [0] * self.num_classes
                    log_conuter = Counter(seq0)
                    for key in log_conuter:
                        seq1[key] = log_conuter[key]

                    seq0 = torch.tensor(seq0, dtype=torch.float).view(
                        -1, self.window_size, self.input_size).to(self.device)
                    seq1 = torch.tensor(seq1, dtype=torch.float).view(
                        -1, self.num_classes, self.input_size).to(self.device)
                    label = torch.tensor(label).view(-1).to(self.device)
                    output = model(features=[seq0, seq1], device=self.device)
                    line_res.append(torch.argsort(output, 1)[0].tolist())
                test_res_abnormal.append(
                    [line, line_res, test_abnormal_loader[line]])

        # Compute precision, recall and F1-measure
        elapsed_time = time.time() - start_time
        np.savez_compressed(self.save_dir + 'res_normal_abnormal.npz',
                            normal=np.array(test_res_normal, dtype=object),
                            abnormal=np.array(test_res_abnormal, dtype=object))
        print('Finished Predicting')
        print('elapsed_time: {}'.format(elapsed_time))

    def predict_supervised(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        start_time = time.time()
        test_logs, test_labels = session_window(self.data_dir,
                                                self.data_prefix,
                                                self.session_size,
                                                datatype='test')
        test_dataset = log_to_tensor(logs=test_logs,
                                     labels=test_labels,
                                     seq=self.sequentials,
                                     quan=self.quantitatives,
                                     sem=self.semantics)
        del test_logs, test_labels
        gc.collect()
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      pin_memory=True)
        del test_dataset
        gc.collect()
        TP, FP, FN, TN = 0, 0, 0, 0
        for log, label in tqdm(self.test_loader):
            features = [
                value.clone().to(self.device) for value in log.values()
            ]
            output = self.model(features=features, device=self.device)
            output = torch.sigmoid(output)[:, 0].cpu().detach().numpy()
            predicted = (output < 0.2).astype(int)
            label = np.array([y.cpu() for y in label])
            TP += ((predicted == 1) * (label == 1)).sum()
            FP += ((predicted == 1) * (label == 0)).sum()
            FN += ((predicted == 0) * (label == 1)).sum()
            TN += ((predicted == 0) * (label == 0)).sum()
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        elapsed_time = time.time() - start_time
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP, FN, P, R, F1))
        print('elapsed_time: {}'.format(elapsed_time))

    def evaluation(self, start=1, end=15, test_abnormal_length=16838):
        data = np.load(self.save_dir + 'res_normal_abnormal.npz',
                       allow_pickle=True)
        normal = data['normal']
        abnormal = data['abnormal']
        for n in range(start, end + 1):
            print('num_candidates:', n)
            TP = 0
            FP = 0
            for res in normal:
                line = res[0]
                c = 0
                for i in range(len(line) - self.window_size):
                    label = line[i + self.window_size]
                    predicted = res[1][c][-n:]
                    c += 1
                    if label not in predicted:
                        FP += res[2]
                        break
            for res in abnormal:
                line = res[0]
                c = 0
                for i in range(len(line) - self.window_size):
                    label = line[i + self.window_size]
                    predicted = res[1][c][-n:]
                    c += 1
                    if label not in predicted:
                        TP += res[2]
                        break
            FN = test_abnormal_length - TP
            P = 100 * TP / (TP + FP)
            R = 100 * TP / (TP + FN)
            F1 = 2 * P * R / (P + R)
            print(
                'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
                .format(FP, FN, P, R, F1))

    # false positive (FP): 706, false negative (FN): 1522, Precision: 95.594%, Recall: 90.961%, F1-measure: 93.220%