#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time

sys.path.append('../../')
import numpy as np
from tqdm import tqdm
from logboost.dataGenerator.sample import read_file


class boost():
    def __init__(self, options):
        self.datapath = options['datapath']
        self.savepath = options['savepath']
        self.data_prefix = options['data_prefix']
        self.alpha = options['alpha']
        self.topk = options['topk']
        self.templates_num = options['templates_num']
        self.mode = options['mode']
        self.size = options['size']
        self.train = []
        self.filter_template_list = []

    def __group_by_len(self, data):
        group = dict()
        for i in data:
            if len(i) not in group:
                group[len(i)] = []
            group[len(i)].append(i)
        return sorted(group.items(), key=lambda x: x[0])

    def __caculate_dist(self, l1, l2):
        max_len = len(l1)
        if max_len != len(l2):
            return max_len
        diff = 0
        for i in range(max_len):
            if l1[i] != l2[i]:
                diff += 1
        return diff / max_len

    def __group_list_by_dist(self, list):
        if not list:
            return 0, dict()
        group = dict()
        group[1] = [list[0]]
        tag = 0
        for i in list[1:]:
            for k in group:
                if self.__caculate_dist(group[k][0], i) < self.alpha:
                    group[k].append(i)
                    tag = 1
            if tag == 0:
                group[len(group) + 1] = [i]
            tag == 0
        return len(group), group

    def __group_by_dist(self, group):
        new_group = dict()
        for _, v in group:
            _, group_list = self.__group_list_by_dist(v)
            for g in group_list:
                new_group[len(new_group) + 1] = group_list[g]
        return sorted(new_group.items(), key=lambda x: x[0])

    def __cal_STPM(self, list):
        matrix = np.zeros([self.templates_num, self.templates_num])
        for i in list:
            for j in range(len(i[1:])):
                matrix[i[j], i[j + 1]] += 1
        for line in range(matrix.shape[0]):
            if matrix[line].sum() != 0:
                matrix[line] = matrix[line] / matrix[line].sum()
        return matrix

    def __cal_STPM_of_group(self, group):
        new_group = dict()
        for k, v in group:
            new_group[k] = self.__cal_STPM(v)
        return sorted(new_group.items(), key=lambda x: x[0])

    def __cal_seq2matrix_dist(self, seq, matrix):
        dist = 0
        for i in range(len(seq[1:])):
            dist += matrix[seq[i], seq[i + 1]]
        return dist

    def __fast_caculate_SAG(self, dist_group, STPM_of_group):
        dist = 0
        matrix = np.zeros([self.templates_num, self.templates_num])
        for _, p in STPM_of_group:
            matrix += p
        for _, seq in dist_group:
            dist += self.__cal_seq2matrix_dist(seq[0], matrix) / len(seq[0])
        return dist

    def __cal_SAG(self, data):
        group = self.__group_by_len(data)
        dist_group = self.__group_by_dist(group)
        STPM_of_group = self.__cal_STPM_of_group(dist_group)
        sum_dist = self.__fast_caculate_SAG(dist_group, STPM_of_group)
        return sum_dist

    def __filter_template(self, data, template):
        if type(template) != list:
            template = [template]
        new_data = []
        for i in data:
            # fast
            dt = (np.zeros(len(i)) == 1)
            for t in template:
                dt += (i == t)
            i = np.delete(i, dt)
            new_data.append(i)
        return np.array(new_data, dtype=object)

    def __top_template_effect(self):
        starttime = time.time()
        base_SAG = self.__cal_SAG(self.train)
        print('Base_SAG', base_SAG)
        template_effect = []
        for template in tqdm(range(1, self.templates_num + 1),
                             desc='Analysing top template effect'):
            new_train_data = self.__filter_template(self.train, template)
            SAG_i = self.__cal_SAG(new_train_data)
            if (SAG_i - base_SAG) != 0:
                template_effect.append([template, SAG_i - base_SAG])
        print('Top template analysis time:', time.time() - starttime)
        return sorted(template_effect, key=lambda x: x[1])

    def __save_boost_dataset(self, filename, data):
        with open(filename, 'w') as f:
            for line in data:
                f.write(' '.join(map(str, line)) + '\n')

    def __random_train_data(self):
        path = self.datapath + self.data_prefix
        normal = np.array(read_file(path + 'test_normal', 'normal'),
                          dtype=object)
        abnormal = np.array(read_file(path + 'test_abnormal', 'abnormal'),
                            dtype=object)
        X = np.hstack([normal, abnormal])
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        return X[shuffle_indices][:self.size]

    def boost_analysis(self):
        if self.mode == 'deep':
            self.train = read_file(self.datapath + self.data_prefix + 'train',
                                   'train')
        elif self.mode == 'swiss':
            self.train = self.__random_train_data()
        else:
            print('Error mode, options in deep or swiss.')
        top_effect = self.__top_template_effect()
        print('Template scores:')
        for t in top_effect:
            print(t)
        self.filter_template_list = [i[0] for i in top_effect[:self.topk]]
        print('Recommed filter template:', self.filter_template_list)

    def boost_dataset(self):
        normal = read_file(self.datapath + self.data_prefix + 'test_normal',
                           'normal')
        abnormal = read_file(
            self.datapath + self.data_prefix + 'test_abnormal', 'abnormal')
        train_ = self.__filter_template(self.train, self.filter_template_list)
        normal_ = self.__filter_template(normal, self.filter_template_list)
        abnormal_ = self.__filter_template(abnormal, self.filter_template_list)
        os.makedirs(self.savepath, exist_ok=True)
        self.__save_boost_dataset(self.savepath + 'boost_hdfs_train', train_)
        self.__save_boost_dataset(self.savepath + 'boost_hdfs_test_normal',
                                  normal_)
        self.__save_boost_dataset(self.savepath + 'boost_hdfs_test_abnormal',
                                  abnormal_)
        print('Boost datasets saved:', self.savepath + 'boost_*')
        return train_, normal_, abnormal_
