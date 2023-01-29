import time
import numpy as np
from logboost.utils.utils import *
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from logboost.dataGenerator.sample import read_file


class xbglog():
    def __init__(self, options):
        self.datapath = options['datapath']
        self.data_prefix = options['data_prefix']
        self.train_size = options['train_size']
        self.type = options['type']
        self.max_len = 0
        self.xgb = XGBClassifier(
            booster=options['booster'],
            objective=options['objective'],
            num_class=options['num_class'],
            gamma=options['gamma'],
            max_depth=options['max_depth'],
            reg_lambda=options['reg_lambda'],
            subsample=options['subsample'],
            colsample_bytree=options['colsample_bytree'],
            min_child_weight=options['min_child_weight'],
            eta=options['eta'],
            seed=options['seed'],
            nthread=options['nthread'],
            use_label_encoder=options['use_label_encoder'])
        self.normal = self.abnormal = None
        self.X_train = self.Y_train = self.X_test = self.Y_test = None
        self.n1 = self.n2 = 0

    def load_dataset(self):
        path = self.datapath + self.data_prefix
        self.normal = np.array(read_file(path + 'test_normal', 'normal', self.type),
                               dtype=object)
        self.abnormal = np.array(read_file(path + 'test_abnormal', 'abnormal', self.type),
                                 dtype=object)
        self.max_len = get_max_seq_len(self.normal, self.abnormal)

    def session_random(self):
        normal_y = np.array([0 for i in self.normal])
        abnormal_y = np.array([1 for i in self.abnormal])
        if self.type == 'seq':
            X = np.hstack([self.normal, self.abnormal])
        else:
            X = np.vstack([self.normal, self.abnormal])
        Y = np.hstack([normal_y, abnormal_y])
        Y = Y.astype(np.int32)
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        X = X[shuffle_indices]
        Y = Y[shuffle_indices]
        X_train = X[:self.train_size]
        self.Y_train = Y[:self.train_size]
        X_test = X[self.train_size:]
        self.Y_test = Y[self.train_size:]
        self.n1 = self.Y_test.tolist().count(0)
        self.n2 = self.Y_test.tolist().count(1)
        if self.type == 'seq':
            self.X_train = padding_data(X_train, self.max_len)
            self.X_test = padding_data(X_test, self.max_len)
        else:
            self.X_train = X_train
            self.X_test = X_test
        print('train shape:', self.X_train.shape, self.Y_train.shape,
              'test shape:', self.X_test.shape, self.Y_test.shape)
        print(
            'abnormal num:', len(self.Y_train[self.Y_train == 1]), '/',
            self.train_size, '=',
            np.round(
                len(self.Y_train[self.Y_train == 1]) / self.train_size * 100,
                3), '%')

    def train(self):
        begin = time.time()
        self.xgb.fit(self.X_train, self.Y_train, eval_metric='auc')
        print('train time:', time.time() - begin)

    def predict(self):
        begin = time.time()
        pre = self.xgb.predict(self.X_test)
        res = evaluation_model('XGBoost', pre, self.Y_test, self.n1, self.n2)
        print('predict time:', time.time() - begin)
        return res


class rflog():
    def __init__(self, options):
        self.datapath = options['datapath']
        self.data_prefix = options['data_prefix']
        self.train_size = options['train_size']
        self.type = options['type']
        self.max_len = 0
        self.rf = RandomForestClassifier(
            n_estimators=options['n_estimators'],
            max_depth=options['max_depth'],
            min_samples_split=options['min_samples_split'],
            random_state=options['random_state'])
        self.normal = self.abnormal = None
        self.X_train = self.Y_train = self.X_test = self.Y_test = None
        self.n1 = self.n2 = 0

    def load_dataset(self):
        path = self.datapath + self.data_prefix
        self.normal = np.array(read_file(path + 'test_normal', 'normal', self.type),
                               dtype=object)
        self.abnormal = np.array(read_file(path + 'test_abnormal', 'abnormal', self.type),
                                 dtype=object)
        self.max_len = get_max_seq_len(self.normal, self.abnormal)

    def session_random(self):
        normal_y = np.array([0 for i in self.normal])
        abnormal_y = np.array([1 for i in self.abnormal])
        if self.type == 'seq':
            X = np.hstack([self.normal, self.abnormal])
        else:
            X = np.vstack([self.normal, self.abnormal])
        Y = np.hstack([normal_y, abnormal_y])
        Y = Y.astype(np.int32)
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        X = X[shuffle_indices]
        Y = Y[shuffle_indices]
        X_train = X[:self.train_size]
        self.Y_train = Y[:self.train_size]
        X_test = X[self.train_size:]
        self.Y_test = Y[self.train_size:]
        self.n1 = self.Y_test.tolist().count(0)
        self.n2 = self.Y_test.tolist().count(1)
        if self.type == 'seq':
            self.X_train = padding_data(X_train, self.max_len)
            self.X_test = padding_data(X_test, self.max_len)
        else:
            self.X_train = X_train
            self.X_test = X_test
        print('train shape:', self.X_train.shape, self.Y_train.shape,
              'test shape:', self.X_test.shape, self.Y_test.shape)
        print(
            'abnormal num:', len(self.Y_train[self.Y_train == 1]), '/',
            self.train_size, '=',
            np.round(
                len(self.Y_train[self.Y_train == 1]) / self.train_size * 100,
                3), '%')

    def train(self):
        begin = time.time()
        self.rf.fit(self.X_train, self.Y_train)
        print('train time:', time.time() - begin)

    def predict(self):
        begin = time.time()
        pre = self.rf.predict(self.X_test)
        res = evaluation_model('RandomForest', pre, self.Y_test, self.n1,
                               self.n2)
        print('predict time:', time.time() - begin)
        return res
