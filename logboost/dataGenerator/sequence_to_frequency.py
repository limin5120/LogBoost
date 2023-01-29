import numpy as np
from tqdm import tqdm
from collections import Counter


def read_file(filename):
    log_sequences = []
    with open(filename, 'r') as f:
        for line in tqdm(f.readlines()):
            line = np.array(list(map(int, line.split())))
            log_sequences.append(line)
    return np.array(log_sequences, dtype=object)


def sequence2frequency(data, max_len):
    new_data = []
    for i in data:
        frq = [0] * max_len
        log_conuter = Counter(i)
        for key in log_conuter:
            frq[key] = log_conuter[key] / len(i)
        new_data.append(np.array(frq))
    return np.array(new_data, dtype=object)


def construct_data(path, prefix, max_len):
    train = read_file(path + prefix + 'train')
    normal = read_file(path + prefix + 'test_normal')
    abnormal = read_file(path + prefix + 'test_abnormal')
    normal = np.hstack([train, normal])
    normal_ = sequence2frequency(normal, max_len)
    abnormal_ = sequence2frequency(abnormal, max_len)
    save_data(path, 'frq_' + prefix + 'test_normal', normal_)
    save_data(path, 'frq_' + prefix + 'test_abnormal', abnormal_)


def save_data(path, filename, data):
    with open(path + filename, 'w') as f:
        for line in data:
            f.write(' '.join(map(str, line)) + '\n')


def deep_seq2frq():
    max_len = 29
    path = '../../data/hdfs/deep/'
    prefix = 'hdfs_'
    construct_data(path, prefix, max_len)
    path = '../../data/hdfs/boost_deep_/'
    prefix = 'boost_hdfs_'
    construct_data(path, prefix, max_len)


def swiss_seq2frq():
    max_len = 49
    path = '../../data/hdfs/swiss/'
    prefix = 'swiss_hdfs_'
    construct_data(path, prefix, max_len)
    path = '../../data/hdfs/boost_swiss/'
    prefix = 'boost_hdfs_'
    construct_data(path, prefix, max_len)


def spark_seq2frq():
    max_len = 159
    path = '../../data/spark/swiss/'
    prefix = 'swiss_spark_'
    construct_data(path, prefix, max_len)
    path = '../../data/spark/boost_swiss/'
    prefix = 'boost_spark_'
    construct_data(path, prefix, max_len)


if __name__ == "__main__":
    deep_seq2frq()
    swiss_seq2frq()
    spark_seq2frq()