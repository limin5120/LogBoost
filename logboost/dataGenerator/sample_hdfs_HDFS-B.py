import numpy as np


def load_src_data(path, filename):
    src_data = np.load(path + filename, allow_pickle=True)
    src_normal = src_data['normal']
    src_abnormal = src_data['abnormal']
    return src_normal, src_abnormal


def save_src_data(path, filename, data):
    with open(path + filename, 'w') as f:
        for line in data:
            f.write(' '.join(map(str, line)) + '\n')


def random_train_data(data, size, seed):
    np.random.seed(seed)
    shuffle_index = np.random.permutation(np.arange(len(data)))
    return data[shuffle_index][:size]


if __name__ == "__main__":
    path = '../../data/hdfs/swiss/'
    filename = 'hdfs_seq_normal_abnormal.npz'
    src_normal, src_abnormal = load_src_data(path, filename)
    save_src_data(path, 'swiss_hdfs_test_normal', src_normal)
    save_src_data(path, 'swiss_hdfs_test_abnormal', src_abnormal)
    save_src_data(path, 'swiss_hdfs_train', src_normal[:4855])
