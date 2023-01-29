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


def select_train_data(data):
    parts = [
        0, 730, 1408, 2137, 2828, 3672, 4294, 5046, 6027, 6478, 7183, 7910,
        8625, 9465, 9999, 10856, 11582, 12288, 12985, 13708, 14440, 15174,
        15850, 16544, 17289, 18032, 18692, 19419, 20117
    ]
    new_data = []
    for i in parts:
        new_data = np.hstack([new_data, data[i:i + 7]])
    return new_data


if __name__ == "__main__":
    path = '../../data/spark/boost/'
    filename = 'opt_spark_seq_normal_abnormal_assemble.npz'
    src_normal, src_abnormal = load_src_data(path, filename)
    save_src_data(path, 'boost_spark_test_normal', src_normal)
    save_src_data(path, 'boost_spark_test_abnormal', src_abnormal)
    random_train = select_train_data(src_normal)
    save_src_data(path, 'boost_spark_train', random_train)
