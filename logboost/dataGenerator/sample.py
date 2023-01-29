import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter


def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict


def read_file(filename, datatype, type='seq'):
    log_sequences = []
    if type == 'seq':
        with open(filename, 'r') as f:
            for line in tqdm(f.readlines(), desc=datatype + " loading..."):
                line = np.array(list(map(int, line.split())))
                log_sequences.append(line)
    else:
        with open(filename, 'r') as f:
            for line in tqdm(f.readlines(), desc=datatype + " loading..."):
                line = np.array(list(map(float, line.split())))
                log_sequences.append(line)
    return log_sequences


def down_sample(logs, labels, sample_ratio):
    print('down sampling val...')
    max_len = len(labels)
    max_num = int(max_len * sample_ratio)
    labels = np.array(labels)
    shuffle_index = np.random.permutation(np.arange(max_len))
    sample_labels = list(labels[shuffle_index][:max_num])
    sample_logs = {}
    for key in logs.keys():
        tmp_logs = np.array(logs[key])
        sample_logs[key] = list(tmp_logs[shuffle_index][:max_num])

    return sample_logs, sample_labels


def trp(l, n):
    r = l[:n]
    if len(r) < n:
        r.extend(list([0]) * (n - len(r)))
    return r


def sliding_window(data_dir,
                   data_prefix,
                   datatype,
                   featuretype,
                   window_size,
                   num_classes,
                   is_fix,
                   sample_ratio=1):
    '''
    dataset structure
        result_logs(dict):
            result_logs['feature0'] = list()
            result_logs['feature1'] = list()
            ...
        labels(list)
    '''
    num_sessions = 0
    result_logs = {}
    if featuretype[0]:
        result_logs['Sequentials'] = []
    if featuretype[1]:
        result_logs['Quantitatives'] = []
    if featuretype[2]:
        result_logs['Semantics'] = []
        event2semantic_vec = read_json(data_dir + 'event2semantic_vec.json')
    labels = []
    if datatype == 'train':
        data_dir += data_prefix + 'train'
    if datatype == 'val':
        data_dir += data_prefix + 'test_normal'

    with open(data_dir, 'r') as f:
        for line in tqdm(f.readlines(), desc=datatype + " sampling..."):
            num_sessions += 1
            if is_fix:
                line = tuple(map(int, line.strip().split()))
            else:
                line = tuple(
                    map(lambda n: n - 1, map(int,
                                             line.strip().split())))
            for i in range(len(line) - window_size):
                Sequential_pattern = list(line[i:i + window_size])

                if featuretype[1]:
                    Quantitative_pattern = [0] * num_classes
                    log_counter = Counter(Sequential_pattern)
                    for key in log_counter:
                        Quantitative_pattern[key] = log_counter[key]
                    Quantitative_pattern = np.array(
                        Quantitative_pattern)[:, np.newaxis]
                    result_logs['Quantitatives'].append(Quantitative_pattern)

                if featuretype[2]:
                    Semantic_pattern = []
                    for event in Sequential_pattern:
                        if event == 0:
                            Semantic_pattern.append([-1] * 300)
                        else:
                            Semantic_pattern.append(
                                event2semantic_vec[str(event - 1)])
                    result_logs['Semantics'].append(Semantic_pattern)

                if featuretype[0]:
                    Sequential_pattern = np.array(
                        Sequential_pattern)[:, np.newaxis]
                    result_logs['Sequentials'].append(Sequential_pattern)

                labels.append(line[i + window_size])

    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    print('File {}, number of sessions {}'.format(data_dir, num_sessions))
    print('File {}, number of seqs {}'.format(data_dir,
                                              len(result_logs['Sequentials'])))

    return result_logs, labels


def session_window(data_dir, data_prefix, session_size, datatype):
    result_logs = {}
    result_logs['Semantics'] = []
    event2semantic_vec = read_json(data_dir + 'event2semantic_vec.json')
    labels = []

    if datatype == 'train':
        data_dir += data_prefix + 'robust_log_train.csv'
    elif datatype == 'val':
        data_dir += data_prefix + 'robust_log_valid.csv'
    elif datatype == 'test':
        data_dir += data_prefix + 'robust_log_test.csv'

    train_df = pd.read_csv(data_dir)
    for _, item in tqdm(train_df.iterrows(), total=len(train_df)):
        ori_seq = list(map(int, item['Sequence'].split()))
        Sequential_pattern = trp(ori_seq, session_size)

        Semantic_pattern = []
        for event in Sequential_pattern:
            if event == 0:
                Semantic_pattern.append([-1] * 300)
            else:
                Semantic_pattern.append(event2semantic_vec[str(event - 1)])

        result_logs['Semantics'].append(Semantic_pattern)
        labels.append(int(item['label']))

    print('Number of sessions({}): {}'.format(data_dir,
                                              len(result_logs['Semantics'])))
    return result_logs, labels