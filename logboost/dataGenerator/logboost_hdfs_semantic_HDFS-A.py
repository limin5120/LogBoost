import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm


def __load_data(data):
    new_data = []
    for _, item in tqdm(data.iterrows(), total=len(data)):
        ori_seq = np.array(list(map(int, item['Sequence'].split())))
        new_data.append(ori_seq)
    return new_data


def __group_by_len(data):
    group = dict()
    for i in data:
        if len(i) not in group:
            group[len(i)] = []
        group[len(i)].append(i)
    return sorted(group.items(), key=lambda x: x[0])


def __caculate_dist(l1, l2):
    max_len = len(l1)
    if max_len != len(l2):
        return max_len
    diff = 0
    for i in range(max_len):
        if l1[i] != l2[i]:
            diff += 1
    return diff / max_len


def __group_list_by_dist(list):
    if not list:
        return 0, dict()
    group = dict()
    group[1] = [list[0]]
    tag = 0
    for i in list[1:]:
        for k in group:
            if __caculate_dist(group[k][0], i) < 0.2:
                group[k].append(i)
                tag = 1
        if tag == 0:
            group[len(group) + 1] = [i]
        tag == 0
    return len(group), group


def __group_by_dist(group):
    new_group = dict()
    for _, v in group:
        _, group_list = __group_list_by_dist(v)
        for g in group_list:
            new_group[len(new_group) + 1] = group_list[g]
    return sorted(new_group.items(), key=lambda x: x[0])


def __cal_STPM(list):
    matrix = np.zeros([templates_num, templates_num])
    for i in list:
        for j in range(len(i[1:])):
            matrix[i[j], i[j + 1]] += 1
    for line in range(matrix.shape[0]):
        if matrix[line].sum() != 0:
            matrix[line] = matrix[line] / matrix[line].sum()
    return matrix


def __cal_STPM_of_group(group):
    new_group = dict()
    for k, v in group:
        new_group[k] = __cal_STPM(v)
    return sorted(new_group.items(), key=lambda x: x[0])


def __cal_seq2matrix_dist(seq, matrix):
    dist = 0
    for i in range(len(seq[1:])):
        dist += matrix[seq[i], seq[i + 1]]
    return dist


def __fast_caculate_SAG(dist_group, STPM_of_group):
    dist = 0
    matrix = np.zeros([templates_num, templates_num])
    for _, p in STPM_of_group:
        matrix += p
    for _, seq in dist_group:
        dist += __cal_seq2matrix_dist(seq[0], matrix) / len(seq[0])
    return dist


def __cal_SAG(data):
    group = __group_by_len(data)
    dist_group = __group_by_dist(group)
    STPM_of_group = __cal_STPM_of_group(dist_group)
    sum_dist = __fast_caculate_SAG(dist_group, STPM_of_group)
    return sum_dist


def __filter_template(data, template):
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


def __top_template_effect():
    starttime = time.time()
    base_SAG = __cal_SAG(train)
    print('Base_SAG', base_SAG)
    template_effect = []
    for template in tqdm(range(1, templates_num + 1),
                         desc='Analysing top template effect'):
        new_train_data = __filter_template(train, template)
        SAG_i = __cal_SAG(new_train_data)
        if (SAG_i - base_SAG) != 0:
            template_effect.append([template, SAG_i - base_SAG])
    print('Top template analysis time:', time.time() - starttime)
    return sorted(template_effect, key=lambda x: x[1])


def __save_boost_dataset(filename, data):
    with open(filename, 'w') as f:
        for line in data:
            f.write(' '.join(map(str, line)) + '\n')


def boost_analysis():
    top_effect = __top_template_effect()
    print('Template scores:')
    for t in top_effect:
        print(t)
    filter_template_list = [i[0] for i in top_effect[:topk]]
    print('Recommed filter template:', filter_template_list)
    return filter_template_list


def boost_dataset(data, filename):
    seq = __load_data(data)
    seq = __filter_template(seq, filter_template_list)
    new_data = []
    for i in range(len(data)):
        new_data.append([" ".join([str(t) for t in seq[i]]), data['label'][i]])
    new_data_df = pd.DataFrame(new_data, columns=['Sequence', 'label'])
    new_data_df.to_csv(save + filename, index=False)


if __name__ == "__main__":
    templates_num = 29
    topk = 2
    path = '../../data/hdfs/robust/'
    save = '../../data/hdfs/boost_robust/'
    os.makedirs(save, exist_ok=True)
    filename1 = 'robust_log_train.csv'
    filename2 = 'robust_log_test.csv'
    filename3 = 'robust_log_valid.csv'
    src_train = pd.read_csv(path + filename1)
    src_test = pd.read_csv(path + filename2)
    src_valid = pd.read_csv(path + filename3)
    train = __load_data(src_train)
    filter_template_list = boost_analysis()
    boost_dataset(src_train, 'boost_robust_log_train.csv')
    boost_dataset(src_test, 'boost_robust_log_test.csv')
    boost_dataset(src_valid, 'boost_robust_log_valid.csv')
