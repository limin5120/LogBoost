import io
import json
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_vectors(fname, words):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    for line in tqdm(fin, total=n):
        tokens = line.rstrip().split(' ')
        if tokens[0] in words:
            words[tokens[0]] = list(map(float, tokens[1:]))
    return words


def count_words(csv):
    words = {}
    sentences = {}
    for idx, item in csv.iterrows():
        sentences[idx] = item['Template']
        for i in item['Template'].split():
            if i not in words:
                words[i] = []
    return words, sentences


def count_word_in_sentences(word, sentences):
    count = 0
    for i, s in sentences.items():
        if word in s.split():
            count += 1
    return count


def cal_TFIDF(word2vec, sentences):
    sentenceTFIDF = {}
    sum_sentences = len(sentences)
    for i, s in sentences.items():
        line = s.split()
        len_line = len(line)
        s_vec = np.zeros(300)
        for k in line:
            TF = line.count(k) / len_line
            IDF = sum_sentences / count_word_in_sentences(k, sentences)
            k_vec = np.array(word2vec[k])
            if len(k_vec) != 0:
                s_vec += k_vec * TF * IDF
        sentenceTFIDF[str(i)] = list(s_vec / len_line)
    return sentenceTFIDF


def generate_semantic_vec():
    templates = pd.read_csv('./hdfs_template_word.csv')
    words, sentences = count_words(templates)
    word2vec = load_vectors('./crawl-300d-2M.vec', words)
    event2semantic = cal_TFIDF(word2vec, sentences)

    with open('./event2semantic_vec_hdfs.json', 'w') as js:
        json.dump(event2semantic, js)


def read_file(filename):
    log_sequences = []
    with open(filename, 'r') as f:
        for line in tqdm(f.readlines()):
            line = np.array(list(map(int, line.split())))
            log_sequences.append(line)
    return np.array(log_sequences, dtype=object)


def save_data(path, normal, abnormal):
    data = []
    for i in normal:
        data.append([" ".join([str(t) for t in i]), 0])
    for i in abnormal:
        data.append([" ".join([str(t) for t in i]), 1])
    columns = ['Sequence', 'label']
    data_df = pd.DataFrame(data, columns=columns)
    data_df.to_csv(path, index=False)


def generate_robust_csv():
    path = '../../data/hdfs/swiss/'
    boost_path = '../../data/hdfs/boost_swiss/'
    train = read_file(path + 'swiss_hdfs_train')
    train_boost = read_file(boost_path + 'boost_hdfs_train')
    normal = read_file(path + 'swiss_hdfs_test_normal')
    normal_boost = read_file(boost_path + 'boost_hdfs_test_normal')
    abnormal = read_file(path + 'swiss_hdfs_test_abnormal')
    abnormal_boost = read_file(boost_path + 'boost_hdfs_test_abnormal')

    normal_ = np.hstack([train, normal])
    normal_boost_ = np.hstack([train_boost, normal_boost])

    train_normal = normal_[:6000]
    valid_normal = normal_[6000:7000]
    test_normal = normal_[7000:]
    train_abnormal = abnormal[:6000]
    valid_abnormal = abnormal[6000:7000]
    test_abnormal = abnormal[7000:]

    boost_train_normal = normal_boost_[:6000]
    boost_valid_normal = normal_boost_[6000:7000]
    boost_test_normal = normal_boost_[7000:]
    boost_train_abnormal = abnormal_boost[:6000]
    boost_valid_abnormal = abnormal_boost[6000:7000]
    boost_test_abnormal = abnormal_boost[7000:]

    savepath = '../../data/hdfs/'
    save_data(savepath + '/robust_swiss/robust_log_train.csv', train_normal,
              train_abnormal)
    print(train_normal.shape, train_abnormal.shape)
    save_data(savepath + '/boost_robust_swiss/boost_robust_log_train.csv',
              boost_train_normal, boost_train_abnormal)
    print(boost_train_normal.shape, boost_train_abnormal.shape)
    save_data(savepath + '/robust_swiss/robust_log_valid.csv', valid_normal,
              valid_abnormal)
    print(valid_normal.shape, valid_abnormal.shape)
    save_data(savepath + '/boost_robust_swiss/boost_robust_log_valid.csv',
              boost_valid_normal, boost_valid_abnormal)
    print(boost_valid_normal.shape, boost_valid_abnormal.shape)
    save_data(savepath + '/robust_swiss/robust_log_test.csv', test_normal,
              test_abnormal)
    print(test_normal.shape, test_abnormal.shape)
    save_data(savepath + '/boost_robust_swiss/boost_robust_log_test.csv',
              boost_test_normal, boost_test_abnormal)
    print(boost_test_normal.shape, boost_test_abnormal.shape)


if __name__ == "__main__":
    generate_semantic_vec()
    generate_robust_csv()