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
    templates = pd.read_csv('./spark_template_word.csv')
    words, sentences = count_words(templates)
    word2vec = load_vectors('./crawl-300d-2M.vec', words)
    event2semantic = cal_TFIDF(word2vec, sentences)

    with open('./event2semantic_vec_spark.json', 'w') as js:
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
    path = '../../data/spark/swiss/'
    boost_path = '../../data/spark/boost_swiss/'
    train = read_file(path + 'swiss_spark_train')
    train_boost = read_file(boost_path + 'boost_spark_train')
    normal = read_file(path + 'swiss_spark_test_normal')
    normal_boost = read_file(boost_path + 'boost_spark_test_normal')
    abnormal = read_file(path + 'swiss_spark_test_abnormal')
    abnormal_boost = read_file(boost_path + 'boost_spark_test_abnormal')
    train_num = 3000
    valid_num = 50
    train_abnormal_num = 200
    valid_abnormal_num = 50
    shuffle_indices = np.random.permutation(np.arange(len(normal)))
    tmp_normal = normal[shuffle_indices]
    tmp_boost_normal = normal_boost[shuffle_indices]
    shuffle_indices = np.random.permutation(np.arange(len(abnormal)))
    tmp_abnormal = abnormal[shuffle_indices]
    tmp_boost_abnormal = abnormal_boost[shuffle_indices]
    train_abnormal = tmp_abnormal[:train_abnormal_num]
    train_boost_abnormal = tmp_boost_abnormal[:train_abnormal_num]
    train_normal = np.hstack([train, tmp_normal[:train_num]])
    train_boost_normal = np.hstack([train_boost, tmp_boost_normal[:train_num]])
    valid_normal = tmp_normal[train_num:train_num + valid_num]
    valid_boost_normal = tmp_boost_normal[:valid_num]
    valid_abnormal = tmp_abnormal[train_abnormal_num:train_abnormal_num +
                                  valid_abnormal_num]
    valid_boost_abnormal = tmp_boost_abnormal[
        train_abnormal_num:train_abnormal_num + valid_abnormal_num]
    normal_ = tmp_normal[train_num + valid_num:]
    boost_normal_ = tmp_boost_normal[train_num + valid_num:]
    abnormal_ = tmp_abnormal[train_abnormal_num + valid_num:]
    boost_abnormal_ = tmp_boost_abnormal[train_abnormal_num + valid_num:]
    save_data('../../data/spark/robust/robust_log_train.csv', train_normal,
              train_abnormal)
    print(train_normal.shape, train_abnormal.shape)
    save_data('../../data/spark/boost_robust/boost_robust_log_train.csv',
              train_boost_normal, train_boost_abnormal)
    print(train_boost_normal.shape, train_boost_abnormal.shape)
    save_data('../../data/spark/robust/robust_log_valid.csv', valid_normal,
              valid_abnormal)
    print(valid_normal.shape, valid_abnormal.shape)
    save_data('../../data/spark/boost_robust/boost_robust_log_valid.csv',
              valid_boost_normal, valid_boost_abnormal)
    print(valid_boost_normal.shape, valid_boost_abnormal.shape)
    save_data('../../data/spark/robust/robust_log_test.csv', normal_,
              abnormal_)
    print(normal_.shape, abnormal_.shape)
    save_data('../../data/spark/boost_robust/boost_robust_log_test.csv',
              boost_normal_, boost_abnormal_)
    print(boost_normal_.shape, boost_abnormal_.shape)


if __name__ == "__main__":
    generate_semantic_vec()
    generate_robust_csv()