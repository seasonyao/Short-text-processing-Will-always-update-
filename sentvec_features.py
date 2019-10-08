#!/usr/bin/env python
#encoding=utf-8

import sys
import numpy as np
import getopt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from gensim.models import KeyedVectors
from sklearn.decomposition import TruncatedSVD

from sklearn.externals import joblib

import pandas as pd

from wordseg import Wordsegmenter
from sentence import Sentence

from sklearn.metrics.pairwise import paired_euclidean_distances
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics.pairwise import paired_manhattan_distances

def get_notional_tokens(sent):
    sent_notion = Sentence()
    for token in sent.basic_words:
        if token.postag == 'a':
            sent_notion.basic_words.append(token)
        elif token.postag == 'd':
            sent_notion.basic_words.append(token)
        elif token.postag == 'v':
            sent_notion.basic_words.append(token)
        elif token.postag[0] == 'n':
            sent_notion.basic_words.append(token)
    return sent_notion

def load_vocab(vocab_path):
    vocab_dict = {}
    with open(vocab_path, 'r') as fin:
        for line in fin:
            line = line.strip('\r\n').decode("utf-8")
            parts = line.split('\r')
            if len(parts) != 2:
                continue
            vocab_dict[parts[0]] = int(parts[1])
    return vocab_dict

def calc_sif_word_weights(vocab_dict, a=1e-3):
    if a <= 0:  
        a = 1.0
    word2weight = {}
    N = sum(vocab_dict.itervalues())
    for key, value in vocab_dict.iteritems():
        word2weight[key] = a / (a + value / N)
    return word2weight

def get_weighted_sentence_embedding(word2vec, dim, word_weights, sentence):
    emb_sum = np.zeros(dim, dtype=np.float32)
    count = 0
    for word in sentence:
        if word in word2vec:
            emb_sum += word2vec[word] * word_weights.get(word, 1.0)
            count += 1
    if count == 0:
        return np.nan
    return emb_sum / len(sentence)

#all words' weight is equal to 1
def get_naive_sentence_embedding(word2vec, dim, sentence):
    emb_sum = np.zeros(dim, dtype=np.float32)
    count = 0
    for word in sentence:
        if word in word2vec:
            emb_sum += word2vec[word] * 1
            count += 1
    if count == 0:
        return np.nan
    return emb_sum / len(sentence)

def calc_naive_sif_embedding(word2vec, dim, word_weights, model, sentence):
    sentence_weight_vecs_temp = []
    sentence_naive_vecs_temp = []

    sentence_weight_vec = get_weighted_sentence_embedding(word2vec, dim, word_weights, sentence)
    sentence_naive_vec = get_naive_sentence_embedding(word2vec, dim, sentence)
    
    if sentence_weight_vec is not np.nan:
        sentence_weight_vecs_temp.append(sentence_weight_vec)
    else:
        sentence_weight_vecs_temp.append([np.nan]*dim)
    
    if sentence_naive_vec is not np.nan:
        sentence_naive_vecs_temp.append(sentence_naive_vec)
    else:
        sentence_naive_vecs_temp.append([np.nan]*dim)
    
    pc = model.components_
    X_weight = np.asarray(sentence_weight_vecs_temp, dtype=np.float32)
    sentence_weight_vecs = X_weight - X_weight.dot(pc.transpose()).dot(pc)
    
    sentence_naive_vecs = np.asarray(sentence_naive_vecs_temp, dtype=np.float32)

    return sentence_weight_vecs, sentence_naive_vecs

def get_sentvec_features(word2vec, word_weights, model, sent_l, sent_r, signature=""):
    feature_dict = {}
    
    ql = ["%s" %(word.term.decode('utf8')) for word in sent_l.basic_words]
    qr = ["%s" %(word.term.decode('utf8')) for word in sent_r.basic_words]

    dim = word2vec.vectors.shape[1]

    sif_vec_ql_weight, naive_vec_ql_weight = calc_naive_sif_embedding(word2vec, dim, word_weights, model, ql)
    sif_vec_qr_weight, naive_vec_qr_weight = calc_naive_sif_embedding(word2vec, dim, word_weights, model, qr)
    
    if signature:
        signature = signature + "_"
    
    #calculate sif feature
    if np.isnan(sif_vec_ql_weight[0][0]) or np.isnan(sif_vec_qr_weight[0][0]):
        feature_dict[signature+ 'sif_sentvec_weight_PED'] = 1.0
        feature_dict[signature+ 'sif_sentvec_weight_PCD'] = 1.0
        feature_dict[signature+ 'sif_sentvec_weight_PMD'] = 1000.0
    else:
        sif_sentvec_weight_PED = paired_euclidean_distances(sif_vec_ql_weight, sif_vec_qr_weight)
        feature_dict[signature+ 'sif_sentvec_weight_PED'] = float(sif_sentvec_weight_PED[0])
        
        sif_sentvec_weight_PCD = paired_cosine_distances(sif_vec_ql_weight, sif_vec_qr_weight)
        feature_dict[signature+ 'sif_sentvec_weight_PCD'] = float(sif_sentvec_weight_PCD[0])
        
        sif_sentvec_weight_PMD = paired_manhattan_distances(sif_vec_ql_weight, sif_vec_qr_weight)
        feature_dict[signature+ 'sif_sentvec_weight_PMD'] = float(sif_sentvec_weight_PMD[0])
        
    #calculate naive feature
    if np.isnan(naive_vec_ql_weight[0][0]) or np.isnan(naive_vec_qr_weight[0][0]):
        feature_dict[signature+ 'avg_sentvec_weight_PED'] = 1.0
        feature_dict[signature+ 'avg_sentvec_weight_PCD'] = 1.0
        feature_dict[signature+ 'avg_sentvec_weight_PMD'] = 1000.0
    else:
        naive_sentvec_weight_PED = paired_euclidean_distances(naive_vec_ql_weight, naive_vec_qr_weight)
        feature_dict[signature+ 'avg_sentvec_weight_PED'] = float(naive_sentvec_weight_PED[0])
        
        naive_sentvec_weight_PCD = paired_cosine_distances(naive_vec_ql_weight, naive_vec_qr_weight)
        feature_dict[signature+ 'avg_sentvec_weight_PCD'] = float(naive_sentvec_weight_PCD[0])
        
        naive_sentvec_weight_PMD = paired_manhattan_distances(naive_vec_ql_weight, naive_vec_qr_weight)
        feature_dict[signature+ 'avg_sentvec_weight_PMD'] = float(naive_sentvec_weight_PMD[0])

    #calculate wmdistance
    feature_dict[signature+ 'WMD'] = word2vec.wmdistance(ql, qr)
    
    return feature_dict

def extract_features(wordseg, ql, qr, sent_word2vec, word_weights, sent_model):
    sent_l = Sentence()
    sent_l.raw_form = ql
    sent_l.base_form = ql
    wordseg_out = wordseg.segment(ql) 
    sent_l.basic_words = wordseg_out.basic_words
    
    sent_r = Sentence()
    sent_r.raw_form = qr
    sent_r.base_form = qr
    wordseg_out = wordseg.segment(qr) 
    sent_r.basic_words = wordseg_out.basic_words

    l_notion = get_notional_tokens(sent_l)
    r_notion = get_notional_tokens(sent_r)
    
    feature_dict = {}
    
    sentvec_features = get_sentvec_features(sent_word2vec, word_weights, sent_model, sent_l, sent_r)
    feature_dict.update(sentvec_features)
    sentvec_features = get_sentvec_features(sent_word2vec, word_weights, sent_model, sent_l, sent_r, "notion")
    feature_dict.update(sentvec_features)
    
    for k,value in feature_dict.items():
        print(k)
        print(value)
    
    return feature_dict

if __name__ == "__main__":
    #word2vec_path = "./data/word2vec.query.bin"
    word2vec_path = "./data/word2vec_normalized.query.bin"
    vocab_path = "./data/word2vec.query.vocab"
    model_path = "./data/sif.model"
    
    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    #word2vec.init_sims(replace=True)#这边是归一化词向量，不加这行的话，计算的距离数据可能会非常大
    #word2vec.save("./data/word2vec_normalized.query.bin",binary=True)
    
    vocab_dict = load_vocab(vocab_path)
    word_weights = calc_sif_word_weights(vocab_dict)
    sif_model = joblib.load(model_path)

    wordseg = Wordsegmenter("./bin/pyseg.so", "./bin/qsegconf.ini")
    in_path = "./data/yao_test_data.txt"
    data = pd.read_csv(in_path, sep="\t", dtype='str', names=['qid', 'ql', 'qr', 'label'])
    
    sentvec_features = data[['ql', 'qr']].apply(lambda row: extract_features(wordseg, row['ql'], row['qr'], word2vec, word_weights, sif_model), axis=1)
 