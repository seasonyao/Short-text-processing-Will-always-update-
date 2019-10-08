#!/usr/bin/env python
#encoding=utf-8
from __future__ import print_function
import sys
import cPickle as  pickle
import getopt

import pandas as pd

from collections import Counter
from sentence import Sentence

from wordseg import Wordsegmenter
from utils import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from marisa_vectorizers import MarisaCountVectorizer, MarisaTfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score

from sklearn.metrics.pairwise import paired_euclidean_distances
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics.pairwise import paired_manhattan_distances

from sklearn.linear_model import LogisticRegression


class VectorModels(object):
    def __init__(self):
        self.tfidf1 = pickle.load(open('./data/yao_saver/ql_qr_tfidf1.pkl','rb'))
        self.count1 = pickle.load(open('./data/yao_saver/ql_qr_count1.pkl','rb'))
        self.hash1_18 = pickle.load(open('./data/yao_saver/ql_qr_hash1_18.pkl','rb'))
        self.hash2_18 = pickle.load(open('./data/yao_saver/ql_qr_hash2_18.pkl','rb'))
        self.hash1_20 = pickle.load(open('./data/yao_saver/ql_qr_hash1_20.pkl','rb'))
        self.hash2_20 = pickle.load(open('./data/yao_saver/ql_qr_hash2_20.pkl','rb'))

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
        
def save_fit_result(texts_ql_qr):
    marisa_count1 = MarisaCountVectorizer()
    marisa_tfidf1 = MarisaTfidfVectorizer()
    hashing1_18 = HashingVectorizer(n_features=2**20)
    hashing1_20 = HashingVectorizer(n_features=2**24)
    hashing2_18 = HashingVectorizer(ngram_range=(1,2),n_features=2**20)
    hashing2_20 = HashingVectorizer(ngram_range=(1,2),n_features=2**24)
    
    ql_qr_tfidf1 = marisa_tfidf1.fit(texts_ql_qr)
    with open('./data/yao_saver/ql_qr_tfidf1.pkl','wb') as a:
        pickle.dump(ql_qr_tfidf1, a)
    print('ql_qr_tfidf1 is ok')
    
    ql_qr_count1 = marisa_count1.fit(texts_ql_qr)
    with open('./data/yao_saver/ql_qr_count1.pkl','wb') as a:
        pickle.dump(ql_qr_count1, a)
    print('ql_qr_count1 is ok')
    
    ql_qr_hash1_18 = hashing1_18.fit(texts_ql_qr)
    with open('./data/yao_saver/ql_qr_hash1_18.pkl','wb') as a:
        pickle.dump(ql_qr_hash1_18, a)
    print('ql_qr_hash1_18 is ok')
    
    ql_qr_hash2_18 = hashing2_18.fit(texts_ql_qr)
    with open('./data/yao_saver/ql_qr_hash2_18.pkl','wb') as a:
        pickle.dump(ql_qr_hash2_18, a)
    print('ql_qr_hash2_18 is ok')
    
    ql_qr_hash1_20 = hashing1_20.fit(texts_ql_qr)
    with open('./data/yao_saver/ql_qr_hash1_20.pkl','wb') as a:
        pickle.dump(ql_qr_hash1_20, a)
    print('ql_qr_hash1_20 is ok')
    
    ql_qr_hash2_20 = hashing2_20.fit(texts_ql_qr)
    with open('./data/yao_saver/ql_qr_hash2_20.pkl','wb') as a:
        pickle.dump(ql_qr_hash2_20, a)
    print('ql_qr_hash2_20 is ok')
    
    
#--------------------------------------------------------------------------------------------------------------------------------------------------     
        
def split_ql_qr(wordseg, ql, qr):
    tokenized_ql = []
    tokenized_qr = []
    sent_l = Sentence()
    sent_l.raw_form = ql
    sent_l.base_form = ql
    wordseg_out = wordseg.segment(ql) 
    sent_l.basic_words = wordseg_out.basic_words
    items=[]
    for item in sent_l.basic_words:
        items.append(item.term)
    tokenized_ql.append(items)
    
    sent_r = Sentence()
    sent_r.raw_form = qr
    sent_r.base_form = qr
    wordseg_out = wordseg.segment(qr) 
    sent_r.basic_words = wordseg_out.basic_words
    items=[]
    for item in sent_r.basic_words:
        items.append(item.term)
    tokenized_qr.append(items)
    
    return tokenized_ql,tokenized_qr
    
def get_split_ql_qr_sentence(wordseg, ql, qr):
    texts_ql = []
    texts_qr = []
    tokenized_ql,tokenized_qr = split_ql_qr(wordseg, ql, qr)

    for sentence in tokenized_ql:
        texts_ql.append(" ".join(sentence))
    
    for sentence in tokenized_qr:
        texts_qr.append(" ".join(sentence))

    return texts_ql, texts_qr
#-------------------------------------------------------------------------------------------------------
def get_count_tfidf_hash(ql, qr, vectorModels):
    ql_tfidf1 = vectorModels.tfidf1.transform(ql)
    ql_count1 = vectorModels.count1.transform(ql)
    ql_hash1_20 = vectorModels.hash1_20.transform(ql)
    ql_hash2_18 = vectorModels.hash2_18.transform(ql)
    ql_hash2_20 = vectorModels.hash2_20.transform(ql)
    ql_hash1_18 = vectorModels.hash1_18.transform(ql)

    
    qr_tfidf1 = vectorModels.tfidf1.transform(qr)
    qr_count1 = vectorModels.count1.transform(qr)
    qr_hash1_18 = vectorModels.hash1_18.transform(qr)
    qr_hash1_20 = vectorModels.hash1_20.transform(qr)
    qr_hash2_18 = vectorModels.hash2_18.transform(qr)
    qr_hash2_20 = vectorModels.hash2_20.transform(qr)
    
    return ql_tfidf1, ql_count1, ql_hash1_18, ql_hash1_20, ql_hash2_18, ql_hash2_20, qr_tfidf1, qr_count1, qr_hash1_18, qr_hash1_20, qr_hash2_18, qr_hash2_20

def get_tfidf_count_hash_features(ql, qr, vectorModels, signature=""):
    texts_ql = []
    texts_qr = []
    
    res = ["%s" %(token.term) for token in ql.basic_words]
    texts_ql.append(" ".join(res))
    
    res = ["%s" %(token.term) for token in qr.basic_words]
    texts_qr.append(" ".join(res))

    feature_dict = {}
    
    ql_tfidf1, ql_count1, ql_hash1_18, ql_hash1_20, ql_hash2_18, ql_hash2_20, qr_tfidf1, qr_count1, qr_hash1_18, qr_hash1_20, qr_hash2_18, qr_hash2_20 = get_count_tfidf_hash(texts_ql, texts_qr, vectorModels)
    
    if signature:
        signature = signature + "_"
        
    tfidf1_PED = paired_euclidean_distances(ql_tfidf1, qr_tfidf1)
    feature_dict[signature+ 'tfidf1_PED'] = float(tfidf1_PED[0])
    
    count1_PED = paired_euclidean_distances(ql_count1, qr_count1)
    feature_dict[signature+ 'count1_PED'] = float(count1_PED[0])
    
    hash1_18_PED = paired_euclidean_distances(ql_hash1_18, qr_hash1_18)
    feature_dict[signature+ 'hash1_18_PED'] = float(hash1_18_PED[0])
    
    hash1_20_PED = paired_euclidean_distances(ql_hash1_20, qr_hash1_20)
    feature_dict[signature+ 'hash1_20_PED'] = float(hash1_20_PED[0])
    #-------------------------------------------------------------------
    
    tfidf1_PCD = paired_cosine_distances(ql_tfidf1, qr_tfidf1)
    feature_dict[signature+ 'tfidf1_PCD'] = float(tfidf1_PCD[0])
    
    count1_PCD = paired_cosine_distances(ql_count1, qr_count1)
    feature_dict[signature+ 'count1_PCD'] = float(count1_PCD[0])
    
    hash1_18_PCD = paired_cosine_distances(ql_hash1_18, qr_hash1_18)
    feature_dict[signature+ 'hash1_18_PCD'] = float(hash1_18_PCD[0])
    
    hash1_20_PCD = paired_cosine_distances(ql_hash1_20, qr_hash1_20)
    feature_dict[signature+ 'hash1_20_PCD'] = float(hash1_20_PCD[0])
    
    hash2_18_PCD = paired_cosine_distances(ql_hash2_18, qr_hash2_18)
    feature_dict[signature+ 'hash2_18_PCD'] = float(hash2_18_PCD[0])
    
    hash2_20_PCD = paired_cosine_distances(ql_hash2_20, qr_hash2_20)
    feature_dict[signature+ 'hash2_20_PCD'] = float(hash2_20_PCD[0])
    #------------------------------------------------------------------
    
    tfidf1_PMD = paired_manhattan_distances(ql_tfidf1, qr_tfidf1)
    feature_dict[signature+ 'tfidf1_PMD'] = float(tfidf1_PMD)
    
    count1_PMD = paired_manhattan_distances(ql_count1, qr_count1)
    feature_dict[signature+ 'count1_PMD'] = float(count1_PMD)
    
    hash1_18_PMD = paired_manhattan_distances(ql_hash1_18, qr_hash1_18)
    feature_dict[signature+ 'hash1_18_PMD'] = float(hash1_18_PMD)
    
    hash1_20_PMD = paired_manhattan_distances(ql_hash1_20, qr_hash1_20)
    feature_dict[signature+ 'hash1_20_PMD'] = float(hash1_20_PMD)
    
    hash2_18_PMD = paired_manhattan_distances(ql_hash2_18, qr_hash2_18)
    feature_dict[signature+ 'hash2_18_PMD'] = float(hash2_18_PMD)
    
    hash2_20_PMD = paired_manhattan_distances(ql_hash2_20, qr_hash2_20)
    feature_dict[signature+ 'hash2_20_PMD'] = float(hash2_20_PMD)
    
    return feature_dict



def extract_features(wordseg, ql, qr, tfidf_count_hash_vectorModels):
    sent_l = Sentence()
    sent_l.raw_form = ql
    sent_l.base_form = ql
    wordseg_out = wordseg.segment(ql) 
    sent_l.basic_words = wordseg_out.basic_words
    
    feature_dict = {}
    sent_r = Sentence()
    sent_r.raw_form = qr
    sent_r.base_form = qr
    wordseg_out = wordseg.segment(qr) 
    sent_r.basic_words = wordseg_out.basic_words
    
    l_notion = get_notional_tokens(sent_l)
    r_notion = get_notional_tokens(sent_r)
    
    count_tfidf_hash_features = get_tfidf_count_hash_features(sent_l, sent_r, tfidf_count_hash_vectorModels)
    feature_dict.update(count_tfidf_hash_features)
    
    notion_count_tfidf_hash_features = get_tfidf_count_hash_features(l_notion, r_notion, tfidf_count_hash_vectorModels, "notion")
    feature_dict.update(notion_count_tfidf_hash_features)
    
    for k in feature_dict:
        print(k)
    
    return feature_dict

def main(vectorModels):
    feature_dict = {}
    
    action = None
    train_in_path = ""
    pred_in_path = ""
    opts, args = getopt.getopt(sys.argv[1:], "a:", ["save_in_path=", "train_in_path="])
    for op, value in opts:
        if op == "-a":
            action = value
        if op == "--save_in_path":
            train_in_path = value
        if op == "--train_in_path":
            pred_in_path = value
            
    if action == "save":
        if train_in_path:
            f = open(train_in_path,'r')
            texts = f.readlines()
            f.close()
            save_fit_result(texts)
            
    if action == "train":
        if pred_in_path:
            wordseg = Wordsegmenter("./bin/pyseg.so", "./bin/qsegconf.ini")
            in_path = "./data/yao_test_data.txt"
            data = pd.read_csv(in_path, sep="\t", dtype='str', names=['qid', 'ql', 'qr', 'label'])
            vectorModels = VectorModels()
            count_tfidf_hash_features = data[['ql', 'qr']].apply(lambda row: extract_features(wordseg, row['ql'], row['qr'], vectorModels), axis=1)
            feature_dict.update(count_tfidf_hash_features)
            

if __name__ == "__main__":
    #main()
    
    wordseg = Wordsegmenter("./bin/pyseg.so", "./bin/qsegconf.ini")
    in_path = "./data/yao_test_data.txt"
    data = pd.read_csv(in_path, sep="\t", dtype='str', names=['qid', 'ql', 'qr', 'label'])
    vectorModels = VectorModels()
    count_tfidf_hash_features = data[['ql', 'qr']].apply(lambda row: extract_features(wordseg, row['ql'], row['qr'], vectorModels), axis=1)
    
    for k in count_tfidf_hash_features:
        print(k)
    
