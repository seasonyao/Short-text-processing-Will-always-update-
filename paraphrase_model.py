#!/usr/bin/env python
#encoding=utf-8

import sys
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

from gensim.models import KeyedVectors

import pandas as pd

from wordseg import Wordsegmenter
from sentence import Sentence

from lexical_features import calc_lexical_features
from count_tfidf_hash_features import get_tfidf_count_hash_features
from count_tfidf_hash_features import VectorModels
from sentvec_features import get_sentvec_features
from sentvec_features import load_vocab

def extract_features(wordseg, ql, qr, tfidf_count_hash_vectorModels, sent_word2vec, sent_vocab_dict, sent_model):

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

    lexical_features = calc_lexical_features(sent_l, sent_r)
    feature_dict.update(lexical_features)    
    
    count_tfidf_hash_features = get_tfidf_count_hash_features(sent_l, sent_r, tfidf_count_hash_vectorModels)
    feature_dict.update(count_tfidf_hash_features)
    
    sentvec_features = get_sentvec_features(sent_word2vec, sent_vocab_dict, sent_model, sent_l, sent_r)
    feature_dict.update(sentvec_features)
    
    return feature_dict

def process():
    wordseg = Wordsegmenter("./bin/pyseg.so", "./bin/qsegconf.ini")
    for line in sys.stdin:
        line = line.strip('\r\n')
        parts = line.split('\t')
        ql = parts[1]
        qr = parts[2]
        feature_dict = extract_features(ql, qr)        
        print "{}\t{}".format(line, json.dumps(feature_dict))

def train():
    wordseg = Wordsegmenter("./bin/pyseg.so", "./bin/qsegconf.ini")
    in_path = "./data/paraphrase_man_annotation.txt"
    
    sent_word2vec_path = "./data/word2vec.query.bin"
    sent_vocab_path = "./data/word2vec.query.vocab"
    sent_model_path = "./data/sif.model"
    
    sent_word2vec = KeyedVectors.load_word2vec_format(sent_word2vec_path, binary=True)
    sent_vocab_dict = load_vocab(sent_vocab_path)
    sent_model = joblib.load(sent_model_path)
    
    #input tfidf count and hash model
    tfidf_count_hash_vectorModels = VectorModels()
    
    data = pd.read_csv(in_path, sep="\t", dtype='str', names=['qid', 'ql', 'qr', 'label'])

    
    X = data[['ql', 'qr']].apply(lambda row: extract_features(wordseg, row['ql'], row['qr'], tfidf_count_hash_vectorModels, sent_word2vec, sent_vocab_dict, sent_model), axis=1)
    print("get all vector")
    X = pd.DataFrame(list(X))
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model = LinearSVC()
    
    model.fit(X_train, y_train)
    
    for x in model.coef_[0]:
        print(x)

    model_path = "./model/paraphrase.svm_model"
    joblib.dump(model, model_path)
    y_preds = model.predict(X_test)
    mean_f1 = f1_score(y_test, y_preds, average='micro')
    print mean_f1
    print classification_report(y_test, y_preds, target_names=["paraphrase", "other"])
    
    feature_names = X.columns.values.tolist()
    for feature_name, coef in zip(feature_names, model.coef_.ravel()):
        print "%s\t%f"% ( feature_name, coef)

if __name__ == "__main__":
    train()
    
