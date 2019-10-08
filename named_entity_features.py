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
from collections import defaultdict

from wordseg import Wordsegmenter
from sentence import Sentence


def ner_by_postag(sent):
    feature_map = {
            "nh": "person", 
            "ni": "organization", 
            "ns": "geography", 
            "nd": "direction", 
            "nl": "location", 
            "nt": "temporal",
            "j": "abbreviation", 
            "m": "numeral", 
            "mq": "numeral_quantifier",
        }
    
    entity_dict = defaultdict(list) 
    for token in sent.basic_words:
        if token.postag in feature_map:
            feature_name = feature_map[token.postag]
            entity_dict[feature_name].append((token.term, feature_name, token.offset, token.length))
    return entity_dict

def ner_by_dict(sent, ner_dict):
    basic_words = [token.term for token in sent.basic_words]
    basic_words_num = len(basic_words) 
    i = 0
    j = 0
    entity_dict = defaultdict(list)
    while i < basic_words_num: 
        j = min(i+5, basic_words_num) 
        while j > i:
            cand_word = "".join([w for w in basic_words[i:j]])
            ucand_word = cand_word.decode("utf-8")
            if ucand_word in ner_dict:
                ne_type = ''.join(ner_dict.get(ucand_word))
                entity_dict[ne_type].append((cand_word, ne_type,i,j-i))
                j = j - 1
                break
            j = j - 1 
        i = j + 1
    return  entity_dict

def ner(sent, ner_dict):
    entities_postag = ner_by_postag(sent)
    entities_dict = ner_by_dict(sent, ner_dict)
    entities_merge = entities_postag.copy()

    for key,value in entities_dict.items():
        entities_merge[key] = value
    
    return entities_merge

def calc_entity_syn_map(entity_dict, syn_dict):
    entity_dict_new = defaultdict(list)
      
    for ent_type, entities in entity_dict.items():
        for entity in entities:
            entity_dict_new[ent_type].append(syn_dict.get(entity[0],entity[0]))
    return entity_dict_new
    
def get_ner_features(sent_l, sent_r, ner_dict, syn_dict):
    entities_l = ner(sent_l, ner_dict)
    entities_r = ner(sent_r, ner_dict)
    
    entities_l =  calc_entity_syn_map(entities_l, syn_dict)
    entities_r =  calc_entity_syn_map(entities_r, syn_dict)
    
    feature_dict = {}
    feature_names = ["person", "organization", "geography", "direction", "location", "temporal"
            "abbreviation", "numeral", "numeral_quantifier","season"]

    entity_all_l = []
    entity_all_r = []

    for feature_name in feature_names:
        entity_intersect = list(set(entities_l[feature_name]).intersection(set(entities_r[feature_name])))#交集
        entity_union = list(set(entities_l[feature_name]).union(set(entities_r[feature_name])))#并集
        entity_all_l.extend(entities_l[feature_name])
        entity_all_r.extend(entities_r[feature_name])
        feature_dict["ner_" + feature_name+ "_same"] = (len(entity_intersect)+0.1)/(len(entity_union)+0.1)
    
    ner_entity_all_same = (len(set(entity_all_l)&set(entity_all_r)) + 0.1) / (len(set(entity_all_l)|set(entity_all_r))+0.1)
    feature_dict["ner_entity_all_same"] = ner_entity_all_same  
    
    return feature_dict



def extract_features(wordseg, ql, qr, ner_dict, syn_dict):
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
    
    feature_dict = {}
    
    entities_features = get_ner_features(sent_l, sent_r, ner_dict, syn_dict)
    feature_dict.update(entities_features)
    
    return feature_dict

if __name__ == "__main__":
    wordseg = Wordsegmenter("./bin/pyseg.so", "./bin/qsegconf.ini")
    in_path = "./data/yao_test_data.txt"
    data = pd.read_csv(in_path, sep="\t", dtype='str', names=['qid', 'ql', 'qr', 'label'])
    
    ner_dict = {}
    ner_dict_path = "./data/ner.dict"
    with open(ner_dict_path, "r") as f:
        for item in f:
            key, value = item.split('\n')[0].split('\t')
            ner_dict[key.decode("utf8")] = value
     
    syn_dict = {}
    syn_dict_path = "./data/syn.dict"
    with open(syn_dict_path, "r") as f:
        for item in f:
            parts = item.split('\t')
            target = parts[0]
            for source  in parts[1:]:
                syn_dict[source] = target

    named_entities_features = data[['ql', 'qr']].apply(lambda row: extract_features(wordseg, row['ql'], row['qr'], ner_dict, syn_dict), axis=1)
    
    for k in named_entities_features:
        print k