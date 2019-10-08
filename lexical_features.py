#!/usr/bin/env python
#encoding=utf-8
from __future__ import print_function
import sys
import json
from collections import Counter
from sentence import Sentence

from wordseg import Wordsegmenter
from utils import *
import pandas as pd

def get_periph(sent,q):
    sent_periph = Sentence()
    sent_periph_adj = Sentence()
    sent_periph_adv = Sentence()
    sent_periph_noun = Sentence()
    sent_periph_verb = Sentence()
    for token in sent.basic_words:
        if token.term not in q:
            sent_periph.basic_words.append(token)
            if token.postag == 'a':
                sent_periph_adj.basic_words.append(token)
            if token.postag == 'd':
                sent_periph_adv.basic_words.append(token)
            if token.postag == 'v':
                sent_periph_verb.basic_words.append(token)
            if token.postag[0] == 'n':
                sent_periph_noun.basic_words.append(token)
    return sent_periph, sent_periph_adj, sent_periph_adv, sent_periph_noun, sent_periph_verb

def calc_length_features(sl, sr, signature=""):
    counter_l = Counter(sl)
    counter_r = Counter(sr)
    #res = ["%s:%d" %(key,value) for key,value in counter_l.items()]
    #print(" ".join(res))
    #print(" ".join(counter_l.keys()))
    
    len_l = len(sl) # |A|
    #print(len_l)
    len_r = len(sr) # |B|
    
    len_union = sum((counter_l | counter_r).values())     # |A∪B|
    len_intersect = sum((counter_l & counter_r).values()) # |A∩B|
    
    len_diff = abs(len_l - len_r) # ||A|-|B||
    len_l_diff_r = sum((counter_l - counter_r).values()) # |A−B|
    len_r_diff_l = sum((counter_r - counter_l).values()) # |B−A|
    
    if len_l != 0:
        len_diff_l_ratio = 1.0 * len_diff / len_l # ||A|-|B||/|A|
        len_r_diff_l_ratio = 1.0 * len_r_diff_l / len_l  # |B−A|/|B|
    else:
        len_diff_l_ratio = 0
        len_r_diff_l_ratio = 0
        
    if len_r != 0:    
        len_diff_r_ratio = 1.0 * len_diff / len_r # ||A|-|B||/|B|
        len_l_diff_r_ratio = 1.0 * len_l_diff_r / len_r  # |A−B|/|A|
    else:
        len_diff_r_ratio = 0
        len_l_diff_r_ratio = 0

    feature_dict = {}

    if signature:
        signature = signature + "_"
    feature_names = ["len_l", "len_r", "len_union", "len_intersect", "len_diff",
            "len_l_diff_r", "len_r_diff_l", "len_l_diff_r_ratio", "len_r_diff_l_ratio"]

    for feature_name in feature_names: 
        feature_dict[signature + feature_name] = locals()[feature_name]
    
    return feature_dict

def calc_ngram_features(sl, sr, ngram_range=(1, 3), signature=""):
    feature_dict = {}
    if signature:
        signature = signature + "_"

    for n in range(*ngram_range):
        ngram_sl = list(ngrams(sl, n))
        ngram_sr = list(ngrams(sr, n))
        jaccard = jaccard_coefficient(ngram_sl, ngram_sr)
        feature_name = "{}{}gram_jaccard".format(signature, n)
        feature_dict[feature_name] = jaccard

        dice = dice_coefficient(ngram_sl, ngram_sr)
        feature_name = "{}{}gram_dice".format(signature, n)
        feature_dict[feature_name] = dice 

        overlap = overlap_coefficient(ngram_sl, ngram_sr)
        feature_name = "{}{}gram_overlap".format(signature, n)
        feature_dict[feature_name] = overlap
        

    return feature_dict

def calc_sequence_features(sl, sr, signature=""):
    feature_dict = {}
    
    if signature:
        signature = signature + "_"
    
    longest_common_subsequence_v = longest_common_subsequence(sl, sr)
    longest_common_substring_v = longest_common_substring(sl, sr)
    longest_common_prefix_v = longest_common_prefix(sl, sr)
    longest_common_suffix_v = longest_common_suffix(sl, sr)
    levenshtein_distance_v = levenshtein_distance(sl, sr)
    damerau_levenshtein_distance_v = damerau_levenshtein_distance(sl, sr)

    len_sl = len(sl)
    len_sr = len(sr)

    if len_sl != 0:
        longest_common_subsequence_l_ratio = 1.0 * longest_common_subsequence_v / len_sl
        longest_common_substring_l_ratio = 1.0 * longest_common_substring_v / len_sl
        longest_common_prefix_l_ratio = 1.0 * longest_common_prefix_v / len_sl
        longest_common_suffix_l_ratio = 1.0 * longest_common_suffix_v / len_sl
        levenshtein_distance_l_ratio = 1.0 * levenshtein_distance_v / len_sl
        damerau_levenshtein_distance_l_ratio = 1.0 * damerau_levenshtein_distance_v / len_sl
    else:
        longest_common_subsequence_l_ratio = 0
        longest_common_substring_l_ratio = 0
        longest_common_prefix_l_ratio = 0
        longest_common_suffix_l_ratio = 0
        levenshtein_distance_l_ratio = 0
        damerau_levenshtein_distance_l_ratio = 0
    
    if len_sr != 0:
        longest_common_subsequence_r_ratio = 1.0 * longest_common_subsequence_v / len_sr
        longest_common_substring_r_ratio = 1.0 * longest_common_substring_v / len_sr
        longest_common_prefix_r_ratio = 1.0 * longest_common_prefix_v / len_sr
        longest_common_suffix_r_ratio = 1.0 * longest_common_suffix_v / len_sr
        levenshtein_distance_r_ratio = 1.0 * levenshtein_distance_v / len_sr
        damerau_levenshtein_distance_r_ratio = 1.0 * damerau_levenshtein_distance_v / len_sr
    else:
        longest_common_subsequence_r_ratio = 0
        longest_common_substring_r_ratio = 0
        longest_common_prefix_r_ratio = 0
        longest_common_suffix_r_ratio = 0
        levenshtein_distance_r_ratio = 0
        damerau_levenshtein_distance_r_ratio = 0
    
    feature_names = ["longest_common_subsequence_v", "longest_common_substring_v", 
            "longest_common_prefix_v", "longest_common_suffix_v", 
            "levenshtein_distance_v", "damerau_levenshtein_distance_v", 
            "longest_common_subsequence_l_ratio", "longest_common_substring_l_ratio",
            "longest_common_prefix_l_ratio", "longest_common_suffix_l_ratio",
            "levenshtein_distance_l_ratio", "damerau_levenshtein_distance_l_ratio",
            "longest_common_subsequence_r_ratio", "longest_common_substring_r_ratio",
            "longest_common_prefix_r_ratio", "longest_common_suffix_r_ratio",
            "levenshtein_distance_r_ratio", "damerau_levenshtein_distance_r_ratio"]

    for feature_name in feature_names: 
        feature_dict[signature + feature_name] = locals()[feature_name]

    return feature_dict
    

def calc_lexical_features(sent_l, sent_r):
    feature_dict = {}

    uchars_l = sent_l.base_form.decode("utf8")
    uchars_r = sent_r.base_form.decode("utf8")
    
    #print(uchars_l)
    
    char_length_features = calc_length_features(uchars_l, uchars_r, signature="char")#9
    feature_dict.update(char_length_features)

    char_ngram_features = calc_ngram_features(uchars_l, uchars_r, ngram_range=(1,5), signature="char")#12
    feature_dict.update(char_ngram_features)
    
    char_sequence_features = calc_sequence_features(uchars_l, uchars_r, signature="char")#18
    feature_dict.update(char_sequence_features)
    
    basic_word_terms_l = [token.term for token in sent_l.basic_words]
    basic_word_terms_r = [token.term for token in sent_r.basic_words]
    basic_word_length_features = calc_length_features(basic_word_terms_l, basic_word_terms_r, signature="basic_word")#9
    feature_dict.update(basic_word_length_features)

    basic_word_ngram_features = calc_ngram_features(basic_word_terms_l, basic_word_terms_r, ngram_range=(1,3), signature="basic_word")#6
    feature_dict.update(basic_word_ngram_features)

    basic_word_sequence_features = calc_sequence_features(basic_word_terms_l, basic_word_terms_r, signature="basic_word") #18
    feature_dict.update(basic_word_sequence_features)

    postags_l = [token.postag for token in sent_l.basic_words]
    postags_r = [token.postag for token in sent_r.basic_words]
    postag_length_features = calc_length_features(postags_l, postags_r, signature="postag")#9
    feature_dict.update(postag_length_features)

    postag_ngram_features = calc_ngram_features(postags_l, postags_r, ngram_range=(1,3), signature="postag")#6
    feature_dict.update(postag_ngram_features)


    return feature_dict

def calc_periph_lexical_features(l_periph, r_periph):
    sent_l_periph, sent_l_periph_adj, sent_l_periph_adv, sent_l_periph_noun, sent_l_periph_verb = l_periph
    sent_r_periph, sent_r_periph_adj, sent_r_periph_adv, sent_r_periph_noun, sent_r_periph_verb = r_periph
    
    feature_dict = {}
    
    periph_basic_word_terms_l = [token.term for token in sent_l_periph.basic_words]
    periph_basic_word_terms_r = [token.term for token in sent_r_periph.basic_words]
    periph_basic_word_length_features = calc_length_features(periph_basic_word_terms_l, periph_basic_word_terms_r, signature="periph_basic_word")
    feature_dict.update(periph_basic_word_length_features)
    
    periph_adj_basic_word_terms_l = [token.term for token in sent_l_periph_adj.basic_words]
    periph_adj_basic_word_terms_r = [token.term for token in sent_r_periph_adj.basic_words]
    periph_adj_basic_word_length_features = calc_length_features(periph_adj_basic_word_terms_l, periph_adj_basic_word_terms_r, signature="periph_adj_basic_word")
    feature_dict.update(periph_adj_basic_word_length_features)
    
    periph_adv_basic_word_terms_l = [token.term for token in sent_l_periph_adv.basic_words]
    periph_adv_basic_word_terms_r = [token.term for token in sent_r_periph_adv.basic_words]
    periph_adv_basic_word_length_features = calc_length_features(periph_adv_basic_word_terms_l, periph_adv_basic_word_terms_r, signature="periph_adv_basic_word")
    feature_dict.update(periph_adv_basic_word_length_features)
    
    periph_verb_basic_word_terms_l = [token.term for token in sent_l_periph_verb.basic_words]
    periph_verb_basic_word_terms_r = [token.term for token in sent_r_periph_verb.basic_words]
    periph_verb_basic_word_length_features = calc_length_features(periph_verb_basic_word_terms_l, periph_verb_basic_word_terms_r, signature="periph_verb_basic_word")
    feature_dict.update(periph_verb_basic_word_length_features)

    periph_noun_basic_word_terms_l = [token.term for token in sent_l_periph_noun.basic_words]
    periph_noun_basic_word_terms_r = [token.term for token in sent_r_periph_noun.basic_words]
    periph_noun_basic_word_length_features = calc_length_features(periph_noun_basic_word_terms_l, periph_noun_basic_word_terms_r, signature="periph_noun_basic_word")
    feature_dict.update(periph_noun_basic_word_length_features)
    
    return feature_dict

def extract_features(wordseg, ql, qr):
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
    
    l_periph = get_periph(sent_l, qr)
    r_periph = get_periph(sent_r, ql)
    
    feature_dict = {}
    
    lexical_features = calc_lexical_features(sent_l, sent_r)
    feature_dict.update(lexical_features)
    
    periph_lexical_features = calc_periph_lexical_features(l_periph, r_periph)
    feature_dict.update( periph_lexical_features)
    
    return feature_dict
'''
def test():
    in_path = ""
    wordseg = Wordsegmenter("./bin/pyseg.so", "./bin/qsegconf.ini")

    #for line in sys.stdin:
    line = "00d5d3eddce90af913a5fd2d10d67b03	用精子库精子需要多少能怀孕	用精子库的精子需要多少钱	0"
    line = line.strip('\r\n')
    parts = line.split('\t')
    ql = parts[1]
    qr = parts[2]
  
    sent_l = Sentence()
    sent_l.raw_form = ql
    sent_l.base_form = ql
    wordseg_out = wordseg.segment(ql) 
    #sent_l.basic_words = wordseg_out.basic_words
    sent_l.basic_words = wordseg_out.long_words

    sent_r = Sentence()
    sent_r.raw_form = qr
    sent_r.base_form = qr
    wordseg_out = wordseg.segment(qr) 
    sent_r.basic_words = wordseg_out.long_words

    lexical_features = calc_lexical_features(sent_l, sent_r)
    #print "{}\t{}".format(line, json.dumps(lexical_features))
    print(len(lexical_features))
'''


if __name__ == "__main__":
    #test()
    wordseg = Wordsegmenter("./bin/pyseg.so", "./bin/qsegconf.ini")
    in_path = "./data/yao_test_data.txt"
    data = pd.read_csv(in_path, sep="\t", dtype='str', names=['qid', 'ql', 'qr', 'label'])

    lexical_features = data[['ql', 'qr']].apply(lambda row: extract_features(wordseg, row['ql'], row['qr']), axis=1)
    
    for k in lexical_features:
        print(k)
    
