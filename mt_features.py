#!/usr/bin/env python
#encoding=utf-8

import sys

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.ribes_score import sentence_ribes


cc = SmoothingFunction()

def calc_mt_features(sent_l, sent_r):
    feature_dict = {}
    uchars_l = sent_l.base_form.decode("utf8")
    uchars_r = sent_r.base_form.decode("utf8")
    
    char_mt_features = calc_seq_mt_features(uchars_l, uchars_r, signature="char")
    feature_dict.update(char_mt_features)

    
    basic_word_terms_l = [token.term for token in sent_l.basic_words]
    basic_word_terms_r = [token.term for token in sent_r.basic_words]
    basic_word_mt_features = calc_seq_mt_features(basic_word_terms_l, basic_word_terms_r, signature="basic_word")
    feature_dict.update(basic_word_mt_features)
    return feature_dict

def calc_seq_mt_features(ql, qr, signature=""):
    bleu_score_l = sentence_bleu([ql], qr, smoothing_function=cc.method3) #NIST smoothing
    bleu_score_r = sentence_bleu([qr], ql, smoothing_function=cc.method3)
    gleu_score_l = sentence_gleu(ql, qr)
    gleu_score_r = sentence_gleu(qr, ql)
    
    try:
        ribes_score_l = sentence_ribes([ql], qr)
    except ZeroDivisionError:
        ribes_score_l = 0
    try:
        ribes_score_r = sentence_ribes([qr], ql)
    except ZeroDivisionError:
        ribes_score_r = 0

    feature_dict = {}

    if signature:
        signature = signature + "_"
    feature_names = ["bleu_score_l", "bleu_score_r", "gleu_score_l", "gleu_score_r",
            "ribes_score_l", "ribes_score_r"]

    for feature_name in feature_names: 
        feature_dict[signature + feature_name] = locals()[feature_name]
    
    return feature_dict



def test():
    ql = ['a', 'b', 'c', 'c', 'd']
    qr = ['b', 'c', 'd']

    print calc_mt_features(ql, qr)


if __name__ == "__main__":
    test()
