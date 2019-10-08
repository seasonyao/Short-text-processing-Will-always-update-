#!/usr/bin/env python
#encoding=utf-8

import sys
import json
import getopt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib
import operator
import xgboost as xgb

import pandas as pd

from wordseg import Wordsegmenter
from sentence import Sentence

from gensim.models import KeyedVectors

import eli5
import eli5.sklearn

from lexical_features import calc_lexical_features
from lexical_features import calc_periph_lexical_features
from lexical_features import get_periph

from mt_features import calc_mt_features

from count_tfidf_hash_features import get_tfidf_count_hash_features
from count_tfidf_hash_features import VectorModels

from sentvec_features import get_sentvec_features
from sentvec_features import load_vocab

from named_entities_features import get_ner_features
from named_entities_features import load_ner_dict

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

def extract_features(wordseg, ql, qr, tfidf_count_hash_vectorModels, sent_word2vec, sent_vocab_dict, sent_model, ner_dict, syn_dict):
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
    
    l_periph = get_periph(sent_l, qr)
    r_periph = get_periph(sent_r, ql)
    #----------------------------------------------------------------------------------------------------------
    
    lexical_features = calc_lexical_features(sent_l, sent_r)
    feature_dict.update(lexical_features)
    periph_lexical_features = calc_periph_lexical_features(l_periph, r_periph)
    feature_dict.update(periph_lexical_features)

    mt_features = calc_mt_features(sent_l, sent_r)
    feature_dict.update(mt_features)
    
    count_tfidf_hash_features = get_tfidf_count_hash_features(sent_l, sent_r, tfidf_count_hash_vectorModels)
    feature_dict.update(count_tfidf_hash_features)
    notion_count_tfidf_hash_features = get_tfidf_count_hash_features(l_notion, r_notion, tfidf_count_hash_vectorModels, signature="notion")
    feature_dict.update(notion_count_tfidf_hash_features)
    
    sentvec_features = get_sentvec_features(sent_word2vec, sent_vocab_dict, sent_model, sent_l, sent_r)
    feature_dict.update(sentvec_features)
    sentvec_features = get_sentvec_features(sent_word2vec, sent_vocab_dict, sent_model, l_notion, r_notion, signature="notion")
    feature_dict.update(sentvec_features)
    
    ner_features = get_ner_features(sent_l, sent_r, ner_dict, syn_dict)
    feature_dict.update(ner_features)

    return feature_dict

def extract():
    wordseg = Wordsegmenter("./bin/pyseg.so", "./bin/qsegconf.ini")
    
    sent_word2vec_path = "./data/word2vec.query.bin"
    sent_vocab_path = "./data/word2vec.query.vocab"
    sent_model_path = "./data/sif.model"
    
    sent_word2vec = KeyedVectors.load_word2vec_format(sent_word2vec_path, binary=True)
    sent_vocab_dict = load_vocab(sent_vocab_path)
    sent_model = joblib.load(sent_model_path)
    
    tfidf_count_hash_vectorModels = VectorModels()
    
    ner_dict_path = "./data/ner.dict"
    syn_dict_path = "./data/syn.dict"
    ner_dict, syn_dict = load_ner_dict(ner_dict_path, syn_dict_path)
    
    for line in sys.stdin:
        line = line.strip("\r\n")
        parts = line.split("\t")
        ql = parts[1]
        qr = parts[2]
        feature_dict = extract_features(wordseg, ql, qr, tfidf_count_hash_vectorModels, sent_word2vec, sent_vocab_dict, sent_model, ner_dict, syn_dict)

        print "{}\t{}".format(line, json.dumps(feature_dict))

def train_svm(in_path, model_path):
    data = [line.strip("\r\n").split("\t") for line in open(in_path)]
    data = pd.DataFrame(data, columns=["qid", "ql", "qr", "label", "features"])
    #data = pd.read_csv(in_path, sep="\t", dtype="str", names=["qid", "ql", "qr", "label", "features"])
    X_features = data["features"].apply(lambda o: json.loads(o)).apply(pd.Series)

    feature_names = X_features.columns.values.tolist()
    X = pd.concat([data[["qid", "ql", "qr"]], X_features], axis=1)
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model = SVC(kernel="linear",probability=True)
    X_train_features = X_train[feature_names]
    model.fit(X_train_features, y_train)

    joblib.dump(model, model_path)

    X_test_features = X_test[feature_names]
    y_preds = model.predict(X_test_features)

    mean_f1 = f1_score(y_test, y_preds, average="micro")
    print mean_f1
    print classification_report(y_test, y_preds, target_names=["paraphrase", "other"])
    
    feature_importance_list = sorted(zip(feature_names, model.coef_.ravel()), key=lambda o:o[1], reverse=True)
    for feature_name, coef in feature_importance_list:
        print "%s\t%f"% (feature_name, coef)
    X_test.loc[:,"label"] = y_test
    X_test.loc[:,"predict"] = y_preds
    X_test.to_csv("test.predict", header=True, sep="\t", mode="w")

def display_scores(scores): 
    print "Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores))

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print "Model with rank: {0}".format(i)
            print "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results["mean_test_score"][candidate],
                  results["std_test_score"][candidate])
            print "Parameters: {0}".format(results["params"][candidate])

def train_xgb(in_path, model_path):
    data = [line.strip("\r\n").split("\t") for line in open(in_path)]
    data = pd.DataFrame(data, columns=["qid", "ql", "qr", "label", "features"])
    #data = pd.read_csv(in_path, sep="\t", dtype="str", names=["qid", "ql", "qr", "label", "features"])
    X_features = data["features"].apply(lambda o: json.loads(o)).apply(pd.Series)
    feature_names = X_features.columns.values.tolist()
    feature_names.sort()
    X_features = X_features[feature_names]
    X = pd.concat([data[["qid", "ql", "qr"]], X_features], axis=1)
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    ##参数
    params={
        "booster":"gbtree",
        "n_estimators": 1000,
        "objective":"binary:logistic",
        "eval_metric": "auc", 
        "eta": 0.007, # 如同学习率
        "min_child_weight":3, 
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        "max_depth":6, # 构建树的深度，越大越容易过拟合
        "gamma":0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
        "subsample":0.7, # 随机采样训练样本
        "colsample_bytree":0.7, # 生成树时进行的列采样 
        "lambda":2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        #"alpha":0, # L1 正则项参数
        #"scale_pos_weight":1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。
        #"num_class":10, # 类别数，多分类与 multisoftmax 并用
        "seed": 42, #随机种子
        "silent":0 ,
        "n_jobs":12,
    }
    num_rounds = 100 # 迭代次数
    print "\t".join(feature_names)
    X_train_features = X_train[feature_names]
    X_test_features = X_test[feature_names]

    #训练模型并保存
    # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_features, y_train, early_stopping_rounds=10, eval_set=[(X_test_features, y_test)], verbose=True)
    #model.save_model("./model/xgb.model") # 用于存储训练出的模型

    print "best best_ntree_limit",model.best_ntree_limit 

    joblib.dump(model, model_path)
    y_preds = model.predict(X_test_features, ntree_limit=model.best_ntree_limit)

    mean_f1 = f1_score(y_test, y_preds, average="micro")
    print mean_f1
    print classification_report(y_test, y_preds, target_names=["paraphrase", "other"])

    feature_importance = list(zip(feature_names, model.feature_importances_))
    feature_importance = sorted(feature_importance, key=operator.itemgetter(1), reverse=True)
    for k, v in feature_importance:
        print "%s\t%.6f" % (k, v)

    #X_test.loc[:,"label"] = y_test
    #X_test.loc[:,"predict"] = y_preds
    #X_test.to_csv("test.predict", header=True, sep="\t", mode="w")

def predict(model_path):
    wordseg = Wordsegmenter("./bin/pyseg.so", "./bin/qsegconf.ini")
    sent_word2vec_path = "./data/word2vec.query.bin"
    sent_vocab_path = "./data/word2vec.query.vocab"
    sent_model_path = "./data/sif.model"
    
    sent_word2vec = KeyedVectors.load_word2vec_format(sent_word2vec_path, binary=True)
    sent_vocab_dict = load_vocab(sent_vocab_path)
    sent_model = joblib.load(sent_model_path)
    
    tfidf_count_hash_vectorModels = VectorModels()
    
    ner_dict_path = "./data/ner.dict"
    syn_dict_path = "./data/syn.dict"
    ner_dict, syn_dict = load_ner_dict(ner_dict_path, syn_dict_path)

    ner_dict_path = "./data/ner.dict"
    syn_dict_path = "./data/syn.dict"
    ner_dict, syn_dict = load_ner_dict(ner_dict_path, syn_dict_path)
    
    model = joblib.load(model_path)

    feature_names = [] 
    column_names = ["qid", "ql", "qr"]
    #reader = pd.read_csv(in_path, sep="\t", dtype="str", names=column_names, chunksize=100)
    reader = pd.read_csv(sys.stdin, sep="\t", dtype="str", names=column_names, chunksize=100)
    first_chunk = True

    feature_extractor = lambda row: extract_features(wordseg, row["ql"], row["qr"], tfidf_count_hash_vectorModels, sent_word2vec, sent_vocab_dict, sent_model, ner_dict, syn_dict)
    for data in reader:
        _ = data.fillna("", inplace=True)

        X = data[["ql", "qr"]].apply(feature_extractor, axis=1)
        X_features = X.apply(pd.Series)
        feature_names = X_features.columns.values.tolist()
        feature_names.sort()
        X_features = X_features[feature_names]
        y_preds = model.predict_proba(X_features, ntree_limit=model.best_ntree_limit)
        y_preds = map(lambda o:o[1], y_preds)
        data = pd.concat([data, X_features], axis=1)
        data = data.assign(predict=y_preds)
        #if first_chunk:
        #    data.to_csv(in_path + ".predict", header=True, sep="\t", mode="w")
        #    first_chunk = False
        #else:
        #    data.to_csv(in_path + ".predict", header=False, sep="\t", mode="a")
        data.to_csv(sys.stdout, header=False, sep="\t")

def explain(model_path):
    wordseg = Wordsegmenter("./bin/pyseg.so", "./bin/qsegconf.ini")
    sent_word2vec_path = "./data/word2vec.query.bin"
    sent_vocab_path = "./data/word2vec.query.vocab"
    sent_model_path = "./data/sif.model"
    
    sent_word2vec = KeyedVectors.load_word2vec_format(sent_word2vec_path, binary=True)
    sent_vocab_dict = load_vocab(sent_vocab_path)
    sent_model = joblib.load(sent_model_path)
    
    tfidf_count_hash_vectorModels = VectorModels()

    ner_dict_path = "./data/ner.dict"
    syn_dict_path = "./data/syn.dict"
    ner_dict, syn_dict = load_ner_dict(ner_dict_path, syn_dict_path)

    model = joblib.load(model_path)

    pd.set_option('display.max_rows',None)

    explain = eli5.explain_weights(model,top=None)
    explain = eli5.format_as_text(explain)
    print explain

    feature_names = [] 
    column_names = ["qid", "ql", "qr"]
    #reader = pd.read_csv(in_path, sep="\t", dtype="str", names=column_names, chunksize=100)
    reader = pd.read_csv(sys.stdin, sep="\t", dtype="str", names=column_names, chunksize=1)
    first_chunk = True
    feature_extractor = lambda row: extract_features(wordseg, row["ql"], row["qr"], tfidf_count_hash_vectorModels, sent_word2vec, sent_vocab_dict, sent_model, ner_dict, syn_dict)
    for data in reader:
        _ = data.fillna("", inplace=True)

        X = data[["ql", "qr"]].apply(feature_extractor, axis=1)
        X_features = X.apply(pd.Series)
        feature_names = X_features.columns.values.tolist()
        X_features = X_features[feature_names]
        y_preds = model.predict_proba(X_features, ntree_limit=model.best_ntree_limit)
        y_preds = map(lambda o:o[1], y_preds)
        data = pd.concat([data, X_features], axis=1)
        data = data.assign(predict=y_preds)
        
        #if first_chunk:
        #    data.to_csv(in_path + ".predict", header=True, sep="\t", mode="w")
        #    first_chunk = False
        #else:
        #    data.to_csv(in_path + ".predict", header=False, sep="\t", mode="a")
        data.to_csv(sys.stdout, header=False, sep="\t")
        explain = eli5.explain_prediction(model,X_features.iloc[0])
        explain = eli5.format_as_text(explain)
        print explain
        print X_features.iloc[0]


def main():
    action = None
    in_path = ""
    model_path = ""
    opts, args = getopt.getopt(sys.argv[1:], "a:", ["in_path=", "model_path="])
    for op, value in opts:
        if op == "-a":
            action = value
        if op == "--in_path":
            in_path = value
        if op == "--model_path":
            model_path = value
    
    if action == "extract":
        extract()
    if action == "train_xgb":
        if in_path and model_path:
            train_xgb(in_path, model_path)
    if action == "predict":
        if model_path:
            predict(model_path)

    if action == "explain":
        if model_path:
            explain(model_path)

if __name__ == "__main__":
    main()
