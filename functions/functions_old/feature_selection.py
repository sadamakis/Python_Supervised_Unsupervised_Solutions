"""
# This is a set of machine learning tools developed using Python
"""

import numpy as np
import pandas as pd
from evaluation import gini
from evaluation import gini_weight
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from decorators import time_function


class SelectBest(object):
    def __init__(self, df, target):
                    
                    '''
                    expect input as following:
                                    df: array_like 
                                                    dataframe used for gini calculation
                                    target: string
                                                    target variable
                    '''
                    
                    self.df = df
                    self.target = target
                    
    @time_function
    def best_univar_gini(self, feats, n=1):
                    '''
                    input:
                                    class initial input df, feats, target, top 
                    output: list
                                    top features with descending absolute gini value.  
                    '''    
                    feat_gini = dict()
                    for x in feats:
                                    feat_gini[x] = gini(self.df[[x, self.target]].values)
                    rank = sorted(feat_gini.items(), key=lambda t: abs(t[1]), reverse=True)
                    return [x[0] for x in rank][:n]
    
    @time_function
    def top_rf_feat(self, feats, model=RandomForestClassifier(n_estimators=200, max_depth=5,random_state=1234), n=1):
                    '''
                    input:
                                    1. class initial input df, feats, target, top 
                                    2. model: 
                                                                    random forest model with pre defined hyperparameters.     
                                                                    
                    output: list
                                    top features with descending variable importance in random forest model.  
                    '''   
                    model.fit(self.df[feats], self.df[self.target])
                    feat_importance = dict(zip(feats, model.feature_importances_)) 
                    rank = sorted(feat_importance.items(), key=lambda t: abs(t[1]), reverse=True)
                    return [x[0] for x in rank][:n]
                    
    @time_function
    def top_gbm_feat(self, feats, model=GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=1234), n=1):
                    '''
                    input:
                                    1. class initial input df, feats, target, top 
                                    2. model: 
                                                                    GBM model with pre defined hyperparameters.     
                                                                    
                    output: list
                                    top features with descending variable importance in GBM model.  
                    '''   
                    model.fit(self.df[feats], self.df[self.target])
                    feat_importance = dict(zip(feats, model.feature_importances_))  
                    rank = sorted(feat_importance.items(), key=lambda t: abs(t[1]), reverse=True)
                    return [x[0] for x in rank][:n]
    
    @time_function
    def top_lgbm_feat(self, feats, model=LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=1234, n_jobs=6), n=1):
                    '''
                    input:
                                    1. class initial input df, feats, target, top 
                                    2. model: 
                                                                    lightGBM model with pre defined hyperparameters.     
                                                                    
                    output: list
                                    top features with descending variable importance in GBM model.  
                    '''   
                    model.fit(self.df[feats], self.df[self.target])
                    feat_importance = dict(zip(feats, model.feature_importances_))  
                    rank = sorted(feat_importance.items(), key=lambda t: abs(t[1]), reverse=True)
                    return rank, [x[0] for x in rank][:n]
    
    @time_function
    def top_svc_feat(self, feats, model=LinearSVC(C=0.01, penalty="l1", dual=False,random_state=42), n=1):
                    '''
                    input:
                                    1. class initial input dev, feats, target, top 
                       
                                    2. model: 
                                                    support vector classification with pre defined hyperparameters. 
                                                    
                    output: list
                                    top features with descending absolute coefficient in SVC model.  
                    '''    
                    model.fit(self.df[feats], self.df[self.target])
                    feat_coef = dict()
                    fc=model.coef_[0]
    
                    for i,x in enumerate(feats):
                                    feat_coef[x] =fc[i]
                                    
                    rank = sorted(feat_coef.items(), key=lambda t: abs(t[1]), reverse=True)
                    return [x[0] for x in rank][:n]
    
    @time_function
    def top_lr_feat(self, feats, model=LogisticRegression(C=0.01, penalty="l1",random_state=42), n=1):
                    '''
                    input:
                                    1. class initial input dev, feats, target, top 
                                    2. model: 
                                                    logistic regression model with pre defined hyperparameters.      
                                                    
                    output: list
                                    top features with descending absolute coefficient in logistic regression model.  
                    '''
                    
                    model.fit(self.df[feats], self.df[self.target])
                    feat_coef = dict()
                    fc = model.coef_[0]
                    
                    for i,x in enumerate(feats):
                                    feat_coef[x] =fc[i]
                                    
                    rank = sorted(feat_coef.items(), key=lambda t: abs(t[1]), reverse=True)
                    return [x[0] for x in rank][:n]
    
    @time_function
    def get_best(self, remaining_feats,oos, model, classification):
                    best_feat, best_gini = " ", 0
                    for v in remaining_feats:
                                    left = remaining_feats[:]
                                    left.remove(v)
                                    model.fit(self.df[left], self.df[self.target])
                                    if classification==True:
                                                    oos['score']=model.predict_proba(oos[left].values)[:, 1]
                                    else:
                                                    oos['score']=model.predict(oos[left].values)
                                    gini_v = gini(oos[['score', self.target]].values)
                                    if gini_v > best_gini:
                                                    best_gini = gini_v
                                                    best_feat = v
                    return best_feat, best_gini
                    
    @time_function
    def backward_recur(self, feats, oos, model, min_feats=5, classification=True):
                    '''
                    input:
                                    1. class initial input dev, feats, target, top 
                                    
                                    2. oos: array_like
                                                                    cross validation dataset
                                    3. model: 
                                                                    model used for backward selection. eg. logistic regression or random forest
                                    4. min_feats: int
                                                                    minimum number of features to keep
                                    5. classification: Boolean (True or False)
                                                                    if a model is a classification model or not.
                                                                    
                    output: list
                                    remaining features after backward selection. 
                    '''
                    keep = feats[:]
                    best_gini =  0
                    
                    for i in range(len(feats)-min_feats):
                                    remove_feat, gini_i = self.get_best(keep, oos, model,classification)
                                    
                                    if (gini_i <= best_gini) or (len(keep)<=2):
                                                    return keep
                                    else:
                                                    print('step i =', i+1, 'feature removed:', remove_feat, 'gini:',gini_i)
                                                    keep.remove(remove_feat)
                    return keep

# COPIED
class SelectBest_weight(object):
    def __init__(self, df, target, weight):
                    
                    '''
                    expect input as following:
                                    df: array_like 
                                                    dataframe used for gini calculation
                                    target: string
                                                    target variable
                                    weight: string
                                                    weight variable
                    '''
                    
                    self.df = df
                    self.target = target
                    self.weight = weight
                    
    @time_function
    def best_univar_gini(self, feats, n=1):
                    '''
                    input:
                                    class initial input df, feats, target, top 
                    output: list
                                    top features with descending absolute gini value.  
                    '''    
                    feat_gini = dict()
                    for x in feats:
#                        feat_gini[x] = gini_weight(self.df[[x, self.target, self.weight]].values)
                        feat_gini[x] = abs(2*roc_auc_score(self.df[self.target].values, self.df[x].values, sample_weight=self.df[self.weight].values)-1)
                    rank = sorted(feat_gini.items(), key=lambda t: abs(t[1]), reverse=True)
                    return rank, [x[0] for x in rank][:n]
    
    @time_function
    def top_rf_feat(self, feats, model=RandomForestClassifier(n_estimators=200, max_depth=5,random_state=1234), n=1):
                    '''
                    input:
                                    1. class initial input df, feats, target, top 
                                    2. model: 
                                                                    random forest model with pre defined hyperparameters.     
                                                                    
                    output: list
                                    top features with descending variable importance in random forest model.  
                    '''   
                    model.fit(self.df[feats], self.df[self.target], sample_weight=self.df[self.weight])
                    feat_importance = dict(zip(feats, model.feature_importances_)) 
                    rank = sorted(feat_importance.items(), key=lambda t: abs(t[1]), reverse=True)
                    return rank, [x[0] for x in rank][:n]
                    
    @time_function
    def top_gbm_feat(self, feats, model=GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=1234), n=1):
                    '''
                    input:
                                    1. class initial input df, feats, target, top 
                                    2. model: 
                                                                    GBM model with pre defined hyperparameters.     
                                                                    
                    output: list
                                    top features with descending variable importance in GBM model.  
                    '''   
                    model.fit(self.df[feats], self.df[self.target], sample_weight=self.df[self.weight])
                    feat_importance = dict(zip(feats, model.feature_importances_))  
                    rank = sorted(feat_importance.items(), key=lambda t: abs(t[1]), reverse=True)
                    return rank, [x[0] for x in rank][:n]
    
    @time_function
    def top_lgbm_feat(self, feats, model=LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=1234, n_jobs=6), n=1):
                    '''
                    input:
                                    1. class initial input df, feats, target, top 
                                    2. model: 
                                                                    lightGBM model with pre defined hyperparameters.     
                                                                    
                    output: list
                                    top features with descending variable importance in lightGBM model.  
                    '''   
                    model.fit(self.df[feats], self.df[self.target], sample_weight=self.df[self.weight])
                    feat_importance = dict(zip(feats, model.feature_importances_))  
                    rank = sorted(feat_importance.items(), key=lambda t: abs(t[1]), reverse=True)
                    return rank, [x[0] for x in rank][:n]
    
    @time_function
    def top_svc_feat(self, feats, model=LinearSVC(C=0.01, penalty="l1", dual=False,random_state=42), n=1):
                    '''
                    input:
                                    1. class initial input dev, feats, target, top 
                       
                                    2. model: 
                                                    support vector classification with pre defined hyperparameters. 
                                                    
                    output: list
                                    top features with descending absolute coefficient in SVC model.  
                    '''    
                    model.fit(self.df[feats], self.df[self.target], sample_weight=self.df[self.weight])
                    feat_coef = dict()
                    fc=model.coef_[0]
    
                    for i,x in enumerate(feats):
                                    feat_coef[x] =fc[i]
                                    
                    rank = sorted(feat_coef.items(), key=lambda t: abs(t[1]), reverse=True)
                    return rank, [x[0] for x in rank][:n]
    
    @time_function
    def top_lr_feat(self, feats, model=LogisticRegression(C=0.01, penalty="l1",random_state=42), n=1):
                    '''
                    input:
                                    1. class initial input dev, feats, target, top 
                                    2. model: 
                                                    logistic regression model with pre defined hyperparameters.      
                                                    
                    output: list
                                    top features with descending absolute coefficient in logistic regression model.  
                    '''
                    
                    model.fit(self.df[feats], self.df[self.target], sample_weight=self.df[self.weight])
                    feat_coef = dict()
                    fc = model.coef_[0]
                    
                    for i,x in enumerate(feats):
                                    feat_coef[x] =fc[i]
                                    
                    rank = sorted(feat_coef.items(), key=lambda t: abs(t[1]), reverse=True)
                    return rank, [x[0] for x in rank][:n]
    
    @time_function
    def get_best(self, remaining_feats,oos, model, classification):
                    best_feat, best_gini = " ", 0
                    for v in remaining_feats:
                                    left = remaining_feats[:]
                                    left.remove(v)
                                    model.fit(self.df[left], self.df[self.target], sample_weight=self.df[self.weight])
                                    if classification==True:
                                                    oos['score']=model.predict_proba(oos[left].values)[:, 1]
                                    else:
                                                    oos['score']=model.predict(oos[left].values)
#                                    gini_v = gini_weight(oos[['score', self.target, self.weight]].values)
                                    gini_v = abs(2*roc_auc_score(oos[self.target].values, oos['score'].values, sample_weight=oos[self.weight].values)-1)
                                    if gini_v > best_gini:
                                                    best_gini = gini_v
                                                    best_feat = v
                    return best_feat, best_gini

    @time_function
    def backward_recur(self, feats, oos, model, min_feats=5, classification=True):
                    '''
                    input:
                                    1. class initial input dev, feats, target, top 
                                    
                                    2. oos: array_like
                                                                    cross validation dataset
                                    3. model: 
                                                                    model used for backward selection. eg. logistic regression or random forest
                                    4. min_feats: int
                                                                    minimum number of features to keep
                                    5. classification: Boolean (True or False)
                                                                    if a model is a classification model or not.
                                                                    
                    output: list
                                    remaining features after backward selection. 
                    '''
                    keep = feats[:]
                    best_gini =  0
                    
                    for i in range(len(feats)-min_feats):
                                    remove_feat, gini_i = self.get_best(keep, oos, model,classification)
                                    
                                    if (gini_i <= best_gini) or (len(keep)<=2):
                                                    return keep
                                    else:
                                                    print('step i =', i+1, 'feature removed:', remove_feat, 'gini:',gini_i)
                                                    keep.remove(remove_feat)
                    return keep

