# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 22:16:12 2019

@author: Shubham
"""

import pandas as pd
from sklearn import preprocessing, metrics
import xgboost as xgb
from time import time
import pickle
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split

def train_model(X_train, y_train, X_test):
    # Label Encoding
    for f in X_train.columns:
        if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(X_train[f].values) + list(X_test[f].values))
            X_train[f] = lbl.transform(list(X_train[f].values))
            X_test[f] = lbl.transform(list(X_test[f].values)) 
    
    # SMOTE with edited nearest neighbour 
    smote_enn = SMOTEENN(random_state=0)
    X_train, y_train = smote_enn.fit_resample(X_train, y_train)
    
    X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2)
    
    print(type(X_train))
    
    print("Start training classfier")
    start = time()
#    clf = xgb.XGBClassifier(n_estimators=500,
#                            n_jobs=4,
#                            max_depth=9, 
#                            learning_rate=0.05,
#                            subsample=0.9,
#                            colsample_bytree=0.9,
#                            missing=-999)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=4,
                                 random_state=0)
    
    clf.fit(X_train, y_train)
    print("Ended classifier training")
    end = time()
    print("Total training time is {} seconds".format(end-start))
    
    # save trained model
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    
    return X_test.as_matrix(), X_val, y_val, clf

def calc_val_acc(clf, X_val, y_val):
    pred = clf.predict(X_val)
    return sum(pred == y_val)/len(pred)

def infer_model(clf, X_test):
    sample_submission = pd.read_csv('../data/sample_submission.csv')
    sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]
    sample_submission.to_csv('../data/simple_xgboost.csv')
    
if __name__ == "__main__":
    train = pd.read_csv("../data/train_merged.csv")    
    test = pd.read_csv("../data/test_merged.csv")
    
    y_train = train['isFraud'].copy()

    # Drop target, fill in NaNs 
    X_train = train.drop('isFraud', axis=1)
    X_test = test.copy()
    X_train = X_train.fillna(-999)
    X_test = X_test.fillna(-999)
    
    X_test_mod, X_val, y_val, clf = train_model(X_train, y_train, X_test)
    infer_model(clf, X_test_mod)
