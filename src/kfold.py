import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import xgboost as xgb
from scipy import interp
import pandas as pd
import time
import numpy as np
import gc

import sys
import wordbatch
from wordbatch.extractors import WordHash
from wordbatch.models import FM_FTRL
from wordbatch.data_utils import *
import threading
from contextlib import contextmanager
    
    
def kfold_train(model, k, train, predictors, target ,categorical, params, logger, name, i_start=-1):
    kf = StratifiedKFold(n_splits=k, shuffle = True, random_state = 47)
            
    i=0
    for train_index, test_index in kf.split(train[predictors], train[target]):
        if i <= i_start:
            i += 1
            continue
        logger.print_log('cv_%d start' % (i))
        model(train.iloc[train_index], train.iloc[test_index], predictors, 
              target, categorical, params, logger, name + '_cv_' + str(i))
        
        logger.print_log('cv_%d finished.' % (i))
        gc.collect()
        i += 1
    logger.print_log('done.') 

    
def kfold_lgb_train_predict(model_files, k, train, predictors, target, logger, name):
    kf = StratifiedKFold(n_splits=k, shuffle = True, random_state = 47)
    scores = []
    predictions = np.zeros((train.shape[0], 1))
            
    i=0
    for train_index, test_index in kf.split(train[predictors], train[target]):
        model = lgb.Booster(model_file=model_files[i])
        predictions[test_index, 0] = model.predict(train[predictors].iloc[test_index])
        score = roc_auc_score(train[target].iloc[test_index].astype(float), predictions[test_index, 0])
        scores.append(score)
        logger.print_log('cv_%d scores: %f' % (i, score))
        i += 1
        gc.collect()

    logger.print_log('mean scores: %f' % np.mean(scores)) 
    sub = pd.DataFrame()
    sub['id'] = train[predictors[1]]
    sub['predictions'] = predictions
    del sub['id']
    output_file = '../output/' + name + '_train.csv'
    logger.print_log("writing to file <" + output_file + ">...")
    sub.to_csv(output_file, index=False)
    logger.print_log("done.")
    
def kfold_xgb_train_predict(model_files, k, train, predictors, target, logger, name):
    kf = StratifiedKFold(n_splits=k, shuffle = True, random_state = 47)
    scores = []
    predictions = np.zeros((train.shape[0], 1))
            
    i=0
    for train_index, test_index in kf.split(train[predictors], train[target]):
        model = xgb.Booster(model_file=model_files[i])
        with open(model_files[i][:-4] + '_best_it.txt', 'r+') as f:  
            n_estimators = int(f.read())
        dtrain = xgb.DMatrix(train[predictors].iloc[test_index])
        predictions[test_index, 0] = model.predict(dtrain, ntree_limit=n_estimators + 1)
        score = roc_auc_score(train[target].iloc[test_index].astype(float), predictions[test_index, 0])
        scores.append(score)
        logger.print_log('cv_%d scores: %f' % (i, score))
        i += 1
        gc.collect()

    logger.print_log('mean scores: %f' % np.mean(scores)) 
    sub = pd.DataFrame()
    sub['id'] = train[predictors[1]]
    sub['predictions'] = predictions
    del sub['id']
    output_file = '../output/' + name + '_train.csv'
    logger.print_log("writing to file <" + output_file + ">...")
    sub.to_csv(output_file, index=False)
    logger.print_log("done.")

def kfold_rf_train_predict(model_files, k, train, predictors, target, logger, name):
    kf = StratifiedKFold(n_splits=k, shuffle = True, random_state = 47)
    scores = []
    predictions = np.zeros((train.shape[0], 1))
            
    i=0
    for train_index, test_index in kf.split(train[predictors], train[target]):
        with open('../models/' + model_files[i],'rb+') as f:
            rf = pickle.load(f)
        predictions[test_index, 0] = rf.predict_proba(test_df[predictors].iloc[test_index])[:,1]
        score = roc_auc_score(train[target].iloc[test_index].astype(float), predictions[test_index, 0])
        scores.append(score)
        logger.print_log('cv_%d scores: %f' % (i, score))
        i += 1
        gc.collect()

    logger.print_log('mean scores: %f' % np.mean(scores)) 
    sub = pd.DataFrame()
    sub['id'] = train[predictors[1]]
    sub['predictions'] = predictions
    del sub['id']
    output_file = '../output/' + name + '_train.csv'
    logger.print_log("writing to file <" + output_file + ">...")
    sub.to_csv(output_file, index=False)
    logger.print_log("done.")

    
def kfold_fm_ftrl_train_predict(model_files, k, train, predictors, target, logger, name):
    kf = StratifiedKFold(n_splits=k, shuffle = True, random_state = 47)
    scores = []
    predictions = np.zeros((train.shape[0], 1))
            
    i=0
    for train_index, test_index in kf.split(train.drop(target, 1), train[target]):
        with open('../models/' + model_files[i] + '_clf.txt','rb+') as f:
            clf = pickle.load(f)
        with open('../models/' + model_files[i] + '_wb.txt','rb+') as f:
            wb = pickle.load(f)
        str_array, labels, weights = df2csr(wb, train.drop(target, 1).iloc[test_index])
        X = wb.transform(str_array)
        predictions[test_index, 0] = clf.predict(X)
        score = roc_auc_score(train[target].iloc[test_index].astype(float), predictions[test_index, 0])
        scores.append(score)
        logger.print_log('cv_%d scores: %f' % (i, score))
        i += 1
        gc.collect()
    
    logger.print_log('mean scores: %f' % np.mean(scores)) 
    sub = pd.DataFrame()
    sub['id'] = train[predictors[1]]
    sub['predictions'] = predictions
    del sub['id']
    output_file = '../output/' + name + '_train.csv'
    logger.print_log("writing to file <" + output_file + ">...")
    sub.to_csv(output_file, index=False)
    logger.print_log("done.")

    
def kfold_plot(train, predictors, target, model, k):
    kf = StratifiedKFold(n_splits=k)
    scores = []
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    exe_time = []
    
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue'])
    lw = 2
    
    xtrain = train[predictors]
    ytrain = train[target]
        
    i=0
    for (train_index, test_index), color in zip(kf.split(xtrain, ytrain), colors):
        X_train, X_test = xtrain.iloc[train_index], xtrain.iloc[test_index]
        y_train, y_test = ytrain.iloc[train_index], ytrain.iloc[test_index]
        
        begin_t = time.time()
        predictions = model(train.iloc[train_index], train.iloc[test_index], predictors, target, i)
        end_t = time.time()
        exe_time.append(round(end_t-begin_t, 3))

        score = roc_auc_score(y_test.astype(float), predictions)
        scores.append(score)
        print_log('cv_%d scores: %f' % (i, score)) 

        fpr, tpr, thresholds = roc_curve(y_test, predictions)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')
    
    mean_tpr /= kf.get_n_splits(train, ytrain)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc='lower right')
    plt.show()
    
    print_log('mean scores: %f' % np.mean(scores)) 
    print_log('mean model process time: %f s' % np.mean(exe_time))
    
    return scores, np.mean(scores), np.mean(exe_time)