import lightgbm as lgb
import xgboost as xgb
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle


import pandas as pd
import time
import numpy as np
import gc
import pickle
import default_config

from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

##################### LightGBM #######################

def lightgbm_model(train_df, val_df, predictors, target, categorical, params, logger, name):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':'auc',
        'learning_rate': 0.01,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0
    }
    
    lgb_params.update(params)
    
    if logger:
        logger.print_log("preparing validation datasets")

    xgtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    xgvalid = lgb.Dataset(val_df[predictors].values, label=val_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    evals_results = {}
    
    if logger:
        logger.print_log("training...")
    bst = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgvalid], 
                     valid_names=['valid'], 
                     evals_result=evals_results, 
                     num_boost_round=1000,
                     early_stopping_rounds=30,
                     verbose_eval=10, 
                     feval=None)
    
    n_estimators = bst.best_iteration
    bst.save_model('../models/' + str(name) + '.txt', num_iteration=bst.best_iteration)
    if logger:
        logger.print_log("Model Report")
        logger.print_log("n_estimators : %d" % n_estimators)
        logger.print_log("auc: %f" % evals_results['valid']['auc'][n_estimators-1])
    
    
    lgb.plot_importance(bst)
    plt.gcf().savefig('../models/' + str(name) + '_feature_importance.png')
    
    predictions = bst.predict(val_df[predictors], num_iteration=bst.best_iteration)
    return predictions

def lgb_predict(logger, model_file, test_file, output_file, predictors):
    logger.print_log('loading test data...')
    test_df = pd.read_csv(test_file , dtype=default_config.dtypes)
    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')
    
    logger.print_log("Predicting...")
    model = lgb.Booster(model_file=model_file)
    sub['is_attributed'] = model.predict(test_df[predictors])
    logger.print_log("writing to file <" + output_file + ">...")
    if output_file[-2:] == "gz":
        sub.to_csv(output_file, index=False, compression='gzip')
    else :
        sub.to_csv(output_file, index=False)
    logger.print_log("done.")

def lgb_predict_test(logger, model_file, test_df, output_file, predictors):
    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')
    
    logger.print_log("Predicting...")
    model = lgb.Booster(model_file=model_file)
    sub['is_attributed'] = model.predict(test_df[predictors])
    logger.print_log("writing to file <" + output_file + ">...")
    sub.to_csv(output_file, index=False, compression='gzip')
    logger.print_log("done.")

    
def lgb_cv_predict(logger, model_files, test_file, output_file, predictors):
    logger.print_log('loading test data...')
    test_df = pd.read_csv(test_file , dtype=default_config.dtypes)
    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')
    
    logger.print_log("Predicting...")
    sub['is_attributed'] = 0
    for model_file in model_files:
        model = lgb.Booster(model_file=model_file)
        sub['is_attributed'] += model.predict(test_df[predictors])
    sub['is_attributed'] = sub['is_attributed'] / len(model_files)
    logger.print_log("writing to file <" + output_file + ">...")
    sub.to_csv(output_file, index=False)
    logger.print_log("done.")    

##################### XGBoost #######################
def xgboost_model(train_df, val_df, predictors, target, categorical, params, logger, name):
    logger.print_log("preparing validation datasets")
    dtrain = xgb.DMatrix(train_df[predictors], train_df[target])
    dvalid = xgb.DMatrix(val_df[predictors], val_df[target])
    del train_df, val_df
    gc.collect()
    logger.print_log("training...")
    bst = xgb.train(params, dtrain, 1000, [(dvalid, 'valid')], maximize=True, early_stopping_rounds = 30, verbose_eval=10)

    bst.save_model('../models/' + str(name) + '.txt')
    
    xgb.plot_importance(bst)
    plt.gcf().savefig('../models/' + str(name) + '_feature_importance.png')
    
    n_estimators = bst.best_iteration

    with open('../models/' + str(name) + '_best_it.txt', 'w+') as f:  
        f.write(str(n_estimators))
    
    if logger:
        logger.print_log("Model Report")
        logger.print_log("n_estimators : %d" % n_estimators)
        logger.print_log("auc : %f" % bst.best_score)
    
#    dtest = xgb.DMatrix(val_df[predictors])
#    predictions = bst.predict(dtest, ntree_limit=n_estimators + 1)
#    return predictions

def xgb_predict(logger, model_file, test_file, output_file, predictors):
    logger.print_log('loading test data...')
    test_df = pd.read_csv(test_file , dtype=default_config.dtypes)
    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')
    
    logger.print_log("Predicting...")
    model = xgb.Booster(model_file=model_file)
    with open(model_file[:-4] + '_best_it.txt', 'r+') as f:  
        n_estimators = int(f.read())
    dtest = xgb.DMatrix(test_df[predictors])
    sub['is_attributed'] = model.predict(dtest, ntree_limit=n_estimators + 1)
    logger.print_log("writing to file <" + output_file + ">...")
    if output_file[-2:] == "gz":
        sub.to_csv(output_file, index=False, compression='gzip')
    else :
        sub.to_csv(output_file, index=False)
    logger.print_log("done.")

def xgb_cv_predict(logger, model_files, test_file, output_file, predictors):
    logger.print_log('loading test data...')
    test_df = pd.read_csv(test_file , dtype=default_config.dtypes)
    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')
    
    logger.print_log("Predicting...")
    sub['is_attributed'] = 0
    for model_file in model_files:
        model = xgb.Booster(model_file=model_file)
        dtest = xgb.DMatrix(test_df[predictors])
        with open(model_file[:-4] + '_best_it.txt', 'r+') as f:  
            n_estimators = int(f.read())
        sub['is_attributed'] += model.predict(dtest, ntree_limit=n_estimators + 1)
    sub['is_attributed'] = sub['is_attributed'] / len(model_files)
    logger.print_log("writing to file <" + output_file + ">...")
    sub.to_csv(output_file, index=False)
    logger.print_log("done.") 

    
##################### RF #######################
def rf_model(train_df, val_df, predictors, target, categorical, params, logger, name):
    train_df = train_df.fillna(0)
    val_df = val_df.fillna(0)
    rf = RandomForestClassifier(n_jobs=params['n_jobs'],
                                oob_score=params['oob_score'],
                                n_estimators=params['n_estimators'], 
                                max_depth=params['max_depth'], 
                                random_state=params['seed'],
                                verbose=2)
    rf.fit(train_df[predictors], train_df[target])
    
    with open('../models/' + str(name) + '.txt','wb+') as f:
        pickle.dump(rf, f)
        
    predictions = rf.predict_proba(val_df[predictors])[:,1]
    auc = roc_auc_score(val_df[target], predictions)
    logger.print_log('val auc: %f' % auc)
    
    importances = rf.feature_importances_
 
    logger.print_log ("Sorted Feature Importance:")
    sorted_feature_importance = sorted(zip(importances, list(train_df[predictors])), reverse=True)
    for item in sorted_feature_importance :
        logger.print_log("%6f, %s" % (item[0], str(item[1])))
    return predictions

def rf_predict(logger, model_file, test_file, output_file, predictors):
    logger.print_log('loading test data...')
    test_df = pd.read_csv(test_file, dtype=default_config.dtypes)
    test_df = test_df.fillna(0)
    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')
    
    logger.print_log("Predicting...")
    with open('../models/' + model_file,'rb+') as f:
        rf = pickle.load(f)
    sub['is_attributed'] = rf.predict_proba(test_df[predictors])[:,1]
    logger.print_log("writing to file <" + output_file + ">...")
    if output_file[-2:] == "gz":
        sub.to_csv(output_file, index=False, compression='gzip')
    else :
        sub.to_csv(output_file, index=False)
    logger.print_log("done.")

def rf_cv_predict(logger, model_files, test_file, output_file, predictors):
    logger.print_log('loading test data...')
    test_df = pd.read_csv(test_file , dtype=default_config.dtypes)
    test_df = test_df.fillna(0)
    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')
    
    logger.print_log("Predicting...")
    sub['is_attributed'] = 0
    for model_file in model_files:
        with open('../models/' + model_file + '_clf.txt','rb+') as f:
            rf = pickle.load(f)

        sub['is_attributed'] += rf.predict_proba(test_df[predictors])[:,1]
    sub['is_attributed'] = sub['is_attributed'] / len(model_files)
    logger.print_log("writing to file <" + output_file + ">...")
    sub.to_csv(output_file, index=False)
    logger.print_log("done.") 
    
##################### FM-FTRL #######################
import sys
import wordbatch
from wordbatch.extractors import WordHash
from wordbatch.models import FM_FTRL
from wordbatch.data_utils import *
import threading
import pandas as pd
from sklearn.metrics import roc_auc_score
import time
import numpy as np
import gc
from contextlib import contextmanager

import os, psutil
def cpuStats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30
    print('memory GB:', memoryUse)

def fit_batch(clf, X, y, w):  clf.partial_fit(X, y, sample_weight=w)

def predict_batch(clf, X):  return clf.predict(X)

def evaluate_batch(clf, X, y, rcount):
    auc= roc_auc_score(y, predict_batch(clf, X))
    print(rcount, "ROC AUC:", auc)
    return auc

def df_add_counts(df, cols, tag="_count"):
    arr_slice = df[cols].values
    unq, unqtags, counts = np.unique(np.ravel_multi_index(arr_slice.T, arr_slice.max(0) + 1),
                                     return_inverse=True, return_counts=True)
    df["_".join(cols)+tag] = counts[unqtags]
    return df

def df_add_uniques(df, cols, tag="_unique"):
    gp = df[cols].groupby(by=cols[0:len(cols) - 1])[cols[len(cols) - 1]].nunique().reset_index(). \
        rename(index=str, columns={cols[len(cols) - 1]: "_".join(cols)+tag})
    df= df.merge(gp, on=cols[0:len(cols) - 1], how='left')
    return df

def df2csr(wb, df, pick_hours=None):
    df.reset_index(drop=True, inplace=True)
    df['click_time']= pd.to_datetime(df['click_time'])
    dt= df['click_time'].dt
    df['day'] = dt.day.astype('uint8')
    df['hour'] = dt.hour.astype('uint8')
    del(dt)
    df= df_add_counts(df, ['ip', 'day', 'hour'])
    df= df_add_counts(df, ['ip', 'app'])
    df= df_add_counts(df, ['ip', 'app', 'os'])
    df= df_add_counts(df, ['ip', 'device'])
    df= df_add_counts(df, ['app', 'channel'])
    df= df_add_uniques(df, ['ip', 'channel'])

    D= 2**26
    df['category'] = (df['ip'].astype(str) + "_" + df['app'].astype(str) + "_" + df['device'].astype(str) \
                     + "_" + df['os'].astype(str)).apply(hash) % D
    click_buffer= np.full(D, 3000000000, dtype=np.uint32)
    df['epochtime']= df['click_time'].astype(np.int64) // 10 ** 9
    next_clicks= []
    for category, time in zip(reversed(df['category'].values), reversed(df['epochtime'].values)):
        next_clicks.append(click_buffer[category]-time)
        click_buffer[category]= time
    del(click_buffer)
    df['next_click']= list(reversed(next_clicks))

    for fea in ['ip_day_hour_count','ip_app_count','ip_app_os_count','ip_device_count',
            'app_channel_count','next_click','ip_channel_unique']: 
                df[fea]= np.log2(1 + df[fea].values).astype(int)

    str_array= ("I" + df['ip'].astype(str) \
        + " A" + df['app'].astype(str) \
        + " D" + df['device'].astype(str) \
        + " O" + df['os'].astype(str) \
        + " C" + df['channel'].astype(str) \
        + " WD" + df['day'].astype(str) \
        + " H" + df['hour'].astype(str) \
        + " AXC" + df['app'].astype(str)+"_"+df['channel'].astype(str) \
        + " OXC" + df['os'].astype(str)+"_"+df['channel'].astype(str) \
        + " AXD" + df['app'].astype(str)+"_"+df['device'].astype(str) \
        + " IXA" + df['ip'].astype(str)+"_"+df['app'].astype(str) \
        + " AXO" + df['app'].astype(str)+"_"+df['os'].astype(str) \
        + " IDHC" + df['ip_day_hour_count'].astype(str) \
        + " IAC" + df['ip_app_count'].astype(str) \
        + " AOC" + df['ip_app_os_count'].astype(str) \
        + " IDC" + df['ip_device_count'].astype(str) \
        + " AC" + df['app_channel_count'].astype(str) \
        + " NC" + df['next_click'].astype(str) \
        + " ICU" + df['ip_channel_unique'].astype(str)
      ).values
    if 'is_attributed' in df.columns:
        labels = df['is_attributed'].values
        weights = np.multiply([1.0 if x == 1 else 0.2 for x in df['is_attributed'].values],
                              df['hour'].apply(lambda x: 1.0 if x in pick_hours else 0.5))
    else:
        labels = []
        weights = []
    return str_array, labels, weights

class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self):
        threading.Thread.join(self)
        return self._return

    
def fm_ftrl_model(train_df, val_df, predictors, target, categorical, params, logger, name):
    mean_auc= 0
    batchsize = 10000000
    D = 2 ** 20

    wb = wordbatch.WordBatch(None, extractor=(WordHash, {"ngram_range": (1, 1), "analyzer": "word",
                                                         "lowercase": False, "n_features": D,
                                                         "norm": None, "binary": True})
                             , minibatch_size=batchsize // 80, procs=8, freeze=True, timeout=1800, verbose=0)

    clf = FM_FTRL(alpha=0.05, beta=0.1, L1=0.0, L2=0.0, D=D, alpha_fm=0.02, L2_fm=0.0, init_fm=0.01, weight_fm=1.0,
                  D_fm=8, e_noise=0.0, iters=2, inv_link="sigmoid", e_clip=1.0, threads=4, use_avx=1, verbose=0)

    p = None
    rcount = 0
    while rcount < train_df.shape[0]:
        if rcount + batchsize > train_df.shape[0] :
            df_c = train_df[rcount:]
        else :
            df_c = train_df[rcount: rcount + batchsize]
        rcount += batchsize
        
        str_array, labels, weights= df2csr(wb, df_c, pick_hours={4, 5, 10, 13, 14})
        del(df_c)
        if p != None:
            p.join()
            del(X)
        gc.collect()
        X= wb.transform(str_array)
        del(str_array)
        if rcount % (2 * batchsize) == 0:
            if p != None:  p.join()
            p = threading.Thread(target=evaluate_batch, args=(clf, X, labels, rcount))
            p.start()
        logger.print_log("Training %d" % rcount)
        if p != None:  p.join()
        p = threading.Thread(target=fit_batch, args=(clf, X, labels, weights))
        p.start()
        if rcount == 130000000:  break
    if p != None:  p.join()

    with open('../models/' + str(name) + '_clf.txt','wb+') as f:
        pickle.dump(clf, f)

    with open('../models/' + str(name) + '_wb.txt','wb+') as f:
        pickle.dump(wb, f)
    
    str_array, labels, weights = df2csr(wb, val_df.drop(target, 1))
    X = wb.transform(str_array)
    predictions = clf.predict(X)
    auc = roc_auc_score(val_df[target], predictions)
    logger.print_log('val auc: %f' % auc)
    return predictions

def fm_ftrl_predict(logger, model_file, test_file, output_file, predictors):
    p = None
    click_ids= []
    test_preds = []
    rcount = 0
    batchsize = 10000000
    
    logger.print_log("Predicting...")
    with open('../models/' + model_file + '_clf.txt','rb+') as f:
        clf = pickle.load(f)
    with open('../models/' + model_file + '_wb.txt','rb+') as f:
        wb = pickle.load(f)
        
    for df_c in pd.read_csv(test_file, engine='c', chunksize=batchsize,
                            sep=",", dtype=default_config.dtypes):
        rcount += batchsize
        logger.print_log(rcount)
        str_array, labels, weights = df2csr(wb, df_c)
        click_ids += df_c['click_id'].tolist()
        logger.print_log('finish df2csr')
        del(df_c)
        if p != None:
            test_preds += list(p.join())
            del (X)
        gc.collect()
        X = wb.transform(str_array)
        logger.print_log('finish transform')
        del (str_array)
        p = ThreadWithReturnValue(target=predict_batch, args=(clf, X))
        p.start()
    if p != None:  test_preds += list(p.join())
        
    logger.print_log("writing to file <" + output_file + ">...")
    sub = pd.DataFrame({"click_id": click_ids, 'is_attributed': test_preds})
    sub.to_csv(output_file, index=False, compression='gzip')
    logger.print_log("done.")
    
def fm_ftrl_cv_predict(logger, model_files, test_file, output_file, predictors):
    logger.print_log('loading test data...')
    test_df = pd.read_csv(test_file , dtype=default_config.dtypes)
    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')
    
    logger.print_log("Predicting...")
    sub['is_attributed'] = 0
    for model_file in model_files:
        p = None
        click_ids= []
        test_preds = []
        rcount = 0
        batchsize = 10000000
        with open('../models/' + model_file + '_clf.txt','rb+') as f:
            clf = pickle.load(f)
        with open('../models/' + model_file + '_wb.txt','rb+') as f:
            wb = pickle.load(f)
        for df_c in pd.read_csv(test_file, engine='c', chunksize=batchsize,
                                sep=",", dtype=dtypes):
            rcount += batchsize
            if rcount % (10 * batchsize) == 0:
                print(rcount)
            str_array, labels, weights = df2csr(wb, df_c)
            click_ids+= df_c['click_id'].tolist()
            del(df_c)
            if p != None:
                test_preds += list(p.join())
                del (X)
            gc.collect()
            X = wb.transform(str_array)
            del (str_array)
            p = ThreadWithReturnValue(target=predict_batch, args=(clf, X))
            p.start()
        if p != None:  test_preds += list(p.join())

        sub['is_attributed'] += clf.predict(X)

    sub['is_attributed'] = sub['is_attributed'] / len(model_files)
    logger.print_log("writing to file <" + output_file + ">...")
    sub.to_csv(output_file, index=False)
    logger.print_log("done.")
    
    