import pandas as pd
import time
import numpy as np
import gc
from sklearn.cross_validation import train_test_split
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def stacking_val(settings, logger):
    logger.print_log("getting stacker CV...")
    X_train = settings['X_train']
    y_train = settings['y_train']
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, 
                                                        test_size=settings['test_size'], 
                                                        random_state=settings['seed'])
    settings['stacker'].fit(X_train, y_train)
    logger.print_log('train\'s auc: '+ str(roc_auc_score(y_train, settings['stacker'].predict_proba(X_train)[:,1])))
    logger.print_log('valid\'s auc: '+ str(roc_auc_score(y_test, settings['stacker'].predict_proba(X_test)[:,1])))

def stacking_fit(settings, logger):
    X_train = settings['X_train']
    y_train = settings['y_train']
    logger.print_log(str(settings['cvdata'].corr()))
    settings['stacker'].fit(X_train, y_train)
    try:
        logger.print_log(settings['stacker'].coef_)
        weights = settings['stacker'].coef_/settings['stacker'].coef_.sum()
        scores = [roc_auc_score(y_train, expit(settings['cvdata'][c]))  for c in settings['cvdata'].columns]
        names = [settings['base_models'][c] for c in settings['cvdata'].columns]
        lb = [ settings['lb_scores'][c] for c in settings['cvdata'].columns ]
        df = pd.DataFrame( data={'LB score': lb, 'CV score':scores, 'weight':weights.reshape(-1)}, index=names)
        logger.print_log(df)
    except:
        pass
    logger.print_log('Train score: '+ str(roc_auc_score( y_train, settings['stacker'].predict_proba(X_train)[:,1] )))
    return settings
    
def stacking_predict(settings, logger):    
    final_sub = pd.DataFrame()
    subs = {m:pd.read_csv(settings['sub_test_file'][m]).rename({'is_attributed':m},axis=1) for m in settings['base_models']}
    first_model = list(settings['base_models'].keys())[0]
    final_sub['click_id'] = subs[first_model]['click_id']

    df = subs[first_model]
    for m in subs:
        if m != first_model:
            df = df.merge(subs[m], on='click_id')  # being careful in case clicks are in different order

            X_test = np.array( df.drop(['click_id'],axis=1).clip(settings['almost_zero'],settings['almost_one']).apply(logit) )
    
    final_sub['is_attributed'] = settings['stacker'].predict_proba(X_test)[:,1]    
    output_file = '../output/' + settings['title'] + settings['subtitle'] + '.csv.gz'
    logger.print_log("writing to file <" + output_file + ">...")
    final_sub.to_csv(output_file, index=False, float_format='%.9f', compression='gzip')
    logger.print_log("done.")