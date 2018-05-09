k_fold = 3

settings_list = ['title', 'subtitle', 'model', 'train_file', 'sub_train_file', 'sub_train_file', 'test_file', 'seed', 'target', 'predictors', 'categorical']

dtypes = {
    'ip'                       : 'uint32',
    'app'                      : 'uint16',
    'device'                   : 'uint16',
    'os'                       : 'uint16',
    'channel'                  : 'uint16',
    'is_attributed'            : 'uint8',
    'click_id'                 : 'uint32',
    'time'                     : 'uint8',
    'day'                      : 'uint8',

    'ip_tcount'                : 'int64',
    'ip_app_count'             : 'int64',
    'ip_app_os_count'          : 'int64',
    'ip_tchan_count'           : 'float64',
    'ip_app_os_var'            : 'float64',
    'ip_app_channel_var_day'   : 'float64',
    'ip_app_channel_mean_time' : 'float64',

    'n_channels'               : 'int64',

    'hour'                     : 'uint8',
    'app_channel'              : 'float64',
    'app_device_os'            : 'float64',
    'channel_device_os'        : 'float64',
    'channel_hour'             : 'float64',
    'app_hour'                 : 'float64',


    'nip_day_test_hh'          : 'uint32',
    'nip_day_hh'               : 'uint16',
    'nip_hh_os'                : 'uint16',
    'nip_hh_app'               : 'uint16',
    'nip_hh_dev'               : 'uint32',
    'n_app'                    : 'uint32',
    
    'UsrappCount'              : 'uint32',
    'UsrCount'                 : 'uint32',
    'UsrappNewness'            : 'int64',
    'UsrNewness'               : 'int64',
    
    'X0'                       : 'int64',
    'X1'                       : 'int64',
    'X2'                       : 'int64',
    'X3'                       : 'int64',
    'X4'                       : 'int64',
    'X5'                       : 'int64',
    'X6'                       : 'int64',
    'X7'                       : 'int64',
    'X8'                       : 'int64',
    'category'                 : 'int64',
    'epochtime'                : 'int64',
    'nextClick'                : 'int64',
    'nextClick_shift'          : 'float64',
}


target = 'is_attributed'

predictors = ['app','device','os', 'channel', 'time', 'day', 
              'ip_tcount', 'ip_tchan_count', 'ip_app_count',
              'ip_app_os_count', 'ip_app_os_var',
              'ip_app_channel_var_day','ip_app_channel_mean_time']

categorical = ['app', 'device', 'os', 'channel', 'time', 'day']


lightgbm_params = {
    'learning_rate': 0.15,
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 7,  # 2^max_depth - 1
    'max_depth': 4,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight':99, # because training data is extremely unbalanced 
    'device': 'gpu',
    'nthread': 8,
    'seed': 47, 
}

rf_params = {
    'seed': 47,
}

xgboost_params = {
    'eta': 0.3,
    'tree_method': "hist",
    'grow_policy': "lossguide",
    'max_leaves': 1400,  
    'max_depth': 0, 
    'subsample': 0.9, 
    'colsample_bytree': 0.7, 
    'colsample_bylevel':0.7,
    'min_child_weight':0,
    'alpha':4,
    'objective': 'binary:logistic', 
    'scale_pos_weight':9,
    'eval_metric': 'auc', 
    'nthread':8,
    'random_state': 47, 
    'silent': True
}