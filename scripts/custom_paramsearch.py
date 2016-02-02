#parameter search by cross validating on only most recent data, folds are by month

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.grid_search import ParameterGrid
import kaggle_xgb
import calc_ndcg
import dataEngr
import pickle

def cv_bymonth(xgbInput):
    """Select folds for cross validation as all cases that occurred in a given with month 
    in 2014, with sessions data.
    Only cases in 2014 have sessions data and the last test case is on June 30.
    """
    
    for i in range(1,7):
        condition = 'dac_year == 2014 & dac_month == @i & action_counts != -1'
        valid_mask = xgbInput.trainDf.index.isin(xgbInput.trainDf.query(condition).index)
        valid_indx = np.where(valid_mask)
        train_indx = np.where(~valid_mask)
        yield train_indx, valid_indx

#read in data and do feature engineering for all columns but the target
xgbInput = dataEngr.clfInput()
xgbInput.get_sessionsFtr()
xgbInput.users_ftrEng()
xgbInput.one_hot()
#xgbInput.binarize_targets()
xgbInput.split_forcv()

param = {'num_class': 12, 'silent': 1, 'objective': 'multi:softprob'}

param_grid = {}
param_grid['eta'] = [.20]
param_grid['max_depth'] = [6]
param_grid['subsample'] = [.5, .7]
param_grid['colsample_bytree'] = [.6, .8, 1]
results = {}
err_out = {}
nrounds = 50

#set up dataframe to store cross-validation from each fold
cv_train = pd.DataFrame()
cv_error = pd.DataFrame()

#set up dataframe to store mean/stdev. after cross validation
cv_tofile = pd.DataFrame()

#set up dataframe to store the parameters used for cross validation
col_names = list(param_grid.iterkeys())
df_params = pd.DataFrame(columns = col_names)

for cnt, p in enumerate(list(ParameterGrid(param_grid))):
    print cnt
    param.update(p)
#store errors from each month by doing cv
    cv_train = pd.DataFrame()
    cv_valid = pd.DataFrame()
    for train_indx, valid_indx in cv_bymonth(xgbInput):   
        dtrain = xgb.DMatrix(xgbInput.train_X[train_indx], label = xgbInput.train_Y[train_indx],
                    missing = -1)
        dvalid = xgb.DMatrix(xgbInput.train_X[valid_indx], label = xgbInput.train_Y[valid_indx],
                    missing = -1)
        evallist = [(dtrain, 'train'), (dvalid, 'eval')]
        bst = xgb.train(param, dtrain, nrounds, evallist, feval = calc_ndcg.evalerror, evals_result = results)
        cv_train = pd.concat([cv_train, pd.Series(results['train']['error'])], axis = 1)
        cv_error = pd.concat([cv_error, pd.Series(results['eval']['error'])], axis = 1)
        
#take the mean and standard deviation of training and validation error and pickle those results
    err_out['test-error-mean' + str(cnt)] = cv_train.astype('float').mean(axis = 1)
    err_out['test-error-std' + str(cnt)] = cv_train.astype('float').std(axis = 1)
    err_out['train-error-mean' + str(cnt)] = cv_error.astype('float').mean(axis = 1)
    err_out['train-error-std' + str(cnt)] = cv_error.astype('float').std(axis = 1)
    cv_tofile = pd.concat([cv_tofile, pd.DataFrame(err_out)], axis = 1)
    pd.to_pickle(cv_tofile, 'cv_results/actions_e20/errors_search2.p')
    
#output the parameters that were used
    df_params = df_params.append(p, ignore_index= True)
    pd.to_pickle(df_params, 'cv_results/actions_e20/params_search2.p')        