#do cross-validation on just the training users with sessions data

import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
import calc_ndcg
import run_model
import xgboost as xgb
import pickle
import fetch_data

def full_cv(train_X, train_Y, fname, param, nrounds):

    xg_train = xgb.DMatrix(train_X, label = train_Y, missing = -1)
    scores_df = xgb.cv(param, xg_train, nrounds, nfold = 10, feval = calc_ndcg.evalerror, 
        early_stopping_rounds = 10)
    scores_df.to_pickle('cv_results/'+fname)

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

def main():

    #load input data for xgboost
    xgbInput = fetch_data.clfInput()
    xgbInput.sessions_ftrEng()
    xgbInput.users_ftrEng()
    xgbInput.one_hot()
    xgbInput.split_data()

    param = {'num_class': 12, 'objective': 'multi:softprob', 'seed': 0}
    param['eta'] = 0.20
    param['max_depth'] = 6
    param['subsample'] = .5
    param['col_sample_bytree'] = .6
    results = {}
    cv_train = pd.DataFrame()
    cv_valid = pd.DataFrame()
    nrounds = 40
    for train_indx, valid_indx in cv_bymonth(xgbInput):
        dtrain = xgb.DMatrix(xgbInput.train_X[train_indx], label = xgbInput.train_Y[train_indx],
                    missing = -1)
        dvalid = xgb.DMatrix(xgbInput.train_X[valid_indx], label = xgbInput.train_Y[valid_indx],
                    missing = -1)
        evallist = [(dtrain, 'train'), (dvalid, 'eval')]
        bst = xgb.train(param, dtrain, nrounds, evallist, feval = calc_ndcg.evalerror, evals_result = results)
        cv_train = pd.concat([cv_train, pd.Series(results['train']['error'])], axis = 1)
        cv_valid = pd.concat([cv_valid, pd.Series(results['eval']['error'])], axis = 1)
        pd.to_pickle(cv_train, 'cv_results/sessions_e20_25n/tr_err_av.p')
        pd.to_pickle(cv_valid, 'cv_results/sessions_e20_25n/val_err_av.p')

    full_cv(xgbInput.train_X, xgbInput.train_Y, 'fulltr_err_av.p', param, nrounds)    
        
if __name__ == '__main__':
        main()    
