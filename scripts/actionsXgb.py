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
xgbInput.split_bySess()

param = {'num_class': 12, 'silent': 1, 'objective': 'multi:softprob'}

param_grid = {}
param_grid['eta'] = [.13]
param_grid['max_depth'] = [6]
param_grid['subsample'] = [.6, .9, 1]
param_grid['colsample_bytree'] = [.7, 1]
nrounds = 200

#set up dataframe to store cross-validation results form each iteration
col_names = ['test-error-mean', 'test-error-std', 'train-error-mean', 
            'train-error-std', 'num_boost_rounds']
df_cv = pd.DataFrame(columns = col_names)

#set up dataframe to store the parameters used for cross validation
col_names = list(param_grid.iterkeys())
df_params = pd.DataFrame(columns = col_names)   

for cnt, p in enumerate(list(ParameterGrid(param_grid))):
    param.update(p)
#store errors from each month by doing cv
    dtrain = xgb.DMatrix(xgbInput.sesstrain_X, label = xgbInput.sesstrain_Y,
                missing = -1)
    cv = xgb.cv(param, dtrain, nrounds, nfold=10, feval= calc_ndcg.eval_all, early_stopping_rounds= 10)

#store parameters and results in respective dataframes
#append the last row (lowest error) of the results 
#index contains the number of iterations
    df_cv = df_cv.append(cv.iloc[-1,:], ignore_index= True)
    df_cv.iloc[-1, -1] = cv.index[-1]

    df_params = df_params.append(p, ignore_index= True)
    df_cv.to_pickle('cv_results/actionsXgb/errors_search1.p')        
    df_params.to_pickle(df_params, 'cv_results/actionsXgb/params_search1.p')