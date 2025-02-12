##Parameter Grid Search for Xgboost
##We do 10 fold Cross Validation to find the mean test error for each set 
##of parameter values. We use the custom ndcg error function defined by Kaggle.

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.grid_search import ParameterGrid
import kaggle_xgb
import calc_ndcg

#read in data and do feature engineering for all columns but the target
xgbInput = dataEngr.clfInput()
xgbInput.get_sessionsFtr()
xgbInput.users_ftrEng()
xgbInput.one_hot()
#xgbInput.binarize_targets()
xgbInput.split_data()

xg_train = xgb.DMatrix(xgbInput.train_X, label = xgbInput.train_Y, missing = -1)

param = {'num_class': 12, 'silent': 1, 'objective': 'multi:softprob'}
#set the number of rounds to be high since early stopping is enabled
nround = 200

param_grid = {}
param_grid['eta'] = [.16, .20]
param_grid['max_depth'] = [6, 7, 8]
param_grid['subsample'] = [.7, .8, .9]
param_grid['colsample_bytree'] = [.3, .6]
#param_grid['max_delta_step'] = [0, 1, 3]

#set up dataframe to store cross-validation results form each iteration
col_names = ['test-error-mean', 'test-error-std', 'train-error-mean', 
            'train-error-std', 'num_boost_rounds']
df_cv = pd.DataFrame(columns = col_names)

#set up dataframe to store parameters from each iteration
col_names = list(param_grid.iterkeys())
df_params = pd.DataFrame(columns = col_names)

for cnt, p in enumerate(list(ParameterGrid(param_grid))):
    print cnt
    #run xgboost with a set of parameters from the grid
    param.update(p)
    cv = xgb.cv(param, xg_train, nround, nfold = 10, feval = calc_ndcg.evalerror, 
        early_stopping_rounds = 10)
    #store parameters and results in respective dataframes
    #append the last row (lowest error) of the results 
    #index contains the number of iterations
    df_cv = df_cv.append(cv.iloc[-1,:], ignore_index= True)
    df_cv.iloc[-1, -1] = cv.index[-1]
    df_params = df_params.append(p, ignore_index= True)
    df_cv.to_pickle('cv_results/actions_e20/params1_err.p')
    df_params.to_pickle('cv_results/actions_e20/params1.p')