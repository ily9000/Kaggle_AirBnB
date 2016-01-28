####noSessions/
Used the features and the feature engineering from the kaggle forked script.
  
	1. cv_results:
		Used sk_learn Kfold and did an intiial grid search by using a for loop to better understand the parameters.
	2. params.p  scores.p
		Grid parameter search for xgboost model with sklearn ParameterGrid. Row-wise correspondence betweeen parameters in params.p and cross-val error in scores.p. Ran using xgb.cv, with early_stopping_rounds = 10, and nround = 100

 
####sessions_e20_25n/
This model included features for booking request, message post, number of actions, total secs elapsed, and if one of the top five devices, (Mac Desktop, Windows Desktop, iPhone, Android Phone, and iPad Tablet) were used.
	1. fulldata.p:
		- 5 fold CV on the entire data with parameters used for 'sessions_e20_25n' model. Run using xgb.cv, with early_stopping_rounds = 10, and nround = 35.  
		- param['eta'] = 0.20, param['max_depth'] = 6, param['subsample'] = .5, param['col_sample_bytree'] = .6
	2. 'train_err.p', 'validate_err.p' (training set error, validation set error)  
		- Custom cross validation using watchlist argument and the below condition to query the training data. Iterate through `i` and set the validation fold as the cases form that month.   
		- condition = 'dac_year == 2014 & dac_month == @i & action_counts != -1'  
		- param['eta'] = 0.20, param['max_depth'] = 6, param['subsample'] = .5, param['col_sample_bytree'] = .6, nrounds = 35  
	3. 'train_err2.p', 'validate_err2.p'
		- Repeated above but ran with eta = .05, and for nrounds = 200.
