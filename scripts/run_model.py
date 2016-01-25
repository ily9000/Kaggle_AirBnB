import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
import kaggle_xgb
import pickle
import data

def train_xgb(train_X, train_Y, p, nrounds):
	"""Run xgb.train on the test data 
	Parameters:
		train_X (2D np.array): features of the training samples
		test_Y (1D np.array): targets of the training samples
		p (dict): It will update the paramter dictionary that will be passed as
			an argument to xgb.train.
		nrounds: It will be passed as the value of the num_boost_rounds parameter 
			in xgb.train
	Returns:
		A booster of XGBoost (a trained model object)
	"""

	xg_train = xgb.DMatrix(train_X, label = train_Y, missing = -1)

	#set up parameters
	#use softmax multi-class classification
	param = {'num_class': 12, 'objective': 'multi:softprob', 'seed': 0}
	param.update(p)

	bst = xgb.train(param, xg_train)
	return bst

def get_submission(bst, test_X, test_users, le) :
	"""Use the xgboost model to get the 5 targets with the highest probability
		for each test sample.
	Parameters:
		bst (xgboost.Booster): trained XGBoost model
		test_X (np.array): features of the test sample
		test_users (list): user id for each corresponding test samples
		le (preprocessing.LabelEncoder): LabelEncoder used to transform
			the countries (classes) to numerical targets.
	"""

	xg_test = xgb.DMatrix(test_X, missing = -1)
	pred = bst.predict(xg_test)

	#select the five countries with highest probabilities for each user
	cntr = []
	for person in pred:
	    cntr += le.inverse_transform(person.argsort()[::-1][:5]).tolist()

	#repeat each user id five times
	idx = [[i]*5 for i in test_users]
	idx = np.ravel(idx).tolist()

	#prepare submission and submit
	submission = pd.DataFrame(np.column_stack([idx, cntr]), columns = ['id', 'country'])
	return submission

def main():	

    xgbInput = fetch_data.clfInput()
    xgbInput.sessions_ftrEng()
    xgbInput.users_ftrEng()
    xgbInput.one_hot()
    xgbInput.split_data()

    #parameters to use to train the model
    param = {}
    param['eta'] = 0.20
    param['max_depth'] = 6
    param['subsample'] = .5
    param['col_sample_bytree'] = .6
    nrounds = 25

    bst = train_xgb(train_X, train_Y, param, nrounds)
    with open('../xgbmodels/sessions_e20_25n.p', 'wb') as f:
        pickle.dump(bst, f)

    #predict and get submissions
    test_X = all_df.loc[test_data.index, :].values
    submission = get_submission(bst, test_X, test_data.index, targets_le)
    submission.to_csv('../submissions/sessions_e20_25n.csv', index=False)

if __name__ == '__main__':
    main()