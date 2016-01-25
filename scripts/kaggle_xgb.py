#forked from Kaggle but used XGBoost library instead of Scikit-learn
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing

def feature_eng(all_df):
    '''Split the dates into day, month, year, convert extreme age values to missing, 
    and do one hot encoding. Forked from a kaggle script.
    
    Parameters:
        all_df: dataframe with concatenated test and training data
    
    Returns:
        dataframe with new columns for one hot encoded features'''

    #drop date of first booking since that value is not found in the test data    
    all_df.drop('date_first_booking', axis = 1, inplace = True)
    all_df = all_df.fillna(-1)

    #date_account_created
    dac = np.vstack(all_df.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
    all_df['dac_year'] = dac[:,0]
    all_df['dac_month'] = dac[:,1]
    all_df['dac_day'] = dac[:,2]
    all_df.drop('date_account_created', axis=1, inplace = True)

    #timestamp_first_active
    tfa = np.vstack(all_df.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
    all_df['tfa_year'] = tfa[:,0]
    all_df['tfa_month'] = tfa[:,1]
    all_df['tfa_day'] = tfa[:,2]
    all_df.drop('timestamp_first_active', axis=1, inplace = True)

    #age
    av = all_df.age.values
    all_df['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

    #one hot encoding of features
    print 'number of columns before one hot encoding', all_df.shape[1]
    ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
    all_df = pd.get_dummies(all_df, columns = ohe_feats)
    print 'number of columns after one hot encoding', all_df.shape[1]

    return all_df

def main():
#merge the training set and test set on all axis except target
	train_full = pd.read_csv('data/train_users_2.csv', index_col = 'id')
	test = pd.read_csv('data/test_users.csv', index_col = 'id')
	merge_data = pd.concat([train_full, test], axis = 0, join = 'inner')

	merge_data = feature_eng(merge_data)

	#convert targets to numerical values
	targets = train_full['country_destination']
	targets_le = preprocessing.LabelEncoder()
	targets = targets_le.fit_transform(targets)
	print 'number of columns after one hot encoding', merge_data.shape[1]

	#set up data for xgboost
	train_X = merge_data.ix[train_full.index, :].values
	train_Y = targets
	test_X = merge_data.ix[test.index, :].values

	xg_train = xgb.DMatrix(train_X, label = train_Y, missing = -1)
	xg_test = xgb.DMatrix(test_X, missing = -1)

	#set up parameters
	param = {'num_class': 12, 'objective': 'multi:softprob', 'seed': 0}
	#use softmax multi-class classification
	param['eta'] = 0.3
	param['max_depth'] = 6
	param['subsample'] = .5
	param['col_sample_bytree'] = .5
	nrounds = 25

	bst = xgb.train(param, xg_train, nrounds)
	bst.save_model('kaggle_model.model')

	#get predictions
	pred = bst.predict(xg_test)

	#taking the five countries with highest probabilities for each user
	cntr = []
	for person in pred:
	    cntr += targets_le.inverse_transform(person.argsort()[::-1][:5]).tolist()

	#repeat each user id five times
	idx = [[i]*5 for i in test.index]
	idx = np.ravel(idx).tolist()

	#prepare submission and submit
	submission = pd.DataFrame(np.column_stack([idx, cntr]), columns = ['id', 'country'])
	submission.to_csv('submissions/kaggle_model2.csv', index=False)

if __name__ == '__main__':
	main()
