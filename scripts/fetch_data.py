import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
import kaggle_xgb
import pickle

class clfInput():
    """Features for both train and test set and targets for the training set
        Public Methods:
            sessions_ftrEng
            users_ftrEng
            one_hot
            encode_targets
    """

    def __init__(self, path = '../data/'):
        """
        Args:
            path (Optional[str]): path to the directory where the .csv files 
        Attributes:
            trainDf (DataFrame): data for training set, initial features  
                are from train_users_2.csv file
            testDf (DataFrame): data for test set, initial features 
                are from test_users.csv file
            sessionsDf (DataFrame): sessions.csv
            allDf (DataFrame): features from both the training and test set 
        """
        self.path = path
        self.trainDf = pd.read_csv(self.path + 'train_users_2.csv', index_col = 'id')
        self.testDf = pd.read_csv(self.path + 'test_users.csv', index_col = 'id')
        #merge the training set and the test set on all columns except target
        self.allDf = pd.concat([self.trainDf, self.testDf], axis = 0, join = 'inner')

    def sessions_ftrEng(self):
        """Feature engineering for data in sessions.csv"""

        sessionsDf = pd.read_csv(self.path + 'sessions.csv')
        sessionsDf.dropna(subset=['user_id'], inplace = True)

        #booking request action type
        self.allDf['booking_request'] = -1
        self.allDf.loc[self.allDf.index.isin(sessionsDf.user_id), 'booking_request'] = 0
        users = self.allDf.index.isin(sessionsDf[sessionsDf.action_type == 'booking_request'].user_id)
        self.allDf.loc[users, 'booking_request'] = 1

        #action_detail 
        self.allDf['message_post'] = -1
        self.allDf.loc[self.allDf.index.isin(sessionsDf.user_id), 'message_post'] = 0
        users = self.allDf.index.isin(sessionsDf[sessionsDf.action_detail == 'message_post'].user_id)
        self.allDf.loc[users, 'message_post'] = 1

        self.allDf['action_counts'] = -1
        s = sessionsDf.groupby('user_id')['device_type'].count()
        self.allDf.loc[s.index, 'action_counts'] = s

        s = sessionsDf.groupby('user_id')['secs_elapsed'].sum()
        self.allDf.loc[s.index, 'secs_elapsed'] = s
        #self.allDf = self.allDf.fillna(-1)

        #sessions devices: top 5 excluding '-unknown-'
        devices = ['Mac Desktop', 'Windows Desktop', 'iPhone', 'Android Phone', 'iPad Tablet']
        for dev in devices:
            self.allDf[dev] = -1
            self.allDf.loc[self.allDf.index.isin(sessionsDf.user_id), dev] = 0
            users = sessionsDf.groupby('device_type')['user_id'].unique()[dev]
            self.allDf.loc[users, dev] = 1

        actionsDf = pd.read_pickle('../data/actions.p')
        self.allDf = pd.concat([self.allDf, actionsDf], axis = 1, join = 'outer')

    def split_data(self):
        """Split the combined dataframe into training and test sets.
        Create the training and test arrays for the classifer."""

        self.trainDf = pd.concat([self.allDf.loc[self.trainDf.index], self.trainDf['country_destination']], axis = 1)
        self.testDf = self.allDf.loc[self.testDf.index]
        self.train_X = self.trainDf.iloc[:,:-1].values
        self.test_X = self.testDf.values
        if not hasattr(self, 'train_Y'):
            self.encode_targets()

    def users_ftrEng(self):
        """Transform date and age columns in users data
        Split dates into year month day, and for timestamps split similarly 
            but also incude the hour.
        Add the day of week using date_account_created.
        Delete data_first_booking since it is null in test set.
        """

        self.allDf.drop('date_first_booking', axis = 1, inplace = True)
        self.allDf = self.allDf.fillna(-1)

        #date_account_created
        dac = pd.to_datetime(self.allDf['date_account_created'])
        self.allDf['dac_year'] = dac.dt.year
        self.allDf['dac_month'] = dac.dt.month
        self.allDf['dac_day'] = dac.dt.day
        self.allDf['dac_dayofweek'] = dac.dt.dayofweek
        self.allDf.drop('date_account_created', axis=1, inplace = True)

        #timestamp_first_active
        tfa = self.allDf['timestamp_first_active'].astype('str')
        tfa = pd.to_datetime(tfa, format='%Y%m%d%H%M%S')    
        self.allDf['tfa_year'] = tfa.dt.year
        self.allDf['tfa_month'] = tfa.dt.month
        self.allDf['tfa_day'] = tfa.dt.day
        self.allDf['tfa_hour'] = tfa.dt.hour
        self.allDf.drop('timestamp_first_active', axis=1, inplace = True)

        #age
        #av = self.allDf.age.values
        #self.allDf.loc[self.allDf.query(1000 > 'age' > 100].index, 'age'] = 105
        #self.allDf.loc[self.allDf.query('age' > 1000').index, 'age'] = 110
        #self.allDf['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

    def _rmbrowsers():
        """remove browsers only found in either test or training set"""
        pass

    def binarize_targets(self, true_cntr = 'NDF'):
        """split targets into NDF and non NDF"""
        
        self.targets = self.trainDf['country_destination']
        self.train_Y = (self.targets == 'NDF').astype(int)
    
    def one_hot(self):
        """one hot encoding of features"""

        print 'number of columns before one hot encoding', self.allDf.shape[1]
        ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 
                    'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 
                    'signup_app', 'first_device_type', 'first_browser']
        self.allDf = pd.get_dummies(self.allDf, columns = ohe_feats)
        print 'number of columns after one hot encoding', self.allDf.shape[1]

    def encode_targets(self):
        """Convert targets to numerical labels
            Attributes:
                targets (1D np.array): country destination of users from training set
                le (sklearn.preprocessing.LabelEncoder): Used to convert targets to numberical labels"""

        self.targets = self.trainDf['country_destination']
        self.le = preprocessing.LabelEncoder()
        self.train_Y = self.le.fit_transform(self.targets)
