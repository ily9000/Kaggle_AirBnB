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

    def split_bySess(self):
        """Subset the training set with just those who have actions"""

        #there is no guarantee the order of allDf matches trainDf, so use index not integers
        if not hasattr(self, 'train_Y'):
            self.encode_targets()
        users = list(set(self.sessUsrs) & set(self.trainDf.index))
        self.sesstrain_X = self.trainDf.loc[users].values
        #find indices of training set users which have sessions data
        #subset target array
        mask = self.trainDf.index.get_indexer(users)
        self.sesstrain_Y = self.train_Y[mask]
        
    def split_data(self, update_trainDf = False):
        """Split the combined dataframe into training and test sets.
        Will convert to numerical labels if not already converted.
        Create the training and test arrays for the classifier.
        If cross validation will be done, set update_trainDf to True, and the
        training set dataframe will be updated with the engineered features."""
       
        #there is no guarantee in the order of allDf, so use index not integers
        if not hasattr(self, 'train_Y'):
            self.encode_targets()
        if update_trainDf:
            self.trainDf = pd.concat([self.allDf.loc[self.trainDf.index], self.trainDf['country_destination']], axis = 1)
        #self.testDf = self.allDf.loc[self.testDf.index]
        self.train_X = self.allDf.loc[self.trainDf.index,:].values
        self.test_X = self.allDf.loc[self.testDf.index, :].values

    def get_sessionsFtr(self):
        """Load and merge the sessions features with user data"""
        
        actionsDf = pd.read_pickle('../data/actions3.p')
        self.allDf = pd.concat([actionsDf, self.allDf], axis = 1, join = 'outer')
        self.allDf.drop(['p4total'], axis = 1, inplace = True)
        self.sessUsrs = actionsDf.index
    
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
        self.allDf.loc[self.allDf.query('age > 1000').index, 'age'] = 200
        self.allDf.loc[self.allDf['age']<16, 'age'] = -1

    def binarize_targets(self, true_cntr = 'NDF'):
        """split targets into NDF and non NDF"""
        
        self.targets = self.trainDf['country_destination']
        self.train_Y = (self.targets == 'NDF').astype(int)
    
    def one_hot(self):
        """One hot encode the features and remove features exclusive to test or training set"""

        print 'number of columns before one hot encoding', self.allDf.shape[1]       
        ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 
                    'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 
                    'signup_app', 'first_device_type', 'first_browser']
        self.allDf = pd.get_dummies(self.allDf, columns = ohe_feats)
        #delete exclusive features
        for col in ohe_feats:
            rm_val = set(self.trainDf[col].unique()) ^ set(self.testDf[col].unique())
            for ftr in rm_val:
                del(self.allDf[col + '_' + str(ftr)])            
        print 'number of columns after one hot encoding', self.allDf.shape[1]

    def encode_targets(self):
        """Convert targets to numerical labels
            Attributes:
                targets (1D np.array): country destination of users from training set
                le (sklearn.preprocessing.LabelEncoder): Used to convert targets to numberical labels"""

        self.targets = self.trainDf['country_destination']
        self.le = preprocessing.LabelEncoder()
        self.train_Y = self.le.fit_transform(self.targets)
