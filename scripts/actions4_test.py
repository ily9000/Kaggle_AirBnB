"""Feature engineering for data in sessions.csv"""
import pandas as pd
import numpy as np

train_data = pd.read_csv('../data/train_users_2.csv', index_col = 'id')
test_data = pd.read_csv('../data/test_users.csv', index_col = 'id')
sessionsDf = pd.read_csv('../data/sessions.csv')
#drop missing users, substitute 'none' for NaN actions
sessionsDf.dropna(subset=['user_id'], inplace = True)
sessionsDf.fillna({'action_type': 'none', 'action': 'none','action_detail': 'none'}, inplace = True)

#this action is unique to test set, we change it to match with training set
sessionsDf.replace({'action': {'this_hosting_reviews_3000': 'this_hosting_reviews'}}, inplace = True)

#merge these actions with action_detail
#actions that correspond to message_post
msgActs = sessionsDf.loc[sessionsDf.action_detail == 'message_post', 'action'].unique().tolist()
#pending action & pending action detail: 'pending_pending' has a 1 to 1 relationship with booking request
#merge impression & book so it has p4 in it's name, we will use this to sum all action details corresponding with p4
#impression and book only have one action detail: p4, so there will be no loss
#need to include click and patch
actions = ['index', 'show', 'create', 'reviews', 'delete', 'recommendations', 'update', 'pending', 'impressions', 'book'] 
actions += msgActs
mergedActs = sessionsDf.action + '_' + sessionsDf.action_detail + '1'
#ignoring patch: action_detail is unknown in test set but has 2 addition ones in training set
for a in actions:
    sessionsDf.loc[sessionsDf.action == a, 'action'] = mergedActs[sessionsDf.action == a]

#these actions are exclusive to one of the sets and can be dropped
train_actions = sessionsDf[sessionsDf.user_id.isin(train_data.index)].action.unique()
test_actions = sessionsDf[sessionsDf.user_id.isin(test_data.index)].action.unique()
a_ignore = set(train_actions) ^ set(test_actions)    
        
#build a dataframe with counts for each action
#using device_type column to count the number of actions
grouped_acts = sessionsDf.groupby(['user_id', 'action'])
action_cnts = grouped_acts.apply(lambda x: x['device_type'].count())
action_cnts = action_cnts.unstack(level = -1)
action_cnts.fillna(0, inplace=True)

#sum booking requests, message posts, and p4
#booking_request has 2 actions: pending and at_checkpoint
a_details = ['message_post', 'pending_pending|at_checkpoint',  'p4', 'payout']
for a in a_details:
    action_cnts[a + 'total'] = action_cnts.loc[:, action_cnts.columns.str.contains(a)].sum(axis = 1)
    
action_cnts.drop(a_ignore, axis = 1, inplace = True)
    
#merge action counts with other sessions features, 
#there should be no missing users.
allDf = pd.DataFrame(index = sessionsDf.user_id.unique())
allDf = pd.concat([allDf, action_cnts], axis = 1, join = 'outer')

#get total number of actions
s = sessionsDf.groupby('user_id')['device_type'].count()
allDf.loc[s.index, 'action_counts'] = s

#get total number seconds elapsed, fill NaN with missing
s = sessionsDf.groupby('user_id')['secs_elapsed'].sum()
allDf.loc[s.index, 'secs_elapsed'] = s
allDf = allDf.fillna(-1)

#devices: total counts, and proportional values
dev_counts = sessionsDf.groupby(['user_id']).apply(lambda x: x['device_type'].value_counts())
n = len(dev_counts.index.get_level_values(-1).unique())
allDf = pd.concat([allDf, dev_counts.unstack()], axis = 1)
allDf = allDf.fillna(0)
allDf.columns = allDf.columns[:-n].tolist() + list(allDf.columns[-n:] + '_counts')

dev_prop = sessionsDf.groupby(['user_id']).apply(lambda x: x['device_type'].value_counts(normalize = True))
allDf = pd.concat([allDf, dev_counts.unstack()], axis = 1)
allDf = allDf.fillna(0)
allDf.columns = allDf.columns[:-n].tolist() + list(allDf.columns[-n:] + '_prop')

pd.to_pickle(allDf, '../data/actions4.p')
#dev_time = sessions.groupby(['user_id', 'action']).apply(lambda x: x['device_type'].value_counts())