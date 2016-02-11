import pandas as pd
import numpy as np

xgbInput = dataEngr.clfInput()
xgbInput.get_sessionsFtr()
xgbInput.users_ftrEng()
xgbInput.one_hot()
#xgbInput.binarize_targets()
xgbInput.split_data()

xg_test = xgb.DMatrix(test_X, missing = -1)

#ALL
with open('../xgbmodels/final_1/actions_all.p') as f:
    bst_all = pickle.load(f)

pred = bst.predict(xg_test)

#select the five countries with highest probabilities for each user
cntr = []
for person in pred:
    cntr += xgbInput.le.inverse_transform(person.argsort()[::-1][:5]).tolist()

#repeat each user id five times
idx = [[i]*5 for i in test_users]
idx = np.ravel(idx).tolist()

sub_all = pd.DataFrame(np.column_stack([idx, cntr]), columns = ['id', 'country'])

#US
with open('../xgbmodels/final_1/actions_us.p') as f:
    bst_all = pickle.load(f)

pred = bst.predict(xg_test)

#select the five countries with highest probabilities for each user
cntr = []
for person in pred:
    cntr += xgbInput.le.inverse_transform(person.argsort()[::-1][:5]).tolist()

#repeat each user id five times
idx = [[i]*5 for i in test_users]
idx = np.ravel(idx).tolist()

#prepare US/NDF submission
sub_us = pd.DataFrame(np.column_stack([idx, cntr]), columns = ['id', 'country'])

#switch position according to US/NDF trained tree
for user in sub_all.id.unique():
    if all(sub_all[sub_all.id == user].iloc[0:2,1].isin(['NDF', 'US'])):
        cntr_us = sub_us.loc[(sub_us.id == user), 'country']
        main_cntr = sub_all.loc[sub_all.id == user, 'country']
        if (cntr_us[cntr_us == 'US'].index[0] > cntr_us[cntr_us == 'NDF'].index[0]):
            main_cntr[0] = 'NDF'
            main_cntr[1] = 'US'
            sub_all.loc[sub_all.id == user, 'country'] = main_cntr
        if (cntr_us[cntr_us == 'US'].index[0] < cntr_us[cntr_us == 'NDF'].index[0]):
            main_cntr[0] = 'US'
            main_cntr[1] = 'NDF'
            sub_all.loc[sub_all.id == user, 'country'] = main_cntr   