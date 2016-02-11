import pandas as pd
import numpy as np


sub_us = pd.read_csv('../submissions/actions2_e13_96n.csv')

#prepare US/NDF submission
sub_all = pd.read_csv('../submissions/final/actions_e10_130n.csv')

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

pd.to_pickle('../submissions/final/actions_merged1.csv')