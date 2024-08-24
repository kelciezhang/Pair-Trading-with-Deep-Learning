# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 10:54:02 2021

@author: kelcie zhang
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

data_path_ori = '1112_1min_2'
data_path_output = 'data'
ts_len = 150
ret_len = 10
num_feature_each = 6
num_asset = 2

if not os.path.exists(data_path_output):
    os.mkdir(data_path_output)

pv_file = os.listdir(data_path_ori)
for i in range(len(pv_file)):
    data_i = pd.read_csv(data_path_ori + '/' + pv_file[i])
    data_i.columns = ['date'] + data_i.columns[1:].tolist()
    if i == 0:
        # extract date index
        dates = pd.Index(data_i[data_i['for_trade'] == 1].date.values).sort_values()
    else:
        dates = pd.Index(set(dates).intersection(set(data_i[data_i['for_trade'] == 1].date.values))).sort_values()
dates = dates[ts_len:]

dates = dates[-500:]
features_list = []
labels_list = []

for f in pv_file:
    print(f)
    data_i = pd.read_csv(data_path_ori + '/' + f)
    data_i.columns = ['date'] + data_i.columns[1:].tolist()
    # drop duplicated data
    old_len = len(data_i)
    data_i = data_i.drop_duplicates(['date', 'ID'])
    if old_len - len(data_i) > 0:
        print(old_len - len(data_i), ' rows have been dropped. ')
    # calculate features
    features = np.zeros((len(dates) - ret_len, num_feature_each, ts_len))
    for i in tqdm(range(len(dates[:-ret_len]))):
        d = dates[i]
        row_id = data_i[(data_i.date == d) & (data_i.for_trade == 1)].index[0]
        features_i = data_i.iloc[(row_id - ts_len): (row_id + 1), :].iloc[:, 1:7]
        features_i = features_i.pct_change().iloc[1:, :]
        features[i, :, :] = features_i.values.T
    features_list.append(features)
    # calculate labels
    cls_price = data_i[data_i['for_trade'] == 1]
    cls_price = cls_price.set_index('date', drop=True)
    cls_price = cls_price.loc[dates]
    cls_price = cls_price['close'].reset_index(drop=True)
    labels = (- cls_price.diff(-ret_len) / cls_price).values
    labels = labels[:-ret_len]
    labels_list.append(labels)
    
        
all_features = np.hstack(features_list)
all_labels = np.vstack(labels_list).T

all_features = np.load('data/all_features.npy')
all_labels = np.load('data/all_labels.npy')

all_features[np.where(all_features == np.inf)] = 0
all_features[np.where(all_features == -np.inf)] = 0
all_features[np.isnan(all_features)] = 0
all_labels[np.where(all_labels == np.inf)] = 0
all_labels[np.where(all_labels == -np.inf)] = 0
all_labels[np.isnan(all_labels)] = 0
# feature_na_index = sum(sum(np.isnan(all_features.transpose(1,2,0))))
# label_na_index = sum(np.isnan(all_labels).T)
# na_index = feature_na_index + label_na_index



np.save('all_features', all_features)
np.save('all_labels', all_labels)

        
        
        
    

