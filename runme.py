# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:35:04 2021

@author: kelcie zhang 
"""

import numpy as np
import os
import configparser
from model import Model1
from utils import ARGS, load_data, load_data_args, load_train_args, load_model_args
from train_model import PairTradingNN


if __name__ == "__main__":
    config = "params.ini"
    data_args = load_data_args(config)
    train_args = load_train_args(config)
    model_args = load_model_args(config)
    
    features, rets = load_data(data_args)
    
    # 生成数据时如果已经处理空值和无穷大，下面四行就删掉就好
    features[np.where(np.isnan(features))] = 0
    rets[np.where(np.isnan(rets))] = 0
    features[np.where(features == np.inf)] = 0
    rets[np.where(rets == np.inf)] = 0
    
    print(features.shape) 
    print(rets.shape) 


    model = Model1(model_args)

    pairtradingnn = PairTradingNN(model,train_args)

    pairtradingnn.fit(features, rets, 5)


