# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:12:55 2021

@author: kelcie zhang
"""
import torch.nn as nn
from layers import LSTMFeaturesLayer, Layer
import torch.nn.functional as func

class Model1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.feature_extract = LSTMFeaturesLayer(args)
        self.optimize_layer = Layer(args)
    def forward(self,x):
        x = self.feature_extract(x)
        x = x[:,-1,:]
        x = self.optimize_layer(x)
        return x

