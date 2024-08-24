# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:21:36 2021

@author: kelcie zhang
"""

import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

class LSTMFeaturesLayer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=args.num_features * args.num_asset, 
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            batch_first=True
        )

    def forward(self, x):
        x, (_, _) = self.lstm(x)
        return x
    
class Layer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.max_weight = args.max_weight
        self.num_asset = args.num_asset

    def forward(self, x):
        w = cp.Variable(x.shape)
        mu = cp.Parameter(x.shape)
        cons = [
            cp.abs(cp.sum(w)) <= 0.1,
            w >= -0.5, 
            w <= 0.5
            ]
        obj = cp.Minimize(
            #-mu @ w - cp.sum(cp.entr(w))
            cp.sum_squares(w-mu)
        )
        prob = cp.Problem(
            obj,
            cons,
        )
        assert prob.is_dpp()

        # convex optimization layer
        optimize_layer = CvxpyLayer(
            prob,
            parameters=[mu],
            variables=[w],
        )

        x, = optimize_layer(
            x,
            solver_args={
                # "solve_method":cp.ECOS,
                #"n_jobs_forward":1,
                "eps":1e-4,
                "max_iters":10000
            }
            )
        return x
