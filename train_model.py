# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:14:35 2021

@author: kelcie zhang
"""

import copy
import random
from time import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from diffcp.cone_program import SolverError
import matplotlib.pyplot as plt

from utils import MyDataset
plt.switch_backend('agg')

class PairTradingNN():

    def __init__(self,model,train_args):

        self.model = model
        self.train_args = train_args

        # determine optimizer
        if self.train_args.optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.train_args.lr
            )
        elif self.train_args.optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.train_args.lr
            )
        else:
            raise NotImplementedError(f"optimizer {self.train_args.optimizer} is not supported!")

        # determine the device and load the model on it
        self.device = torch.device(
            f"cuda:{self.model_args.GPU}" if torch.cuda.is_available() and self.model_args.GPU >= 0 else "cpu"
            )
        self.model.to(self.device)

        # determine the seed
        if train_args.seed is not None:
            np.random.seed(train_args.seed)
            torch.manual_seed(train_args.seed)
            # torch.cuda.manual_seed(seed)
            random.seed(train_args.seed)
            # torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.deterministic = True

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def reward_fn(self, weights, rets): # (100, 2), (100, 2)
        return torch.prod(torch.sum(rets * weights, 1) + 1) -1

    def train(self, train_loader):
        
        self.model.train()
        solvererror_counter = 0
        for i, data in enumerate(train_loader): 
            features = data[0].to(self.device).float()
            rets = data[1].to(self.device).float() 

            try:
                weights = self.model(features)
            except SolverError:
                # in case of SolverError, use the weight of last date
                print("SolverError occurs at times: ",solvererror_counter)
                solvererror_counter += 1
                if solvererror_counter == 10:
                    print("solvererror exceed")
                    raise RuntimeError
                continue
            
    
            self.train_optimizer.zero_grad()
            loss = - self.reward_fn(weights, rets)
            loss.backward()
            self.train_optimizer.step()

        return loss

    def valid(self, valid_loader):
        
        self.model.eval()

        rets = torch.zeros((len(valid_loader),self.train_args.batch_size))
        for i, data in enumerate(valid_loader):
            features = data[0].to(self.device).float()
            rets = data[1].to(self.device).float()
            
            weights = self.model(features)
            
            rets[i] = - self.reward_fn(weights, rets)

        return torch.prod(rets + 1) - 1

    def fit(self, features, rets, cut_epoch):
        features = features.transpose(0, 2, 1)
        train_dataset = MyDataset(features[:int(len(features*0.8))], rets[:int(len(features*0.8))])
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_args.batch_size,
            shuffle=False
        )

        valid_dataset = MyDataset(features[int(len(features*0.8)):], rets[int(len(features*0.8)):])
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.train_args.batch_size,
            shuffle=False
        )

        stop_steps = 0
        best_score = np.inf
        best_epoch = 0
        list_of_train_scores = []
        list_of_valid_scores = []



        for epoch in range(self.train_args.n_epochs):
            t1 = time()
            print(f"Epoch{epoch}:")
            
            train_score = self.train(train_loader)
            valid_score = self.valid(valid_loader)
            
            print(f"train score {train_score:.6f}, valid score {valid_score:.6f}")

            t2 = time()
            print(f'epoch: {epoch}', (f'time: {(t2 - t1):.4f}'))

            list_of_train_scores.append(train_score.detach().numpy())
            list_of_valid_scores.append(valid_score.detach().numpy())
            if (valid_score < best_score) and (epoch > self.train_args.cut_epoch): # ??
                best_score = valid_score
                stop_steps = 0
                best_epoch = epoch
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.train_args.early_stop:
                    print("early_stop")
                    break
            
                
        print(f"best score:{best_score:.6f} @ {best_epoch}")

        self.model.load_state_dict(best_param)

        if self.use_gpu:
            torch.cuda.empty_cache()

        return 0
