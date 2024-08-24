import copy
import torch
from torch.utils.data import Dataset
import numpy as np
import configparser
import pandas as pd

class ARGS():
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

class MyDataset(Dataset):
    def __init__(self, features, rets):
        self.features = copy.deepcopy(features)
        self.rets = copy.deepcopy(rets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.rets[idx])

def load_data_args(cfg_file):
    conf = configparser.ConfigParser()
    conf.read(cfg_file)
    args = ARGS(
        features_path=conf.get("data","features_path"),
        rets_path=conf.get("data","rets_path")
    )
    return args

def load_data(args):
    features = np.load(args.features_path)
    rets = np.load(args.rets_path)
    return features, rets

def load_train_args(cfg_file):
    conf = configparser.ConfigParser()
    conf.read(cfg_file)
    args = ARGS(
        early_stop=conf.getint("train","early_stop"),
        optimizer=conf.get("train","optimizer"),
        batch_size=conf.getint("train","batch_size"),
        n_epochs=conf.getint("train","n_epochs"),
        lr=conf.getfloat("train","lr"),
        seed=conf.getint("train","seed"),
        period=conf.getint("train","period"),
        cut_epoch=conf.getint("train","cut_epoch"),
    )
    return args



def load_model_args(cfg_file):
    conf = configparser.ConfigParser()
    conf.read(cfg_file)
    model_args=ARGS(
        GPU = conf.getint("model","GPU"),
        num_features=conf.getint("model","num_features"),
        num_asset=conf.getint("model","num_asset"),
        num_layers=conf.getint("model","num_layers"),
        input_size=conf.getint("model","input_size"),
        hidden_size=conf.getint("model","hidden_size"),
        cvxver = conf.getint("model","cvxver"),
        budget_cons = conf.getboolean("model","budget_cons"),
        lowerbound = conf.getfloat("model","lowerbound"),
        upperbound = conf.getfloat("model","upperbound"),
        weightbound=conf.getfloat("model","weightbound"),
        max_weight=conf.getfloat("model","max_weight")
        )
    return model_args


