import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import math
import argparse
from Sampler import Sampler_train, data_tune
from sklearn import metrics
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, input_dim, n_hidden_1, n_hidden_2, output_dim, droprate):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.mlp = torch.nn.Sequential(nn.Linear(input_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.Dropout(droprate), nn.ReLU(),
                                       nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.Dropout(droprate), nn.ReLU(),
                                       nn.Linear(n_hidden_2, output_dim), nn.BatchNorm1d(output_dim), nn.Dropout(droprate), nn.ReLU(),
                                       nn.Linear(self.output_dim, 1))

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, val=0)

    def forward(self, x):

        return self.mlp(x)


class FeatureAdapter(nn.Module):
    def __init__(self, in_dim, num_head, temperature):
        super(FeatureAdapter, self).__init__()
        self.num_head = num_head
        self.P = nn.Parameter(torch.empty(num_head, in_dim))
        nn.init.kaiming_uniform_(self.P, a=math.sqrt(5))
        self.heads = nn.ModuleList([nn.Linear(in_dim, in_dim, bias=True) for _ in range(num_head)])
        self.temperature = temperature

    def forward(self, x):
        s_hat = torch.stack([F.cosine_similarity(x, self.P[i], dim=-1) for i in range(self.num_head)], dim=-1)
        s = F.softmax(s_hat / self.temperature, dim=-1)
        weighted_features = sum([s[:, i].unsqueeze(-1) * self.heads[i](x) for i in range(self.num_head)])
        return x + weighted_features


class LabelAdapter(nn.Module):
    def __init__(self, x_dim, num_head, temperature, hid_dim):
        super(LabelAdapter, self).__init__()
        self.num_head = num_head
        self.linear = nn.Linear(x_dim, hid_dim, bias=False)
        self.P = nn.Parameter(torch.empty(num_head, hid_dim))
        nn.init.kaiming_uniform_(self.P, a=math.sqrt(5))
        self.heads = nn.ModuleList([nn.Linear(1, 1, bias=True) for _ in range(num_head)])
        self.weight = nn.Parameter(torch.empty(1, num_head))
        self.bias = nn.Parameter(torch.ones(1, num_head) / num_head)
        init.uniform_(self.weight, 0.75, 1.25)
        self.temperature = temperature

    def forward(self, x, y, inverse):
        v = self.linear(x.reshape(len(x), -1))
        gate = F.cosine_similarity(v.unsqueeze(1), self.P.unsqueeze(0), dim=-1)
        gate = F.softmax(gate / self.temperature, dim=-1)

        if inverse:
            adapted_y = (gate * (y.view(-1, 1) - self.bias) / (self.weight + 1e-9)).sum(-1)
        else:
            adapted_y = (gate * (self.weight + 1e-9) * y.view(-1, 1) + self.bias).sum(-1)

        return adapted_y


class DataAdapter(nn.Module):
    def __init__(self, args, Adapter):
        super(DataAdapter, self).__init__()
        self.predictor = MLP(args.input_dim, args.n_hidden_1, args.n_hidden_2, args.output_dim, args.droprate)
        self.FeatureAdapter = FeatureAdapter(args.input_dim, num_head=4, temperature=5)
        self.LabelAdapter = LabelAdapter(args.input_dim, num_head=4, temperature=5, hid_dim=16)
        self.Adapter = Adapter
        self.optimizer_inner = optim.Adam(self.predictor.parameters(), lr=args.base_lr)

        if Adapter == 'FeatureAdapter':
            self.optimizer_outer = optim.Adam(list(self.predictor.parameters())+list(self.FeatureAdapter.parameters()), lr=args.meta_lr)
        elif Adapter == 'LabelAdapter':
            self.optimizer_outer = optim.Adam(list(self.predictor.parameters()) + list(self.LabelAdapter.parameters()), lr=args.meta_lr)

        # self.criterion = nn.HuberLoss(delta=1.0)
        self.criterion = nn.MSELoss()

    def inner_loop(self, support_x, support_y, inverse=False):
        if self.Adapter == 'FeatureAdapter':
            adap_x = self.FeatureAdapter(support_x)
            pred = self.predictor(adap_x)
            loss_s = self.criterion(pred.flatten(), support_y)
            loss_reg = self.criterion(pred.flatten(), support_y)
            return loss_s, loss_reg, pred

        elif self.Adapter == 'LabelAdapter':
            pred = self.predictor(support_x)
            adap_y = self.LabelAdapter(support_x, support_y, inverse)
            loss_s = self.criterion(pred.flatten(), adap_y)
            loss_reg = self.criterion(pred.flatten(), support_y)
            return loss_s, loss_reg, pred

    def outer_loop(self, query_x, query_y, inverse=True):
        if self.Adapter == 'FeatureAdapter':
            adap_x = self.FeatureAdapter(query_x)
            pred = self.predictor(adap_x)
            loss_q = self.criterion(pred.flatten(), query_y)
            return loss_q, pred

        elif self.Adapter == 'LabelAdapter':
            pred = self.predictor(query_x)
            pred = self.LabelAdapter(query_x, pred, inverse)
            loss_q = self.criterion(pred.flatten(), query_y)
            return loss_q, pred


