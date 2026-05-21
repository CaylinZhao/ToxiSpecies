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
    """
    Feature Adapter — transforms the feature space via multi-head attention.
    Uses learnable prototype vectors P as "anchors": computes cosine similarity
    between input features and each prototype to derive attention weights, then
    applies multi-head linear transformations with a residual connection to
    preserve the original information.
    """
    def __init__(self, in_dim, num_head, temperature):
        super(FeatureAdapter, self).__init__()
        self.num_head = num_head                      # number of attention heads (default 4)
        self.P = nn.Parameter(torch.empty(num_head, in_dim))  # learnable prototype vectors
        nn.init.kaiming_uniform_(self.P, a=math.sqrt(5))
        self.heads = nn.ModuleList([nn.Linear(in_dim, in_dim, bias=True) for _ in range(num_head)])  # multi-head linear transforms
        self.temperature = temperature                # temperature coefficient controlling softmax sharpness

    def forward(self, x):
        # Compute cosine similarity between input features and each prototype → attention weights
        s_hat = torch.stack([F.cosine_similarity(x, self.P[i], dim=-1) for i in range(self.num_head)], dim=-1)
        s = F.softmax(s_hat / self.temperature, dim=-1)
        # Weighted combination of multi-head outputs with residual connection
        weighted_features = sum([s[:, i].unsqueeze(-1) * self.heads[i](x) for i in range(self.num_head)])
        return x + weighted_features


class LabelAdapter(nn.Module):
    """
    Label Adapter — applies adaptive scaling and shifting to label values
    to handle distribution differences across tasks. Uses a gating mechanism
    to fuse multi-head linear transforms (y' = weight * y + bias), mapping
    toxicity values of different tasks to a unified scale to mitigate
    task-to-task label distribution mismatch.

    Parameters:
        inverse=True  → inverse transform: maps adapted values back to original scale (outer loop prediction)
        inverse=False → forward transform: maps original labels to adapted space (inner loop training)
    """
    def __init__(self, x_dim, num_head, temperature, hid_dim):
        super(LabelAdapter, self).__init__()
        self.num_head = num_head
        self.linear = nn.Linear(x_dim, hid_dim, bias=False)  # project features to gating hidden space
        self.P = nn.Parameter(torch.empty(num_head, hid_dim))  # learnable gating prototypes
        nn.init.kaiming_uniform_(self.P, a=math.sqrt(5))
        self.heads = nn.ModuleList([nn.Linear(1, 1, bias=True) for _ in range(num_head)])
        self.weight = nn.Parameter(torch.empty(1, num_head))   # scaling factor
        self.bias = nn.Parameter(torch.ones(1, num_head) / num_head)  # bias term
        init.uniform_(self.weight, 0.75, 1.25)
        self.temperature = temperature

    def forward(self, x, y, inverse):
        # Compute gating weights: cosine similarity between projected features and prototypes
        v = self.linear(x.reshape(len(x), -1))
        gate = F.cosine_similarity(v.unsqueeze(1), self.P.unsqueeze(0), dim=-1)
        gate = F.softmax(gate / self.temperature, dim=-1)

        if inverse:
            # Inverse transform: y_original = (y_adapted - bias) / weight
            adapted_y = (gate * (y.view(-1, 1) - self.bias) / (self.weight + 1e-9)).sum(-1)
        else:
            # Forward transform: y_adapted = weight * y_original + bias
            adapted_y = (gate * (self.weight + 1e-9) * y.view(-1, 1) + self.bias).sum(-1)

        return adapted_y


class DataAdapter(nn.Module):
    """
    ToxiSpecies core model — Data Adapter.

    Integrates an MLP molecular property predictor with two adaptation strategies:
      - FeatureAdapter: transforms the feature space (for tasks with large feature distribution shifts)
      - LabelAdapter:   transforms the label space (for tasks with large label scale differences)

    Meta-training uses bilevel optimization:
      - inner_loop: updates predictor parameters on the support set (task-level fast adaptation)
      - outer_loop: updates adapter parameters on the query set (cross-task meta-knowledge learning)
    """
    def __init__(self, args, Adapter):
        super(DataAdapter, self).__init__()
        self.predictor = MLP(args.input_dim, args.n_hidden_1, args.n_hidden_2, args.output_dim, args.droprate)
        self.FeatureAdapter = FeatureAdapter(args.input_dim, num_head=4, temperature=5)
        self.LabelAdapter = LabelAdapter(args.input_dim, num_head=4, temperature=5, hid_dim=16)
        self.Adapter = Adapter
        # Inner-loop optimizer: updates only predictor params (task adaptation on support set)
        self.optimizer_inner = optim.Adam(self.predictor.parameters(), lr=args.base_lr)

        # Outer-loop optimizer: updates predictor + adapter params (meta-learning on query set)
        if Adapter == 'FeatureAdapter':
            self.optimizer_outer = optim.Adam(list(self.predictor.parameters())+list(self.FeatureAdapter.parameters()), lr=args.meta_lr)
        elif Adapter == 'LabelAdapter':
            self.optimizer_outer = optim.Adam(list(self.predictor.parameters()) + list(self.LabelAdapter.parameters()), lr=args.meta_lr)

        self.criterion = nn.MSELoss()

    def inner_loop(self, support_x, support_y, inverse=False):
        """
        Inner loop: fast task-level adaptation on the support set.
        Only predictor parameters are updated (via inner_optimizer); adapter params stay fixed.
        """
        if self.Adapter == 'FeatureAdapter':
            # Feature adaptation: transform feature space first, then predict
            adap_x = self.FeatureAdapter(support_x)
            pred = self.predictor(adap_x)
            loss_s = self.criterion(pred.flatten(), support_y)
            loss_reg = self.criterion(pred.flatten(), support_y)  # regularization = raw prediction loss
            return loss_s, loss_reg, pred

        elif self.Adapter == 'LabelAdapter':
            # Label adaptation: predict first, then transform in label space
            pred = self.predictor(support_x)
            adap_y = self.LabelAdapter(support_x, support_y, inverse)
            loss_s = self.criterion(pred.flatten(), adap_y)       # adaptation loss
            loss_reg = self.criterion(pred.flatten(), support_y)  # raw loss (regularization)
            return loss_s, loss_reg, pred

    def outer_loop(self, query_x, query_y, inverse=True):
        """
        Outer loop: compute meta-learning loss on the query set.
        Adapter + predictor parameters are updated (via outer_optimizer).
        """
        if self.Adapter == 'FeatureAdapter':
            adap_x = self.FeatureAdapter(query_x)
            pred = self.predictor(adap_x)
            loss_q = self.criterion(pred.flatten(), query_y)
            return loss_q, pred

        elif self.Adapter == 'LabelAdapter':
            pred = self.predictor(query_x)
            pred = self.LabelAdapter(query_x, pred, inverse)  # inverse=True: map predictions back to original scale
            loss_q = self.criterion(pred.flatten(), query_y)
            return loss_q, pred


