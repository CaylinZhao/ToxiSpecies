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
from Model import DataAdapter
from Train import train, test_da


def argument():
    parser = argparse.ArgumentParser(description='PyTorch implementation of Meta_Tox')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--decay', type=float, default=0, help='weight decay (default: 0)')
    parser.add_argument('--runseed', type=int, default=4, help='Seed for minibatch selection, random initialization.')

    parser.add_argument('--batch_size', type=int, default=24, help='input batch size for training (default: 32)')
    parser.add_argument('--episodes', type=int, default=60, help='number of episodes to train (default: 200)')
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--meta_lr', type=float, default=1e-3)
    parser.add_argument('--update_step_inner', type=int, default=5)
    parser.add_argument('--update_step_test', type=int, default=5)

    parser.add_argument('--input_dim', type=int, default=2048, help='input dimensions (default: 200, 167, 1024, 3705)')
    parser.add_argument('--n_hidden_1', type=int, default=512, help='input dimensions (default: 300)')
    parser.add_argument('--n_hidden_2', type=int, default=256, help='input dimensions (default: 300)')
    parser.add_argument('--output_dim', type=int, default=128, help='input dimensions (default: 100)')
    parser.add_argument('--droprate', type=float, default=0.1)

    parser.add_argument('--n_q_train', type=int, default=48, help='size of the query train dataset')
    parser.add_argument('--k_shot_train', type=int, default=108, help='size of the train support dataset')
    parser.add_argument('--k_shot_test', type=int, default=16, help='size of the test support dataset')

    args = parser.parse_args()

    return args


def ensemble(setting, args):

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    torch.cuda.manual_seed_all(args.runseed)

    adapters = ['FeatureAdapter', 'LabelAdapter']

    model_fa = DataAdapter(args, adapters[0])
    model_la = DataAdapter(args, adapters[1])

    for seed in range(20):
        label_tune, feature_tune, label_test, feature_test = data_tune(args.k_shot_test, setting, 'test', seed=seed)
        test_da(label_tune, feature_tune, label_test, feature_test, model_fa, model_la, args, setting, seed)

    df_list = [pd.read_csv('Results/Experiment setting/seeds/DoubleAdapter_Setting_' + str(setting) + '_s' + str(args.runseed) + '_s' + str(seed) + '.csv', index_col=0) for seed in range(20)]
    combined_df = pd.concat(df_list, axis=0)
    result = combined_df.groupby(combined_df.index).mean()
    result = result.reindex(['R2 score', 'RMSE', 'MAE', 'Pearson', 'Spearman'])
    result = result.round(3)
    result.to_csv('Results/Experiment setting/DoubleAdapter_' + str(setting) + '_s' + str(args.runseed) + '.csv')

    print('ok')


for setting in ['1_1', '1_2', '2_1', '2_2', '2_3']:
    args = argument()
    if setting == '1_1':
        args.k_shot_train = 32
        args.n_q_train = 90
    elif setting == '1_2':
        args.k_shot_train = 48
        args.n_q_train = 108
    elif setting == '2_1':
        args.k_shot_train = 33
        args.n_q_train = 90
    elif setting == '2_2':
        args.k_shot_train = 35
        args.n_q_train = 90
    elif setting == '2_3':
        args.k_shot_train = 32
        args.n_q_train = 60

    for s in range(5):
        args.runseed = s
        ensemble(setting, args)



