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
from Train import train, test
import random


def set_seed(seed):
    # Make randomness deterministic for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def argument():
    parser = argparse.ArgumentParser(description='PyTorch implementation of Meta_Tox')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--decay', type=float, default=0, help='weight decay (default: 0)')
    parser.add_argument('--runseed', type=int, default=2, help='Seed for minibatch selection, random initialization.')
    parser.add_argument('--setting', type=str, default=None, choices=['1', '2', '3', '4'], help='run only one setting when specified')

    parser.add_argument('--batch_size', type=int, default=24, help='input batch size for training (default: 32)')
    parser.add_argument('--episodes', type=int, default=60, help='number of episodes to train (default: 200)')
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--meta_lr', type=float, default=1e-3)
    parser.add_argument('--update_step_inner', type=int, default=5)
    parser.add_argument('--update_step_test', type=int, default=5)
    parser.add_argument('--eval_seeds', type=int, default=20, help='number of evaluation seeds')
    parser.add_argument('--use_early_stopping', action='store_true', default=False, help='enable early stopping')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience based on valid MAE')
    parser.add_argument('--min_epochs', type=int, default=30, help='minimum episodes before early stopping')

    parser.add_argument('--input_dim', type=int, default=2048, help='input dimensions (default: 200, 167, 1024, 3705)')
    parser.add_argument('--n_hidden_1', type=int, default=512, help='input dimensions (default: 300)')
    parser.add_argument('--n_hidden_2', type=int, default=256, help='input dimensions (default: 300)')
    parser.add_argument('--output_dim', type=int, default=128, help='input dimensions (default: 100)')
    parser.add_argument('--droprate', type=float, default=0.1)

    parser.add_argument('--n_q_train', type=int, default=90, help='size of the query train dataset')
    parser.add_argument('--k_shot_train', type=int, default=32, help='size of the train support dataset')
    parser.add_argument('--k_shot_test', type=int, default=16, help='size of the test support dataset')

    args = parser.parse_args()

    return args


def main(setting, args):

    Adapters = ['FeatureAdapter', 'LabelAdapter']
    Adapter = Adapters[0]

    set_seed(args.runseed)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    model = DataAdapter(args, Adapter).to(device)
    lr_tag = 'lr' + str(args.base_lr)

    support_loaders, query_loaders, file_train = Sampler_train(args.k_shot_train, args.n_q_train, setting, seed=args.runseed)
    label_shot, feature_shot, label_valid, feature_valid = data_tune(args.k_shot_test, setting, 'valid', seed=args.runseed)
    train(support_loaders, query_loaders, label_shot, feature_shot, label_valid, feature_valid, model, args, setting, Adapter)

    for split_name in ['valid', 'test']:
        for seed in range(args.eval_seeds):
            label_tune, feature_tune, label_eval, feature_eval = data_tune(args.k_shot_test, setting, split_name, seed=seed)
            test(label_tune, feature_tune, label_eval, feature_eval, model, args, setting, Adapter, seed, split_name=split_name)

        df_list = [pd.read_csv('Results/Experiment setting/seeds/' + str(Adapter) + '_Setting_' + str(setting) + '_' + lr_tag + '_' + split_name + '_s' + str(args.runseed) + '_s' + str(seed) + '.csv', index_col=0) for seed in range(args.eval_seeds)]
        combined_df = pd.concat(df_list, axis=0)
        result = combined_df.groupby(combined_df.index).mean()
        result = result.reindex(['R2 score', 'RMSE', 'MAE', 'Pearson', 'Spearman'])
        result = result.round(3)
        result.to_csv('Results/Experiment setting/' + str(Adapter) + '_' + str(setting) + '_' + lr_tag + '_' + split_name + '_s' + str(args.runseed) + '.csv')

    print('ok', flush=True)


if __name__ == '__main__':

    args = argument()
    settings = [args.setting] if args.setting else ['1', '2', '3', '4']

    for setting in settings:
        if setting == '2':
            args.k_shot_train = 32
            args.n_q_train = 90
        elif setting == '1':
            args.k_shot_train = 48
            args.n_q_train = 108
        elif setting == '3':
            args.k_shot_train = 33
            args.n_q_train = 90
        elif setting == '4':
            args.k_shot_train = 32
            args.n_q_train = 60

        for s in range(5):
            args.runseed = s
            set_seed(args.runseed)
            main(setting, args)

