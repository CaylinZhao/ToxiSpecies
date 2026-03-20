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


def argument():
    """
    Parses command line arguments for the Meta_Tox framework using Label Adapter.
    """
    parser = argparse.ArgumentParser(description='PyTorch implementation of Meta_Tox')
    # Device and environment reproducibility
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--decay', type=float, default=0, help='weight decay (default: 0)')
    parser.add_argument('--runseed', type=int, default=2, help='Seed for minibatch selection, random initialization.')

    # Global optimization parameters
    parser.add_argument('--batch_size', type=int, default=24, help='input batch size for training (default: 32)')
    parser.add_argument('--episodes', type=int, default=60, help='number of episodes to train (default: 200)')
    parser.add_argument('--base_lr', type=float, default=1e-3, help='Learning rate for task adaptation (inner loop)')
    parser.add_argument('--meta_lr', type=float, default=1e-3, help='Learning rate for global meta-update (outer loop)')
    parser.add_argument('--update_step_inner', type=int, default=5, help='Number of gradient steps for adaptation in training')
    parser.add_argument('--update_step_test', type=int, default=5, help='Number of gradient steps for adaptation in inference')

    # Backbone network architecture
    parser.add_argument('--input_dim', type=int, default=2048, help='Molecular fingerprint input dimension')
    parser.add_argument('--n_hidden_1', type=int, default=512, help='First layer hidden units')
    parser.add_argument('--n_hidden_2', type=int, default=256, help='Second layer hidden units')
    parser.add_argument('--output_dim', type=int, default=128, help='Output mapping dimension')
    parser.add_argument('--droprate', type=float, default=0.1, help='Dropout rate for all layers')

    # Data sampling configuration for Meta-tasks
    parser.add_argument('--n_q_train', type=int, default=90, help='Size of the Query Set in Meta-Training phase')
    parser.add_argument('--k_shot_train', type=int, default=32, help='Size of the Support Set in Meta-Training phase')
    parser.add_argument('--k_shot_test', type=int, default=16, help='Size of the Support Set in Meta-Testing phase')

    args = parser.parse_args()

    return args


def main(setting, args):
    """
    Main training and evaluation pipeline for Label Adapter (LA).
    """
    Adapters = ['FeatureAdapter', 'LabelAdapter']
    Adapter = Adapters[1]  # Select Label Adapter strategy

    # Global Random Seed setting for reproducibility
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    torch.cuda.manual_seed_all(args.runseed)

    # Initialize a DataAdapter instance configured for LA
    model = DataAdapter(args, Adapter)

    # Sample tasks and data for training and validation scenarios
    support_loaders, query_loaders, file_train = Sampler_train(args.k_shot_train, args.n_q_train, setting, seed=args.runseed)
    label_shot, feature_shot, label_valid, feature_valid = data_tune(args.k_shot_test, setting, 'valid', seed=args.runseed)
    
    # Entrance for the Meta-Training process
    train(support_loaders, query_loaders, label_shot, feature_shot, label_valid, feature_valid, model, args, setting, Adapter)

    # Performance evaluation across multiple independent test seeds (20 trials)
    for seed in range(20):
        label_tune, feature_tune, label_test, feature_test = data_tune(args.k_shot_test, setting, 'test', seed=seed)
        test(label_tune, feature_tune, label_test, feature_test, model, args, setting, Adapter, seed)

    # Post-experiment result aggregation and statistical averaging
    df_list = [pd.read_csv('Results/Experiment setting/seeds/' + str(Adapter) + '_Setting_' + str(setting) + '_s' + str(args.runseed) + '_s' + str(seed) + '.csv', index_col=0) for seed in range(20)]
    combined_df = pd.concat(df_list, axis=0)
    result = combined_df.groupby(combined_df.index).mean()
    result = result.reindex(['R2 score', 'RMSE', 'MAE', 'Pearson', 'Spearman'])
    result = result.round(3)
    # Save optimized results to file
    result.to_csv('Results/Experiment setting/' + str(Adapter) + '_' + str(setting) + '_s' + str(args.runseed) + '.csv')

    print('ok')


if __name__ == '__main__':

    for setting in ['2_1']:
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

        for s in [3]:
            args.runseed = s
            main(setting, args)

