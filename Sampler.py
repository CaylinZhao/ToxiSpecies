import random
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def Sampler_train(k_shot, q_num, setting, seed):

    support_loaders = []
    query_loaders = []

    file_train = pd.read_csv('Data/3.Task split/Setting_' + str(setting) + '/tasks_train.csv')
    data_train = pd.read_csv('Data/3.Task split/Setting_' + str(setting) + '/data_train.csv')
    random.seed(seed)
    data_train = data_train.sample(frac=1).reset_index(drop=True)

    for name in file_train.iloc[:, 0]:
        this_task = data_train[data_train['Label_name'] == name]

        random.seed(seed)

        support_list = random.sample(range(len(this_task)), k_shot)
        q_all = [m for m in range(len(this_task)) if m not in support_list]
        query_list = random.sample(q_all, q_num)

        support_label = torch.tensor(np.array(this_task.iloc[support_list, 2])).float()
        query_label = torch.tensor(np.array(this_task.iloc[query_list, 2])).float()

        support_feature = torch.tensor(np.array(this_task.iloc[support_list, 4:])).float()
        query_feature = torch.tensor(np.array(this_task.iloc[query_list, 4:])).float()

        support_dataset = TensorDataset(support_feature, support_label)
        query_dataset = TensorDataset(query_feature, query_label)

        support_loader = DataLoader(support_dataset, batch_size=k_shot, shuffle=False, num_workers=1)
        query_loader = DataLoader(query_dataset, batch_size=q_num, shuffle=False, num_workers=1)

        support_loaders.append(support_loader)
        query_loaders.append(query_loader)

    return support_loaders, query_loaders, file_train


def data_tune(k_shot, setting, phase, seed):

    label, feature = [], []
    label_tune, feature_tune = [], []

    file_test = pd.read_csv('Data/3.Task split/Setting_' + str(setting) + '/tasks_'+str(phase)+'.csv')
    data_test = pd.read_csv('Data/3.Task split/Setting_' + str(setting) + '/data_'+str(phase)+'.csv')
    random.seed(seed)
    data_test = data_test.sample(frac=1).reset_index(drop=True)
    for name in file_test.iloc[:, 0]:
        this_task = data_test[data_test['Label_name'] == name]

        random.seed(seed)

        support_list = random.sample(range(len(this_task)), k_shot)
        query_list = [m for m in range(len(this_task)) if m not in support_list]

        label_tune.append(this_task.iloc[support_list, 2])
        feature_tune.append(this_task.iloc[support_list, 4:])

        label.append(this_task.iloc[query_list, 2])
        feature.append(this_task.iloc[query_list, 4:])

    return label_tune, feature_tune, label, feature


def data_cl(k_shot, setting, phase, seed, name):

    data_test = pd.read_csv('Data/3.Task split/Setting_' + str(setting) + '/data_'+str(phase)+'.csv')
    random.seed(seed)
    data_test = data_test.sample(frac=1).reset_index(drop=True)
    this_task = data_test[data_test['Label_name'] == name]

    random.seed(seed)

    support_list = random.sample(range(len(this_task)), k_shot)
    query_list = [m for m in range(len(this_task)) if m not in support_list]

    label_tune = this_task.iloc[support_list, 2]
    feature_tune = this_task.iloc[support_list, 4:]

    label = this_task.iloc[query_list, 2]
    feature = this_task.iloc[query_list, 4:]

    compound = this_task.iloc[query_list, 3]
    cid = this_task.iloc[query_list, 0]

    return label_tune, feature_tune, label, feature, compound, cid
