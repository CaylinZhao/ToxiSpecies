import random
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def Sampler_train(k_shot, q_num, setting, seed):
    """
    Samples support and query sets for Meta-Training.
    
    Args:
        k_shot (int): Number of samples in the support set per task.
        q_num (int): Number of samples in the query set per task.
        setting (str): Experimental scenario identifier (e.g., '1_1').
        seed (int): Random seed for reproducibility.
        
    Returns:
        support_loaders (list): List of DataLoaders for support sets across all training tasks.
        query_loaders (list): List of DataLoaders for query sets across all training tasks.
        file_train (DataFrame): List of task metadata.
    """
    support_loaders = []
    query_loaders = []

    file_train = pd.read_csv('Data/3.Task split/Setting_' + str(setting) + '/tasks_train.csv')
    data_train = pd.read_csv('Data/3.Task split/Setting_' + str(setting) + '/data_train.csv')
    random.seed(seed)
    # Shuffle the dataset to ensure diverse batches
    data_train = data_train.sample(frac=1).reset_index(drop=True)

    for name in file_train.iloc[:, 0]:
        # Filter samples belonging to the current task (species/endpoint)
        this_task = data_train[data_train['Label_name'] == name]

        random.seed(seed)

        # Randomly sample support indices and use the rest for query set
        support_list = random.sample(range(len(this_task)), k_shot)
        q_all = [m for m in range(len(this_task)) if m not in support_list]
        query_list = random.sample(q_all, q_num)

        # Toxicity labels are typically in the 3rd column (index 2)
        support_label = torch.tensor(np.array(this_task.iloc[support_list, 2])).float()
        query_label = torch.tensor(np.array(this_task.iloc[query_list, 2])).float()

        # Features (molecular fingerprints) start from the 5th column (index 4)
        support_feature = torch.tensor(np.array(this_task.iloc[support_list, 4:])).float()
        query_feature = torch.tensor(np.array(this_task.iloc[query_list, 4:])).float()

        support_dataset = TensorDataset(support_feature, support_label)
        query_dataset = TensorDataset(query_feature, query_label)

        # Use batch_size = k_shot to process the whole support set in one step of the inner loop
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

