import random
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def _worker_init_fn(worker_id):
    """Fix DataLoader worker random seed for reproducible multi-process data loading."""
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)


def Sampler_train(k_shot, q_num, setting, seed):
    """
    Meta-training data sampler.

    For each training task, randomly sample k_shot support examples and q_num
    query examples from that task's data. The support set is used for inner-loop
    task adaptation; the query set for outer-loop meta-loss computation.

    Args:
        k_shot: number of support samples per task (few-shot K)
        q_num:  number of query samples per task
        setting: data split scheme (1/2/3/4)
        seed:   random seed for reproducible sampling
    Returns:
        support_loaders: list of support DataLoaders, one per task
        query_loaders:   list of query DataLoaders, one per task
        file_train:      training task list
    """
    support_loaders = []
    query_loaders = []

    file_train = pd.read_csv('Data/3.Task split/Setting_' + str(setting) + '/tasks_train.csv')
    data_train = pd.read_csv('Data/3.Task split/Setting_' + str(setting) + '/data_train.csv')
    random.seed(seed)
    data_train = data_train.sample(frac=1).reset_index(drop=True)  # shuffle data order

    for name in file_train.iloc[:, 0]:
        this_task = data_train[data_train['Label_name'] == name]

        random.seed(seed)

        # Non-overlapping sampling: support and query sets are disjoint
        support_list = random.sample(range(len(this_task)), k_shot)
        q_all = [m for m in range(len(this_task)) if m not in support_list]
        query_list = random.sample(q_all, q_num)

        support_label = torch.tensor(np.array(this_task.iloc[support_list, 2])).float()
        query_label = torch.tensor(np.array(this_task.iloc[query_list, 2])).float()

        support_feature = torch.tensor(np.array(this_task.iloc[support_list, 4:])).float()
        query_feature = torch.tensor(np.array(this_task.iloc[query_list, 4:])).float()

        support_dataset = TensorDataset(support_feature, support_label)
        query_dataset = TensorDataset(query_feature, query_label)

        support_loader = DataLoader(support_dataset, batch_size=k_shot, shuffle=False, num_workers=1, worker_init_fn=_worker_init_fn)
        query_loader = DataLoader(query_dataset, batch_size=q_num, shuffle=False, num_workers=1, worker_init_fn=_worker_init_fn)

        support_loaders.append(support_loader)
        query_loaders.append(query_loader)

    return support_loaders, query_loaders, file_train


def data_tune(k_shot, setting, phase, seed):
    """
    Evaluation-stage fine-tuning data sampler.

    For each task in validation/test set, randomly sample k_shot support examples
    (for fine-tuning) and the remaining query examples (for evaluation).
    Unlike the training stage, the query set here contains all samples of the task
    except those used for support.

    Args:
        k_shot: number of support samples
        setting: data split scheme
        phase:  'valid' or 'test'
        seed:   random seed (different seeds produce different support/query splits)
    """
    label, feature = [], []
    label_tune, feature_tune = [], []

    file_test = pd.read_csv('Data/3.Task split/Setting_' + str(setting) + '/tasks_'+str(phase)+'.csv')
    data_test = pd.read_csv('Data/3.Task split/Setting_' + str(setting) + '/data_'+str(phase)+'.csv')
    random.seed(seed)
    data_test = data_test.sample(frac=1).reset_index(drop=True)
    for name in file_test.iloc[:, 0]:
        this_task = data_test[data_test['Label_name'] == name]

        random.seed(seed)

        # Non-overlapping sampling
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
