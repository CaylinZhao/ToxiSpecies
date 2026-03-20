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
from early_stopping import EarlyStopping


def train(support_loaders, query_loaders, label_shot, feature_shot, label_valid, feature_valid, model, args, setting, Adapter):
    """
    Core Meta-Learning Training Loop (MAML approach).
    
    Args:
        support_loaders: Task-specific support sets for inner-loop adaptation.
        query_loaders: Task-specific query sets for outer-loop meta-update.
        model: Base neural network with adapter layers.
        args: Hyperparameters (learning rates, update steps, etc.).
        Adapter: Type of adapter used (FA or LA).
    """
    alpha = 0.1
    MAE_min = 100.0
    loss_e_list = []
    # Loop through meta-episodes (Outer Loop)
    for e in range(args.episodes):
        loss_q_all = 0.0
        # Iterate over tasks in the current meta-batch
        for t in range(len(support_loaders)):
            loss_reg = 0.0
            # Inner Loop: Fast adaptation to the specific task 't' using Support Set
            for i in range(args.update_step_inner):
                for _, batch in enumerate(support_loaders[t]):
                    support_x, support_y = batch[0], batch[1]
                    loss_s, loss_reg, _ = model.inner_loop(support_x, support_y, inverse=False)

                    model.optimizer_inner.zero_grad()
                    loss_s.backward()
                    model.optimizer_inner.step()

            for _, batch in enumerate(query_loaders[t]):
                query_x, query_y = batch[0], batch[1]
                loss_q, pred = model.outer_loop(query_x, query_y, inverse=True)
                loss_q_all += (loss_q.item() + alpha * loss_reg.item())

        loss_t_q = torch.tensor(loss_q_all /len(query_loaders), requires_grad=True)
        loss_e_list.append(loss_t_q.item())

        model.optimizer_outer.zero_grad()
        loss_t_q.backward()
        model.optimizer_outer.step()

        model1 = copy.deepcopy(model)
        MAE = valid(label_shot, feature_shot, label_valid, feature_valid, model1, args)
        if MAE < MAE_min:
            MAE_min = MAE
            best_epoch = e
            torch.save(model.state_dict(), 'Models/'+str(Adapter)+'_Setting_' + str(setting) + '_s'+str(args.runseed)+'.pth')
        # early_stopping(MAE, model)
        # if early_stopping.early_stop:
        #     print("Early stopping triggered")
        #     break
        print('(', setting, args.runseed, ') episode ', e, '---train loss ', loss_t_q, '---valid MAE', MAE.round(3))

    fig, ax = plt.subplots()
    ax.plot(range(1, len(loss_e_list) + 1), loss_e_list, label='query loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend(loc="best")
    plt.savefig('Results/Experiment setting/loss curve/'+str(Adapter)+'_Setting_' + str(setting) + '_s'+str(args.runseed)+'.png', dpi=300, bbox_inches='tight')


def valid(label_shot, feature_shot, label_valid, feature_valid, model, args):

    MAE = 0.0

    for t in range(len(label_shot)):
        for i in range(args.update_step_test):
            support_x, support_y = torch.tensor(np.array(feature_shot[t])).float(), torch.tensor(np.array(label_shot[t])).float()
            loss_s, loss_reg, _ = model.inner_loop(support_x, support_y, inverse=False)
            model.optimizer_inner.zero_grad()
            loss_s.backward()
            model.optimizer_inner.step()
        model.eval()
        with torch.no_grad():
            query_x = torch.tensor(np.array(feature_valid[t])).float()
            query_y = torch.tensor(np.array(label_valid[t])).float()

            _, y_pred = model.outer_loop(query_x, query_y, inverse=True)
            y_true = query_y.view(y_pred.shape)
            MAE += metrics.mean_absolute_error(y_true, y_pred)

    return MAE/len(label_shot)


def test(label_shot, feature_shot, label_test, feature_test, model, args, setting, Adapter, seed):

    r2_score, RMSE, MAE, PCC, SCC = [], [], [], [], []

    for t in range(len(label_shot)):
        model.load_state_dict(torch.load('Models/'+str(Adapter)+'_Setting_' + str(setting) + '_s'+str(args.runseed)+'.pth'))

        for i in range(args.update_step_test):
            support_x, support_y = torch.tensor(np.array(feature_shot[t])).float(), torch.tensor(np.array(label_shot[t])).float()
            loss_s, loss_reg, _ = model.inner_loop(support_x, support_y, inverse=False)

            model.optimizer_inner.zero_grad()
            loss_s.backward()
            model.optimizer_inner.step()

        model.eval()
        with torch.no_grad():
            query_x = torch.tensor(np.array(feature_test[t])).float()
            query_y = torch.tensor(np.array(label_test[t])).float()

            _, y_pred = model.outer_loop(query_x, query_y, inverse=True)
            y_true = query_y.view(y_pred.shape)

            MAE.append(metrics.mean_absolute_error(y_true, y_pred))
            RMSE.append(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
            r2_score.append(metrics.r2_score(y_true, y_pred))
            PCC.append(pearsonr(np.array(y_true).flatten(), np.array(y_pred).flatten())[0])
            SCC.append(spearmanr(np.array(y_true).flatten(), np.array(y_pred).flatten())[0])

    file_test = pd.read_csv('Data/3.Task split/Setting_' + str(setting) + '/tasks_test.csv')
    results = pd.DataFrame([r2_score, RMSE, MAE, PCC, SCC],
                           index=['R2 score', 'RMSE', 'MAE', 'Pearson', 'Spearman'],
                           columns=file_test.iloc[:, 0].tolist())
    results['mean'] = [np.mean(results.loc[i]) for i in results.index]
    results['std'] = [np.std(results.loc[i]) for i in results.index]
    results.to_csv('Results/Experiment setting/seeds/'+str(Adapter)+'_Setting_' + str(setting) + '_s'+str(args.runseed)+'_s'+str(seed)+'.csv')


def test_da(label_shot, feature_shot, label_test, feature_test, model_fa, model_la, args, setting, seed):

    r2_score, RMSE, MAE, PCC, SCC = [], [], [], [], []

    for t in range(len(label_shot)):
        model_fa.load_state_dict(torch.load('Models/FeatureAdapter_Setting_' + str(setting) + '_s'+str(args.runseed)+'.pth'))
        model_la.load_state_dict(torch.load('Models/LabelAdapter_Setting_' + str(setting) + '_s'+str(args.runseed)+'.pth'))

        for i in range(args.update_step_test):
            support_x, support_y = torch.tensor(np.array(feature_shot[t])).float(), torch.tensor(np.array(label_shot[t])).float()
            loss_s, loss_reg, _ = model_fa.inner_loop(support_x, support_y, inverse=False)
            model_fa.optimizer_inner.zero_grad()
            loss_s.backward()
            model_fa.optimizer_inner.step()

            loss_s, loss_reg, _ = model_la.inner_loop(support_x, support_y, inverse=False)
            model_la.optimizer_inner.zero_grad()
            loss_s.backward()
            model_la.optimizer_inner.step()

        model_fa.eval()
        model_la.eval()
        with torch.no_grad():
            query_x = torch.tensor(np.array(feature_test[t])).float()
            query_y = torch.tensor(np.array(label_test[t])).float()

            _, y_pred_fa = model_fa.outer_loop(query_x, query_y, inverse=True)
            _, y_pred_la = model_la.outer_loop(query_x, query_y, inverse=True)
            y_pred = torch.mean(torch.stack([y_pred_fa.squeeze(-1), y_pred_la]), dim=0)

            y_true = query_y.view(y_pred_fa.shape)

            MAE.append(metrics.mean_absolute_error(y_true, y_pred))
            RMSE.append(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
            r2_score.append(metrics.r2_score(y_true, y_pred))
            PCC.append(pearsonr(np.array(y_true).flatten(), np.array(y_pred).flatten())[0])
            SCC.append(spearmanr(np.array(y_true).flatten(), np.array(y_pred).flatten())[0])

    file_test = pd.read_csv('Data/3.Task split/Setting_' + str(setting) + '/tasks_test.csv')
    results = pd.DataFrame([r2_score, RMSE, MAE, PCC, SCC],
                           index=['R2 score', 'RMSE', 'MAE', 'Pearson', 'Spearman'],
                           columns=file_test.iloc[:, 0].tolist())
    results['mean'] = [np.mean(results.loc[i]) for i in results.index]
    results['std'] = [np.std(results.loc[i]) for i in results.index]
    results.to_csv('Results/Experiment setting/seeds/DoubleAdapter_Setting_' + str(setting) + '_s'+str(args.runseed)+'_s'+str(seed)+'.csv')


def test_cl(label_shot, feature_shot, label_test, feature_test, model_fa, model_la, args, setting, seed):

    model_fa.load_state_dict(torch.load('Models/FeatureAdapter_Setting_' + str(setting) + '_s'+str(args.runseed)+'.pth'))
    model_la.load_state_dict(torch.load('Models/LabelAdapter_Setting_' + str(setting) + '_s'+str(args.runseed)+'.pth'))

    for i in range(args.update_step_test):
        support_x, support_y = torch.tensor(np.array(feature_shot)).float(), torch.tensor(np.array(label_shot)).float()
        loss_s, loss_reg, _ = model_fa.inner_loop(support_x, support_y, inverse=False)
        model_fa.optimizer_inner.zero_grad()
        loss_s.backward()
        model_fa.optimizer_inner.step()

        loss_s, loss_reg, _ = model_la.inner_loop(support_x, support_y, inverse=False)
        model_la.optimizer_inner.zero_grad()
        loss_s.backward()
        model_la.optimizer_inner.step()

    model_fa.eval()
    model_la.eval()
    with torch.no_grad():
        query_x = torch.tensor(np.array(feature_test)).float()
        query_y = torch.tensor(np.array(label_test)).float()

        _, y_pred_fa = model_fa.outer_loop(query_x, query_y, inverse=True)
        _, y_pred_la = model_la.outer_loop(query_x, query_y, inverse=True)
        y_emb = torch.stack([y_pred_fa.squeeze(-1), y_pred_la])
        y_pred = torch.mean(y_emb, dim=0)

    return y_pred, y_emb, model_fa, model_la

