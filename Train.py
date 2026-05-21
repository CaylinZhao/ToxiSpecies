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


def _models_dir(args):
    return getattr(args, 'models_dir', 'Models')


def _lr_tag(args):
    return getattr(args, 'lr_tag', '') or ('lr' + str(args.base_lr))


def train(support_loaders, query_loaders, label_shot, feature_shot, label_valid, feature_valid, model, args, setting, Adapter):
    """
    Meta-training main loop.

    Uses MAML-style bilevel optimization:
      1. Inner Loop: for each task, take several gradient steps on the support set
         to rapidly adapt to that task (update predictor parameters)
      2. Outer Loop: for a batch of tasks, compute meta-loss on the query set,
         backpropagate, and update adapter + predictor parameters so the model
         learns "how to quickly adapt to new tasks"

    After each episode, evaluate on the validation set and save the checkpoint
    with the best MAE.
    """
    alpha = 0.1  # regularization weight: balances adaptation loss and raw prediction loss
    loss_e_list = []
    device = next(model.parameters()).device
    lr_tag = 'lr' + str(args.base_lr)
    best_mae = float('inf')
    md = _models_dir(args)
    best_path = md + '/' + str(Adapter) + '_Setting_' + str(setting) + '_' + lr_tag + '_s' + str(args.runseed) + '.pth'

    early_stopping = None
    if getattr(args, 'use_early_stopping', False):
        early_stopping = EarlyStopping(
            patience=args.patience,
            delta=1e-4,
            verbose=True,
            save_path=best_path,
            min_epochs=args.min_epochs,
        )

    for e in range(args.episodes):
        print('(', setting, args.runseed, ') episode ', e, 'start', flush=True)
        model.optimizer_outer.zero_grad()
        outer_loss_total = 0.0
        # ---- Iterate over all training tasks ----
        for t in range(len(support_loaders)):
            loss_reg = 0.0
            # Inner loop: update_step_inner gradient steps on the support set
            for i in range(args.update_step_inner):
                for _, batch in enumerate(support_loaders[t]):
                    support_x, support_y = batch[0].to(device), batch[1].to(device)
                    loss_s, loss_reg, _ = model.inner_loop(support_x, support_y, inverse=False)

                    model.optimizer_inner.zero_grad()
                    loss_s.backward()
                    model.optimizer_inner.step()

            # Outer loop: compute meta-loss on the query set
            for _, batch in enumerate(query_loaders[t]):
                query_x, query_y = batch[0].to(device), batch[1].to(device)
                loss_q, pred = model.outer_loop(query_x, query_y, inverse=True)
                task_outer_loss = loss_q + alpha * loss_reg.detach()  # add regularization term
                task_outer_loss.backward()
                outer_loss_total += task_outer_loss.detach().item()

        loss_t_q = torch.tensor(outer_loss_total / len(query_loaders), device=device)
        loss_e_list.append(loss_t_q.item())

        model.optimizer_outer.step()  # outer-loop parameter update

        # Evaluate on validation set, save best model
        model1 = copy.deepcopy(model)
        MAE = valid(label_shot, feature_shot, label_valid, feature_valid, model1, args)
        if MAE < best_mae:
            best_mae = MAE
            torch.save(model.state_dict(), best_path)

        if early_stopping is not None:
            early_stopping(MAE, model, epoch=e)
            if early_stopping.early_stop:
                print('Early stopping triggered', flush=True)
                break

        print('(', setting, args.runseed, ') episode ', e, '---train loss ', loss_t_q, '---valid MAE', round(MAE, 3), 'end', flush=True)

    # Plot training loss curve
    fig, ax = plt.subplots()
    ax.plot(range(1, len(loss_e_list) + 1), loss_e_list, label='query loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend(loc="best")
    plt.savefig('Results/Experiment setting/loss curve/'+str(Adapter)+'_Setting_' + str(setting) + '_' + lr_tag + '_s'+str(args.runseed)+'.png', dpi=300, bbox_inches='tight')


def valid(label_shot, feature_shot, label_valid, feature_valid, model, args):

    MAE = 0.0
    device = next(model.parameters()).device
    base_state = copy.deepcopy(model.state_dict())
    base_optimizer_state = copy.deepcopy(model.optimizer_inner.state_dict())

    for t in range(len(label_shot)):
        model.load_state_dict(base_state)
        model.optimizer_inner.load_state_dict(base_optimizer_state)
        model.train()
        for i in range(args.update_step_test):
            support_x = torch.tensor(np.array(feature_shot[t])).float().to(device)
            support_y = torch.tensor(np.array(label_shot[t])).float().to(device)
            loss_s, loss_reg, _ = model.inner_loop(support_x, support_y, inverse=False)
            model.optimizer_inner.zero_grad()
            loss_s.backward()
            model.optimizer_inner.step()
        model.eval()
        with torch.no_grad():
            query_x = torch.tensor(np.array(feature_valid[t])).float().to(device)
            query_y = torch.tensor(np.array(label_valid[t])).float().to(device)

            _, y_pred = model.outer_loop(query_x, query_y, inverse=True)
            y_true = query_y.view(y_pred.shape).detach().cpu().numpy()
            y_pred_np = y_pred.detach().cpu().numpy()
            MAE += metrics.mean_absolute_error(y_true, y_pred_np)

    model.load_state_dict(base_state)
    model.optimizer_inner.load_state_dict(base_optimizer_state)

    return MAE/len(label_shot)


def test(label_shot, feature_shot, label_test, feature_test, model, args, setting, Adapter, seed, split_name='test'):

    r2_score, RMSE, MAE, PCC, SCC = [], [], [], [], []
    device = next(model.parameters()).device
    lr_tag = _lr_tag(args)

    model.load_state_dict(torch.load(_models_dir(args) + '/'+str(Adapter)+'_Setting_' + str(setting) + '_' + lr_tag + '_s'+str(args.runseed)+'.pth', map_location=device))
    base_state = copy.deepcopy(model.state_dict())
    base_optimizer_state = copy.deepcopy(model.optimizer_inner.state_dict())

    for t in range(len(label_shot)):
        model.load_state_dict(base_state)
        model.optimizer_inner.load_state_dict(base_optimizer_state)
        model.train()

        for i in range(args.update_step_test):
            support_x = torch.tensor(np.array(feature_shot[t])).float().to(device)
            support_y = torch.tensor(np.array(label_shot[t])).float().to(device)
            loss_s, loss_reg, _ = model.inner_loop(support_x, support_y, inverse=False)

            model.optimizer_inner.zero_grad()
            loss_s.backward()
            model.optimizer_inner.step()

        model.eval()
        with torch.no_grad():
            query_x = torch.tensor(np.array(feature_test[t])).float().to(device)
            query_y = torch.tensor(np.array(label_test[t])).float().to(device)

            _, y_pred = model.outer_loop(query_x, query_y, inverse=True)
            y_true = query_y.view(y_pred.shape).detach().cpu().numpy()
            y_pred_np = y_pred.detach().cpu().numpy()

            MAE.append(metrics.mean_absolute_error(y_true, y_pred_np))
            RMSE.append(np.sqrt(metrics.mean_squared_error(y_true, y_pred_np)))
            r2_score.append(metrics.r2_score(y_true, y_pred_np))
            PCC.append(pearsonr(np.array(y_true).flatten(), np.array(y_pred_np).flatten())[0])
            SCC.append(spearmanr(np.array(y_true).flatten(), np.array(y_pred_np).flatten())[0])

    file_test = pd.read_csv('Data/3.Task split/Setting_' + str(setting) + '/tasks_' + str(split_name) + '.csv')
    results = pd.DataFrame([r2_score, RMSE, MAE, PCC, SCC],
                           index=['R2 score', 'RMSE', 'MAE', 'Pearson', 'Spearman'],
                           columns=file_test.iloc[:, 0].tolist())
    results['mean'] = [np.mean(results.loc[i]) for i in results.index]
    results['std'] = [np.std(results.loc[i]) for i in results.index]
    results.to_csv('Results/Experiment setting/seeds/'+str(Adapter)+'_Setting_' + str(setting) + '_' + lr_tag + '_' + split_name + '_s'+str(args.runseed)+'_s'+str(seed)+'.csv')


def test_da(label_shot, feature_shot, label_test, feature_test, model_fa, model_la, args, setting, seed, split_name='test'):

    r2_score, RMSE, MAE, PCC, SCC = [], [], [], [], []
    device = next(model_fa.parameters()).device
    lr_tag = _lr_tag(args)

    model_fa.load_state_dict(torch.load(_models_dir(args) + '/FeatureAdapter_Setting_' + str(setting) + '_' + lr_tag + '_s'+str(args.runseed)+'.pth', map_location=device))
    model_la.load_state_dict(torch.load(_models_dir(args) + '/LabelAdapter_Setting_' + str(setting) + '_' + lr_tag + '_s'+str(args.runseed)+'.pth', map_location=device))
    base_state_fa = copy.deepcopy(model_fa.state_dict())
    base_state_la = copy.deepcopy(model_la.state_dict())
    base_optimizer_state_fa = copy.deepcopy(model_fa.optimizer_inner.state_dict())
    base_optimizer_state_la = copy.deepcopy(model_la.optimizer_inner.state_dict())

    for t in range(len(label_shot)):
        model_fa.load_state_dict(base_state_fa)
        model_la.load_state_dict(base_state_la)
        model_fa.optimizer_inner.load_state_dict(base_optimizer_state_fa)
        model_la.optimizer_inner.load_state_dict(base_optimizer_state_la)
        model_fa.train()
        model_la.train()

        for i in range(args.update_step_test):
            support_x = torch.tensor(np.array(feature_shot[t])).float().to(device)
            support_y = torch.tensor(np.array(label_shot[t])).float().to(device)
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
            query_x = torch.tensor(np.array(feature_test[t])).float().to(device)
            query_y = torch.tensor(np.array(label_test[t])).float().to(device)

            _, y_pred_fa = model_fa.outer_loop(query_x, query_y, inverse=True)
            _, y_pred_la = model_la.outer_loop(query_x, query_y, inverse=True)
            y_pred = torch.mean(torch.stack([y_pred_fa.squeeze(-1), y_pred_la]), dim=0)
            y_true = query_y.view(y_pred_fa.shape).detach().cpu().numpy()
            y_pred_np = y_pred.detach().cpu().numpy()

            MAE.append(metrics.mean_absolute_error(y_true, y_pred_np))
            RMSE.append(np.sqrt(metrics.mean_squared_error(y_true, y_pred_np)))
            r2_score.append(metrics.r2_score(y_true, y_pred_np))
            PCC.append(pearsonr(np.array(y_true).flatten(), np.array(y_pred_np).flatten())[0])
            SCC.append(spearmanr(np.array(y_true).flatten(), np.array(y_pred_np).flatten())[0])

    file_test = pd.read_csv('Data/3.Task split/Setting_' + str(setting) + '/tasks_' + str(split_name) + '.csv')
    results = pd.DataFrame([r2_score, RMSE, MAE, PCC, SCC],
                           index=['R2 score', 'RMSE', 'MAE', 'Pearson', 'Spearman'],
                           columns=file_test.iloc[:, 0].tolist())
    results['mean'] = [np.mean(results.loc[i]) for i in results.index]
    results['std'] = [np.std(results.loc[i]) for i in results.index]
    results.to_csv('Results/Experiment setting/seeds/DoubleAdapter_Setting_' + str(setting) + '_' + lr_tag + '_' + split_name + '_s'+str(args.runseed)+'_s'+str(seed)+'.csv')


def test_cl(label_shot, feature_shot, label_test, feature_test, model_fa, model_la, args, setting, seed):

    device = next(model_fa.parameters()).device
    lr_tag = _lr_tag(args)
    model_fa.load_state_dict(torch.load(_models_dir(args) + '/FeatureAdapter_Setting_' + str(setting) + '_' + lr_tag + '_s'+str(args.runseed)+'.pth', map_location=device))
    model_la.load_state_dict(torch.load(_models_dir(args) + '/LabelAdapter_Setting_' + str(setting) + '_' + lr_tag + '_s'+str(args.runseed)+'.pth', map_location=device))

    # Reset inner optimizer: include adapter params so cross-domain adaptation
    # can update both the feature/label transformations and the predictor.
    inner_lr = getattr(args, 'base_lr', 1e-3)
    model_fa.optimizer_inner = torch.optim.Adam(
        list(model_fa.predictor.parameters()) + list(model_fa.FeatureAdapter.parameters()),
        lr=inner_lr,
    )
    model_la.optimizer_inner = torch.optim.Adam(
        list(model_la.predictor.parameters()) + list(model_la.LabelAdapter.parameters()),
        lr=inner_lr,
    )

    model_fa.train()
    model_la.train()

    # Use more adaptation steps for cross-domain transfer when k_shot is larger
    n_steps = getattr(args, 'update_step_test', 5)
    if getattr(args, 'k_shot_test', 16) >= 32:
        n_steps = max(n_steps, 10)
    if getattr(args, 'k_shot_test', 16) >= 64:
        n_steps = max(n_steps, 20)

    for i in range(n_steps):
        support_x = torch.tensor(np.array(feature_shot)).float().to(device)
        support_y = torch.tensor(np.array(label_shot)).float().to(device)
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
        query_x = torch.tensor(np.array(feature_test)).float().to(device)
        query_y = torch.tensor(np.array(label_test)).float().to(device)

        _, y_pred_fa = model_fa.outer_loop(query_x, query_y, inverse=True)
        _, y_pred_la = model_la.outer_loop(query_x, query_y, inverse=True)
        y_emb = torch.stack([y_pred_fa.squeeze(-1), y_pred_la])
        y_pred = torch.mean(y_emb, dim=0)

    return y_pred, y_emb, model_fa, model_la

