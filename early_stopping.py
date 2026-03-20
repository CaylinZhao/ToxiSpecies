import torch
import torch.nn as nn
import torch.optim as optim


class EarlyStopping:
    """
    Early stopping helper to terminate training when validation MAE doesn't improve.
    
    Args:
        patience (int): Number of checks to wait after last improvement.
        delta (float): Minimum change to qualify as an improvement.
        verbose (bool): Whether to log progress.
        save_path (str): Path to save the best model checkpoint.
    """
    def __init__(self, patience=5, delta=0.0001, verbose=False, save_path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.save_path = save_path
        self.best_mae = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, mae, model):
        """Processes the current validation result and updates the stopping state."""
        if self.best_mae is None:
            self.best_mae = mae
            self.save_checkpoint(mae, model)
        elif mae > self.best_mae - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_mae = mae
            self.save_checkpoint(mae, model)
            self.counter = 0

    def save_checkpoint(self, mae, model):
        torch.save(model.state_dict(), self.save_path)
        self.best_mae = mae

