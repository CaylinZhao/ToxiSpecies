import argparse

def argument():
    parser = argparse.ArgumentParser(description='PyTorch implementation of Meta_Tox')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=24, help='input batch size for training (default: 32)')
    parser.add_argument('--episodes', type=int, default=80, help='number of episodes to train (default: 200)')
    parser.add_argument('--input_dim', type=int, default=2048, help='input dimensions (default: 200, 167, 1024, 3705)')
    parser.add_argument('--n_hidden_1', type=int, default=512, help='input dimensions (default: 300)')
    parser.add_argument('--n_hidden_2', type=int, default=256, help='input dimensions (default: 300)')
    parser.add_argument('--output_dim', type=int, default=128, help='input dimensions (default: 100)')
    parser.add_argument('--droprate', type=float, default=0.1)
    parser.add_argument('--decay', type=float, default=0, help='weight decay (default: 0)')
    parser.add_argument('--runseed', type=int, default=0, help='Seed for minibatch selection, random initialization.')
    parser.add_argument('--n_q_train', type=int, default=60, help='size of the query train dataset')
    parser.add_argument('--k_shot_train', type=int, default=60, help='size of the train support dataset')
    parser.add_argument('--k_shot_test', type=int, default=10, help='size of the test support dataset')
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--meta_lr', type=float, default=1e-3)
    parser.add_argument('--update_step_inner', type=int, default=3)
    parser.add_argument('--update_step_outer', type=int, default=5)
    parser.add_argument('--update_step_test', type=int, default=5)
    parser.add_argument('--num_train_tasks', type=int, default=12, help='# of training tasks')

    args = parser.parse_args()

    return args
