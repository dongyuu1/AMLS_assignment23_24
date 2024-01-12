import argparse
import os
from A import launch as Alaunch
from B import launch as Blaunch
from A import defaults as Adefaults
from B import defaults as Bdefaults
import random
import numpy as np
import torch


def load_args():
    """
    Load the config parameters of this task
    :return: The corresponding config parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='A',
                        help='which task to perform. possible choices: A, B')
    parser.add_argument('--feature', type=str, default='none',
                        help='the method of extracting features. possible choices: lbp, hog, hog+lbp, none')
    parser.add_argument('--clf', type=str, default='CNN',
                        help='which type of classifier to be used. possible choices: Logreg, SVM, RandomForest, CNN')
    parser.add_argument('--mode', type=str, default='train_test',
                        help='which mode the programme will run in. possible choices: train_test, search')

    args = parser.parse_args()

    return args


def load_config(args):
    """
    Construct the configuration object used for running

    :param args: The config parameters of this task
    :return: The total configuration parameters
    """

    # Setup cfg.
    assert args.task == 'A' or args.task == 'B'
    base_path = os.path.abspath('.')
    if args.task == 'A':
        cfg = Adefaults.get_cfg()
        data_path = os.path.join(base_path, 'Datasets', 'pneumoniamnist.npz')
    else:
        cfg = Bdefaults.get_cfg()
        data_path = os.path.join(base_path, 'Datasets', 'pathmnist.npz')

    cfg.MODE = args.mode
    cfg.FEAT.METHOD = args.feature
    cfg.MODEL = args.clf
    cfg.DATA_PATH = data_path
    return cfg





if __name__ == '__main__':
    # Load the input arguments
    args = load_args()
    # Merge the input arguments in the configuration.
    cfg = load_config(args)

    # Fix random seed
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    #sys.stdout = open(cfg.LOG_PATH, 'a')

    if args.task == 'A':
        Alaunch.run(cfg)

    else:
        Blaunch.run(cfg)
