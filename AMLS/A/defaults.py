# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# The path of the dataset, to be determined after the choice of the task in main.py
_C.DATA_PATH = ''

# Set which mode the programme will run in, possible selections: 'search', 'train_test'.
# In 'search' mode, the programme will find and return the best set of hyperparameters of a model.
# In 'train_test' mode, the programme will run the normal training-testing flow and return the relevant scores.
_C.MODE = 'search'

# The path of file saving the running result.
_C.LOG_PATH = './A/log.txt'

# The model to be selected.
_C.MODEL = 'MLP'

# The random seed
_C.SEED = 0

# -----------------------------------------------------------------------------
# Config of Feature Engineering
# -----------------------------------------------------------------------------
_C.FEAT = CfgNode()

_C.FEAT.METHOD = 'none'

# The config of local binary pattern
_C.FEAT.LBP = CfgNode()

# The radius of circle (spatial resolution of the operator)
_C.FEAT.LBP.R = 3

# Number of circularly symmetric neighbour set points (quantization of the angular space).
_C.FEAT.LBP.P = 8 * _C.FEAT.LBP.R

# The config of histogram of oriented gradients
_C.FEAT.HOG = CfgNode()

# The pixels contained in each cell
_C.FEAT.HOG.PIXELS_PER_CELL = (8, 8)

# The cells contained in each block
_C.FEAT.HOG.CELLS_PER_BLOCK = (4, 4)

# Resize an image before the feature extraction
_C.FEAT.RESIZE = (64, 64)

# The number of dimension after applying PCA
_C.FEAT.REDUCED_DIM = 128

_C.FEAT.PCA = True

# ---------------------------------------------------------------------------- #
# Config of Logistic Regression
# ---------------------------------------------------------------------------- #
_C.LOGREG = CfgNode()

# The maximum number of iteration of logistic regression model
_C.LOGREG.MAX_ITER = 10000

# The solver of the logistic regression model
_C.LOGREG.SOLVER = 'newton-cg'

# The number of trial times when performing random grid search
_C.LOGREG.SEARCH_ITER = 15

_C.LOGREG.C = 1.0
# ---------------------------------------------------------------------------- #
# Config of Support Vector Machine
# ---------------------------------------------------------------------------- #
_C.SVM = CfgNode()

# The maximum number of iteration of support vector machine
_C.SVM.MAX_ITER = -1

# The kernel of the support vector machine
_C.SVM.KERNEL = 'rbf'

# The number of trial times when performing random grid search
_C.SVM.SEARCH_ITER = 15

# Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
_C.SVM.C = 1.0

# ---------------------------------------------------------------------------- #
# Config of Random Forest
# ---------------------------------------------------------------------------- #

_C.FOREST = CfgNode()

# The number of decision trees
_C.FOREST.ESTIMATOR = 55

# The maximum depth of each decision tree
_C.FOREST.DEPTH = 20

# The minimum number of samples required to be at a leaf node
_C.FOREST.MIN_LEAF = 5

# The minimum number of samples required to create a new split
_C.FOREST.MIN_SPLIT = 5

# The number of trial times when performing random grid search
_C.FOREST.SEARCH_ITER = 90


# ---------------------------------------------------------------------------- #
# Config of Convolutional Neural Network
# ---------------------------------------------------------------------------- #

_C.CNN = CfgNode()

# Which device to be used to run the programme
_C.CNN.DEVICE = 'cuda:0'

# The learning rate
_C.CNN.LR = 1e-3

# The loss function
_C.CNN.LOSS_FUNC = 'cross_entropy'

# The type of optimizer
_C.CNN.OPTIM = 'adamw'

# The maximum epoch
_C.CNN.EPOCH = 100

# The interval between which the information is logged
_C.CNN.LOG_INTERVAL = 1

# Rhe interval between which the evaluation is performed
_C.CNN.EVAL_INTERVAL = 10

# The number of trial time when performing hyperparameter search
_C.CNN.SEARCH_ITER = 10

# Whether to use pretrained model
_C.CNN.PRETRAINED = False

# The batch size
_C.CNN.BATCH = 128

# The number of data classes
_C.CNN.NUM_CLASS = 2
# ---------------------------------------------------------------------------- #
# Config of Hyperparameter Search
# ---------------------------------------------------------------------------- #

_C.SEARCH = CfgNode()

# The number of folds when performing K-fold cross validation
_C.SEARCH.FOLDS = 5

# The scoring metric
_C.SEARCH.SCORING = 'accuracy'

# The intensity of message reported
_C.SEARCH.VERBOSE = 2

def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
