import numpy as np
from .models import LogisticReg, SVM, RandomForest, CNN
from .feature_engineering import FeatureExtractor
from .utils import *
import time

def run(cfg):

    # Get the data used for training/testing.
    # The data is a list including [x_train, y_train, x_val, y_val, x_test, y_test]
    data = get_data(cfg)

    # In this mode, the programme searches for the optimal model parameters promising best validation score and
    # returns the according parameter set.
    if cfg.MODE == 'search':
        print('Mode selected: optimal parameters searching')
        param_search(data, cfg)

    # In this mode, the programme normally trains the model and evaluates it using test dataset.
    elif cfg.MODE == 'train_test':
        print('Mode selected: normal training-testing flow')
        normal_train_test(data, cfg)


def get_data(cfg):
    """
    The function is to load the data from the file, perform feature extraction and return the corresponding results.
    :param cfg: The configuration object
    :return: The train/val/test data after feature extraction
    """
    # Load train/val/test data
    data = np.load(cfg.DATA_PATH)

    x_train = data['train_images']
    y_train = data['train_labels']

    x_val = data['val_images']
    y_val = data['val_labels']

    x_test = data['test_images']
    y_test = data['test_labels']

    # Instantiate the feature extractor to perform feature engineering
    fe = FeatureExtractor(cfg)

    # Perform feature engineering except for CNN
    if cfg.MODEL != 'CNN':

        # Extract the Histogram of Oriented Gradient Features
        if cfg.FEAT.METHOD == 'hog':
            print('Extracting HOG features of the training set')
            x_train = fe.extract_hog_features(x_train)
            print('Extracting HOG features of the validation set')
            x_val = fe.extract_hog_features(x_val)
            print('Extracting HOG features of the testing set')
            x_test = fe.extract_hog_features(x_test)

        # Extract the Local Binary Pattern Features
        elif cfg.FEAT.METHOD == 'lbp':
            print('Extracting LBP features of the training set')
            x_train = fe.extract_lbp_features(x_train)
            print('Extracting LBP features of the validation set')
            x_val = fe.extract_lbp_features(x_val)
            print('Extracting LBP features of the testing set')
            x_test = fe.extract_lbp_features(x_test)

        # In this case, the lbp features are extracted first, then we extract hog features from them.
        elif cfg.FEAT.METHOD == 'hog+lbp':
            print('Extracting HOG features of the training set')
            x_train = fe.extract_hog_and_lbp_features(x_train)
            print('Extracting HOG features of the validation set')
            x_val = fe.extract_hog_and_lbp_features(x_val)
            print('Extracting HOG features of the testing set')
            x_test = fe.extract_hog_and_lbp_features(x_test)

    data = [x_train, y_train, x_val, y_val, x_test, y_test]
    return data


def dataset_construction(data):
    [x_train, y_train, x_val, y_val, x_test, y_test] = data

        
def param_search(data, cfg):
    """
    This is to search the optimal parameter set of the model promising the best validation score.
    :param data: The input dataset
    :param cfg: The configuration object
    """
    [x_train, y_train, x_val, y_val, x_test, y_test] = data

    # Concatenate the original training and validation data to perform k-fold cross validation
    x_train = np.concatenate((x_train, x_val), axis=0)
    y_train = np.concatenate((y_train, y_val), axis=0)

    if cfg.MODEL == 'CNN':

        x_train = numpy_to_pil(x_train)
        x_test = numpy_to_pil(x_test)
    else:
        # Flatten the data. N*H*W --> N*D
        x_train = np.reshape(x_train, [x_train.shape[0], -1])
        x_test = np.reshape(x_test, [x_test.shape[0], -1])

    y_train = y_train.ravel()
    y_test = y_test.ravel()


    data_train = [x_train, y_train]
    data_test = [x_test, y_test]

    if cfg.MODEL == 'Logreg':
        print('Model used: Logistic Regression')
        # Initialize the model object
        logisreg = LogisticReg(cfg)

        # Search for the optimal parameter set
        logisreg.random_search(data_train)

        # Use test set to evaluate the performance of the model with the optimal parameters
        logisreg.random_search_best_test_score(data_test)

    elif cfg.MODEL == 'SVM':
        print('Model used: Support Vector Machine')
        svm = SVM(cfg)
        svm.random_search(data_train)
        svm.random_search_best_test_score(data_test)

    elif cfg.MODEL == 'RandomForest':
        print('Model used: Random Forest')
        forest = RandomForest(cfg)
        forest.random_search(data_train)
        forest.random_search_best_test_score(data_test)

    else:
        print('Model used: Convolutional Neural Network')
        cnn = CNN(cfg)
        cnn.random_search(data_train)


def normal_train_test(data, cfg):
    """
    This task performs normal training-testing flow
    :param data: The input dataset
    :param cfg: The configuration object
    """
    [x_train, y_train, x_val, y_val, x_test, y_test] = data

    if cfg.MODEL == 'CNN':
        x_train = numpy_to_pil(x_train)
        x_val = numpy_to_pil(x_val)
        x_test = numpy_to_pil(x_test)

    else:
        # Flatten the data. N*H*W --> N*D
        x_train = np.reshape(x_train, [x_train.shape[0], -1])
        x_val = np.reshape(x_val, [x_val.shape[0], -1])
        x_test = np.reshape(x_test, [x_test.shape[0], -1])

    # Squeeze the extra dimension of labels. N*1 --> N
    y_train = y_train.ravel()
    y_val = y_val.ravel()
    y_test = y_test.ravel()

    # For different models, get the dataset with respective format.
    #if cfg.MODEL == 'CNN':
    #    dataset_train = TaskDataset(cfg, x_train, y_train)
    #    dataset_val = TaskDataset(cfg, x_val, y_val)
    #    dataset_test = TaskDataset(cfg, x_test, y_test)
    #else:
    data_train = [x_train, y_train]
    data_val = [x_val, y_val]
    data_test = [x_test, y_test]

    # Instantiate the classifier model
    if cfg.MODEL == 'Logreg':
        print('Model used: Logistic Regression')
        clf = LogisticReg(cfg)
    elif cfg.MODEL == 'SVM':
        print('Model used: Support Vector Machine')
        clf = SVM(cfg)
    elif cfg.MODEL == 'RandomForest':
        print('Model used: Random Forest')
        clf = RandomForest(cfg)
    else:
        print('Model used: Convoluitonal neural network')
        clf = CNN(cfg)

    # Perform training-testing flow
    clf.train(data_train, data_val)
    time_start = time.time()
    clf.test(data_test)
    time_finish = time.time()
    time_diff = time_finish - time_start
    print("The time spent during inference: " + str(time_diff) + 's')

