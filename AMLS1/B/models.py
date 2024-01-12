from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from torchvision import models
import random
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from .utils import *
import time

class LogisticReg:
    def __init__(self, config):
        self.cfg = config
        self.scaler = MinMaxScaler()

        if self.cfg.FEAT.PCA:
            self.PCA = PCA(n_components=self.cfg.FEAT.REDUCED_DIM)

        self.clf = LogisticRegression(solver=self.cfg.LOGREG.SOLVER,
                                      C=self.cfg.LOGREG.C,
                                      max_iter=self.cfg.LOGREG.MAX_ITER)

    def random_search(self, train_data):
        [x_train, y_train] = train_data

        print('Start randomized searching')

        # Create the pipeline for hyperparameter searching
        if self.cfg.FEAT.PCA:
            pipe = Pipeline([('scaler', MinMaxScaler()),
                             ('PCA', PCA(n_components=self.cfg.FEAT.REDUCED_DIM)),
                             ('logisticreg', LogisticRegression(max_iter=self.cfg.LOGREG.MAX_ITER))])
        else:
            pipe = Pipeline([('scaler', MinMaxScaler(),
                              'logisicreg', LogisticRegression(max_iter=self.cfg.LOGREG.MAX_ITER))])

        # Set the tunable hyperparameters and their scopes
        param_grid = {'logisticreg__C': [0.2, 0.4, 0.6, 0.8, 1.0],
                      'logisticreg__solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag']
                      }
        # Use randomized search with maximum search iteration to reduce the search time
        self.grid = RandomizedSearchCV(pipe,
                                       param_distributions=param_grid,
                                       n_iter=self.cfg.SVM.SEARCH_ITER,
                                       cv=self.cfg.SEARCH.FOLDS,
                                       scoring=self.cfg.SEARCH.SCORING,
                                       verbose=self.cfg.SEARCH.VERBOSE)

        self.grid.fit(x_train, y_train)

        print('The best set of the parameters is:')
        print(self.grid.best_params_)

    def random_search_best_test_score(self, test_data):
        [x_test, y_test] = test_data
        print('The test score of the best set of parameters is: {}'.format(self.grid.score(x_test, y_test)))

    def train(self, train_data, val_data):

        # Unpack the training/validation data
        [x_train, y_train] = train_data
        [x_val, y_val] = val_data

        # Normalize the data
        x_train = self.scaler.fit_transform(x_train)
        x_val = self.scaler.transform(x_val)

        # Perform dimensionality reduction using PCA
        if self.cfg.FEAT.PCA:
            x_train = self.PCA.fit_transform(x_train)
            x_val = self.PCA.transform(x_val)

        # Model training
        print('Start training')
        self.clf.fit(x_train, y_train)

        # Model validation
        print('Start validation')
        y_predict = self.clf.predict(x_val)

        # Print the validation result
        print('The accuracy on the validation set is: {}'.format(accuracy_score(y_val, y_predict)))
        print('The confusion matrix is:')
        print(confusion_matrix(y_val, y_predict))

    def test(self, test_data):
        # Unpack test data
        [x_test, y_test] = test_data

        # Normalize the data
        x_test = self.scaler.transform(x_test)

        # Perform dimensionality reduction
        x_test = self.PCA.transform(x_test)

        # Model test
        print('Start test')
        y_predict = self.clf.predict(x_test)

        # Print test result
        print('The accuracy on the test set is: {}'.format(accuracy_score(y_test, y_predict)))
        print('The confusion matrix is: ')
        print(confusion_matrix(y_test, y_predict))


class SVM:
    def __init__(self, config):
        self.cfg = config
        self.scaler = MinMaxScaler()

        if self.cfg.FEAT.PCA:
            self.PCA = PCA(n_components=self.cfg.FEAT.REDUCED_DIM)

        self.clf = SVC(kernel=self.cfg.SVM.KERNEL,
                       max_iter=self.cfg.SVM.MAX_ITER,
                       C=self.cfg.SVM.C)

    def random_search(self, train_data):
        [x_train, y_train] = train_data

        n_feat = x_train.shape[1]
        print('Start randomized searching')

        if self.cfg.FEAT.PCA:
            pipe = Pipeline([('scaler', MinMaxScaler()),
                             ('PCA', PCA(n_components=self.cfg.FEAT.REDUCED_DIM)),
                             ('SVM', SVC(max_iter=-1))])
        else:
            pipe = Pipeline([('scaler', MinMaxScaler()),
                             'SVM', SVC(max_iter=-1)])

        param_grid = {'SVM__C': [0.2, 0.4, 0.6, 0.8, 1.0],
                      'SVM__kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
                      }

        self.grid = RandomizedSearchCV(pipe,
                                       param_distributions=param_grid,
                                       n_iter=self.cfg.SVM.SEARCH_ITER,
                                       cv=self.cfg.SEARCH.FOLDS,
                                       scoring=self.cfg.SEARCH.SCORING,
                                       verbose=self.cfg.SEARCH.VERBOSE)

        self.grid.fit(x_train, y_train)

        print('The best set of the parameters is:')
        print(self.grid.best_params_)

    def random_search_best_test_score(self, test_data):
        [x_test, y_test] = test_data
        print('The test score of the best set of parameters is: {}'.format(self.grid.score(x_test, y_test)))

    def train(self, train_data, val_data):
        [x_train, y_train] = train_data
        [x_val, y_val] = val_data

        x_train = self.scaler.fit_transform(x_train)
        x_val = self.scaler.transform(x_val)

        if self.cfg.FEAT.PCA:
            x_train = self.PCA.fit_transform(x_train)
            x_val = self.PCA.transform(x_val)

        print('Start training')
        self.clf.fit(x_train, y_train)

        print('Start validating')
        y_predict = self.clf.predict(x_val)

        print('The accuracy on the validation set is: {}'.format(accuracy_score(y_val, y_predict)))
        print('The confusion matrix is:')
        print(confusion_matrix(y_val, y_predict))

    def test(self, test_data):
        [x_test, y_test] = test_data

        x_test = self.scaler.transform(x_test)
        x_test = self.PCA.transform(x_test)

        print('Start testing')
        y_predict = self.clf.predict(x_test)

        print('The accuracy on the test set is: {}'.format(accuracy_score(y_test, y_predict)))
        print('The confusion matrix is: ')
        print(confusion_matrix(y_test, y_predict))


class RandomForest:
    def __init__(self, config):
        self.cfg = config
        self.scaler = MinMaxScaler()

        if self.cfg.FEAT.PCA:
            self.PCA = PCA(n_components=self.cfg.FEAT.REDUCED_DIM)

        self.clf = RandomForestClassifier(n_estimators=self.cfg.FOREST.ESTIMATOR,
                                          max_depth=self.cfg.FOREST.DEPTH,
                                          min_samples_leaf=self.cfg.FOREST.MIN_LEAF,
                                          min_samples_split=self.cfg.FOREST.MIN_SPLIT)

    def random_search(self, train_data):
        [x_train, y_train] = train_data
        n_feat = x_train.shape[1]
        print('Start randomized searching')
        if self.cfg.FEAT.PCA:
            pipe = Pipeline([('scaler', MinMaxScaler()),
                             ('PCA', PCA(n_components=self.cfg.FEAT.REDUCED_DIM)),
                             ('forest', RandomForestClassifier())])
        else:
            pipe = Pipeline([('scaler', MinMaxScaler(),
                              'forest', RandomForestClassifier())])

        param_grid = {'forest__n_estimators': [30, 55, 80, 105, 150],
                      'forest__max_depth': [5, 20, 35, 50, 65],
                      'forest__min_samples_leaf': [5, 20, 35, 50, 65],
                      'forest__min_samples_split': [5, 20, 35, 50, 65]
                      }

        self.grid = RandomizedSearchCV(pipe,
                                       param_distributions=param_grid,
                                       n_iter=self.cfg.SVM.SEARCH_ITER,
                                       cv=self.cfg.SEARCH.FOLDS,
                                       scoring=self.cfg.SEARCH.SCORING,
                                       verbose=self.cfg.SEARCH.VERBOSE)

        self.grid.fit(x_train, y_train)

        print('The best set of the parameters is:')
        print(self.grid.best_params_)

    def random_search_best_test_score(self, test_data):
        [x_test, y_test] = test_data
        print('The test score of the best set of parameters is: {}'.format(self.grid.score(x_test, y_test)))

    def train(self, train_data, val_data):
        [x_train, y_train] = train_data
        [x_val, y_val] = val_data

        x_train = self.scaler.fit_transform(x_train)
        x_val = self.scaler.transform(x_val)
        if self.cfg.FEAT.PCA:
            x_train = self.PCA.fit_transform(x_train)
            x_val = self.PCA.transform(x_val)

        print('Start training')
        self.clf.fit(x_train, y_train)

        print('Start validating')
        y_predict = self.clf.predict(x_val)

        print('The accuracy on the validation set is: {}'.format(accuracy_score(y_val, y_predict)))
        print('The confusion matrix is:')
        print(confusion_matrix(y_val, y_predict))

    def test(self, test_data):
        [x_test, y_test] = test_data

        x_test = self.scaler.transform(x_test)
        if self.cfg.FEAT.PCA:
            x_test = self.PCA.transform(x_test)

        print('Start testing')
        y_predict = self.clf.predict(x_test)

        print('The accuracy on the test set is: {}'.format(accuracy_score(y_test, y_predict)))
        print('The confusion matrix is: ')
        print(confusion_matrix(y_test, y_predict))


class CNN:
    def __init__(self, config):
        self.cfg = config
        # Use resnet18 as the classification model in this project
        self.clf = models.resnet18(pretrained=self.cfg.CNN.PRETRAINED, num_classes=self.cfg.CNN.NUM_CLASS)
        # Transfer the model to the corresponding device (cpu or cuda:0)
        self.clf.to(device=self.cfg.CNN.DEVICE)

    def train(self, train_data, val_data):
        if not os.path.exists('./B/weights.pt'):
            # Unpack training/validation data
            x_train, y_train = train_data
            x_val, y_val = val_data
            # Build corresponding datasets
            train_dataset = TaskDataset(self.cfg, x_train, y_train, train=True)
            val_dataset = TaskDataset(self.cfg, x_val, y_val, train=False)
            # Initialize the loss function
            if self.cfg.CNN.LOSS_FUNC == 'cross_entropy':
                loss_fn = nn.CrossEntropyLoss()
            else:
                loss_fn = nn.MSELoss()
                # Initialize the optimizer
            if self.cfg.CNN.OPTIM == 'adamw':
                optimizer = torch.optim.AdamW(params=self.clf.parameters(), lr=self.cfg.CNN.LR)
            else:
                optimizer = torch.optim.SGD(params=self.clf.parameters(), lr=self.cfg.CNN.LR)
            scheduler = StepLR(optimizer=optimizer, step_size=20, gamma=0.1, verbose=True)
            self.train_loop(self.clf, train_dataset, val_dataset, optimizer, loss_fn, scheduler)
            torch.save(self.clf, './B/weights.pt')
        else:
            print("Loading trained weights of the model")

    def random_search(self, data):
        x, y = data
        x_train = x[:89996]
        y_train = y[:89996]
        x_val = x[89996:]
        y_val = y[89996:]
        train_dataset = TaskDataset(self.cfg, x_train, y_train, train=True)
        val_dataset = TaskDataset(self.cfg, x_val, y_val, train=False)

        param_grid = {'optimizer': ['sgd', 'adamw'],
                      'loss_fn': ['MSE', 'cross_entropy'],
                      'lr': [1e-6, 1e-5, 1e-4]}
        search_iter = self.cfg.CNN.SEARCH_ITER
        choice_history = []
        val_results = []
        for i in range(search_iter):
            print('Hyperparameter search iteration {}/{}'.format(str(i + 1), str(search_iter)))
            while(True):
                optim_choice = random.choice(param_grid['optimizer'])
                loss_fn_choice = random.choice(param_grid['loss_fn'])
                lr_choice = random.choice(param_grid['lr'])
                choice = (optim_choice, loss_fn_choice, lr_choice)
                if choice not in choice_history:
                    break
            choice_history.append(choice)

            print('The current hyperparameter choice is:')
            print('Optimizer: {}, loss_function: {} , learning rate: {}'
                  .format(choice[0], choice[1], str(choice[2])))

            classifier = models.resnet18(pretrained=False, num_classes=9)
            classifier.to(device=self.cfg.CNN.DEVICE)
            lr = lr_choice

            if optim_choice == 'sgd':
                optimizer = torch.optim.SGD(classifier.parameters(), lr=lr)
            else:
                optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr)

            scheduler = StepLR(optimizer=optimizer, step_size=20, gamma=0.1, verbose=True)

            if loss_fn_choice == 'MSE':
                loss_fn = nn.MSELoss()
            else:
                loss_fn = nn.CrossEntropyLoss()

            val_accuracy = self.train_loop(classifier, train_dataset, val_dataset, optimizer, loss_fn, scheduler)
            val_results.append(val_accuracy)
        best_accuracy = max(val_results)
        best_choice = choice_history[val_results.index(best_accuracy)]
        print('The best hyperparameter choice is:')
        print('Optimizer: {}, loss_function: {} , learning rate: {}'
              .format(best_choice[0], best_choice[1], str(best_choice[2])))
        print('The corresponding validation accuracy: ' + str(best_accuracy))

    def train_loop(self, classifier, train_dataset, val_dataset, optimizer, loss_fn, scheduler):
        # Create the dataloader of training/validation set

        train_loader = DataLoader(train_dataset, batch_size=self.cfg.CNN.BATCH, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.cfg.CNN.BATCH, shuffle=False, drop_last=False)

        val_accuracy = 0

        for e in range(1, self.cfg.CNN.EPOCH + 1):
            # Switch the mode of the model to training
            classifier.train()
            loss_train = 0
            time1 = time.time()
            for x, y in train_loader:

                b = x.shape[0]

                # Transfer input to the corresponding device
                x = x.to(device=self.cfg.CNN.DEVICE)
                # Modify the shape and type of labels and transfer it to the corresponding device
                y = y.to(device=self.cfg.CNN.DEVICE, dtype=torch.float32)

                # Compute predicted labels
                y_pred = classifier(x)
                # Compute the loss between predicted and true labels
                loss = loss_fn(y_pred, y)

                # Clear the history gradients in the optimizer
                optimizer.zero_grad()
                # Perform back propagation
                loss.backward()
                # Step the optimizer
                optimizer.step()

                # Add loss of this batch to the total loss
                loss_train += loss.item() * b
            time2 = time.time()
            print(time2-time1)
            # Log the training process every a few epochs indicated by cfg.CNN.LOG_INTERVAL
            if e % self.cfg.CNN.LOG_INTERVAL == 0:
                print('epoch:%d/%d train_loss:%f' % (e, self.cfg.CNN.EPOCH, loss_train / len(train_dataset)))

            # Perform validation every a few epochs indicated by cfg.CNN.EVAL_INTERVAL
            if e % self.cfg.CNN.EVAL_INTERVAL == 0:
                # Switch the mode of the model to evaluation mode
                classifier.eval()
                matching_num = 0
                loss_val = 0
                with torch.no_grad():
                    for x, y in val_loader:
                        b = x.shape[0]

                        x = x.to(device=self.cfg.CNN.DEVICE)
                        y = y.to(device=self.cfg.CNN.DEVICE, dtype=torch.float32)

                        y_pred = classifier(x)

                        loss = loss_fn(y_pred, y)
                        loss_val += loss.item() * b

                        # Get accuracy in the batch
                        acc = get_test_score(y_pred, y)

                        # Count the number of TP predictions
                        matching_num += acc * b

                    # Calculate the overall accuracy
                    val_accuracy = matching_num / len(val_dataset)
                print('Validation in the epoch %d loss:%f accuracy:%f' % (e, loss_val / len(val_dataset), val_accuracy))
            scheduler.step()
        print('The accuracy on the validation set is: %f' % val_accuracy)

        return val_accuracy

    def test(self, test_data):
        self.clf = torch.load('./B/weights.pt')
        # Unpack the test data
        x_test, y_test = test_data
        # Build the test dataset
        test_dataset = TaskDataset(self.cfg, x_test, y_test, train=False)
        self.test_loop(test_dataset, self.clf)

    def test_loop(self, test_dataset, classifier):
        # Build the dataloader of the test set
        test_loader = DataLoader(test_dataset, batch_size=self.cfg.CNN.BATCH, shuffle=False, drop_last=False)

        # Create tensors storing all predicted labels and true labels
        total_y_pred = torch.zeros([len(test_dataset), self.cfg.CNN.NUM_CLASS], dtype=torch.float).to(device=self.cfg.CNN.DEVICE)
        total_y = torch.zeros([len(test_dataset), self.cfg.CNN.NUM_CLASS], dtype=torch.float).to(device=self.cfg.CNN.DEVICE)

        # Switch the classifier to the evaluation mode
        classifier.eval()

        matching_num = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                b = x.shape[0]

                x = x.to(device=self.cfg.CNN.DEVICE)
                y = y.to(device=self.cfg.CNN.DEVICE, dtype=torch.float32)
                y_pred = classifier(x)

                # Load the small batch of predicted/true labels into the storage tensors
                if (i + 1) * self.cfg.CNN.BATCH <= len(test_dataset):
                    total_y_pred[i * self.cfg.CNN.BATCH:(i + 1) * self.cfg.CNN.BATCH] = y_pred
                    total_y[i * self.cfg.CNN.BATCH:(i + 1) * self.cfg.CNN.BATCH] = y
                else:
                    total_y_pred[i * self.cfg.CNN.BATCH:] = y_pred
                    total_y[i * self.cfg.CNN.BATCH:] = y

                acc = get_test_score(y_pred, y)
                matching_num += acc * b

            test_accuracy = matching_num / len(test_dataset)
        print('The accuracy on the test set is: %f' % test_accuracy)
        print('The confusion matrix is: ')

        total_y_pred = one_hot_to_indices(total_y_pred).cpu().numpy()
        total_y = one_hot_to_indices(total_y).cpu().numpy()
        # Print the confusion matrix
        print(confusion_matrix(total_y_pred, total_y))


if __name__ == '__main__':
    cfg = 0
    clf = CNN(cfg)



