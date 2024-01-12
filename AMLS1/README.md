# APPLIED MACHINE LEARNING SYSTEM ELEC0134 

This is the code implementation of the Applied Machine Learning System by Dongyu Wang 
(SN:23104424). The code incorporates four classification models: Logistic Regression, 
Support Vector Machine, Random Forest and Convolutional Network\
(In this project the Resnet-18 is specifically chosen). 

The programme has two modes: "search" and "train_test". In "seach" mode, the code runs 
a random grid hyperparameter search to find the best set of hyperparameters for each model 
in each task. In the "train_test" mode, the code normally trains and tests a model with
the hyperparameter found in the "search" mode.

## Code Structure and Description
The main.py is used to launch the programme. It incorporates the selection of tasks,
classification models, feature extraction methods and running mode (which will be explained later).

The code includes two folders A and B, containing the corresponding codes for dealing with
Task A and Task B, respectively. The structure and function of files in folders A and B are exactly the same,
with slight difference of configurations for tackling binary or multi-class classification.

In each folder, the defaults.py contains all the hyperparameters used for the code. 
The launch.py is for constructing the mainframe for launching training-testing flow and 
hyperparameter search. The feature_engineering.py contains different feature extraction methods.
The models.py includes multiple classification model classes. The utils.py contains some 
tool codes. 

## Initialization

Please install the packages required using:
```
pip install -r requirements.txt
```
To install the pytorch, please run the following command line:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
## Dataset
Please place the QMUL-ShoeV2 and QMUL-ChairV2 datasets according to the path:
```
./Datasets/pathmnist.npz(pneumoniamnist.npz)
```


## Hyperparameter Seach
Here we can search for the best hyperparameter of a model:
```
 python main.py --task A --feature hog --clf SVM --mode search
```
The "task" means which task to be tackled (A or B). 

The "feature" means the feature extraction method used (choices: none, hog, lbp, hog+lbp).

The "clf" means the choice of classification model (choices: Logreg, SVM, RandomForest, CNN)

The hyperparameter search outputs the best set of the hyperparameters of a model


## Train-test 
Here we can train and test the model with the found hyperparameter set:
```
python main.py --task A --feature hog --clf SVM --mode train_tests
```
The training-testing flow returns the accuracy and confusion matrix of both validation set
and test set.
