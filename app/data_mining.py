import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn import tree

#Bagging method
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor

#Boosting method
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor

#Random Forest method
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn import model_selection

from sklearn import preprocessing
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def decision_tree(data):
    #load data
    num_features = len(data.columns) - 1

    features = data.ix[:, 0:num_features]
    target = data.ix[:, num_features]

    print(features)
    print(target)
    data_features_train, data_features_test, data_targets_train, data_targets_test = \
        train_test_split(features,
                         target,
                         test_size=0.25)

    # Model declaration
    """
    Parameters to select:
    criterion: "mse"
    max_depth: maximum depth of tree, default: None
    """
    dec_tree_reg = DecisionTreeRegressor(criterion='mse', max_depth=10)
    dec_tree_reg.fit(data_features_train, data_targets_train)

    # Model evaluation
    test_data_predicted = dec_tree_reg.predict(data_features_test)

    error = metrics.mean_absolute_error(data_targets_test, test_data_predicted)

    print('Total Error: ' + str(error))

def mlp_classifier(data):
    #load data
    num_features = len(data.columns) - 1

    features = data.ix[:, 0:num_features]
    targets = data.ix[:, num_features]

    print(features)
    print(targets)

    # Data splitting
    features_train, features_test, targets_train, targets_test = data_splitting(
        features,
        targets,
        0.25)

    # Model declaration
    """
    Parameters to select:
    hidden_layer_sizes: its an array in which each element represents a new layer with "n" neurones on it
            Ex. (3,4) = Neural network with 2 layers: 3 neurons in the first layer and 4 neurons in the second layer
            Ex. (25) = Neural network with one layer and 25 neurons
            Default = Neural network with one layer and 100 neurons
    activation: "identity", "logistic", "tanh" or "relu". Default: "relu"
    solver: "lbfgs", "sgd" or "adam" Default: "adam"
    ###Only used with "sgd":###
    learning_rate_init: Neural network learning rate. Default: 0.001
    learning_rate: Way in which learning rate value change through iterations.
            Values: "constant", "invscaling" or "adaptive"
    momentum: Default: 0.9
    early_stopping: The algorithm automatic stop when the validation score is not improving.
            Values: "True" or "False". Default: False
    """
    neural_net = MLPClassifier(
        hidden_layer_sizes=(25),
        activation="relu",
        solver="adam"
    )
    neural_net.fit(features_train, targets_train.values.ravel())

    # Model evaluation
    test_data_predicted = neural_net.predict(features_test)
    score = metrics.accuracy_score(targets_test, test_data_predicted)

    logger.debug("Model Score: %s", score)

def ensemble_methods_classifiers(data):
    #load data
    num_features = len(data.columns) - 1

    features = data.ix[:, 0:num_features]
    targets = data.ix[:, num_features]

    print(features)
    print(targets)

    # Data splitting
    data_features_train, data_features_test, data_targets_train, data_targets_test = data_splitting(
        features,
        targets,
        0.25)

    # Model declaration
    """
    Parameters to select:
    n_estimators: The number of base estimators in the ensemble.
            Values: Random Forest and Bagging. Default 10
                    AdaBoost. Default: 50
    ###Only for Bagging and Boosting:###
    base_estimator: Base algorithm of the ensemble. Default: DecisionTree
    ###Only for Random Forest:###
    criterion: "entropy" or "gini": default: gini
    max_depth: maximum depth of tree, default: None
    """

    names = ["Bagging Classifier", "AdaBoost Classifier", "Random Forest Classifier", "Decision Tree Regressor", "SVR", "KNeighbors Regressor"]

    models = [
        BaggingClassifier(
            base_estimator=tree.DecisionTreeClassifier(
                criterion='gini',
                max_depth=10)
        ),
        AdaBoostClassifier(
            n_estimators=10,
            base_estimator=tree.DecisionTreeClassifier(
                criterion='gini',
                max_depth=10)
        ),
        RandomForestClassifier(
            criterion='gini',
            max_depth=10
        ),
        tree.DecisionTreeRegressor(
            criterion='mse'
        ),
        SVR(
            kernel='rbf',
            C=1e3,
            gamma=0.1
        ),
        KNeighborsRegressor()
    ]

    for name, em_clf in zip(names, models):
        logger.info("###################---" + name + "---###################")

        em_clf.fit(data_features_train, data_targets_train.values.ravel())

        # Model evaluation
        test_data_predicted = em_clf.predict(data_features_test)

        error = metrics.mean_absolute_error(data_targets_test, test_data_predicted)
        logger.debug('Total Error: %s', error)

def data_splitting(data_features, data_targets, test_size):
    """
    This function returns four subsets that represents training and test data
    :param data: numpy array
    :return: four subsets that represents data train and data test
    """
    data_features_train, data_features_test, data_targets_train, data_targets_test = \
        train_test_split(data_features,
                         data_targets,
                         test_size = test_size)

    return data_features_train, data_features_test, data_targets_train, data_targets_test

if __name__ == '__main__':
    data = pandas.read_csv('../resources/output.csv')
    #decision_tree(data)
    #mlp_classifier(data)
    ensemble_methods_classifiers(data)
