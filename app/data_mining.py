import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import metrics

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def decision_tree(data):
    num_features = len(data.columns) - 1

    features = data[list(range(1, num_features))]
    target = data[[num_features]]

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
    dec_tree_reg = DecisionTreeRegressor(criterion='mse', max_depth=5)
    dec_tree_reg.fit(data_features_train, data_targets_train)

    # Model evaluation
    test_data_predicted = dec_tree_reg.predict(data_features_test)

    error = metrics.mean_absolute_error(data_targets_test, test_data_predicted)

    print('Total Error: ' + str(error))

def mlp_classifier(data):
    #load data
    num_features = len(data.columns) - 1

    features = data[list(range(1, num_features))]
    targets = data[[num_features]]

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
        hidden_layer_sizes=(50),
        activation="relu",
        solver="adam"
    )
    neural_net.fit(features_train, targets_train.values.ravel())

    # Model evaluation
    test_data_predicted = neural_net.predict(features_test)
    score = metrics.accuracy_score(targets_test, test_data_predicted)

    logger.debug("Model Score: %s", score)

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
    mlp_classifier(data)
