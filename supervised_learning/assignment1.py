from time import time

import pandas as pd  # package for data analysis and manipulation
import numpy as np  # package for scientific computing on multidimensional arrays
import matplotlib  # package for creating visualizations
from matplotlib import pyplot as plt
import seaborn as sns  # data visualization library based on matplotlib
import sklearn  # machine learning library
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import scipy  # library for mathematics, science and engineering
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
import collections
import zipfile
import requests
import platform
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC
import json

from sklearn.utils._testing import ignore_warnings

import util

do_grid = True

def dump_metrics_to_file(metrics, filename):
    f = open(filename, "w")
    for key in metrics.keys():
        f.write(key + "\t\t\t")
    f.write('\n')
    for i in range(len(metrics['x_axis'])):
        for key in metrics.keys():
            f.write(str(metrics[key][i]) + "\t\t\t")
        f.write('\n')
    f.close()


def wine_dt_initial(data, features, target):
    model = tree.DecisionTreeClassifier(random_state=234)
    print(model)
    metrics = util.training_by_size(data, features, target, model)
    dump_metrics_to_file(metrics, 'output/wine_dt_initial.csv')

    util.draw(metrics['x_axis'], metrics['train_f1_macro'], metrics['test_f1_macro'], 'percentage', 'f1 score',
              'Decision Tree Initial', 'output/wine_dt_initial')
    util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
              'Decision Tree Initial Time', 'output/wine_dt_initial_time')


def wine_dt_hyper(X_all, Y_all):
    param_range = np.linspace(0.0002, 0.02, 100)
    train_score, test_score = util.get_hyperdata_validation(tree.DecisionTreeClassifier(), X_all, Y_all, "ccp_alpha",
                                                            param_range, 'f1_weighted')
    util.draw(param_range, train_score, test_score, "ccp_alpha", "f1_score", "Decision Tree Cpp Alpha",
              'output/wine_dt_ccp_alpha')
    print("Wine DT ccp_alpha Best param", param_range[test_score.argmax()])
    return param_range[test_score.argmax()]


def wine_dt_param(data, features, target, ccp_alpha):
    model = tree.DecisionTreeClassifier(random_state=234, ccp_alpha=ccp_alpha)
    print(model)
    metrics = util.training_by_size(data, features, target, model)
    dump_metrics_to_file(metrics, 'output/wine_dt_param_{}.csv'.format(ccp_alpha))

    util.draw(metrics['x_axis'], metrics['train_f1_macro'], metrics['test_f1_macro'], 'percentage', 'f1 score',
              'Decision Tree Ccp_alpha({})'.format(ccp_alpha), 'output/wine_dt_param_{}'.format(ccp_alpha))
    util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
              'Decision Tree Ccp_alpha({}) Time'.format(ccp_alpha), 'output/wine_dt_param_{}_time'.format(ccp_alpha))
    return model

def wine_knn_initial(data, features, target):
    model = KNeighborsClassifier()
    print(model)
    metrics = util.training_by_size(data, features, target, model)
    dump_metrics_to_file(metrics, 'output/wine_knn_initial.csv')
    util.draw(metrics['x_axis'], metrics['train_f1_macro'], metrics['test_f1_macro'], 'percentage', 'f1 score',
              'KNN Initial', 'output/wine_knn_initial')
    util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
              'KNN Initial Time', 'output/wine_knn_initial_time')


def wine_knn_hyper(X_all, Y_all):
    param_range = np.linspace(1, 50, 50).astype(int)
    train_score, test_score = util.get_hyperdata_validation(KNeighborsClassifier(), X_all, Y_all, "n_neighbors",
                                                            param_range, 'f1_weighted')
    util.draw(param_range, train_score, test_score, "number of neighbors", "f1_score", "Wine KNN n_neighbors",
              'output/wine_knn_neighbors')
    print("Wine KNN Best param", param_range[test_score.argmax()])
    return param_range[test_score.argmax()]


def wine_knn_neighbor(data, features, target, neighbor):
    model = KNeighborsClassifier(n_neighbors=neighbor)
    print(model)
    metrics = util.training_by_size(data, features, target, model)
    dump_metrics_to_file(metrics, 'output/wine_knn_neighbor_{}.csv'.format(neighbor))
    util.draw(metrics['x_axis'], metrics['train_f1_macro'], metrics['test_f1_macro'], 'percentage', 'f1 score',
              'KNN Neighbor {}'.format(neighbor), 'output/wine_knn_neighbor_{}'.format(neighbor))
    util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
              'KNN Neighbor{} Time'.format(neighbor), 'output/wine_knn_neighbor_{}_time'.format(neighbor))

def wine_knn_metric_algorithm(data, features, target, neighbor):
    algorithms = ['euclidean','manhattan','chebyshev','minkowski']
    for alg in algorithms:
        model = KNeighborsClassifier(n_neighbors=neighbor, metric= alg)
        print(model)
        metrics = util.training_by_size(data, features, target, model)
        dump_metrics_to_file(metrics, 'output/wine_knn_neighbor_{}.csv'.format(neighbor))
        util.draw(metrics['x_axis'], metrics['train_f1_macro'], metrics['test_f1_macro'], 'percentage', 'f1 score',
                  'KNN Algorithm ({})'.format(alg), 'output/wine_knn_metric_{}'.format(alg))
        util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
                  'KNN Algorithm ({}) Time'.format(alg), 'output/wine_knn_metric_{}_time'.format(alg))

def wine_knn_grid(X_all, Y_all):
    param_grid = {
        'n_neighbors': np.linspace(1, 50, 50).astype(int),
        'metric': ['euclidean','manhattan','chebyshev','minkowski'],
        'weights': ['uniform', 'distance']
    }
    knn = KNeighborsClassifier()
    search_f1 = GridSearchCV(knn, param_grid, cv=10,verbose=1, scoring='f1_weighted', n_jobs=-1)
    search_f1.fit(X_all, Y_all)
    print(search_f1.best_estimator_, search_f1.best_score_)
    return search_f1.best_estimator_

def wine_knn_last(data, features, target,model):
    print(model)
    metrics = util.training_by_size(data, features, target, model, initial=40)
    dump_metrics_to_file(metrics, 'output/wine_knn_last.csv')
    util.draw(metrics['x_axis'], metrics['train_f1_macro'], metrics['test_f1_macro'], 'percentage', 'f1 score',
              'KNN', 'output/wine_knn_last')
    util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
              'KNN Time', 'output/wine_knn_last_time')

def wine_svc_initial(data, features, target):
    model = SVC(random_state=234)
    print(model)
    metrics = util.training_by_size(data, features, target, model)
    dump_metrics_to_file(metrics, 'output/wine_svc_initial.csv')
    util.draw(metrics['x_axis'], metrics['train_f1_macro'], metrics['test_f1_macro'], 'percentage', 'f1 score',
              'SVC Initial', 'output/wine_svc_initial')
    util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
              'SVC Initial Time', 'output/wine_svc_initial_time')




def wine_svc_hyper_kernel(data, features, target):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    for kernel in kernels:
        model = SVC(random_state=234, kernel=kernel)
        print(model)
        metrics = util.training_by_size(data, features, target, model)

        util.draw(metrics['x_axis'], metrics['train_f1_macro'], metrics['test_f1_macro'], 'percentage', 'f1 score',
                  'SVM kernel ({})'.format(kernel), "output/wine_svc_kernel_" + kernel)


def wine_svc_degree(data, features, target):
    degreelist = np.linspace(1, 20, 20).astype(int)
    for degree in degreelist:
        model = SVC(random_state=234, kernel='poly', degree=degree)
        print(model)
        metrics = util.training_by_size(data, features, target, model)
        dump_metrics_to_file(metrics, 'output/wine_svc_degree_{}.csv'.format(degree))
        util.draw(metrics['x_axis'], metrics['train_f1_macro'], metrics['test_f1_macro'], 'percentage', 'f1 score',
                  'SVC degree ({})'.format(degree), 'output/wine_svc_degree_{}'.format(degree))

def wine_svc_coeff(data, features, target):
    coeff_list = np.linspace(0.1, 10, 20)
    for coef in coeff_list:
        model = SVC(random_state=234, kernel='poly',degree=2, coef0=coef)
        print(model)
        metrics = util.training_by_size(data, features, target, model)

        util.draw(metrics['x_axis'], metrics['train_f1_macro'], metrics['test_f1_macro'], 'percentage', 'f1 score',
                  'SVC coeff ({})'.format(coef), 'output/wine_svc_coeff_{}'.format(coef))

def wine_svc_C(data, features, target):
    c_list = np.linspace(0.1, 10, 20)
    for c in c_list:
        model = SVC(random_state=234, kernel='poly',degree=1, C=c)
        print(model)
        metrics = util.training_by_size(data, features, target, model)

        util.draw(metrics['x_axis'], metrics['train_f1_macro'], metrics['test_f1_macro'], 'percentage', 'f1 score',
                  'SVC C ({})'.format(c), 'output/wine_svc_c_{}'.format(c))


def wine_svc_grid(X_all, Y_all):
    param_grid = {
        'C': np.linspace(0.1, 1.5, 10),
        'degree':[1,2,3],
        'coef0': np.linspace(0.1, 10, 10)
    }

    svc = SVC(random_state=234, kernel='poly')
    search_f1 = GridSearchCV(svc, param_grid, cv=10,verbose=1, scoring='f1_weighted', n_jobs=-1)
    search_f1.fit(X_all, Y_all)
    print(search_f1.best_estimator_, search_f1.best_score_)
    return search_f1.best_estimator_


def wine_svc_last(data, features, target, model):
    print(model)
    metrics = util.training_by_size(data, features, target, model)
    dump_metrics_to_file(metrics, 'output/wine_svc_last.csv')
    util.draw(metrics['x_axis'], metrics['train_f1_macro'], metrics['test_f1_macro'], 'percentage', 'f1 score',
              'SVC', 'output/wine_svc_last')
    util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
              'SVC Time', 'output/wine_svc_last_time')

def wine_boosting_initial(data, features, target):
    model = AdaBoostClassifier(tree.DecisionTreeClassifier(ccp_alpha=0.03), random_state=234)
    print(model)
    metrics = util.training_by_size(data, features, target, model)
    dump_metrics_to_file(metrics, 'output/wine_boost_initial.csv')
    util.draw(metrics['x_axis'], metrics['train_f1_macro'], metrics['test_f1_macro'], 'percentage', 'f1 score',
              'Boosting Initial', 'output/wine_boost_initial')
    util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
              'Boosting Initial Time', 'output/wine_boost_initial_time')


def wine_boosting_estimator(data, features, target):
    estimator = np.linspace(50, 600, 12).astype(int)
    for n_estimator in estimator:
        model = AdaBoostClassifier(tree.DecisionTreeClassifier(ccp_alpha=0.03), random_state=234, n_estimators=n_estimator)
        print(model)
        metrics = util.training_by_size(data, features, target, model)
        util.draw(metrics['x_axis'], metrics['train_f1_macro'], metrics['test_f1_macro'], 'percentage', 'f1 score',
                  'Boosting Estimator {}'.format(n_estimator), 'output/wine_boost_estimator_{}'.format(n_estimator))
        util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
                  'Boosting Estimator {} Time'.format(n_estimator), 'output/wine_boost_estimator_{}_time'.format(n_estimator))

def wine_boost_hyper(X_all, Y_all):
    param_range = np.linspace(50, 600, 12).astype(int)
    bdt_clf = AdaBoostClassifier(
        tree.DecisionTreeClassifier(ccp_alpha=0.03))
    train_score, test_score = util.get_hyperdata_validation(bdt_clf, X_all, Y_all, "n_estimators",
                                                            param_range, 'f1_weighted')
    util.draw(param_range, train_score, test_score, "n_estimators", "f1_score", "Boost",
              'output/wine_boost_hyper_estimator')
    print("Boost Best param", param_range[test_score.argmax()])
    # Best param 450
    return param_range[test_score.argmax()]


def wine_boost_grid(X_all, Y_all):
    param_grid = {
        'base_estimator__max_depth': [1, 2],
        'algorithm': ['SAMME', 'SAMME.R'],
        'n_estimators': [10, 50, 200, 300, 350, 400, 450, 500, 550, 600, 800],
        'learning_rate': [0.3, 0.5, 0.8, 1.0, 1.2]
    }
    empty_decision_tree = tree.DecisionTreeClassifier()
    adaBoost = AdaBoostClassifier(empty_decision_tree, random_state=234)
    search_f1 = GridSearchCV(adaBoost, param_grid, cv=10,verbose=1, scoring='f1_weighted', n_jobs=-1)
    search_f1.fit(X_all, Y_all)
    print(search_f1.best_estimator_, search_f1.best_score_)
    return search_f1.best_estimator_

def wine_boosting_last(data, features, target,model):
    print(model)
    metrics = util.training_by_size(data, features, target, model)
    dump_metrics_to_file(metrics, 'output/wine_boost_last.csv')
    util.draw(metrics['x_axis'], metrics['train_f1_macro'], metrics['test_f1_macro'], 'percentage', 'f1 score',
              'Boosting', 'output/wine_boost_last')
    util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
              'Boosting Time', 'output/wine_boost_last')

def wine_nn_initial(data, features, target):
    model = MLPClassifier(random_state=234)
    print(model)
    metrics = util.training_by_size(data, features, target, model)
    dump_metrics_to_file(metrics, 'output/wine_nn_initial.csv')
    util.draw(metrics['x_axis'], metrics['train_f1_macro'], metrics['test_f1_macro'], 'percentage', 'f1 score',
              'Neural Network Initial', 'output/wine_nn_initial')
    util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
              'Neural Network Initial Time', 'output/wine_nn_initial_time')

def wine_nn_hiddenlayer(data, features, target):
    param_range = np.linspace(50, 500, 10).astype(int)
    param_tuple = [tuple([i]) for i in param_range]
    for param in param_tuple:
        model = MLPClassifier(random_state=234, hidden_layer_sizes=param)
        print(model)
        metrics = util.training_by_size(data, features, target, model)
        util.draw(metrics['x_axis'], metrics['train_f1_macro'], metrics['test_f1_macro'], 'percentage', 'f1 score',
                  'Neural Network Hidden Layer {}'.format(param), 'output/wine_nn_hiddenlayer_{}'.format(param))
        util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
                  'Neural Network Hidden Layer {} Time'.format(param), 'output/wine_nn_hidden_layer_{}_time'.format(param))

def wine_nn_hyper_hiddenlayer(X_all, Y_all):
    # param_range = np.linspace(1, 100, 100).astype(int)
    param_range = np.linspace(3, 20, 10).astype(int)
    param_tuple = [tuple([i]) for i in param_range]
    model = MLPClassifier(random_state=234, max_iter=600)
    train_score, test_score = util.get_hyperdata_validation(model, X_all, Y_all, "hidden_layer_sizes",
                                                            param_tuple, 'f1_weighted')
    util.draw(param_range, train_score, test_score, "Hidden Layer", "f1 score", 'Neural Network Hidden Layer',
              'output/wine_nn_hiddenlayer')
    print("NN Hidden layer Best param", param_range[test_score.argmax()])
    # Best param 79


def wine_nn_hyper_max_iter(X_all, Y_all):
    param_range = np.linspace(600, 1000, 9).astype(int)

    model = MLPClassifier(random_state=234, max_iter=5000)
    train_score, test_score = util.get_hyperdata_validation(model, X_all, Y_all, "max_iter",
                                                            param_range, 'f1_weighted')
    util.draw(param_range, train_score, test_score, "Max Iteration ", "f1 score", 'Neural Network Max Iteration',
              'output/wine_nn_max_iter')
    print("NN Max iter Best param", param_range[test_score.argmax()])
    return test_score.argmax()


def wine_nn_hyper_learning_rate(data, features, target):

    learning_rate = ['constant', 'invscaling', 'adaptive']
    for learning in learning_rate:
        model = MLPClassifier(random_state=234, learning_rate=learning, solver='sgd', max_iter=1000,
                              learning_rate_init=0.1)
        print(model)
        metrics = util.training_by_size(data, features, target, model)
        util.draw(metrics['x_axis'], metrics['train_f1_macro'], metrics['test_f1_macro'], 'percentage', 'f1 score',
                  'Neural Network Learning Model ({})'.format(learning), "output/wine_nn_learning_rate_" + learning)


def wine_nn_hyper_activation(data, features, target):
    activation_list = ['identity', 'logistic', 'tanh', 'relu']
    for activation in activation_list:
        model = MLPClassifier(random_state=234, activation=activation, solver='sgd', max_iter=1000,
                              learning_rate_init=0.1)
        print(model)
        metrics = util.training_by_size(data, features, target, model)
        util.draw(metrics['x_axis'], metrics['train_f1_macro'], metrics['test_f1_macro'], 'percentage', 'f1 score',
                  'Neural Network Activation Param ({})'.format(activation), "output/wine_nn_activation_" + activation)


def wine_nn_grid(X_all, Y_all):
    param_range = np.linspace(3,20, 17).astype(int)
    hidden_tuple = [tuple([i]) for i in param_range]
    param_grid = {
        'hidden_layer_sizes': hidden_tuple,
        'activation':  ['identity', 'logistic', 'tanh', 'relu'],
        'learning_rate':  ['constant', 'invscaling', 'adaptive'],
        'max_iter': [800]
    }

    nn = MLPClassifier(random_state=234)
    search_f1 = GridSearchCV(nn, param_grid, cv=10,verbose=1, scoring='f1_weighted', n_jobs=-1)
    search_f1.fit(X_all, Y_all)
    print(search_f1.best_estimator_, search_f1.best_score_)
    return search_f1.best_estimator_


def wine_nn_last(data, features, target,model):
    model = MLPClassifier(random_state=234)
    print(model)
    metrics = util.training_by_size(data, features, target, model)
    dump_metrics_to_file(metrics, 'output/wine_nn_last.csv')
    util.draw(metrics['x_axis'], metrics['train_accuracy'], metrics['test_accuracy'], 'percentage', 'accuracy',
              'Neural Network', 'output/wine_nn_last')
    util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
              'Neural Network Time', 'output/wine_nn_last_time')

# DIVORCE TESTS


def divorce_dt_initial(data, features, target):
    model = tree.DecisionTreeClassifier(random_state=234)
    print(model)
    metrics = util.training_by_size(data, features, target, model, )
    dump_metrics_to_file(metrics, 'output/divorce_dt_initial.csv')
    util.draw(metrics['x_axis'], metrics['train_accuracy'], metrics['test_accuracy'], 'percentage', 'accuracy',
              'Decision Tree Initial', 'output/divorce_dt_initial')
    util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
              'Decision Tree Initial Time', 'output/divorce_dt_initial_time')


def divorce_dt_hyper(X_all, Y_all):
    param_range = np.linspace(0.0002, 0.02, 100)
    train_score, test_score = util.get_hyperdata_validation(tree.DecisionTreeClassifier(), X_all, Y_all, "ccp_alpha",
                                                            param_range, 'accuracy')
    util.draw(param_range, train_score, test_score, "ccp_alpha", "accuracy", "Decision Tree Cpp Alpha",
              'output/divorce_dt_ccp_alpha')
    print("Divorce DT ccp_alpha Best param", param_range[test_score.argmax()])
    return param_range[test_score.argmax()]

def divorce_dt_hyper_2(X_all, Y_all):
    param_range = np.linspace(2,200,40).astype(int)
    train_score, test_score = util.get_hyperdata_validation(tree.DecisionTreeClassifier(), X_all, Y_all, "min_samples_split",
                                                            param_range, 'accuracy')
    util.draw(param_range, train_score, test_score, "min_samples_split", "accuracy", "Decision Tree Min Samples",
              'output/divorce_dt_min_samples')
    print("Divorce DT min_samples_split Best param", param_range[test_score.argmax()])
    return param_range[test_score.argmax()]

def divorce_dt_hyper(X_all, Y_all):
    param_range = np.linspace(0.0002, 0.02, 100)
    train_score, test_score = util.get_hyperdata_validation(tree.DecisionTreeClassifier(), X_all, Y_all, "min_samples_split",
                                                            param_range, 'accuracy')
    util.draw(param_range, train_score, test_score, "Min Samples", "f1_score", "Decision Tree Cpp Alpha",
              'output/divorce_dt_min_samples')
    print("Divorce DT ccp_alpha Best param", param_range[test_score.argmax()])
    return param_range[test_score.argmax()]


def divorce_dt_param(data, features, target, min_split):
    model = tree.DecisionTreeClassifier(random_state=234, min_samples_split=min_split)
    print(model)
    metrics = util.training_by_size(data, features, target, model)
    dump_metrics_to_file(metrics, 'output/divorce_dt_minsplit_{}.csv'.format(min_split))

    util.draw(metrics['x_axis'], metrics['train_accuracy'], metrics['test_accuracy'], 'percentage', 'accuracy',
              'Decision Tree Min Split ({})'.format(min_split), 'output/divorce_dt_min_split_{}'.format(min_split))
    util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
              'Decision Tree Min Split ({}) Time'.format(min_split), 'output/divorce_dt_min_split_{}_time'.format(min_split))
    return model


def divorce_knn_initial(data, features, target):
    model = KNeighborsClassifier()
    print(model)
    metrics = util.training_by_size(data, features, target, model)
    dump_metrics_to_file(metrics, 'output/divorce_knn_initial.csv')
    util.draw(metrics['x_axis'], metrics['train_accuracy'], metrics['test_accuracy'], 'percentage', 'accuracy',
              'KNN Initial', 'output/divorce_knn_initial')
    util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
              'KNN Initial Time', 'output/divorce_knn_initial_time')


def divorce_knn_hyper(X_all, Y_all):
    param_range = np.linspace(1, 50, 50).astype(int)
    train_score, test_score = util.get_hyperdata_validation(KNeighborsClassifier(), X_all, Y_all, "n_neighbors",
                                                            param_range, 'accuracy')
    util.draw(param_range, train_score, test_score, "number of neighbors", "accuracy", "KNN n_neighbors",
              'output/divorce_knn_neighbors')
    print("Divorce KNN Best param", param_range[test_score.argmax()])
    return param_range[test_score.argmax()]


def divorce_knn_neighbor(data, features, target, neighbor):
    model = KNeighborsClassifier(n_neighbors=neighbor)
    print(model)
    metrics = util.training_by_size(data, features, target, model)
    dump_metrics_to_file(metrics, 'output/divorce_knn_neighbor_{}.csv'.format(neighbor))
    util.draw(metrics['x_axis'], metrics['train_accuracy'], metrics['test_accuracy'], 'percentage', 'accuracy',
              'KNN Neighbor {}'.format(neighbor), 'output/divorce_knn_neighbor_{}'.format(neighbor))
    util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
              'KNN Neighbor{} Time'.format(neighbor), 'output/divorce_knn_neighbor_{}_time'.format(neighbor))


def divorce_knn_metric_algorithm(data, features, target, neighbor):
    algorithms = ['euclidean','manhattan','chebyshev','minkowski']
    for alg in algorithms:
        model = KNeighborsClassifier(n_neighbors=neighbor, metric=alg)
        print(model)
        metrics = util.training_by_size(data, features, target, model)
        dump_metrics_to_file(metrics, 'output/divorce_knn_metric_{}.csv'.format(alg))
        util.draw(metrics['x_axis'], metrics['train_accuracy'], metrics['test_accuracy'], 'percentage', 'accuracy',
                  'KNN Algorithm ({})'.format(alg), 'output/divorce_knn_metric_{}'.format(alg))
        util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
                  'KNN Algorithm ({}) Time'.format(alg), 'output/divorce_knn_metric_{}_time'.format(alg))

def divorce_knn_grid(X_all, Y_all):
    param_grid = {
        'n_neighbors': np.linspace(1, 50, 50).astype(int),
        'metric': ['euclidean','manhattan','chebyshev','minkowski'],
        'weights': ['uniform', 'distance']
    }
    knn = KNeighborsClassifier()
    search_f1 = GridSearchCV(knn, param_grid, cv=10,verbose=1, scoring='accuracy', n_jobs=-1)
    search_f1.fit(X_all, Y_all)
    print(search_f1.best_estimator_, search_f1.best_score_)
    return search_f1.best_estimator_

def divorce_knn_last(data, features, target,model):

    print(model)
    metrics = util.training_by_size(data, features, target, model)
    dump_metrics_to_file(metrics, 'output/divorce_knn_last.csv')
    util.draw(metrics['x_axis'], metrics['train_accuracy'], metrics['test_accuracy'], 'percentage', 'accuracy',
              'KNN', 'output/divorce_knn_last')
    util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
              'KNN ', 'output/divorce_knn_last_time')


def divorce_svc_initial(data, features, target):
    model = SVC(random_state=234)
    print(model)
    metrics = util.training_by_size(data, features, target, model)
    dump_metrics_to_file(metrics, 'output/divorce_svc_initial.csv')
    util.draw(metrics['x_axis'], metrics['train_accuracy'], metrics['test_accuracy'], 'percentage', 'accuracy',
              'SVC Initial', 'output/divorce_svc_initial')
    util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
              'SVC Initial Time', 'output/divorce_svc_initial_time')


def divorce_svc_hyper_kernel(data, features, target):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    for kernel in kernels:
        model = SVC(random_state=234, kernel=kernel)
        print(model)
        metrics = util.training_by_size(data, features, target, model)

        util.draw(metrics['x_axis'], metrics['train_accuracy'], metrics['test_accuracy'], 'percentage', 'accuracy',
                  'SVM kernel ({})'.format(kernel), "output/divorce_svc_kernel_" + kernel)


def divorce_svc_C(X_all, Y_all):
    param_range = np.linspace(0.1,20, 30)

    train_score, test_score = util.get_hyperdata_validation(SVC(random_state=234, kernel='linear',degree=1), X_all, Y_all, "C",
                                                            param_range, 'accuracy')
    util.draw(param_range, train_score, test_score, "C", "accuracy", "SVC linear kernel with C",
              'output/divorce_svc_C')
    print("Divorce SVC Best param", param_range[test_score.argmax()])
    return param_range[test_score.argmax()]


def divorce_svc_grid(X_all, Y_all):
    param_grid = {
        'degree': np.linspace(1, 10, 10).astype(int),
        'coef0': np.linspace(0, 10, 30).astype(int)
    }
    svc = SVC(random_state=234, kernel='linear')
    search_f1 = GridSearchCV(svc, param_grid, cv=10,verbose=1, scoring='accuracy', n_jobs=-1)
    search_f1.fit(X_all, Y_all)
    print(search_f1.best_estimator_, search_f1.best_score_)
    return search_f1.best_estimator_

def divorce_svc_last(data, features, target, model):
    print(model)
    metrics = util.training_by_size(data, features, target, model)
    dump_metrics_to_file(metrics, 'output/divorce_svc_last.csv')
    util.draw(metrics['x_axis'], metrics['train_accuracy'], metrics['test_accuracy'], 'percentage', 'accuracy',
              'SVC Initial', 'output/divorce_svc_last')
    util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
              'SVC Initial Time', 'output/divorce_svc_last_time')

def divorce_boosting_initial(data, features, target):
    model = AdaBoostClassifier(tree.DecisionTreeClassifier(min_samples_split=15), random_state=234)
    print(model)
    metrics = util.training_by_size(data, features, target, model)
    dump_metrics_to_file(metrics, 'output/divorce_boost_initial.csv')
    util.draw(metrics['x_axis'], metrics['train_accuracy'], metrics['test_accuracy'], 'percentage', 'accuracy',
              'Adaboost Initial', 'output/divorce_boost_initial')
    util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
              'Adaboost Initial Time', 'output/divorce_boost_initial_time')


def divorce_boost_hyper(X_all, Y_all):
    param_range = np.linspace(10, 600, 60).astype(int)
    bdt_clf = AdaBoostClassifier(
        tree.DecisionTreeClassifier(min_samples_split=15))
    train_score, test_score = util.get_hyperdata_validation(bdt_clf, X_all, Y_all, "n_estimators",
                                                            param_range, 'accuracy')
    util.draw(param_range, train_score, test_score, "n_estimators", "accuracy", "Boost",
              'output/divorce_boost_hyper_estimator')
    print("Divorce Boost Best param", param_range[test_score.argmax()])
    # Best param 450
    return param_range[test_score.argmax()]


def divorce_boosting_estimator(data, features, target):
    estimator = np.linspace(10, 600, 12).astype(int)
    for n_estimator in estimator:
        model = AdaBoostClassifier(tree.DecisionTreeClassifier(ccp_alpha=0.03), random_state=234, n_estimators=n_estimator)
        print(model)
        metrics = util.training_by_size(data, features, target, model)
        util.draw(metrics['x_axis'], metrics['train_accuracy'], metrics['test_accuracy'], 'percentage', 'accuracy',
                  'Boosting Estimator {}'.format(n_estimator), 'output/divorce_boost_estimator_{}'.format(n_estimator))
        util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
                  'Boosting Estimator {} Time'.format(n_estimator), 'output/divorce_boost_estimator_{}_time'.format(n_estimator))


def divorce_boost_grid(X_all, Y_all):
    param_grid = {
        'base_estimator__min_samples_split': [2,5,10,15,20],
        'algorithm': ['SAMME', 'SAMME.R'],
        'n_estimators': [10, 50, 200, 300, 350, 400, 450, 500, 550, 600, 800],
        'learning_rate': [0.3, 0.5, 0.8, 1.0, 1.2]
    }
    empty_decision_tree = tree.DecisionTreeClassifier()
    adaBoost = AdaBoostClassifier(empty_decision_tree, random_state=234)
    search_f1 = GridSearchCV(adaBoost, param_grid, cv=10,verbose=1, scoring='accuracy', n_jobs=-1)
    search_f1.fit(X_all, Y_all)
    print("Divorce boots grid")
    print(search_f1.best_estimator_, search_f1.best_score_)
    return search_f1.best_estimator_

def divorce_boosting_last(data, features, target, model):
    print(model)
    metrics = util.training_by_size(data, features, target, model)
    dump_metrics_to_file(metrics, 'output/divorce_boost_last.csv')
    util.draw(metrics['x_axis'], metrics['train_accuracy'], metrics['test_accuracy'], 'percentage', 'accuracy',
              'Adaboost Initial', 'output/divorce_boost_last')
    util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
              'Adaboost Initial Time', 'output/divorce_boost_last_time')

def divorce_nn_initial(data, features, target):
    model = MLPClassifier(random_state=234)
    print(model)
    metrics = util.training_by_size(data, features, target, model)

    dump_metrics_to_file(metrics, 'output/divorce_nn_initial.csv')
    util.draw(metrics['x_axis'], metrics['train_accuracy'], metrics['test_accuracy'], 'percentage', 'accuracy',
              'Neural Network Initial', 'output/divorce_nn_initial')
    util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
              'Neural Network Initial Time', 'output/divorce_nn_initial_time')


def divorce_nn_hyper_hiddenlayer(X_all, Y_all):
    # param_range = np.linspace(1, 100, 100).astype(int)
    param_range = np.linspace(50, 500, 10).astype(int)
    param_tuple = [tuple([i]) for i in param_range]
    model = MLPClassifier(random_state=234, max_iter=5000)
    train_score, test_score = util.get_hyperdata_validation(model, X_all, Y_all, "hidden_layer_sizes",
                                                            param_tuple, 'f1_weighted')
    util.draw(param_range, train_score, test_score, "Hidden Layer", "f1 score", "Hidden Layer",
              'output/divorce_nn_hiddenlayer')
    print("NN Hidden layer Best param", param_range[test_score.argmax()])
    # Best param 79


def divorce_nn_hyper_max_iter(X_all, Y_all):
    # param_range = np.linspace(1, 100, 100).astype(int)
    param_range = np.linspace(50, 5000, 100).astype(int)
    model = MLPClassifier(random_state=234, max_iter=5000)
    train_score, test_score = util.get_hyperdata_validation(model, X_all, Y_all, "max_iter",
                                                            param_range, 'f1_weighted')
    util.draw(param_range, train_score, test_score, "Max Iteration ", "f1 score", "Max Iteration ",
              'output/divorce_nn_max_iter')
    print("NN Max iter Best param", param_range[test_score.argmax()])
    return test_score.argmax()


def divorce_nn_hyper_learning_rate(data, features, target):
    # param_range = np.linspace(1, 100, 100).astype(int)
    learning_rate = ['constant', 'invscaling', 'adaptive']
    for learning in learning_rate:
        model = MLPClassifier(random_state=234, learning_rate=learning, solver='sgd', max_iter=1000,
                              learning_rate_init=0.1)
        print(model)
        metrics = util.training_by_size(data, features, target, model)
        util.draw(metrics['x_axis'], metrics['train_accuracy'], metrics['test_accuracy'], 'percentage', 'accuracy',
                  'Neural Network Learning Model ({})'.format(learning), "output/divorce_nn_learning_rate_" + learning)


def divorce_nn_hyper_activation(data, features, target):
    activation_list = ['identity', 'logistic', 'tanh', 'relu']
    for activation in activation_list:
        model = MLPClassifier(random_state=234, activation=activation, solver='sgd', max_iter=1000,
                              learning_rate_init=0.1)
        print(model)
        metrics = util.training_by_size(data, features, target, model)
        util.draw(metrics['x_axis'], metrics['train_accuracy'], metrics['test_accuracy'], 'percentage', 'accuracy',
                  'Neural Network Activation Param ({})'.format(activation), "output/divorce_nn_activation" + activation)


def divorce_nn_grid(X_all, Y_all):
    param_range = np.linspace(5, 50, 10).astype(int)
    hidden_tuple = [tuple([i]) for i in param_range]
    param_grid = {
        'hidden_layer_sizes': hidden_tuple,
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [30,50,60]
    }

    nn = MLPClassifier(random_state=234)
    search_f1 = GridSearchCV(nn, param_grid, cv=10,verbose=1, scoring='accuracy', n_jobs=-1)
    search_f1.fit(X_all, Y_all)
    print(search_f1.best_estimator_, search_f1.best_score_)
    return search_f1.best_estimator_

def divorce_nn_last(data, features, target,model):

    print(model)
    metrics = util.training_by_size(data, features, target, model)

    dump_metrics_to_file(metrics, 'output/divorce_nn_last.csv')
    util.draw(metrics['x_axis'], metrics['train_accuracy'], metrics['test_accuracy'], 'percentage', 'accuracy',
              'Neural Network ', 'output/divorce_nn_last')
    util.draw(metrics['x_axis'], metrics['fit_time'], metrics['score_time'], 'percentage', 'time',
              'Neural Network Time', 'output/divorce_nn_last_time')


def wine():
    features = ['fixed_acidity', 'volatile_acidity', 'citric_acid',
                'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
                'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    target = 'quality'
    file_name = "data/processed/white_wine_updated.csv"
    initial_data = util.get_data(file_name, features, target)
    realworld_data, data = util.get_realworld_data(initial_data, features, target)
    X_train, X_test, y_train, y_test = util.split_data(data, features, target)


    # DT
    # Run without pruning
    print("Wine")
    print("DT")
    print("initial")
    wine_dt_initial(data, features, target)
    print("hyper")
    best_val = wine_dt_hyper(data[features], data[target])
    print("bestparam")
    model = wine_dt_param(data, features, target,best_val)
    util.prediction_report(model,X_train, X_test, y_train, y_test, realworld_data[features], realworld_data[target])
    print("bestparam 0.02")
    model = wine_dt_param(data, features, target, 0.02)
    util.prediction_report(model, X_train, X_test, y_train, y_test, realworld_data[features], realworld_data[target])

    # KNN
    print("knn")
    print("initial")
    wine_knn_initial(data, features, target)
    print("neighbors ")
    best_val = wine_knn_hyper(data[features], data[target])
    wine_knn_neighbor(data, features, target,8)
    print("algorithms  ")
    wine_knn_metric_algorithm(data, features, target,8)
    if do_grid:
        print("grid ")
        model = wine_knn_grid(data[features], data[target])
        print("last ")
        wine_knn_last(data, features, target,model)
        util.prediction_report(model, X_train, X_test, y_train, y_test, realworld_data[features], realworld_data[target])



    # SVM
    print("svm initial")
    wine_svc_initial(data, features, target)
    print("kernel")
    wine_svc_hyper_kernel(data, features, target)
    print("kernel parameter")
    
    #wine_svc_degree(data, features, target)
    #wine_svc_coeff(data, features, target)
    wine_svc_C(data, features, target)

    if do_grid:
        print("grid")
        model = wine_svc_grid(data[features], data[target])
        print("last")
        wine_svc_last(data, features, target, model)
        util.prediction_report(model, X_train, X_test, y_train, y_test, realworld_data[features], realworld_data[target])
    
    # Boosting
    print("boosting initial")
    wine_boosting_initial(data, features, target)
    print("hyper estimator")
    wine_boosting_estimator(data, features, target)
    wine_boost_hyper(data[features], data[target])
    if do_grid:
        print("grid")
        model = wine_boost_grid(data[features], data[target])
        print("last")
        wine_boosting_last(data, features, target, model)
        util.prediction_report(model, X_train, X_test, y_train, y_test, realworld_data[features], realworld_data[target])


    # NN
    print("nn initial")
    wine_nn_initial(data, features, target)
    print("nn hiddenlayer")
    #wine_nn_hiddenlayer(data, features, target)
    wine_nn_hyper_hiddenlayer(data[features], data[target])
    print("nn maxiter")
    wine_nn_hyper_max_iter(data[features], data[target])
    print("nn learning rate")
    wine_nn_hyper_learning_rate(data, features, target)
    print("nn activation")
    wine_nn_hyper_activation(data, features, target)

    if do_grid:
        print("grid")
        model = wine_nn_grid(data[features], data[target])
        print("last run")
        wine_nn_last(data, features, target, model)
        print("prediction")
        util.prediction_report(model, X_train, X_test, y_train, y_test, realworld_data[features], realworld_data[target])


def divorce():
    features = ['Atr1', 'Atr2', 'Atr3', 'Atr4', 'Atr5', 'Atr6', 'Atr7', 'Atr8', 'Atr9', 'Atr10', 'Atr11', 'Atr12',
                'Atr13', 'Atr14', 'Atr15', 'Atr16', 'Atr17', 'Atr18', 'Atr19', 'Atr20', 'Atr21', 'Atr22', 'Atr23',
                'Atr24', 'Atr25', 'Atr26', 'Atr27', 'Atr28', 'Atr29', 'Atr30', 'Atr31', 'Atr32', 'Atr33', 'Atr34',
                'Atr35', 'Atr36', 'Atr37', 'Atr38', 'Atr39', 'Atr40', 'Atr41', 'Atr42', 'Atr43', 'Atr44', 'Atr45',
                'Atr46', 'Atr47', 'Atr48', 'Atr49', 'Atr50', 'Atr51', 'Atr52', 'Atr53', 'Atr54']
    target = 'Class'
    file_name = "data/processed/divorce_updated.csv"
    initial_data = util.get_data(file_name, features, target)
    realworld_data, data = util.get_realworld_data(initial_data, features, target)
    X_train, X_test, y_train, y_test = util.split_data(data, features, target)
    '''
    # DT
    # Run without pruning
    print ("Decision Tree")
    print ("Initial Run")
    divorce_dt_initial(data, features, target)
    divorce_dt_hyper(data[features], data[target])
    divorce_dt_hyper_2(data[features], data[target])
    #print("With best val")
    #model = divorce_dt_param(data, features, target, best_val)
    #util.prediction_report(model,X_train, X_test, y_train, y_test, realworld_data[features], realworld_data[target])
    print("With pruned val")
    model = divorce_dt_param(data, features, target, 10)
    util.prediction_report(model, X_train, X_test, y_train, y_test, realworld_data[features], realworld_data[target])
   
    # KNN
    print ("KNN")
    print("initial")
    divorce_knn_initial(data, features, target)
    print("neighbors ")
    best_val = divorce_knn_hyper(data[features], data[target])
    divorce_knn_neighbor(data, features, target, best_val)
    print("algorithms  ")
    divorce_knn_metric_algorithm(data, features, target,best_val)
    if do_grid:
        print("grid ")
        model = divorce_knn_grid(data[features], data[target])
        print("last ")
        divorce_knn_last(data, features, target,model)
        util.prediction_report(model, X_train, X_test, y_train, y_test, realworld_data[features], realworld_data[target])
    
    
    # SVM
    print("svm initial")
    divorce_svc_initial(data, features, target)
    print("kernel")
    divorce_svc_hyper_kernel(data, features, target)
    
    divorce_svc_C(data[features], data[target])
    
    if do_grid:
        print("grid")
        model = divorce_svc_grid(data[features], data[target])
        print("last")
        divorce_svc_last(data, features, target, model)
        util.prediction_report(model, X_train, X_test, y_train, y_test, realworld_data[features],
                               realworld_data[target])

    # Boosting
    print("boosting initial")
    divorce_boosting_initial(data, features, target)
    print("hyper")
    divorce_boosting_estimator(data, features, target)
    divorce_boost_hyper(data[features], data[target])
    if do_grid:
        model = divorce_boost_grid(data[features], data[target])
        divorce_boosting_last(data, features, target, model)
        util.prediction_report(model, X_train, X_test, y_train, y_test, realworld_data[features], realworld_data[target])

    
    # NN
    print("nn initial")
    divorce_nn_initial(data, features, target)
    print("nn hiddenlayer")
    divorce_nn_hyper_hiddenlayer(data[features], data[target])
    print("nn maxiter")
    divorce_nn_hyper_max_iter(data[features], data[target])
    print("nn learning rate")
    divorce_nn_hyper_learning_rate(data, features, target)
    print("nn activation")
    divorce_nn_hyper_activation(data, features, target)
    '''
    if do_grid:
        model = divorce_nn_grid(data[features], data[target])
        divorce_nn_last(data, features, target, model)
        util.prediction_report(model, X_train, X_test, y_train, y_test, realworld_data[features], realworld_data[target])

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", category=ConvergenceWarning)

starttime = time()
#wine()
divorce()
endtime = time()
print("Total test run at : {}".format(endtime - starttime))
