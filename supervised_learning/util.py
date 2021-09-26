import time

import pandas as pd               # package for data analysis and manipulation
import numpy as np                # package for scientific computing on multidimensional arrays
import matplotlib                 # package for creating visualizations
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, validation_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import plot_confusion_matrix, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def draw(test_x, train_rate, val_rate, xlabel, ylabel, title,filename):
    plt.plot(test_x, train_rate , "r-+", linewidth=2, label="train")
    plt.plot(test_x, val_rate, "b-+", linewidth=2, label="validation")
    plt.title(title)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.savefig(filename + ".png")
    plt.show()


def confusion_drawing(classifier,X_test, y_test,class_names,normalize, title,filename):
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
    plt.savefig(filename+".png")
    plt.show()

def prediction_report(model,X_train, X_test, y_train, y_test, X_real,y_real):
    start_time = time.time() * 1000
    model.fit(X_train,y_train)
    end_time = time.time() * 1000
    print ("Training Time:",end_time-start_time)
    start_time = time.time() * 1000
    y_pred = model.predict(X_test)
    end_time = time.time() * 1000
    print("Testing Time:", end_time - start_time)

    print(classification_report(y_test, y_pred, digits=5))
    print(confusion_matrix(y_test, y_pred))
    print("--Real World")
    y_pred = model.predict(X_real)
    print(classification_report(y_real, y_pred, digits=5))
    print(confusion_matrix(y_real, y_pred))


def avg(data):
    return sum(data) / len(data)

metric_types = ['fit_time', 'score_time', 'test_f1_macro', 'test_precision_macro', 'test_recall_macro', 'train_f1_macro', 'train_precision_macro', 'train_recall_macro','test_accuracy','train_accuracy']
def initiate_metrics():
    metric={}

    for data in metric_types:
        metric[data]=[]
    metric['x_axis']=[]
    return metric

def append_metrics(metric, data, data_type):
    metric[data_type].append(avg(data[data_type]))

def training_by_size(data, features, target, model, initial=20,max=105,increment=5):
    metrics = initiate_metrics()
    for percentage in range(initial, max, increment):
        size = int(len(data)*percentage/100)
        train = data[:percentage]
        scoring = ['precision_macro', 'precision_micro', 'recall_macro', 'recall_micro', 'f1_macro','f1_micro','accuracy']
        scores = cross_validate(model,train[features], train[target], scoring=scoring, cv=5, return_train_score=True)
        for type in metric_types:
            append_metrics(metrics,scores,type)
        metrics['x_axis'].append(percentage)

    return metrics

def get_hyperdata_validation(model,x ,y,param_name,param_range, scoring):
    train_scores, test_scores = validation_curve(model, x, y, param_name=param_name, param_range=param_range,
                                                 scoring=scoring, verbose=1, cv=10, n_jobs=-1)

    return np.mean(train_scores,axis=1), np.mean(test_scores,axis=1)


def get_data(file_name, features, target, draw=False,figurename="test"):
    test_data = pd.read_csv(file_name)

    print("Data size:",test_data.shape)
    if draw:
        print(test_data[target].unique())
        print(test_data[target].value_counts())
        test_data[target].value_counts().plot(kind='bar')
        plt.savefig(figurename + ".png")
        plt.show()

    scaler = StandardScaler()
    transformed_data = scaler.fit_transform(test_data[features].values)
    transformed_data = pd.DataFrame(transformed_data, columns=features)
    transformed_data[target] = test_data[target]
    return transformed_data;


def split_data(data, features,target,test_size=0.20, random_state=324):
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target],test_size=0.20, random_state=234)
    return X_train, X_test, y_train, y_test

def get_realworld_data(data, features,target):
    skf = StratifiedKFold(n_splits=10, shuffle=False)
    fold = skf.split(data[features], data[target])
    fold = list(fold)
    train_index, test_index = fold[0]
    realworld_data = data.loc[test_index]
    realworld_data.reset_index(drop=True, inplace=True)
    train_data = data.loc[train_index]
    train_data.reset_index(drop=True, inplace=True)
    return realworld_data, train_data