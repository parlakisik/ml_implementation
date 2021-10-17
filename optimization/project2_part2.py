import numpy as np
import pandas as pd
import mlrose_hiive
import logging
import numpy as np
import matplotlib.pyplot as plt
from mlrose_hiive.algorithms.crossovers import UniformCrossOver
from mlrose_hiive.algorithms.mutators import ChangeOneMutator
from mlrose_hiive.fitness import MaxKColor
from mlrose_hiive.opt_probs.discrete_opt import DiscreteOpt
import time
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score
from mlrose_hiive.algorithms.decay import GeomDecay, ExpDecay,ArithDecay

def plot_iteration(title,rh,sa,ga,mimic):
    fig = plt.figure()
    plt.plot(rh, label = 'RH', color="green", lw=2)
    plt.plot(sa, label = 'SA', color="yellow", lw=2)
    plt.plot(ga, label = 'GA', color="red", lw=2)
    plt.plot(mimic, label = 'MIMIC', color="blue", lw=2)
    plt.ylabel('Fitness Value')
    plt.xlabel('Iterations')
    plt.title('Fitness Value VS Iterations ({})'.format(title))
    plt.legend()
    plt.show()

def plot_iteration_nn(title, iteration, rh, sa, ga,gad, ylabel='Fitness Value', xlabel='Iterations',test_data_name="test"):
    fig = plt.figure()
    plt.plot(iteration, rh, label='RH', color="green", lw=2)
    plt.plot(iteration, sa, label='SA', color="yellow", lw=2)
    plt.plot(iteration, ga, label='GA', color="red", lw=2)
    plt.plot(iteration, gad, label='Backpropagation', color="blue", lw=2)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title('Fitness Value VS Iterations ({})'.format(title))
    plt.legend()
    plt.savefig(test_data_name)
    plt.show()

def current_milli_time():
    return round(time.time() * 1000)


def get_data(file_name, features, target, draw=False, figurename="test"):
    test_data = pd.read_csv(file_name)

    print("Data size:", test_data.shape)
    if draw:
        print(test_data[target].unique())
        print(test_data[target].value_counts())
        test_data[target].value_counts().plot(kind='bar')
        plt.savefig(figurename + ".png")
        plt.show()

    scaler = MinMaxScaler()

    transformed_data = scaler.fit_transform(test_data[features].values)
    transformed_data = pd.DataFrame(transformed_data, columns=features)
    transformed_data[target] = test_data[target]
    return transformed_data;

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

def split_data(data, features,target,test_size=0.20, random_state=324):
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target],test_size=0.20, random_state=234)
    return X_train, X_test, y_train, y_test



def get_test_results(nn,X_train, X_test, y_train, y_test,test_type):
    one_hot = OneHotEncoder()
    st = current_milli_time()
    y_train_hot = one_hot.fit_transform(y_train.values.reshape(-1, 1)).todense()
    y_test_hot = one_hot.transform(y_test.values.reshape(-1, 1)).todense()
    nn.fit(X_train, y_train_hot)
    y_train_pred = nn.predict(X_train)
    if test_type == "f1":
        acc_train = f1_score(y_train_hot, y_train_pred,average='macro')
    else:
        acc_train = accuracy_score(y_train_hot, y_train_pred)
    y_test_pred = nn.predict(X_test)
    if test_type == "f1":
        acc_test = f1_score(y_test_hot, y_test_pred,average='macro')
    else:
        acc_test = accuracy_score(y_test_hot, y_test_pred)
    rh_time = current_milli_time()-st
    #print(acc_train,acc_test)
    #print(rh_time)
    return acc_train,acc_test,rh_time

def NN_train(X_train, X_test, y_train, y_test,activation,test_data_name,test_type):
    iter_num = range(1, 1000, 50)

    gad_train = []
    gad_test = []
    gad_time = []

    rhc_train = []
    rhc_test = []
    rhc_time = []

    gal_train = []
    gal_test = []
    gal_time = []

    sal_train = []
    sal_test = []
    sal_time = []

    iteration_set = []

    for iteration in range(1, 500, 50):
        print("Iteration: {}".format(iteration))

        bp_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], activation=activation, algorithm="gradient_descent",
                                            max_iters=iteration,
                                            max_attempts=10, restarts=10, random_state=234, curve=True)

        gad_acc_train, gad_acc_test, gad_acc_time = get_test_results(bp_nn, X_train, X_test, y_train, y_test,test_type)

        gad_train.append(gad_acc_train)
        gad_test.append(gad_acc_test)
        gad_time.append(gad_acc_time)

        rhc_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], activation=activation, algorithm="random_hill_climb",
                                            max_iters=iteration,
                                            max_attempts=10, restarts=10, random_state=234, curve=True)
        rhc_acc_train, rhc_acc_test, rh_time = get_test_results(rhc_nn, X_train, X_test, y_train, y_test,test_type)

        rhc_train.append(rhc_acc_train)
        rhc_test.append(rhc_acc_test)
        rhc_time.append(rh_time)

        ga_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], activation=activation, algorithm="genetic_alg",
                                           max_iters=iteration,
                                           max_attempts=10, random_state=234, curve=True)
        ga_acc_train, ga_acc_test, ga_time = get_test_results(ga_nn, X_train, X_test, y_train, y_test,test_type)

        gal_train.append(ga_acc_train)
        gal_test.append(ga_acc_test)
        gal_time.append(ga_time)

        sa_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], activation=activation, algorithm="simulated_annealing",
                                           max_iters=iteration,max_attempts=10, random_state=234, curve=True)
        sa_acc_train, sa_acc_test, sa_time = get_test_results(sa_nn, X_train, X_test, y_train, y_test,test_type)

        sal_train.append(sa_acc_train)
        sal_test.append(sa_acc_test)
        sal_time.append(sa_time)

        iteration_set.append(iteration)

    plot_iteration_nn("NN Train vs Iteration ({})".format(activation), range(1, 500, 50), rhc_train, sal_train, gal_train,gad_train,
                      ylabel='Fitness Value', xlabel='Iterations', test_data_name="train"+test_data_name)
    plot_iteration_nn("NN Test vs Iteration ({})".format(activation), range(1, 500, 50), rhc_test, sal_test, gal_test,gad_test,
                      ylabel='Fitness Value', xlabel='Iterations', test_data_name="test"+test_data_name)
    plot_iteration_nn("NN Time vs Iteration ({})".format(activation), range(1, 500, 50), rhc_time, sal_time, gal_time,gad_time, ylabel='Time',
                      xlabel='Iterations', test_data_name="time"+test_data_name)


def NN_train_rhc(X_train, X_test, y_train, y_test,test_data_name,test_type):
    iter_num = range(1, 1000, 50)
    rhc10_train = []
    rhc10_test = []
    rhc10_time = []

    rhc20_train = []
    rhc20_test = []
    rhc20_time = []

    rhc30_train = []
    rhc30_test = []
    rhc30_time = []

    iteration_set = []

    for iteration in range(1, 500, 50):
        print("Iteration: {}".format(iteration))
        rhc_nn_10 = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], algorithm="random_hill_climb",
                                            max_iters=iteration,
                                            max_attempts=10, restarts=10, random_state=234, curve=True)
        rhc10_acc_train, rhc10_acc_test, rh10_time = get_test_results(rhc_nn_10, X_train, X_test, y_train, y_test,test_type)

        rhc10_train.append(rhc10_acc_train)
        rhc10_test.append(rhc10_acc_test)
        rhc10_time.append(rh10_time)

        rhc_nn_20 = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], algorithm="random_hill_climb",
                                            max_iters=iteration,
                                            max_attempts=10, restarts=20, random_state=234, curve=True)
        rhc20_acc_train, rhc20_acc_test, rh20_time = get_test_results(rhc_nn_20, X_train, X_test, y_train, y_test,test_type)

        rhc20_train.append(rhc20_acc_train)
        rhc20_test.append(rhc20_acc_test)
        rhc20_time.append(rh20_time)

        rhc_nn_30 = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], algorithm="random_hill_climb",
                                            max_iters=iteration,
                                            max_attempts=10, restarts=30, random_state=234, curve=True)
        rhc30_acc_train, rhc30_acc_test, rh30_time  = get_test_results(rhc_nn_30, X_train, X_test, y_train, y_test,test_type)

        rhc30_train.append(rhc30_acc_train)
        rhc30_test.append(rhc30_acc_test)
        rhc30_time.append(rh30_time)

        iteration_set.append(iteration)

    fig = plt.figure()
    plt.plot(range(1, 500, 50), rhc10_train, label='Restart 10', color="green", lw=2)
    plt.plot(range(1, 500, 50), rhc20_train, label='Restart 20', color="yellow", lw=2)
    plt.plot(range(1, 500, 50), rhc30_train, label='Restart 30', color="red", lw=2)
    plt.ylabel('Fitness Value')
    plt.xlabel('Iterations')
    plt.title('Fitness Train Value VS Iterations')
    plt.legend()
    plt.savefig(test_data_name+"_rhctrain.png")
    plt.show()

    fig = plt.figure()
    plt.plot(range(1, 500, 50), rhc10_test, label='Restart 10', color="green", lw=2)
    plt.plot(range(1, 500, 50), rhc20_test, label='Restart 20', color="yellow", lw=2)
    plt.plot(range(1, 500, 50), rhc30_test, label='Restart 30', color="red", lw=2)
    plt.ylabel('Fitness Value')
    plt.xlabel('Iterations')
    plt.title('Fitness Test Value VS Iterations')
    plt.legend()
    plt.savefig(test_data_name+"_rhctest.png")
    plt.show()

    fig = plt.figure()
    plt.plot(range(1, 500, 50), rhc30_time, label='Restart 10', color="green", lw=2)
    plt.plot(range(1, 500, 50), rhc30_time, label='Restart 20', color="yellow", lw=2)
    plt.plot(range(1, 500, 50), rhc30_time, label='Restart 30', color="red", lw=2)
    plt.ylabel('Time')
    plt.xlabel('Iterations')
    plt.title('Time VS Iterations')
    plt.legend()
    plt.savefig(test_data_name+"_rhctime.png")
    plt.show()


def NN_train_sa(X_train, X_test, y_train, y_test,test_data_name,test_type):
    iter_num = range(1, 1000, 50)
    salgeo_train = []
    salgeo_test = []
    salgeo_time = []

    salexp_train = []
    salexp_test = []
    salexp_time = []

    salarit_train = []
    salarit_test = []
    salarit_time = []

    iteration_set = []

    for iteration in range(1, 500, 50):
        print("Iteration: {}".format(iteration))
        salgeo_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], algorithm="simulated_annealing",schedule=GeomDecay(),
                                           max_iters=iteration, random_state=234, curve=True)
        salgeo_acc_train, salgeo_acc_test, salgeo_acc_time = get_test_results(salgeo_nn, X_train, X_test, y_train, y_test,test_type)

        salgeo_train.append(salgeo_acc_train)
        salgeo_test.append(salgeo_acc_test)
        salgeo_time.append(salgeo_acc_time)

        salexp_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], algorithm="simulated_annealing",schedule=ExpDecay(),
                                           max_iters=iteration, random_state=234, curve=True)
        salexp_acc_train, salexp_acc_test, salexp_acc_time = get_test_results(salexp_nn, X_train, X_test, y_train, y_test,test_type)

        salexp_train.append(salexp_acc_train)
        salexp_test.append(salexp_acc_test)
        salexp_time.append(salexp_acc_time)

        salarit_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], algorithm="simulated_annealing",schedule=ArithDecay(),
                                           max_iters=iteration, random_state=234, curve=True)
        salarit_acc_train, salarit_acc_test, salarit_acc_time = get_test_results(salarit_nn, X_train, X_test, y_train, y_test,test_type)

        salarit_train.append(salarit_acc_train)
        salarit_test.append(salarit_acc_test)
        salarit_time.append(salarit_acc_time)

        iteration_set.append(iteration)
    fig = plt.figure()
    plt.plot(range(1, 500, 50), salgeo_train, label='Geom Decay', color="green", lw=2)
    plt.plot(range(1, 500, 50), salexp_train, label='Exp Decay', color="yellow", lw=2)
    plt.plot(range(1, 500, 50), salarit_train, label='Arith Decay', color="red", lw=2)
    plt.ylabel('Fitness Value')
    plt.xlabel('Iterations')
    plt.title('Fitness Train Value VS Iterations')
    plt.legend()
    plt.savefig(test_data_name+"_saltrain.png")
    plt.show()

    fig = plt.figure()
    plt.plot(range(1, 500, 50), salgeo_test, label='Geom Decay', color="green", lw=2)
    plt.plot(range(1, 500, 50), salexp_test, label='Exp Decay', color="yellow", lw=2)
    plt.plot(range(1, 500, 50), salarit_test, label='Arith Decay', color="red", lw=2)
    plt.ylabel('Fitness Value')
    plt.xlabel('Iterations')
    plt.title('Fitness Test Value VS Iterations')
    plt.legend()
    plt.savefig(test_data_name+"_saltest.png")
    plt.show()

    fig = plt.figure()
    plt.plot(range(1, 500, 50), salgeo_time, label='Geom Decay', color="green", lw=2)
    plt.plot(range(1, 500, 50), salexp_time, label='Exp Decay', color="yellow", lw=2)
    plt.plot(range(1, 500, 50), salarit_time, label='Arith Decay', color="red", lw=2)
    plt.ylabel('Time')
    plt.xlabel('Iterations')
    plt.title('Time VS Iterations')
    plt.legend()
    plt.savefig(test_data_name+"_saltime.png")
    plt.show()


def NN_train_ga(X_train, X_test, y_train, y_test,test_data_name,test_type):
    iter_num = range(1, 1000, 50)
    gal21_train = []
    gal21_test = []
    gal21_time = []

    gal23_train = []
    gal23_test = []
    gal23_time = []

    gal31_train = []
    gal31_test = []
    gal31_time = []

    gal33_train = []
    gal33_test = []
    gal33_time = []

    gal51_train = []
    gal51_test = []
    gal51_time = []

    gal53_train = []
    gal53_test = []
    gal53_time = []


    iteration_set = []

    for iteration in range(1, 500, 50):
        print("Iteration: {}".format(iteration))

        ga_21_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], algorithm="genetic_alg",
                                           max_iters=iteration,pop_size=200, mutation_prob=0.1,
                                           max_attempts=10, random_state=234, curve=True)
        ga21_acc_train, ga21_acc_test, ga21_acc_time = get_test_results(ga_21_nn, X_train, X_test, y_train, y_test,test_type)

        gal21_train.append(ga21_acc_train)
        gal21_test.append(ga21_acc_test)
        gal21_time.append(ga21_acc_time)

        ga_23_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], algorithm="genetic_alg",
                                           max_iters=iteration,pop_size=200, mutation_prob=0.3,
                                           max_attempts=10, random_state=234, curve=True)
        ga23_acc_train, ga23_acc_test, ga23_acc_time = get_test_results(ga_23_nn, X_train, X_test, y_train, y_test,test_type)

        gal23_train.append(ga23_acc_train)
        gal23_test.append(ga23_acc_test)
        gal23_time.append(ga23_acc_time)

        ga_31_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], algorithm="genetic_alg",
                                           max_iters=iteration,pop_size=300, mutation_prob=0.1,
                                           max_attempts=10, random_state=234, curve=True)
        ga31_acc_train, ga31_acc_test, ga31_acc_time = get_test_results(ga_31_nn, X_train, X_test, y_train, y_test,test_type)

        gal31_train.append(ga31_acc_train)
        gal31_test.append(ga31_acc_test)
        gal31_time.append(ga31_acc_time)

        ga_33_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], algorithm="genetic_alg",
                                           max_iters=iteration,pop_size=300, mutation_prob=0.3,
                                           max_attempts=10, random_state=234, curve=True)
        ga33_acc_train, ga33_acc_test, ga33_acc_time = get_test_results(ga_33_nn, X_train, X_test, y_train, y_test,test_type)

        gal33_train.append(ga33_acc_train)
        gal33_test.append(ga33_acc_test)
        gal33_time.append(ga33_acc_time)

        ga_51_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], algorithm="genetic_alg",
                                           max_iters=iteration,pop_size=500, mutation_prob=0.1,
                                           max_attempts=10, random_state=234, curve=True)
        ga51_acc_train, ga51_acc_test, ga51_acc_time = get_test_results(ga_51_nn, X_train, X_test, y_train, y_test,test_type)

        gal51_train.append(ga51_acc_train)
        gal51_test.append(ga51_acc_test)
        gal51_time.append(ga51_acc_time)

        ga_53_nn = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], algorithm="genetic_alg",
                                           max_iters=iteration,pop_size=500, mutation_prob=0.3,
                                           max_attempts=10, random_state=234, curve=True)
        ga53_acc_train, ga53_acc_test, ga53_acc_time = get_test_results(ga_51_nn, X_train, X_test, y_train, y_test,test_type)

        gal53_train.append(ga53_acc_train)
        gal53_test.append(ga53_acc_test)
        gal53_time.append(ga53_acc_time)

        iteration_set.append(iteration)
    fig = plt.figure()
    plt.plot(range(1, 500, 50), gal21_train, label='Pop:200, Mut:0.1', color="green", lw=2)
    plt.plot(range(1, 500, 50), gal23_train, label='Pop:200, Mut:0.3', color="yellow", lw=2)
    plt.plot(range(1, 500, 50), gal31_train, label='Pop:300, Mut:0.1', color="red", lw=2)
    plt.plot(range(1, 500, 50), gal33_train, label='Pop:300, Mut:0.3', color="green", lw=2)
    plt.plot(range(1, 500, 50), gal51_train, label='Pop:500, Mut:0.1', color="yellow", lw=2)
    plt.plot(range(1, 500, 50), gal53_train, label='Pop:500, Mut:0.3', color="red", lw=2)
    plt.ylabel('Fitness Value')
    plt.xlabel('Iterations')
    plt.title('Fitness Train Value VS Iterations')
    plt.legend()
    plt.savefig(test_data_name+"_galtrain.png")
    plt.show()

    fig = plt.figure()
    plt.plot(range(1, 500, 50), gal21_test, label='Pop:200, Mut:0.1', color="green", lw=2)
    plt.plot(range(1, 500, 50), gal23_test, label='Pop:200, Mut:0.3', color="yellow", lw=2)
    plt.plot(range(1, 500, 50), gal31_test, label='Pop:300, Mut:0.1', color="red", lw=2)
    plt.plot(range(1, 500, 50), gal33_test, label='Pop:300, Mut:0.3', color="green", lw=2)
    plt.plot(range(1, 500, 50), gal51_test, label='Pop:500, Mut:0.1', color="yellow", lw=2)
    plt.plot(range(1, 500, 50), gal53_test, label='Pop:500, Mut:0.3', color="red", lw=2)
    plt.ylabel('Fitness Value')
    plt.xlabel('Iterations')
    plt.title('Fitness Test Value VS Iterations')
    plt.legend()
    plt.savefig(test_data_name+"_galtest.png")
    plt.show()

    fig = plt.figure()
    plt.plot(iteration, gal21_time, label='Pop:200, Mut:0.1', color="green", lw=2)
    plt.plot(iteration, gal23_time, label='Pop:200, Mut:0.3', color="yellow", lw=2)
    plt.plot(iteration, gal31_time, label='Pop:300, Mut:0.1', color="red", lw=2)
    plt.plot(iteration, gal33_time, label='Pop:300, Mut:0.3', color="green", lw=2)
    plt.plot(iteration, gal51_time, label='Pop:500, Mut:0.1', color="yellow", lw=2)
    plt.plot(iteration, gal53_time, label='Pop:500, Mut:0.3', color="red", lw=2)
    plt.ylabel('Time')
    plt.xlabel('Iterations')
    plt.title('Time VS Iterations')
    plt.legend()
    plt.savefig(test_data_name+"_galtime.png")
    plt.show()


def wine():
    features = ['fixed_acidity', 'volatile_acidity', 'citric_acid',
                'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
                'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    target = 'quality'
    file_name = "./white_wine_updated.csv"
    initial_data = get_data(file_name, features, target)
    X_train, X_test, y_train, y_test = split_data(initial_data, features, target)

    print("wine")

    print ("relu")
    NN_train(X_train, X_test, y_train, y_test,"relu","relu"+"wine","f1")
    print("identity")
    NN_train(X_train, X_test, y_train, y_test, "identity", "identity" + "wine","f1")
    print("sigmoid")
    NN_train(X_train, X_test, y_train, y_test, "sigmoid", "sigmoid" + "wine","f1")
    print("tanh")
    NN_train(X_train, X_test, y_train, y_test, "tanh", "tanh" + "wine","f1")

    print("rhc")
    NN_train_rhc(X_train, X_test, y_train, y_test, "wine","f1")
    print("sa")
    NN_train_sa(X_train, X_test, y_train, y_test, "wine","f1")
    print("ga")
    NN_train_ga(X_train, X_test, y_train, y_test, "wine","f1")

def divorce():
    features = ['Atr1', 'Atr2', 'Atr3', 'Atr4', 'Atr5', 'Atr6', 'Atr7', 'Atr8', 'Atr9', 'Atr10', 'Atr11', 'Atr12',
                'Atr13', 'Atr14', 'Atr15', 'Atr16', 'Atr17', 'Atr18', 'Atr19', 'Atr20', 'Atr21', 'Atr22', 'Atr23',
                'Atr24', 'Atr25', 'Atr26', 'Atr27', 'Atr28', 'Atr29', 'Atr30', 'Atr31', 'Atr32', 'Atr33', 'Atr34',
                'Atr35', 'Atr36', 'Atr37', 'Atr38', 'Atr39', 'Atr40', 'Atr41', 'Atr42', 'Atr43', 'Atr44', 'Atr45',
                'Atr46', 'Atr47', 'Atr48', 'Atr49', 'Atr50', 'Atr51', 'Atr52', 'Atr53', 'Atr54']
    target = 'Class'
    file_name = "./divorce_updated.csv"
    initial_data = get_data(file_name, features, target)
    X_train, X_test, y_train, y_test = split_data(initial_data, features, target)
    print("divorce")

    print("relu")
    NN_train(X_train, X_test, y_train, y_test, "relu", "relu"+"divorce","acc")
    print("identity")
    NN_train(X_train, X_test, y_train, y_test, "identity", "identity" + "divorce","acc")
    print("sigmoid")
    NN_train(X_train, X_test, y_train, y_test, "sigmoid", "sigmoid" + "divorce","acc")
    print("tanh")
    NN_train(X_train, X_test, y_train, y_test, "tanh", "tanh" + "divorce","acc")

    print("rhc")
    NN_train_rhc(X_train, X_test, y_train, y_test, "divorce","acc")
    print("sa")
    NN_train_sa(X_train, X_test, y_train, y_test, "divorce","acc")
    print("ga")
    NN_train_ga(X_train, X_test, y_train, y_test, "divorce","acc")


wine()
divorce()