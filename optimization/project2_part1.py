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
from collections import defaultdict
from mlrose_hiive import TSPOpt
import itertools as it
from mlrose_hiive.algorithms.decay import GeomDecay, ExpDecay,ArithDecay

def plot_iteration(title, rh, sa, ga, mimic):
    fig = plt.figure()
    plt.plot(rh, label='RH', color="green", linewidth=2)
    plt.plot(sa, label='SA', color="yellow", linewidth=2)
    plt.plot(ga, label='GA', color="red", linewidth=2)
    plt.plot(mimic, label='MIMIC', color="blue", linewidth=2)
    plt.ylabel('Fitness')
    plt.xlabel('Iterations')
    plt.title('Fitness Value VS Iterations ({})'.format(title))
    plt.legend()
    plt.savefig(title + ".png")
    plt.show()


def current_milli_time():
    return round(time.time() * 1000)


def max_k_color_generate(seed, number_of_nodes=20, max_connections_per_node=4, max_colors=None):
    np.random.seed(seed)
    # all nodes have to be connected, somehow.
    node_connection_counts = 1 + np.random.randint(max_connections_per_node, size=number_of_nodes)

    node_connections = {}
    nodes = range(number_of_nodes)
    for n in nodes:
        all_other_valid_nodes = [o for o in nodes if (o != n and (o not in node_connections or
                                                                  n not in node_connections[o]))]
        count = min(node_connection_counts[n], len(all_other_valid_nodes))
        other_nodes = sorted(np.random.choice(all_other_valid_nodes, count, replace=False))
        node_connections[n] = [(n, o) for o in other_nodes]

    # check connectivity
    g = nx.Graph()
    g.add_edges_from([x for y in node_connections.values() for x in y])

    for n in nodes:
        cannot_reach = [(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()]
        for s, f in cannot_reach:
            g.add_edge(s, f)
            check_reach = len([(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()])
            if check_reach == 0:
                break

    edges = [(s, f) for (s, f) in g.edges()]
    problem = mlrose_hiive.MaxKColorOpt(edges=edges, length=number_of_nodes, max_colors=max_colors, maximize=True,
                                        source_graph=g)
    return problem

# TSP

def get_distances(coords, truncate=True):
    distances = [(c1, c2, np.linalg.norm(np.subtract(coords[c1], coords[c2])))
                 for c1, c2 in it.product(range(len(coords)), range(len(coords)))
                 if c1 != c2 and c2 > c1]
    if truncate:
        distances = [(c1, c2, int(d)) for c1, c2, d in distances]
    return distances


def list_duplicates_(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return list((indices[1:] for _, indices in tally.items() if len(indices) > 1))

def tsp_generate(seed, number_of_cities, area_width=250, area_height=250):
    np.random.seed(seed)
    x_coords = np.random.randint(area_width, size=number_of_cities)
    y_coords = np.random.randint(area_height, size=number_of_cities)

    coords = list(tuple(zip(x_coords, y_coords)))
    duplicates = list_duplicates_(coords)

    while len(duplicates) > 0:
        for d in duplicates:
            x_coords = np.random.randint(area_width, size=len(d))
            y_coords = np.random.randint(area_height, size=len(d))
            for i in range(len(d)):
                coords[d[i]] = (x_coords[i], y_coords[i])
                pass
        duplicates = list_duplicates_(coords)
    distances = get_distances(coords, False)

    return TSPOpt(coords=coords, distances=distances, maximize=True)



def test_problem(problem, name, version):
    print ("---------{}-{}----".format(name,version))
    print("rh .")
    st = current_milli_time()
    best_state_rh, best_fitness_rh, ft_rh = mlrose_hiive.random_hill_climb(problem,
                                                                           max_attempts=1000, max_iters=500,
                                                                           restarts=10, curve=True, random_state=234)
    rh_time = current_milli_time() - st
    print("sa .")
    st = current_milli_time()
    best_state_sa, best_fitness_sa, ft_sa = mlrose_hiive.simulated_annealing(problem, max_attempts=1000,
                                                                             max_iters=500, curve=True,
                                                                             random_state=234)
    sa_time = current_milli_time() - st
    print("ga .")
    st = current_milli_time()
    best_state_ga, best_fitness_ga, ft_ga = mlrose_hiive.genetic_alg(problem, max_attempts=1000,
                                                                     max_iters=500, curve=True, random_state=234)
    ga_time = current_milli_time() - st

    print("mimic .")
    st = current_milli_time()
    best_state_mimic, best_fitness_mimic, ft_mimic = mlrose_hiive.mimic(problem, max_attempts=1000,
                                                                        max_iters=500, curve=True, random_state=234)
    mimic_time = current_milli_time() - st
    st = current_milli_time()

    plot_iteration(name + version, ft_rh, ft_sa, ft_ga, ft_mimic)
    print("RH :{}, SA: {}, GA: {}, MIMIC: {}".format(best_fitness_rh, best_fitness_sa, best_fitness_ga,
                                                     best_fitness_mimic))
    print("RH :{}, SA: {}, GA: {}, MIMIC: {}".format(rh_time, sa_time, ga_time, mimic_time))

def rhl_tests(problem,problem_name):
    _, _, ft_rh5 = mlrose_hiive.random_hill_climb(problem,max_attempts=1000, max_iters=500,restarts=5, curve=True, random_state=234)
    _, _, ft_rh10 = mlrose_hiive.random_hill_climb(problem, max_attempts=1000, max_iters=500, restarts=10, curve=True,
                                                 random_state=234)
    _, _, ft_rh15 = mlrose_hiive.random_hill_climb(problem, max_attempts=1000, max_iters=500, restarts=15, curve=True,
                                                 random_state=234)
    _, _, ft_rh20 = mlrose_hiive.random_hill_climb(problem, max_attempts=1000, max_iters=500, restarts=20, curve=True,
                                                 random_state=234)

    fig = plt.figure()
    plt.plot(ft_rh5, label='Restart 5', color="green", linewidth=2)
    plt.plot(ft_rh10, label='Restart 10', color="yellow", linewidth=2)
    plt.plot(ft_rh15, label='Restart 15', color="red", linewidth=2)
    plt.plot(ft_rh20, label='Restart 20', color="blue", linewidth=2)
    plt.ylabel('Fitness')
    plt.xlabel('Iterations')
    plt.title('Fitness Value VS Iterations ({})'.format(problem_name))
    plt.legend()
    plt.savefig(problem_name + "_rhl.png")
    plt.show()

def sa_tests(problem,problem_name):
    _, _, ft_sa_geom = mlrose_hiive.simulated_annealing(problem, max_attempts=1000,schedule=GeomDecay(),
                                                                             max_iters=500, curve=True,
                                                                             random_state=234)
    _, _, ft_sa_exp = mlrose_hiive.simulated_annealing(problem, max_attempts=1000, schedule=ExpDecay(),
                                                   max_iters=500, curve=True,
                                                   random_state=234)
    _, _, ft_sa_arith = mlrose_hiive.simulated_annealing(problem, max_attempts=1000, schedule=ArithDecay(),
                                                   max_iters=500, curve=True,
                                                   random_state=234)
    fig = plt.figure()
    plt.plot(ft_sa_geom, label='GeomDecay', color="green", linewidth=2)
    plt.plot(ft_sa_exp, label='ExpDecay', color="yellow", linewidth=2)
    plt.plot(ft_sa_arith, label='ArithDecay', color="red", linewidth=2)
    plt.ylabel('Fitness')
    plt.xlabel('Iterations')
    plt.title('Fitness Value VS Iterations ({})'.format(problem_name))
    plt.legend()
    plt.savefig(problem_name + "_sa.png")
    plt.show()

def ga_tests(problem,problem_name):
    _, _, ft_ga2_7 = mlrose_hiive.genetic_alg(problem, max_attempts=1000,max_iters=500,pop_size=200, pop_breed_percent=0.75 , curve=True, random_state=234)
    _, _, ft_ga2_5 = mlrose_hiive.genetic_alg(problem, max_attempts=1000, max_iters=500, pop_size=200,
                                           pop_breed_percent=0.50, curve=True, random_state=234)
    _, _, ft_ga3_7 = mlrose_hiive.genetic_alg(problem, max_attempts=1000, max_iters=500, pop_size=300,
                                           pop_breed_percent=0.75, curve=True, random_state=234)
    _, _, ft_ga3_5 = mlrose_hiive.genetic_alg(problem, max_attempts=1000, max_iters=500, pop_size=300,
                                           pop_breed_percent=0.50, curve=True, random_state=234)
    _, _, ft_ga5_7 = mlrose_hiive.genetic_alg(problem, max_attempts=1000, max_iters=500, pop_size=500,
                                           pop_breed_percent=0.75, curve=True, random_state=234)
    _, _, ft_ga5_5 = mlrose_hiive.genetic_alg(problem, max_attempts=1000, max_iters=500, pop_size=500,
                                           pop_breed_percent=0.50, curve=True, random_state=234)

    fig = plt.figure()
    plt.plot(ft_ga2_7, label='pop(200) breed(0.75)', color="green", linewidth=2)
    plt.plot(ft_ga2_5, label='pop(200) breed(0.5)', color="yellow", linewidth=2)
    plt.plot(ft_ga3_7, label='pop(300) breed(0.75)', color="red", linewidth=2)
    plt.plot(ft_ga3_5, label='pop(300) breed(0.5)', color="blue", linewidth=2)
    plt.plot(ft_ga5_7, label='pop(500) breed(0.75)', color="black", linewidth=2)
    plt.plot(ft_ga5_5, label='pop(500) breed(0.5)', color="purple", linewidth=2)
    plt.ylabel('Fitness')
    plt.xlabel('Iterations')
    plt.title('Fitness Value VS Iterations ({})'.format(problem_name))
    plt.legend()
    plt.savefig(problem_name + "_ga.png")
    plt.show()

def mimic_tests(problem, problem_name):
    _, _, ft_mimic2_2 = mlrose_hiive.mimic(problem, max_attempts=1000,pop_size=200, keep_pct=0.2,
                                                                        max_iters=500, curve=True, random_state=234)
    _, _, ft_mimic2_4 = mlrose_hiive.mimic(problem, max_attempts=1000,pop_size=200, keep_pct=0.4,
                                                                        max_iters=500, curve=True, random_state=234)
    _, _, ft_mimic3_2 = mlrose_hiive.mimic(problem, max_attempts=1000,pop_size=300, keep_pct=0.2,
                                                                        max_iters=500, curve=True, random_state=234)
    _, _, ft_mimic3_4 = mlrose_hiive.mimic(problem, max_attempts=1000,pop_size=300, keep_pct=0.4,
                                                                        max_iters=500, curve=True, random_state=234)
    _, _, ft_mimic4_2 = mlrose_hiive.mimic(problem, max_attempts=1000,pop_size=500, keep_pct=0.2,
                                                                        max_iters=500, curve=True, random_state=234)
    _, _, ft_mimic4_4 = mlrose_hiive.mimic(problem, max_attempts=1000,pop_size=500, keep_pct=0.4,
                                                                        max_iters=500, curve=True, random_state=234)

    fig = plt.figure()
    plt.plot(ft_mimic2_2, label='pop(200) keep_pct(0.2)', color="green", linewidth=2)
    plt.plot(ft_mimic2_4, label='pop(200) breed(0.4)', color="yellow", linewidth=2)
    plt.plot(ft_mimic3_2, label='pop(300) breed(0.2)', color="red", linewidth=2)
    plt.plot(ft_mimic3_4, label='pop(300) breed(0.4)', color="blue", linewidth=2)
    plt.plot(ft_mimic4_2, label='pop(500) breed(0.2)', color="black", linewidth=2)
    plt.plot(ft_mimic4_4, label='pop(500) breed(0.4)', color="purple", linewidth=2)
    plt.ylabel('Fitness')
    plt.xlabel('Iterations')
    plt.title('Fitness Value VS Iterations ({})'.format(problem_name))
    plt.legend()
    plt.savefig(problem_name + "_mimic.png")
    plt.show()

print("initial")
problem_knapsack = mlrose_hiive.KnapsackGenerator.generate(123)
problem_maxkcolor = max_k_color_generate(123, 20)
problem_flipflop = mlrose_hiive.FlipFlopGenerator.generate(123, size=20)
problem_tsp = tsp_generate(123, 50)
problem_cp = mlrose_hiive.ContinuousPeaksGenerator.generate(123, size=20)

test_problem(problem_knapsack, "Knapsack", "1")
test_problem(problem_maxkcolor, "Max K color", "1")
test_problem(problem_flipflop, "Flip Flop", "1")
test_problem(problem_tsp, "TSP", "1")
test_problem(problem_cp, "Continuos Peak", "1")

print("Second -----")
problem_knapsack = mlrose_hiive.KnapsackGenerator.generate(123, number_of_items_types=50,
                                                           max_item_count=30, max_weight_per_item=25,
                                                           max_value_per_item=10)
problem_maxkcolor = max_k_color_generate(123, number_of_nodes=50)
problem_flipflop = mlrose_hiive.FlipFlopGenerator.generate(123, size=50)
problem_tsp = tsp_generate(123, number_of_cities=100)
problem_cp = mlrose_hiive.ContinuousPeaksGenerator.generate(123, size=50)

test_problem(problem_knapsack, "Knapsack", "2")
test_problem(problem_maxkcolor, "Max K color", "2")
test_problem(problem_flipflop, "Flip Flop", "2")

test_problem(problem_tsp, "TSP", "2")
test_problem(problem_cp, "Continuos Peak", "2")

print("Third -----")

problem_knapsack = mlrose_hiive.KnapsackGenerator.generate(123, number_of_items_types=100,
                                                           max_item_count=30, max_weight_per_item=25,
                                                           max_value_per_item=10)
problem_maxkcolor = max_k_color_generate(123, number_of_nodes=100)
problem_flipflop = mlrose_hiive.FlipFlopGenerator.generate(123, size=100)
problem_tsp = tsp_generate(123, number_of_cities=120)
problem_cp = mlrose_hiive.ContinuousPeaksGenerator.generate(123, size=100)

test_problem(problem_knapsack, "Knapsack", "3")
test_problem(problem_maxkcolor, "Max K color", "3")
test_problem(problem_flipflop, "Flip Flop", "3")
test_problem(problem_tsp, "TSP", "3")

test_problem(problem_cp, "Continuos Peak", "3")

# Part to hyperparameter tests
problem_maxkcolor = max_k_color_generate(123, 20)
problem_flipflop = mlrose_hiive.FlipFlopGenerator.generate(123, size=20)
problem_cp = mlrose_hiive.ContinuousPeaksGenerator.generate(123, size=20)

print("RHL")
rhl_tests(problem_maxkcolor,"MaxKColor")
rhl_tests(problem_flipflop,"FlipFlop")
rhl_tests(problem_cp,"ContinuousPeak")
print("SA")
sa_tests(problem_maxkcolor,"MaxKColor")
sa_tests(problem_flipflop,"FlipFlop")
sa_tests(problem_cp,"ContinuousPeak")
print("GA")
ga_tests(problem_maxkcolor,"MaxKColor")
ga_tests(problem_flipflop,"FlipFlop")
ga_tests(problem_cp,"ContinuousPeak")
print("Mimic")
mimic_tests(problem_maxkcolor, "MaxKColor")
mimic_tests(problem_flipflop, "FlipFlop")
mimic_tests(problem_cp, "ContinuousPeak")