import hiive.mdptoolbox
import hiive.mdptoolbox.example
import hiive.mdptoolbox.mdp
import matplotlib.pyplot as plt
from gym.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv
import re
import gym
import numpy as np
import random
from timeit import default_timer as timer
from datetime import timedelta
import pandas as pd
import time

def policy_evaluation(P, R, policy, test_count=1000, gamma=0.9):
    num_state = P.shape[-1]
    total_episode = num_state * test_count
    # start in each state
    total_reward = 0
    for state in range(num_state):
        state_reward = 0
        for state_episode in range(test_count):
            episode_reward = 0
            disc_rate = 1
            ep = 0
            while True and ep < 10000:
                # take step
                ep += 1
                action = policy[state]
                # get next step using P
                probs = P[action][state]
                candidates = list(range(len(P[action][state])))
                next_state =  np.random.choice(candidates, 1, p=probs)[0]
                # get the reward
                reward = R[state][action] * disc_rate
                episode_reward += reward
                # when go back to 0 ended
                disc_rate *= gamma
                if next_state == 0:
                    break
            state_reward += episode_reward
        total_reward += state_reward
    return total_reward / total_episode


def test_policy(env, policy, n_epoch=1000):
    rewards = []
    episode_counts = []
    for i in range(n_epoch):
        current_state = env.reset()
        ep = 0
        done = False
        episode_reward = 0
        while not done and ep < 10000:
            ep += 1
            act = int(policy[current_state])
            new_state, reward, done, _ = env.step(act)
            episode_reward += reward
            current_state = new_state
        rewards.append(episode_reward)
        episode_counts.append(ep)

    # all done
    return sum(rewards) / len(rewards)


def plot_data(x, y, x_label, y_label, title,savename):
    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.savefig(savename+".png")
    plt.show()



class OpenAI_MDPToolbox:
    """Class to convert Discrete Open AI Gym environemnts to MDPToolBox environments.
    You can find the list of available gym environments here: https://gym.openai.com/envs/#classic_control
    You'll have to look at the source code of the environments for available kwargs; as it is not well documented.
    """

    def __init__(self, openAI_env_name: str, render: bool = False, **kwargs):
        """Create a new instance of the OpenAI_MDPToolbox class
        :param openAI_env_name: Valid name of an Open AI Gym env
        :type openAI_env_name: str
        :param render: whether to render the Open AI gym env
        :type rander: boolean
        """
        self.env_name = openAI_env_name

        self.env = gym.make(self.env_name, **kwargs)
        self.env.reset()

        if render:
            self.env.render()

        self.transitions = self.env.P
        self.actions = int(re.findall(r'\d+', str(self.env.action_space))[0])
        self.states = int(re.findall(r'\d+', str(self.env.observation_space))[0])
        self.P = np.zeros((self.actions, self.states, self.states))
        self.R = np.zeros((self.states, self.actions))
        self.convert_PR()

    def convert_PR(self):
        """Converts the transition probabilities provided by env.P to MDPToolbox-compatible P and R arrays
        """
        for state in range(self.states):
            for action in range(self.actions):
                for i in range(len(self.transitions[state][action])):
                    tran_prob = self.transitions[state][action][i][0]
                    state_ = self.transitions[state][action][i][1]
                    self.R[state][action] += tran_prob * self.transitions[state][action][i][2]
                    self.P[action, state, state_] += tran_prob

def openai(env_name:str, render:bool=False, **kwargs):
    env = OpenAI_MDPToolbox(env_name, render, **kwargs)
    return env.P, env.R


def colors():
    return {
        'S': 'green',
        'F': 'skyblue',
        'H': 'black',
        'G': 'gold',
    }

def directions():
    return {
        3: '⬆',
        2: '➡',
        1: '⬇',
        0: '⬅'
    }


def show_policy_map(title, policy, map_desc, color_map, direction_map,file_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size = 'x-large'
    if policy.shape[1] > 16:
        font_size = 'small'
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1)
            p.set_facecolor(color_map[map_desc[i][j]])
            ax.add_patch(p)
            if map_desc[i][j] == 'H' :
                text = ax.text(x + 0.5, y + 0.5, "", weight='bold', size=font_size,
                               horizontalalignment='center', verticalalignment='center', color='w')
            else:
                text = ax.text(x + 0.5, y + 0.5, direction_map[policy[i, j]], weight='bold', size=font_size,
                           horizontalalignment='center', verticalalignment='center', color='w')

    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    plt.tight_layout()
    plt.show()
    plt.draw()
    fig.savefig(file_name+str('.png'))


def callback_frozenlake(old_s, action, new_s):
    global map_states
    if map_states[new_s] == 'G' or map_states[new_s] == 'H':
        return True
    else:
        return False


def callback_forest(old_s, action, new_s):
    global iter_count
    global check_val
    iter_count += 1
    return iter_count % check_val == 0


def qlearn_mpd(P, R, desc, mdp_name='',
                      max_iters=100000, alpha=0.1, gamma=0.95,
                      plot_grid=False,epsilon=1,forest=False):



    if forest:
        global iter_count
        iter_count = 0
        global check_val
        check_val = 10000
        ql = hiive.mdptoolbox.mdp.QLearning(P, R, gamma=gamma, n_iter=max_iters,
                                            alpha=alpha, iter_callback=callback_forest,
                                            run_stat_frequency=1)
    else:
        global map_states
        map_states = ''.join(desc)
        ql = hiive.mdptoolbox.mdp.QLearning(P, R, gamma=gamma, n_iter=max_iters,
                                            alpha=alpha, iter_callback=callback_frozenlake,
                                            run_stat_frequency=1)

    start_time = time.time()
    ql.run()
    print("Ql finished  time : {}".format((time.time() - start_time)))
    output = ql.run_stats
    result_df = pd.DataFrame(output)
    result_df.set_index('Iteration')



    rolling_rew = result_df['Reward'].rolling(5000).mean()

    rolling_rew.plot(ylabel='Avg Reward',
                     xlabel='Iterations',
                     legend=True,
                     grid=True,
                     title="MDP: {} Gamma:({}) Alpha({})".format(mdp_name, gamma,alpha))
    plt.savefig("{}_qvalue_{}_{}.png".format(mdp_name, gamma,alpha))
    plt.show()

    if plot_grid and not forest:
        data_arr = []
        array_size = len(desc[0])
        for word in desc:
            data_arr.append(list(word))
        print(data_arr)
        show_policy_map("{} Size ({}) Discount ({})".format(mdp_name, array_size, gamma),
                        np.asarray(ql.policy).reshape(array_size, array_size), data_arr, colors(), directions(),
                        "{}_{}_{}".format(mdp_name, array_size, gamma))


def mdp_solver(P, R, mdp_name, info, plot_grid = False, desc=None,isForest=False):



    discount_list = np.linspace(0.1, 1, 20, endpoint=False)
    v_arr = []
    p_arr = []
    iter_arr = []
    time_arr = []
    discount_arr = []
    data_arr = []
    print("Policy Iteration")
    for discount in discount_list:
        pi = hiive.mdptoolbox.mdp.PolicyIteration(P, R, discount, max_iter=1000)
        data = pi.run()
        data_arr.append(data)
        discount_arr.append(discount)
        v_arr.append(np.mean(pi.V))
        p_arr.append(pi.policy)
        iter_arr.append(pi.iter)
        time_arr.append(pi.time)
        if (plot_grid):
            data_arr = []
            array_size = len(desc[0])
            for word in desc:
                data_arr.append(list(word))
            show_policy_map("{} Size ({}) Discount ({})".format(mdp_name, info, discount), np.asarray(pi.policy).reshape(array_size, array_size), data_arr, colors(), directions(),
                            "{}_pi_{}_{}".format(mdp_name, info, discount))
    plot_data(discount_arr, time_arr, "Discount", "Time", "{} Policy ({})".format(mdp_name, info),"{}_{}_policy_time".format(mdp_name, info))
    plot_data(discount_arr, iter_arr, "Discount", "Iteration", "{} Policy ({})".format(mdp_name, info),"{}_{}_policy_iteration".format(mdp_name, info))
    plot_data(discount_arr, v_arr, "Discount", "Average V", "{} Policy ({})".format(mdp_name, info),"{}_{}_policy_average".format(mdp_name, info))
    print("Value Iteration")
    v_arr = []
    p_arr = []
    iter_arr = []
    time_arr = []
    discount_arr = []
    data_arr = []
    for discount in discount_list:
        pi = hiive.mdptoolbox.mdp.ValueIteration(P, R, discount, max_iter=1000)
        data = pi.run()
        data_arr.append(data)
        discount_arr.append(discount)
        v_arr.append(np.mean(pi.V))
        p_arr.append(pi.policy)
        iter_arr.append(pi.iter)
        time_arr.append(pi.time)
        if (plot_grid):
            data_arr = []
            array_size = len(desc[0])
            for word in desc:
                data_arr.append(list(word))
            show_policy_map("{} Size ({}) Discount ({})".format(mdp_name, info, discount), np.asarray(pi.policy).reshape(array_size, array_size), data_arr, colors(), directions(),
                            "{}_vi_{}_{}".format(mdp_name, info, discount))
    plot_data(discount_arr, time_arr, "Discount", "Time", "{} Value ({})".format(mdp_name, info),"{}_{}_value_time".format(mdp_name, info))
    plot_data(discount_arr, iter_arr, "Discount", "Iteration", "{} Value ({})".format(mdp_name, info),"{}_{}_value_iteration".format(mdp_name, info))
    plot_data(discount_arr, v_arr, "Discount", "Average V", "{} Value ({})".format(mdp_name, info),"{}_{}_value_average".format(mdp_name, info))

    print("Q Iteration")
    alpha = [ 0.1,  0.3, 0.5,  0.95, 0.999]
    gamma = [0.001 ,0.75 , 0.95]
    iters = [10000, 30000, 50000, 75000, 100000]
    for a in alpha:
        for g in gamma:
            print("Alpha {} Gamma {}".format(a,g))
            qlearn_mpd(P, R, desc, mdp_name="{}_{}".format(mdp_name,info),alpha=a,gamma=g,  max_iters=1000000,plot_grid=True, forest=isForest)



def forest():
    np.random.seed(234)
    print("Forest 10")
    P, R = hiive.mdptoolbox.example.forest(S=10, r1=4, r2=2, p=0.05)
    mdp_solver(P, R, "Forest", 10,isForest=True)

    print("Forest 1000")
    P, R = hiive.mdptoolbox.example.forest(S=1000, r1=4, r2=2, p=0.05)
    mdp_solver(P, R, "Forest", 1000,isForest=True)


def frozen_lake():
    np.random.seed(234)
    frozenlake_ten = generate_random_map(10, p=0.8)
    frozenlake_twentyfive= generate_random_map(25, p=0.8)


    print("Frozen Lake 10")
    print (frozenlake_ten)
    P, R = openai("FrozenLake-v1", desc=frozenlake_ten)
    mdp_solver(P, R, "Frozen Lake", 10,plot_grid = True,desc=frozenlake_ten)

    print("Frozen Lake 100")
    P, R = openai("FrozenLake-v1", desc=frozenlake_twentyfive)
    mdp_solver(P, R, "Frozen Lake", 25,plot_grid = True,desc=frozenlake_twentyfive)


frozen_lake()
forest()
