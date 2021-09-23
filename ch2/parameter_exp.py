import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

from k_armed import KArmedEnv, KArmedNonstationaryEnv
from agent import EpsGreedyAgent, EpsGreedyConstStepAgent, UCBAgent, GradientAgent

PARAMS = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]
NUM_AGENTS = 4
COLORS = ['r', 'k', 'b', 'g']

def create_agents(param: float, k: int=10, num_agt: int=2000):
    eps_greedy  = EpsGreedyAgent(eps=param, k=k, num_agt=num_agt)
    init_greedy = EpsGreedyConstStepAgent(eps=0, q_init=param, k=k, num_agt=num_agt)
    ucb         = UCBAgent(c=param, k=k, num_agt=num_agt)
    gradient    = GradientAgent(step_size=param, use_baseline=True, k=k, num_agt=num_agt)
    return [eps_greedy, init_greedy, ucb, gradient]

# average reward over last [avg_steps] steps
def param_study(time_steps=1000, avg_steps=1000, env_type=KArmedEnv, k=10, num_env=2000):
    # index where recording starts
    rec_idx = time_steps - avg_steps

    # average reward over last [avg_steps] steps per agent
    sum_rewards = [[0 for _ in range(len(PARAMS))] for _ in range(NUM_AGENTS)] 
    sum_rewards = np.array(sum_rewards, dtype=float) # (# agents, # params)

    # loop over all parameter settings
    for param_i, param in tqdm(enumerate(PARAMS), total=len(PARAMS), desc=f"param study", position=1):

        # create environment
        env = env_type(k=k, num_env=num_env)

        # create agents
        agents      = create_agents(param, k=k, num_agt=num_env)

        # loop over time steps
        for t in tqdm(range(time_steps), desc=f"training (param={param})", position=0, leave=False):

            # loop over agents
            for agt_i, agt in enumerate(agents):

                # act
                a = agt.act()
                r = env.step(a)

                # learn
                agt.learn(a, r)

                # stats recording
                env_avg_r = r.mean() # average over [num_env]
                if t >= rec_idx:
                    sum_rewards[agt_i, param_i] += env_avg_r
    
        avg_rewards = sum_rewards / min(time_steps, avg_steps)

        with open('param_study.pkl', 'wb') as f:
            pickle.dump({
                'parameters': PARAMS[:param_i+1],
                'rewards': avg_rewards[:param_i+1],
            }, f)

# Figure 2.6: parameter study of stationary environment
def plot_stationary():
    # get data
    with open('parameter_stationary.pkl', 'rb') as f:
        data = pickle.load(f)
    parameters = data['parameters']
    rewards = data['rewards']

    # plot curves
    fig = plt.figure()
    ax = fig.gca()
    for i in range(rewards.shape[0]):
        ax.plot(parameters, rewards[i, :], color=COLORS[i])
    ax.set_xscale('log', base=2)
    ax.set_ylim(1, 1.5)
    ax.set_xlabel("eps Q0 c alpha")
    ax.set_ylabel("Average reward over last 1000 steps")
    ax.legend(labels=["eps-greedy", "greedy(alpha=0.1)", "UCB", "gradient"])
    fig.savefig("parameter_stationary.png")

# Exercise 2.11: parameter study of nonstationary environment
def plot_nonstationary():
    # get data
    with open('parameter_nonstationary.pkl', 'rb') as f:
        data = pickle.load(f)
    parameters = data['parameters']
    rewards = data['rewards']

    # plot curves
    fig = plt.figure()
    ax = fig.gca()
    for i in range(rewards.shape[0]):
        ax.plot(parameters, rewards[i, :], color=COLORS[i])
    ax.set_xscale('log', base=2)
    # ax.set_ylim(4, 9)
    ax.set_xlabel("eps Q0 c alpha")
    ax.set_ylabel("Average reward over last 100000 steps")
    ax.legend(labels=["eps-greedy", "greedy(alpha=0.1)", "UCB", "gradient"])
    fig.savefig("parameter_nonstationary.png")

if __name__ == '__main__':
    # param_study(time_steps=200000, avg_steps=100000, env_type=KArmedNonstationaryEnv)
    plot_nonstationary()