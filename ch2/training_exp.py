import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from k_armed import KArmedEnv, KArmedNonstationaryEnv, KArmedShiftedEnv
from agent import EpsGreedyAgent, EpsGreedyConstStepAgent, UCBAgent, GradientAgent

def train_greedy(time_steps=1000, k=10, num_env=2000, eps=0, env_type=KArmedEnv, agt_type=EpsGreedyAgent, q_init=0):
    env = env_type(k=k, num_env=num_env)
    agt = agt_type(eps=eps, k=k, num_agt=num_env, q_init=q_init)

    avg_rewards = np.zeros(time_steps, dtype=float)
    opt_counts = np.zeros(time_steps, dtype=float)
    for i in tqdm(range(time_steps), desc=f"training greedy agent (eps={eps}, Q1={q_init})"):
        # act
        a = agt.act()
        r = env.step(a)
        # learn
        agt.learn(a, r)
        # stats
        avg_r = r.mean()
        avg_rewards[i] = avg_r
        opt_counts[i] = np.sum(np.equal(a, env._get_opt_actions()))
    
    return avg_rewards, opt_counts

def train_ucb(time_steps=1000, k=10, num_env=2000, c=0, env_type=KArmedEnv, agt_type=UCBAgent, q_init=0):
    env = env_type(k=k, num_env=num_env)
    agt = agt_type(c=c, k=k, num_agt=num_env, q_init=q_init)

    avg_rewards = np.zeros(time_steps, dtype=float)
    opt_counts = np.zeros(time_steps, dtype=float)
    for i in tqdm(range(time_steps), desc=f"training UCB agent (c={c}, Q1={q_init})"):
        # act
        a = agt.act()
        r = env.step(a)
        # learn
        agt.learn(a, r)
        # stats
        avg_r = r.mean()
        avg_rewards[i] = avg_r
        opt_counts[i] = np.sum(np.equal(a, env._get_opt_actions()))
    
    return avg_rewards, opt_counts

def train_gradient(time_steps=1000, k=10, num_env=2000, alpha=0.1, env_type=KArmedEnv, agt_type=GradientAgent, use_baseline=True):
    env = env_type(k=k, num_env=num_env)
    agt = agt_type(step_size=alpha, k=k, num_agt=num_env, use_baseline=use_baseline)

    avg_rewards = np.zeros(time_steps, dtype=float)
    opt_counts = np.zeros(time_steps, dtype=float)
    for i in tqdm(range(time_steps), desc=f"training gradient agent (alpha={alpha}, baseline={use_baseline})"):
        # act
        a = agt.act()
        r = env.step(a)
        # learn
        agt.learn(a, r)
        # stats
        avg_r = r.mean()
        avg_rewards[i] = avg_r
        opt_counts[i] = np.sum(np.equal(a, env._get_opt_actions()))
    
    return avg_rewards, opt_counts

# Figure 2.2
def stationary():
    avg_rewards1, opt_counts1 = train_greedy(eps=0.1)
    avg_rewards2, opt_counts2 = train_greedy(eps=0.01)
    avg_rewards3, opt_counts3 = train_greedy(eps=0)
    fig, axes = plt.subplots(2, 1)
    # average rewards
    axes[0].plot(avg_rewards1, color='b')
    axes[0].plot(avg_rewards2, color='r')
    axes[0].plot(avg_rewards3, color='g')
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Average reward")
    axes[0].legend(labels=["eps=0.1", "eps=0.01", "eps=0(greedy)"])
    # optimal actions
    axes[1].plot(opt_counts1/2000, color='b')
    axes[1].plot(opt_counts2/2000, color='r')
    axes[1].plot(opt_counts3/2000, color='g')
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Optimal action(%)")
    axes[1].legend(labels=["eps=0.1", "eps=0.01", "eps=0(greedy)"])
    fig.savefig("stationary.png")

# Exercise 2.5
def nonstationary():
    avg_rewards1, opt_counts1 = train_greedy(eps=0.1, time_steps=10000, env_type=KArmedNonstationaryEnv, agt_type=EpsGreedyAgent)
    avg_rewards2, opt_counts2 = train_greedy(eps=0.1, time_steps=10000, env_type=KArmedNonstationaryEnv, agt_type=EpsGreedyConstStepAgent)
    fig, axes = plt.subplots(2, 1)
    # average rewards
    axes[0].plot(avg_rewards1, color='b')
    axes[0].plot(avg_rewards2, color='r')
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Average reward")
    axes[0].legend(labels=["eps=0.1(sample)", "eps=0.1(weighted)"])
    # optimal actions
    axes[1].plot(opt_counts1/2000, color='b')
    axes[1].plot(opt_counts2/2000, color='r')
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Optimal action(%)")
    axes[1].legend(labels=["eps=0.1(sample)", "eps=0.1(weighted)"])
    fig.savefig("nonstationary2.png")

# Figure 2.3
def init_values():
    avg_rewards1, opt_counts1 = train_greedy(eps=0, agt_type=EpsGreedyConstStepAgent, q_init=5)
    avg_rewards2, opt_counts2 = train_greedy(eps=0.1, agt_type=EpsGreedyConstStepAgent, q_init=0)
    fig, axes = plt.subplots(2, 1)
    # average rewards
    axes[0].plot(avg_rewards1, color='b')
    axes[0].plot(avg_rewards2, color='r')
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Average reward")
    axes[0].legend(labels=["eps=0, Q1=5", "eps=0.1, Q1=0"])
    # optimal actions
    axes[1].plot(opt_counts1/2000, color='b')
    axes[1].plot(opt_counts2/2000, color='r')
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Optimal action(%)")
    axes[1].legend(labels=["eps=0, Q1=5", "eps=0.1, Q1=0"])
    fig.savefig("init_values.png")

# Figure 2.4: UCB v.s. eps-greedy
def action_selection():
    avg_rewards1, opt_counts1 = train_ucb(c=2, agt_type=UCBAgent)
    avg_rewards2, opt_counts2 = train_ucb(c=1, agt_type=UCBAgent)
    avg_rewards3, opt_counts3 = train_greedy(eps=0.1, agt_type=EpsGreedyAgent)
    fig, axes = plt.subplots(2, 1)
    # average rewards
    axes[0].plot(avg_rewards1, color='b', alpha=0.5)
    axes[0].plot(avg_rewards2, color='r', alpha=0.5)
    axes[0].plot(avg_rewards3, color='g', alpha=0.5)
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Average reward")
    axes[0].legend(labels=["UCB c=2", "UCB c=1", "eps-greedy eps=0.1"])
    # optimal actions
    axes[1].plot(opt_counts1/2000, color='b', alpha=0.5)
    axes[1].plot(opt_counts2/2000, color='r', alpha=0.5)
    axes[1].plot(opt_counts3/2000, color='g', alpha=0.5)
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Optimal action(%)")
    axes[1].legend(labels=["UCB c=2", "UCB c=1", "eps-greedy eps=0.1"])
    fig.savefig("action_selection.png")

# Figure 2.5: w v.s. w/o baseline
def baseline():
    avg_rewards1, opt_counts1 = train_gradient(env_type=KArmedShiftedEnv, alpha=0.1, use_baseline=True, time_steps=1000)
    avg_rewards2, opt_counts2 = train_gradient(env_type=KArmedShiftedEnv, alpha=0.4, use_baseline=True, time_steps=1000)
    avg_rewards3, opt_counts3 = train_gradient(env_type=KArmedShiftedEnv, alpha=0.1, use_baseline=False, time_steps=1000)
    avg_rewards4, opt_counts4 = train_gradient(env_type=KArmedShiftedEnv, alpha=0.4, use_baseline=False, time_steps=1000)
    fig, axes = plt.subplots(2, 1)
    # average rewards
    axes[0].plot(avg_rewards1, color='b')
    axes[0].plot(avg_rewards2, color='r')
    axes[0].plot(avg_rewards3, color='g')
    axes[0].plot(avg_rewards4, color='k')
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Average reward")
    axes[0].legend(labels=["alpha=0.1(with baseline)", "alpha=0.4(with baseline)", "alpha=0.1(no baseline)", "alpha=0.4(no baseline)"])
    # optimal actions
    axes[1].plot(opt_counts1/2000, color='b')
    axes[1].plot(opt_counts2/2000, color='r')
    axes[1].plot(opt_counts3/2000, color='g')
    axes[1].plot(opt_counts4/2000, color='k')
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Optimal action(%)")
    axes[1].legend(labels=["alpha=0.1(with baseline)", "alpha=0.4(with baseline)", "alpha=0.1(no baseline)", "alpha=0.4(no baseline)"])
    fig.savefig("baseline.png")

if __name__ == '__main__':
    action_selection()