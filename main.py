import matplotlib.pyplot as plt
from agent import *
from env import BanditEnv
from tqdm import trange
import numpy as np

def q4(k: int, num_samples: int):
    """Q4

    Structure:
        1. Create multi-armed bandit env
        2. Pull each arm `num_samples` times and record the rewards
        3. Plot the rewards (e.g. violinplot, stripplot)

    Args:
        k (int): Number of arms in bandit environment
        num_samples (int): number of samples to take for each arm
    """
    env = BanditEnv(k=k)
    env.reset()
    total_rewards=[]

    for i in range(k):
        j = 0
        rewards_per_arm=[]
        while j<num_samples:
            rewards_per_arm.append(generate_reward(env.means[i]))
            j+=1
        total_rewards.append(rewards_per_arm)

    plt.violinplot(total_rewards,showmeans=True)
    plt.axhline(y=0,linestyle='--',color='k')
    plt.xlabel('Action')
    plt.ylabel('Reward distribution')
    plt.xticks(range(1,k+1))
    plt.show()

def generate_reward(mean):
    ''' This function takes in the generated mean of the corresponding arm and returns the reward

    Args: mean (float)
    '''
    reward= np.random.normal(mean,1)
    return reward

def q6(k: int, trials: int, steps: int):
    """Q6

    Implement epsilon greedy bandit agents with an initial estimate of 0

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    # TODO initialize env and agents here
    env = BanditEnv(k=k)
    agents=[EpsilonGreedy(k,0,0),EpsilonGreedy(k,0,0.01),EpsilonGreedy(k,0,0.1)]
    # agents=[EpsilonGreedy(k,0,0)]
    avg_rewards = np.zeros((len(agents), trials, steps))
    optimal_actions=np.zeros((len(agents),trials,steps))
    max_q=np.zeros((len(agents),trials,steps))
    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        true_means=env.means
        max_q_per_trial=np.max(true_means)
        optimal_action_trial=np.argmax(true_means)
        for ind,agent in enumerate(agents):
            agent.reset()
            s=0

            # TODO For each trial, perform specified number of steps for each type of agent
            while s<steps:
                action=agent.choose_action()
                reward=env.step(action)
                agent.update(action,reward)
                avg_rewards[ind][t][s] = reward
                max_q[ind][t][s]=max_q_per_trial
                if action==optimal_action_trial:
                    optimal_actions[ind][t][s]=1
                s+=1

    std_err = np.std(avg_rewards, axis=1) / np.sqrt(trials)
    optimal_action_per=np.mean(optimal_actions,axis=1)*100
    return avg_rewards,optimal_action_per,max_q,std_err

def q7(k: int, trials: int, steps: int):
    """Q7

    Compare epsilon greedy bandit agents and UCB agents

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    # TODO initialize env and agents here
    env = BanditEnv(k=k)
    agents = [EpsilonGreedy(k,0,0),EpsilonGreedy(k,5,0),EpsilonGreedy(k,0,0.1),EpsilonGreedy(k,5,0.1),UCB(k,0,2,0.1)]
    avg_rewards = np.zeros((len(agents), trials, steps))
    optimal_actions = np.zeros((len(agents), trials, steps))
    max_q = np.zeros((len(agents), trials, steps))
    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        true_means = env.means
        max_q_per_trial = np.max(true_means)
        optimal_action_trial = np.argmax(true_means)
        for ind,agent in enumerate(agents):
            agent.reset()
            s=0
            # TODO For each trial, perform specified number of steps for each type of agent
            while s < steps:
                action = agent.choose_action()
                reward = env.step(action)
                agent.update(action, reward)
                avg_rewards[ind][t][s] = reward
                max_q[ind][t][s] = max_q_per_trial
                if action == optimal_action_trial:
                    optimal_actions[ind][t][s] = 1
                s += 1

    std_err = np.std(avg_rewards, axis=1) / np.sqrt(trials)
    optimal_action_per = np.mean(optimal_actions, axis=1) * 100
    return avg_rewards, optimal_action_per, max_q, std_err


#Displays confidence bands along with average rewards
def confidence_bands_plot(steps, trials, avg_rewards, std_err, max_q,color,label):
    avg_rewards_over_trials=np.mean(avg_rewards, axis=1)
    upper_conf_bound = avg_rewards_over_trials+1.96*std_err
    lower_conf_bound = avg_rewards_over_trials-1.96*std_err

    #calculations for upper bound curve
    std_err_ub = np.std(max_q, axis=1) / np.sqrt(trials)
    max_q_over_trials=np.mean(max_q, axis=1)
    upper_conf_bound_ub= max_q_over_trials + 1.96 * std_err_ub
    lower_conf_bound_ub= max_q_over_trials - 1.96 * std_err_ub

    for ind,rewards in enumerate(avg_rewards_over_trials):
        plt.plot(range(0,steps),rewards,color=color[ind], label=label[ind])
        plt.fill_between(range(steps),lower_conf_bound[ind],upper_conf_bound[ind],alpha=0.2)

    plt.plot(range(0,steps),max_q_over_trials[0], color='black', linestyle='--', label='Upper bound')
    plt.fill_between(range(0,steps),lower_conf_bound_ub[0],upper_conf_bound_ub[0],alpha=0.2)
    plt.title('Average rewards with Confidence intervals')
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()
    plt.show()

def optimal_action_plot(steps,trials,optimal_action_per,color,label):
    for ind,actions in enumerate(optimal_action_per):
        plt.plot(range(0,steps),actions,color=color[ind],label=label[ind])
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.legend()
    plt.show()


def main():
    # TODO run code for all questions
    q4(10,1000)
    q6_steps=2000
    q6_trials=2000
    q6_avg_greedy_rewards,q6_optimal_action_per,q6_upper_bound,q6_std_err=q6(10,q6_trials,q6_steps)
    q6_labels=['epsilon=0', 'epsilon=0.01', 'epsilon=0.1']
    q6_colors= ['green', 'red', 'blue']
    #Plotting confidence bands and average plots
    confidence_bands_plot(q6_steps,q6_trials,q6_avg_greedy_rewards,q6_std_err,q6_upper_bound,q6_colors,q6_labels)
    # Plotting optimal action
    optimal_action_plot(q6_steps,q6_trials,q6_optimal_action_per,q6_colors,q6_labels)

    q7_steps=1000
    q7_trials=2000
    q7_avg_greedy_rewards, q7_optimal_action_per, q7_upper_bound, q7_std_err = q7(10, q7_trials, q7_steps)
    q7_labels = ['Q1=0,eps=0', 'Q1=5,eps=0','Q1=0,eps=0.1','Q1=5,eps=0.1','UCB,c=2']
    q7_colors = ['green', 'red', 'blue','magenta','brown']

    # Plotting confidence bands and average plots
    confidence_bands_plot(q7_steps,q7_trials,q7_avg_greedy_rewards,q7_std_err,q7_upper_bound,q7_colors,q7_labels)
    #Plotting optimal action
    optimal_action_plot(q7_steps,q7_trials,q7_optimal_action_per,q7_colors,q7_labels)

if __name__ == "__main__":
    main()
