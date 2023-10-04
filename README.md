# Multi_armed_bandits

The multi-armed bandit problem was implemented using Epsilon-Greedy and Upper Confidence Bound(UCB1) Algorithm.

## Libraries used
- numpy
- matplotlib

## Description
A 10-armed testbed was generated, reproducing some textbook figures based on this testbed, and performing further experimentation on bandit algorithms. The testbed contains 2000 trials with varying steps for each part, with the true action values q∗(a) for each action/arm in each bandit sampled from a normal distribution N(0,1). When a learning algorithm is applied, action A<sub>t</sub> at time t is selected and reward R<sub>t</sub> is sampled from normal distribution of true mean and variance 1. Performance is evaluated as 'Average Reward' or '%tprimal Action picked' at each step.

## Manually implementing the 10-armed testbed
In this, the q4 function in main.py forms a Bandit testbed using the 'BanditEnv' class. Then it pulls each arm many times and generates rewards from the true means. Finally it plots a violin plot to show the distribution of sampled rewards.

<img src="Images/Reward_distribution.png" alt="Dataset" width="700" height="400">

## Epsilon Greedy Algorithm
Implemented the ε-greedy algorithm with incremental updates. Used 2000 steps with 2000 independent runs. The function q6 in main.py forms the Bandit environment, takes a list of agents with varying epsilon values, and calculates the average reward and the number of times optimal action was taken. In the agent.py file, the EpsilonGreedy class chooses action and then updates it. Update happens dynamically if no step size is given. For the reward plot, an extra constant upper bound line is added corresponding to the best possible average performance in the trials, based on the known true expected rewards q∗(a). 
For each reward curve, confidence bands are also plotted corresponding to (1.96× standard error) of the rewards. The standard error of the mean is defined as the standard deviation divided by √n: (σ/√n)

<img src="Images/e-greedy_rewards.png" alt="Dataset" width="700" height="400">
<img src="Images/e-greedy-action.png" alt="Dataset" width="700" height="400">


 
