# Multi_armed_bandits

The multi-armed bandit problem was implemented using Epsilon-Greedy and Upper Confidence Bound(UCB1) Algorithm.

# Libraries used
- numpy
- matplotlib

## Description
A 10-armed testbed was generated, reproducing some textbook figures based on this testbed, and performing further experimentation on bandit algorithms. The testbed contains 2000 trials with 2000 steps each, with the true action values q∗(a) for each action/arm in each bandit sampled from a normal distribution N(0,1). When a learning algorithm is applied, action A<sub>t</sub> at time t is selected and reward R<sub>t</sub> is sampled from normal distribution of true mean and variance 1. Performance is evaluated as 'Average Reward' or '%tprimal Action picked' at each step.



 
