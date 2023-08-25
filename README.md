# Bernoulli Multi-Armed Bandit Simulation

This repository contains a simulation of various Bernoulli Multi-Armed Bandit (MAB) methods. Bernoulli MAB is a classic problem in probability and decision theory, often used to model scenarios where an agent needs to decide which arm (option) of a multi-armed bandit machine to pull in order to maximize cumulative reward over time.

In this simulation, we explore and compare the performance of different strategies/algorithms for solving the Bernoulli MAB problem. The following algorithms are implemented and analyzed:

1. **Epsilon-Greedy**: A simple algorithm that balances exploration and exploitation by choosing the arm with the highest observed reward with probability (1 - ε), and a random arm with probability ε.

2. **UCB (Upper Confidence Bound)**: An algorithm that selects the arm with the highest upper confidence bound, which is calculated based on the observed rewards and confidence intervals.

3. **Thompson Sampling**: A probabilistic algorithm that maintains a distribution for each arm's expected reward and samples from these distributions to decide which arm to pull.


## Installation

1. Clone this repository:
   git clone https://github.com/ndominutti/mab_simulation.git
   cd your-repo

## Results
Simulation's results are stored in MAB_simulation.ipynb, but as github does not render interactive plotly charts correctly, we recommend to use this <a link='https://chart-studio.plotly.com/~NDOMINUTTI/89.embed'>link</a> instead.
