# Setup instructions
- pip install optuna
- pip install optuna-dashboard
- pip install supersuit
- pip install stable_baselines3[extra]
- pip install sumo-rl

# NB!!! Don't forget to replace the files that need to be replaced in your python sumo-rl pip package

# Set parameters
- Replace type with AEC OR Parallel
- Replace mdl with PPO or DQN
- Set the parameters at the top of each respective file
# Train instructions
- python Multi-Agent-Tuned-Train.py
 or For parallel computing with multi-cpu's
- python Multi-Agent-Train.py

# See hyper parameter tuning results
- optuna-dashboard ./optuna/multi-agent-tuned-using-optuma-{type}-{mdl}

# Simulate trained model, view mean reward and compare with random phases
- python Multi-Agent-Simulation
- Remember to select with model you want to simulate

# Plot graphs
- results are saved in the subplots.pdf file
## Summary of how the model improves over multi-episodes
- python plot.py -f ./results/train/results-Parallel-PPO_conn1
## Look at a specific episode
- python plot.py -f ./results/train/results-Parallel-PPO_conn1_ep2