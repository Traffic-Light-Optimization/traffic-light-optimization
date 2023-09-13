# Setup instructions
- pip install optuna
- pip install optuna-dashboard
- pip install supersuit
- pip install stable_baselines3[extra]
- pip install sumo-rl

# Train instructions
- python Multi-Agent-Tuned-Train

# See hyper parameter tuning results
- Replace type with AEC OR Parallel
- Replace mdl with PPO or DQN
- optuna-dashboard multi-agent-tuned-using-optuma-{type}-{mdl}

# Simulate trained model, view mean reward and compare with random phases
- python Multi-Agent-Tuned-Simulation
- Remember to select with model you want to simulate

# Plot graphs
- Replace type with AEC OR Parallel
- Replace mdl with PPO or DQN
- python plot.py -f CSV/{type}/{mdl}/results_conn0_ep2
- Replace with different episodes