# Setup instructions
- pip install optuna
- pip install optuna-dashboard
- pip install supersuit
- pip install stable_baselines3[extra]
- pip install sumo-rl

# Set parameters
- Replace type with AEC OR Parallel
- Replace mdl with PPO or DQN
- Set the parameters at the top of each respective file
# Train instructions
- python Multi-Agent-Tuned-Train.py

# See hyper parameter tuning results
- optuna-dashboard multi-agent-tuned-using-optuma-{type}-{mdl}

# Simulate trained model, view mean reward and compare with random phases
- python Multi-Agent-Tuned-Simulation
- Remember to select with model you want to simulate

# Plot graphs
- python plot.py -f results_conn1_ep2
- close the window to see additional results