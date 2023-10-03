## Code Setup
### Install SUMO
- You can download SUMO from: https://sumo.dlr.de/docs/Downloads.php

### Install git
- https://git-scm.com/downloads
- Tick the option tha enables git to be added to the path
- Select vscode as the default editor for git
![Alt text](./images/image-1.png)
![Alt text](./images/image.png)
### Install python 3
- https://www.python.org/downloads/
- Tick the option tha enables python to be added to the path
![Alt text](./images/image-2.png)
### Install VS c++
- https://aka.ms/vs/16/release/vc_redist.x64.exe

### Install dependencies
- pip install optuna
- pip install optuna-dashboard
- pip install supersuit
- pip install stable_baselines3[extra]
- pip install sumo-rl
- pip install seaborn

### Create a network file using the netedit application that comes with SUMO (optional)
- Define nodes, edges, traffic lights, etc. 
- Checkout the SUMO tutorials for more information: https://sumo.dlr.de/docs/Tutorials/ 
- Save the network file (.net.xml) 

### Create a route file within the network using netedit (Optional)
- This is done by defining traffic flows and routes within the demand section of netedit.
- Save the route file (.rou.xml)

### Replace the files that need to be replaced in your python sumo-rl pip package
- Type pip show sumo-rl in the terminal 
- Navigate to the sumo-rl folder
- Replace the files that need to be replaced (from the files to be replaced folder) in your python sumo-rl pip package
## Train instructions
- Navigate to the Investigation-Project folder
- Define parameters, model, type, and reward at the top of Multi-Agent-Train.py
- Type python Multi-Agent-Train.py in the terminal
- Note: ensure that your computer has the number of cpu's that you specify in the Multi-Agent-Train.py file

## Simulation instructions
- Define parameters, model, type, and reward at the top of Multi-Agent-Simulation.py that correspond to your trained model
- Type multi-agent-simulation.py in the terminal

## Plot graphs
- results are saved in the subplots.pdf file

### Repair your result files
- python repair.py -f ./results/train/

## Plot one or multiple simulations on the same graph
- python plot.py -f ./results/train/cologne8-PPO-ob11-default_conn1 ./results/greedy/cologne8-camera_conn1 ./etc

# Hyper parameter tuning
## Train instructions
- Navigate to the Investigation-Project folder
- Define parameters, model, type, and reward at the top of Multi-Agent-Tuned-Train.py
- Type python Multi-Agent-Tuned-Train.py in the terminal
- Note: ensure that your computer has the number of cpu's that you specify in the Multi-Agent-Tuned-Train.py file

## View results
- CD into the optuna folder
- Type optuna-dashboard sqlite:///{name}.sqlite3 in the terminal
- Remember to replace name with the name of the database you generated in the optuna folder

#### For a better explanation of the code, checkout the sumo-rl documentation
- https://lucasalegre.github.io/sumo-rl/documentation/sumo_env/
