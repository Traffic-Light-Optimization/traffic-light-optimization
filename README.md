# Code Setup

### Install SUMO
- You can download SUMO from: https://sumo.dlr.de/docs/Downloads.php

### Create a network file using the netedit application that comes with SUMO
- Define nodes, edges, traffic lights, etc. 
- Checkout the SUMO tutorials for more information: https://sumo.dlr.de/docs/Tutorials/ 
- Save the network file (.net.xml) 

### Create a route file within the network using netedit
- This is done by defining traffic flows and routes within the demand section of netedit.
- Save the route file (.rou.xml)

### Install sumo-rl
- You can use 'pip install sumo-rl', this will also install pettingzoo and other dependencies.

### Install stable-baselines3
- You can use 'pip install stable_baselines3'

### Paste the relative path to the network and route files you want to run
- Within the training and simulation code, replace the net_file and route_file arguments with your corresponsing files.
- The example that is currently being used is a single intersection from the nets folder.

### Training and Simulating
- Run the training files first to train and save a model as a zip file
- You can then run the simulation files to load those saved models and evaluate them

### For a better explanation of the code, checkout the sumo-rl documentation
- https://lucasalegre.github.io/sumo-rl/documentation/sumo_env/
