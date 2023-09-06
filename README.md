# Code Setup

## Install SUMO
You can download SUMO from: https://sumo.dlr.de/docs/Downloads.php

## Create a network file using the netedit application that comes with SUMO
Define nodes, edges, traffic lights, etc.
Checkout the SUMO tutorials for more information: https://sumo.dlr.de/docs/Tutorials/
Save the network file (.net.xml)

## Create a route file within the network using netedit
This is done by defining traffic flows and routes within the demand section of netedit.
Save the route file (.rou.xml)

## Install sumo-rl
You can use 'pip install sumo-rl', this will also install pettingzoo and other dependencies.

## Paste the relative path to the network and route files you want to run
Within 'Single_Agent.py', replace the net_file and route_file arguments with your corresponsing files.
The example that is currently being used is 'Single.net.xml' and 'Single.route.xml' which is one intersection.

## For a better explanation of the code, checkout the sumo-rl documentation
https://lucasalegre.github.io/sumo-rl/documentation/sumo_env/
